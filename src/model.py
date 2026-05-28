"""Decoder-only transformer, written by hand.

Design choices (all driven by the Kaggle 2xT4 constraint):
- RMSNorm + pre-norm blocks (cheaper than LayerNorm, no bias).
- RoPE applied to Q/K (no learned positional embedding, no extra params).
- SwiGLU MLP (Llama-style; ~50% larger MLP per layer but better quality/FLOP).
- Attention via F.scaled_dot_product_attention with is_causal=True. Turing
  (T4) doesn't support FlashAttention-2; SDPA picks the memory-efficient
  backend automatically.
- Weight-tied token embedding and output projection.
- No transformers-library imports anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    # MLP hidden dim. If None we use round_to_multiple(8/3 * n_embd, 64).
    mlp_hidden: int | None = None
    # RoPE base; 10000 is standard.
    rope_base: float = 10000.0

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        if self.mlp_hidden is None:
            target = int(8 / 3 * self.n_embd)
            self.mlp_hidden = ((target + 63) // 64) * 64  # round up to multiple of 64

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for numerical stability under AMP.
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(dtype) * self.weight


def _build_rope_cache(seq_len: int, head_dim: int, base: float, device, dtype):
    """Return (cos, sin) of shape (seq_len, head_dim) for RoPE.

    We use the GPT-NeoX / Llama convention where the rotation is applied to
    (x1, x2) pairs given by the first and second halves of the head dim.
    """
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # (seq_len, half)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).to(dtype)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).to(dtype)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: (B, n_head, T, head_dim). cos/sin: (T, head_dim).
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)
        # (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        # SDPA picks mem-efficient / math backend; FA-2 is unsupported on Turing.
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        # (B, n_head, T, head_dim) -> (B, T, C)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(attn)


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        h = cfg.mlp_hidden
        self.w_gate = nn.Linear(cfg.n_embd, h, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, h, bias=False)
        self.w_down = nn.Linear(h, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying.
        self.lm_head.weight = self.tok_emb.weight

        # RoPE cache (buffer so it moves with .to(device) but isn't a param).
        cos, sin = _build_rope_cache(
            seq_len=cfg.block_size, head_dim=cfg.head_dim,
            base=cfg.rope_base, device="cpu", dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 trick: scale residual-stream projections by 1/sqrt(2*n_layer).
        scale = 1.0 / math.sqrt(2.0 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.w_down.weight"):
                with torch.no_grad():
                    p.mul_(scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def num_params(self) -> int:
        # Tied weights only count once.
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.cfg.block_size}")
        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None,
                 top_p: float | None = None,
                 eos_id: int | None = None) -> torch.Tensor:
        """Sampling with temperature / top-k / top-p (nucleus). Greedy if temp==0."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, V)

            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    kth_vals = torch.topk(logits, k, dim=-1).values[..., -1, None]
                    logits = torch.where(logits < kth_vals, torch.full_like(logits, float("-inf")), logits)
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cum = probs.cumsum(dim=-1)
                    # Keep tokens until cumulative prob crosses top_p (always keep the first).
                    mask = cum > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx
