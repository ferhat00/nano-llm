"""Training loop: AMP fp16, grad accum, cosine+warmup, atomic resume, optional DDP."""

from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .data import get_batch, prepare
from .model import GPT, GPTConfig
from .tokenizer import load_tokenizer
from .utils import (
    banner,
    find_latest_checkpoint,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
    snapshot_rng,
)


# ---------------------------------------------------------------------------
# Optimizer + LR schedule
# ---------------------------------------------------------------------------

def configure_optimizer(model: nn.Module, weight_decay: float, lr: float,
                        betas: tuple[float, float], device: str) -> torch.optim.AdamW:
    """AdamW with weight-decay parameter grouping (only >=2D tensors decay)."""
    decay, no_decay = [], []
    seen = set()
    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = device.startswith("cuda")
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)


def get_lr(step: int, *, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_cfg_from(cfg: dict, vocab_size: int) -> GPTConfig:
    m = cfg["model"]
    return GPTConfig(
        vocab_size=vocab_size,
        block_size=int(m["block_size"]),
        n_layer=int(m["n_layer"]),
        n_head=int(m["n_head"]),
        n_embd=int(m["n_embd"]),
        dropout=float(m.get("dropout", 0.0)),
        mlp_hidden=m.get("mlp_hidden"),
        rope_base=float(m.get("rope_base", 10000.0)),
    )


@torch.no_grad()
def estimate_loss(model: nn.Module, data_dir: str, *, batch_size: int, block_size: int,
                  device: str, eval_iters: int) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data_dir, split, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = float(losses.mean())
    model.train()
    return out


def _sample_text(model: nn.Module, tokenizer, *, prompt: str, max_new_tokens: int,
                 temperature: float, top_k: int | None, top_p: float | None,
                 device: str) -> str:
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.eos_id]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    core = model.module if isinstance(model, DDP) else model
    out = core.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=tokenizer.eos_id,
    )
    return tokenizer.decode(out[0].tolist())


# ---------------------------------------------------------------------------
# Single-process / DDP-rank worker
# ---------------------------------------------------------------------------

def _train_loop(cfg: dict, out_dir: str, data_dir: str, tokenizer_path: str,
                rank: int = 0, world_size: int = 1) -> None:
    is_main = (rank == 0)
    train_cfg = cfg["train"]
    device = str(train_cfg["device"])
    if device.startswith("cuda") and world_size > 1:
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)

    # ---- Tokenizer + model ----
    tokenizer = load_tokenizer(tokenizer_path)
    model_cfg = _model_cfg_from(cfg, vocab_size=tokenizer.vocab_size)
    model = GPT(model_cfg).to(device)
    if is_main:
        print(f"[model] params={model.num_params/1e6:.2f}M  cfg={model_cfg}", flush=True)

    if world_size > 1:
        model = DDP(model, device_ids=[rank] if device.startswith("cuda") else None)

    # ---- Optimizer / scaler ----
    optimizer = configure_optimizer(
        model,
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
        lr=float(train_cfg["learning_rate"]),
        betas=(float(train_cfg.get("beta1", 0.9)), float(train_cfg.get("beta2", 0.95))),
        device=device,
    )

    precision = str(train_cfg.get("precision", "fp32")).lower()
    use_amp = precision == "fp16" and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp else nullcontext()
    )

    # ---- Resume ----
    resume_from = cfg.get("resume_from") or find_latest_checkpoint(out_dir)
    start_step = 0
    best_val = float("inf")
    if resume_from and os.path.exists(resume_from):
        if is_main:
            print(f"[resume] loading {resume_from}", flush=True)
        ckpt = load_checkpoint(resume_from, map_location=device)
        core = model.module if isinstance(model, DDP) else model
        core.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if use_amp and "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt["step"]) + 1
        best_val = float(ckpt.get("best_val_loss", best_val))
        if "rng" in ckpt:
            try:
                restore_rng(ckpt["rng"])
            except Exception as e:
                print(f"[resume] RNG restore failed ({e}); continuing with fresh RNG", flush=True)
        if is_main:
            print(f"[resume] continuing from step {start_step}", flush=True)
    elif is_main:
        print("[resume] no checkpoint found; starting from scratch", flush=True)

    # ---- Training loop ----
    max_steps = int(train_cfg["max_steps"])
    micro_batch = int(train_cfg["batch_size"])
    grad_accum = int(train_cfg.get("grad_accum", 1))
    block_size = int(cfg["model"]["block_size"])
    eval_interval = int(train_cfg.get("eval_interval", 100))
    eval_iters = int(train_cfg.get("eval_iters", 20))
    sample_interval = int(train_cfg.get("sample_interval", eval_interval))
    ckpt_interval = int(train_cfg.get("checkpoint_interval", eval_interval))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    keep_last_k = train_cfg.get("keep_last_k", 3)
    lr_max = float(train_cfg["learning_rate"])
    lr_min = float(train_cfg.get("min_lr", lr_max * 0.1))
    warmup = int(train_cfg.get("warmup_steps", 0))

    if start_step >= max_steps:
        if is_main:
            print(f"[train] start_step {start_step} >= max_steps {max_steps}; nothing to do.", flush=True)
        return

    model.train()
    t0 = time.time()
    for step in range(start_step, max_steps):
        lr = get_lr(step, warmup_steps=warmup, max_steps=max_steps, lr=lr_max, min_lr=lr_min)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(grad_accum):
            x, y = get_batch(str(out_dir if False else data_dir), "train",
                             micro_batch, block_size, device)
            with autocast_ctx:
                _, loss = model(x, y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if is_main and (step % 10 == 0 or step == max_steps - 1):
            dt = time.time() - t0
            t0 = time.time()
            print(f"[step {step:6d}/{max_steps}] loss={loss_accum:.4f}  lr={lr:.2e}  dt={dt:.2f}s", flush=True)

        # ---- Periodic eval ----
        if is_main and eval_interval > 0 and (step + 1) % eval_interval == 0:
            losses = estimate_loss(
                model.module if isinstance(model, DDP) else model,
                data_dir,
                batch_size=micro_batch, block_size=block_size,
                device=device, eval_iters=eval_iters,
            )
            ppl_val = math.exp(min(losses["val"], 20))  # cap to avoid overflow
            print(
                f"[eval  {step+1:6d}/{max_steps}] train={losses['train']:.4f}  "
                f"val={losses['val']:.4f}  ppl_val={ppl_val:.1f}", flush=True
            )
            if losses["val"] < best_val:
                best_val = losses["val"]

        # ---- Periodic sample ----
        if is_main and sample_interval > 0 and (step + 1) % sample_interval == 0:
            sample_cfg = cfg.get("sample", {})
            text = _sample_text(
                model, tokenizer,
                prompt=str(sample_cfg.get("prompt", "Once upon a time")),
                max_new_tokens=int(sample_cfg.get("max_new_tokens", 80)),
                temperature=float(sample_cfg.get("temperature", 0.8)),
                top_k=sample_cfg.get("top_k"),
                top_p=sample_cfg.get("top_p"),
                device=device,
            )
            print(f"[sample {step+1:6d}] {text!r}", flush=True)
            model.train()

        # ---- Periodic checkpoint ----
        if is_main and ckpt_interval > 0 and (step + 1) % ckpt_interval == 0:
            core = model.module if isinstance(model, DDP) else model
            state = {
                "model": core.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "step": step,
                "best_val_loss": best_val,
                "cfg": cfg,
                "rng": snapshot_rng(),
            }
            path = save_checkpoint(state, out_dir, step + 1, keep_last_k=keep_last_k)
            print(f"[ckpt  {step+1:6d}] saved -> {path}", flush=True)

    # ---- Final checkpoint if last step wasn't a save step ----
    if is_main and ((max_steps) % max(ckpt_interval, 1) != 0):
        core = model.module if isinstance(model, DDP) else model
        state = {
            "model": core.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "step": max_steps - 1,
            "best_val_loss": best_val,
            "cfg": cfg,
            "rng": snapshot_rng(),
        }
        path = save_checkpoint(state, out_dir, max_steps, keep_last_k=keep_last_k)
        print(f"[ckpt  {max_steps:6d}] final -> {path}", flush=True)


# ---------------------------------------------------------------------------
# DDP wrapper
# ---------------------------------------------------------------------------

def _ddp_worker(rank: int, world_size: int, cfg: dict, out_dir: str, data_dir: str,
                tokenizer_path: str) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        _train_loop(cfg, out_dir, data_dir, tokenizer_path, rank=rank, world_size=world_size)
    finally:
        dist.destroy_process_group()


def train(cfg: dict, out_dir: str, data_dir: str, tokenizer_path: str) -> None:
    """Top-level entry. Routes to single-process or mp.spawn DDP based on config."""
    train_cfg = cfg["train"]
    distributed = bool(train_cfg.get("distributed", False))
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if distributed and n_gpu > 1:
        banner(f"TRAIN (DDP, world_size={n_gpu})", char="-")
        mp.spawn(
            _ddp_worker,
            args=(n_gpu, cfg, out_dir, data_dir, tokenizer_path),
            nprocs=n_gpu,
            join=True,
        )
    else:
        if distributed and n_gpu <= 1:
            print(f"[train] distributed requested but only {n_gpu} GPU available; using single-process path.",
                  flush=True)
        banner("TRAIN (single-process)", char="-")
        _train_loop(cfg, out_dir, data_dir, tokenizer_path, rank=0, world_size=1)
