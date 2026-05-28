"""Generation from a saved checkpoint. Callable from orchestrate.py and as a CLI."""

from __future__ import annotations

import argparse
import os

import torch

from .model import GPT, GPTConfig
from .tokenizer import load_tokenizer
from .utils import find_latest_checkpoint, load_checkpoint


def load_model_and_tokenizer(ckpt_path: str, tokenizer_path: str, device: str):
    tokenizer = load_tokenizer(tokenizer_path)
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    m = cfg["model"]
    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=int(m["block_size"]),
        n_layer=int(m["n_layer"]),
        n_head=int(m["n_head"]),
        n_embd=int(m["n_embd"]),
        dropout=float(m.get("dropout", 0.0)),
        mlp_hidden=m.get("mlp_hidden"),
        rope_base=float(m.get("rope_base", 10000.0)),
    )
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer, cfg


def generate(model, tokenizer, *, prompt: str, max_new_tokens: int,
             temperature: float = 0.8, top_k: int | None = 40, top_p: float | None = 0.9,
             device: str = "cpu") -> str:
    ids = tokenizer.encode(prompt) if prompt else [tokenizer.eos_id]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=tokenizer.eos_id,
    )
    return tokenizer.decode(out[0].tolist())


def run_from_config(cfg: dict, out_dir: str, tokenizer_path: str) -> None:
    """Used by orchestrate.py's `sample` stage."""
    ckpt_path = cfg.get("resume_from") or find_latest_checkpoint(out_dir)
    if not ckpt_path:
        raise FileNotFoundError(f"no checkpoint found in {out_dir!r}; train first.")
    device = str(cfg["train"]["device"])
    model, tokenizer, _ = load_model_and_tokenizer(ckpt_path, tokenizer_path, device)
    sample_cfg = cfg.get("sample", {})
    prompts = sample_cfg.get("prompts") or [sample_cfg.get("prompt", "Once upon a time")]
    for i, prompt in enumerate(prompts):
        text = generate(
            model, tokenizer,
            prompt=str(prompt),
            max_new_tokens=int(sample_cfg.get("max_new_tokens", 100)),
            temperature=float(sample_cfg.get("temperature", 0.8)),
            top_k=sample_cfg.get("top_k"),
            top_p=sample_cfg.get("top_p"),
            device=device,
        )
        print(f"\n--- sample {i+1} (prompt={prompt!r}) ---\n{text}\n", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Sample from a nano-llm checkpoint.")
    ap.add_argument("--ckpt", required=True, help="path to a ckpt_step*.pt file")
    ap.add_argument("--tokenizer", required=True, help="path to tokenizer.json")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, tokenizer, _ = load_model_and_tokenizer(args.ckpt, args.tokenizer, args.device)
    text = generate(
        model, tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
    )
    print(text)


if __name__ == "__main__":
    main()
