"""Single entry point for the nano-llm pipeline.

Stages (in order, each idempotent unless --force):
  1. tokenizer  -- train or load BPE; save tokenizer.json
  2. data       -- pre-tokenize TinyStories to uint16 memmap (.bin files)
  3. train      -- training loop with auto-resume from latest checkpoint
  4. eval       -- final val loss + perplexity on the latest checkpoint
  5. sample     -- generate text from the latest checkpoint

Usage:
  python orchestrate.py --config configs/smoke.yaml
  python orchestrate.py --config configs/small.yaml --force
  python orchestrate.py --config configs/smoke.yaml --stage train
  python orchestrate.py --config configs/small.yaml --train.max_steps=500 --model.n_layer=4
  python orchestrate.py --config configs/small.yaml --resume-from /path/to/ckpt.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

# Make `from src...` imports work when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from src import data as data_mod
from src import sample as sample_mod
from src import tokenizer as tok_mod
from src import train as train_mod
from src.utils import (
    banner,
    dump_config,
    get_device_info,
    load_config,
    require_cuda_if_requested,
    set_seed,
)


STAGES = ("tokenizer", "data", "train", "eval", "sample")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(
        description="nano-llm orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", required=True, help="path to YAML config")
    ap.add_argument("--stage", choices=("all", *STAGES), default="all",
                    help="run only this stage; 'all' runs the full pipeline")
    ap.add_argument("--force", action="store_true",
                    help="ignore stage idempotency (retrain tokenizer, re-tokenize data)")
    ap.add_argument("--resume-from", default=None,
                    help="override config.resume_from with this checkpoint path")
    # Any remaining --key.subkey=value tokens become config overrides.
    args, unknown = ap.parse_known_args(argv)

    overrides: list[str] = []
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if tok.startswith("--") and "=" in tok:
            overrides.append(tok[2:])
            i += 1
        elif tok.startswith("--") and i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
            overrides.append(f"{tok[2:]}={unknown[i + 1]}")
            i += 2
        else:
            raise SystemExit(f"unrecognized argument: {tok!r} (expected --key.subkey=value)")
    return args, overrides


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def _tokenizer_path(out_dir: str) -> str:
    return os.path.join(out_dir, "tokenizer.json")


def _data_dir(cfg: dict) -> str:
    return cfg.get("data", {}).get("data_dir") or os.path.join(cfg["out_dir"], "data")


def _bpe_text_iter(cfg_data: dict, n_docs: int):
    """Yield up to n_docs strings from the dataset's train split."""
    from datasets import load_dataset
    ds = load_dataset(cfg_data["dataset"], split="train")
    n = min(n_docs, len(ds))
    ds = ds.select(range(n))
    field = cfg_data.get("text_field", "text")
    for row in ds:
        yield row[field]


def stage_tokenizer(cfg: dict, force: bool) -> str:
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    tok_path = _tokenizer_path(out_dir)
    cfg_tok = cfg["tokenizer"]

    if os.path.exists(tok_path) and not force and cfg_tok.get("type", "bpe") == "bpe":
        print(f"[tokenizer] found {tok_path} (skip; pass --force to retrain)", flush=True)
        return tok_path

    if cfg_tok.get("type", "bpe") == "gpt2":
        print("[tokenizer] reusing GPT-2 tokenizer (downloading if needed)", flush=True)
        wrapped = tok_mod.load_gpt2()
        wrapped.save(tok_path)
        print(f"[tokenizer] saved -> {tok_path}  vocab={wrapped.vocab_size}", flush=True)
        return tok_path

    n_docs = int(cfg_tok.get("train_subset_docs", 50000))
    print(f"[tokenizer] training BPE vocab={cfg_tok['vocab_size']} on {n_docs} docs", flush=True)
    t0 = time.time()
    wrapped = tok_mod.train_bpe(
        _bpe_text_iter(cfg["data"], n_docs),
        vocab_size=int(cfg_tok["vocab_size"]),
        save_path=tok_path,
    )
    print(f"[tokenizer] saved -> {tok_path}  vocab={wrapped.vocab_size}  ({time.time()-t0:.1f}s)",
          flush=True)
    return tok_path


def stage_data(cfg: dict, tokenizer_path: str, force: bool) -> str:
    data_dir = _data_dir(cfg)
    train_bin = os.path.join(data_dir, "train.bin")
    val_bin = os.path.join(data_dir, "val.bin")
    if os.path.exists(train_bin) and os.path.exists(val_bin) and not force:
        print(f"[data] found {train_bin} and {val_bin} (skip; pass --force to rebuild)", flush=True)
        return data_dir
    tokenizer = tok_mod.load_tokenizer(tokenizer_path)
    info = data_mod.prepare(cfg["data"], tokenizer, data_dir, force=force)
    for split, meta in info.items():
        flag = "rebuilt" if meta["rebuilt"] else "kept"
        print(f"[data] {split}: {meta['tokens']:,} tokens at {meta['path']} ({flag})", flush=True)
    return data_dir


def stage_train(cfg: dict, data_dir: str, tokenizer_path: str) -> None:
    out_dir = cfg["out_dir"]
    train_mod.train(cfg, out_dir, data_dir, tokenizer_path)


def stage_eval(cfg: dict, data_dir: str, tokenizer_path: str) -> None:
    """Compute val loss + perplexity on the latest checkpoint."""
    from src.utils import find_latest_checkpoint
    out_dir = cfg["out_dir"]
    ckpt = cfg.get("resume_from") or find_latest_checkpoint(out_dir)
    if not ckpt:
        print("[eval] no checkpoint; skipping", flush=True)
        return
    device = str(cfg["train"]["device"])
    model, tokenizer, ck_cfg = sample_mod.load_model_and_tokenizer(ckpt, tokenizer_path, device)
    block_size = int(ck_cfg["model"]["block_size"])
    batch_size = int(cfg["train"].get("eval_batch_size", cfg["train"]["batch_size"]))
    eval_iters = int(cfg["train"].get("eval_iters", 20))
    losses = train_mod.estimate_loss(
        model, data_dir,
        batch_size=batch_size, block_size=block_size,
        device=device, eval_iters=eval_iters,
    )
    ppl = math.exp(min(losses["val"], 20))
    print(f"[eval] ckpt={ckpt}  train_loss={losses['train']:.4f}  "
          f"val_loss={losses['val']:.4f}  val_ppl={ppl:.2f}", flush=True)


def stage_sample(cfg: dict, tokenizer_path: str) -> None:
    out_dir = cfg["out_dir"]
    sample_mod.run_from_config(cfg, out_dir, tokenizer_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args, overrides = parse_cli(sys.argv[1:] if argv is None else argv)
    cfg = load_config(args.config, overrides)
    if args.resume_from is not None:
        cfg["resume_from"] = args.resume_from

    banner("nano-llm")
    print(get_device_info(), flush=True)
    print("\n--- resolved config ---")
    print(dump_config(cfg), flush=True)

    require_cuda_if_requested(cfg)
    set_seed(int(cfg.get("seed", 1337)))

    out_dir = cfg["out_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    selected = STAGES if args.stage == "all" else (args.stage,)
    state: dict = {}

    if "tokenizer" in selected:
        banner("STAGE 1 / tokenizer", char="-")
        state["tokenizer_path"] = stage_tokenizer(cfg, force=args.force)
    else:
        state["tokenizer_path"] = _tokenizer_path(out_dir)

    if "data" in selected:
        banner("STAGE 2 / data", char="-")
        state["data_dir"] = stage_data(cfg, state["tokenizer_path"], force=args.force)
    else:
        state["data_dir"] = _data_dir(cfg)

    if "train" in selected:
        banner("STAGE 3 / train", char="-")
        stage_train(cfg, state["data_dir"], state["tokenizer_path"])

    if "eval" in selected:
        banner("STAGE 4 / eval", char="-")
        stage_eval(cfg, state["data_dir"], state["tokenizer_path"])

    if "sample" in selected:
        banner("STAGE 5 / sample", char="-")
        stage_sample(cfg, state["tokenizer_path"])

    banner("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
