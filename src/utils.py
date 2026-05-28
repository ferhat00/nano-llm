"""Config loading, seeding, checkpoint I/O, logging helpers."""

from __future__ import annotations

import glob
import os
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce_scalar(s: str) -> Any:
    # YAML scalar parser handles bool/int/float/null/str cleanly.
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def _parse_overrides(pairs: list[str]) -> dict:
    """Turn ['model.n_layer=4', 'train.device=cpu'] into a nested dict."""
    out: dict = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"override must be key=value, got {p!r}")
        key, val = p.split("=", 1)
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce_scalar(val)
    return out


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg = _deep_merge(cfg, _parse_overrides(overrides))
    return cfg


def dump_config(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device info
# ---------------------------------------------------------------------------

def get_device_info() -> str:
    lines = [f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}"]
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        lines.append(f"cuda devices ({n}):")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            lines.append(f"  [{i}] {props.name}  vram={vram_gb:.1f} GB  cc={props.major}.{props.minor}")
    return "\n".join(lines)


def require_cuda_if_requested(cfg: dict) -> None:
    """Fail loudly if the config asks for CUDA but it's unavailable."""
    device = str(cfg.get("train", {}).get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"config requests train.device={device!r} but CUDA is unavailable. "
            "Override with --train.device=cpu for a CPU run."
        )


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

_CKPT_PATTERN = re.compile(r"^ckpt_step(\d+)\.pt$")


def list_checkpoints(out_dir: str | os.PathLike) -> list[tuple[int, str]]:
    """Return [(step, path), ...] sorted ascending by step."""
    out = []
    if not os.path.isdir(out_dir):
        return out
    for name in os.listdir(out_dir):
        m = _CKPT_PATTERN.match(name)
        if m:
            out.append((int(m.group(1)), os.path.join(out_dir, name)))
    out.sort(key=lambda t: t[0])
    return out


def find_latest_checkpoint(out_dir: str | os.PathLike) -> str | None:
    ckpts = list_checkpoints(out_dir)
    return ckpts[-1][1] if ckpts else None


def save_checkpoint(state: dict, out_dir: str | os.PathLike, step: int,
                    keep_last_k: int | None = None) -> str:
    """Atomic save: write to .tmp in the same dir, fsync, then os.replace."""
    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, f"ckpt_step{step}.pt")
    # NamedTemporaryFile in same dir guarantees os.replace is atomic.
    fd, tmp_path = tempfile.mkstemp(prefix=f"ckpt_step{step}.", suffix=".pt.tmp", dir=str(out_dir))
    os.close(fd)
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    if keep_last_k is not None and keep_last_k > 0:
        ckpts = list_checkpoints(out_dir)
        for _, path in ckpts[:-keep_last_k]:
            try:
                os.remove(path)
            except OSError:
                pass

    return final_path


def load_checkpoint(path: str | os.PathLike, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------------------
# RNG snapshot/restore (for exact resume)
# ---------------------------------------------------------------------------

def snapshot_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def banner(text: str, char: str = "=", width: int = 72) -> None:
    bar = char * width
    print(bar, flush=True)
    print(f"  {text}", flush=True)
    print(bar, flush=True)


def ensure_parent(path: str | os.PathLike) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
