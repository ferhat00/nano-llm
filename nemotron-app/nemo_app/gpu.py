"""GPU VRAM readout via `nvidia-smi`.

The model's weights live in the `llama-server.exe` subprocess, not this Python
process, so `torch.cuda.memory_*` would report ~0 and miss the very thing we want
to watch. `nvidia-smi` reports *total* board usage across all processes, which is
the honest signal: when the server is killed, used VRAM drops.

Per-process VRAM (`--query-compute-apps`) is deliberately not used — on Windows
WDDM (e.g. a consumer RTX 3060) it frequently reports `[Not Supported]`, so we
rely on total used/free instead.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional

# Suppress the brief console window nvidia-smi would otherwise flash on Windows.
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


@dataclass(frozen=True)
class VramStat:
    """A single GPU's memory snapshot, in MiB (as nvidia-smi reports)."""
    index: int
    name: str
    used_mib: int
    total_mib: int
    free_mib: int


def read_vram(device_index: int, timeout_s: float = 3.0) -> Optional[VramStat]:
    """Return a VramStat for `device_index`, or None if it can't be read.

    Returns None on any failure (nvidia-smi missing, no NVIDIA GPU, bad index,
    timeout, unparseable output) so callers can degrade gracefully.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
                "-i", str(device_index),
            ],
            capture_output=True, text=True, timeout=timeout_s,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None

    line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4:
        return None
    name, used, total, free = parts
    try:
        return VramStat(
            index=device_index,
            name=name,
            used_mib=int(used),
            total_mib=int(total),
            free_mib=int(free),
        )
    except ValueError:
        return None
