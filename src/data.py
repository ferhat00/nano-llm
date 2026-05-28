"""Download a corpus, pre-tokenize once to uint16 memmap, sample random windows."""

from __future__ import annotations

import os
from typing import Iterator

import numpy as np
import torch
from tqdm import tqdm

from .tokenizer import TokenizerWrapper


# ---------------------------------------------------------------------------
# Corpus iteration
# ---------------------------------------------------------------------------

def iter_split(cfg_data: dict, split: str) -> Iterator[str]:
    """Yield raw strings from a dataset split, applying optional subset cap."""
    dataset_name = cfg_data["dataset"]
    # HF datasets: roneneldan/TinyStories has 'train' and 'validation' splits.
    from datasets import load_dataset
    hf_split = "validation" if split == "val" else split
    ds = load_dataset(dataset_name, split=hf_split)
    subset_key = f"subset_{split}"
    n = cfg_data.get(subset_key)
    if n is not None:
        ds = ds.select(range(min(int(n), len(ds))))
    text_field = cfg_data.get("text_field", "text")
    for row in ds:
        yield row[text_field]


# ---------------------------------------------------------------------------
# Pre-tokenization (write uint16 memmap)
# ---------------------------------------------------------------------------

def _split_bin_path(data_dir: str, split: str) -> str:
    return os.path.join(data_dir, f"{split}.bin")


def prepare(cfg_data: dict, tokenizer: TokenizerWrapper, data_dir: str | os.PathLike,
            force: bool = False) -> dict:
    """Pre-tokenize train and val into uint16 memmaps. Idempotent."""
    if tokenizer.vocab_size > 65535:
        raise ValueError(
            f"vocab_size={tokenizer.vocab_size} doesn't fit in uint16; "
            "drop vocab_size to <=65535 or change the memmap dtype."
        )
    os.makedirs(data_dir, exist_ok=True)
    info: dict = {}
    for split in ("train", "val"):
        path = _split_bin_path(str(data_dir), split)
        if os.path.exists(path) and not force:
            n = os.path.getsize(path) // np.dtype(np.uint16).itemsize
            info[split] = {"path": path, "tokens": int(n), "rebuilt": False}
            continue

        # Two-pass: first count tokens, then write into a fixed-size memmap.
        # This avoids holding the whole corpus in RAM.
        eos = tokenizer.eos_id
        total = 0
        for text in tqdm(iter_split(cfg_data, split), desc=f"tokenize/count {split}", unit="doc"):
            ids = tokenizer.encode(text)
            total += len(ids) + 1  # +1 for EOS

        mm = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))
        idx = 0
        for text in tqdm(iter_split(cfg_data, split), desc=f"tokenize/write {split}", unit="doc"):
            ids = tokenizer.encode(text)
            ids.append(eos)
            mm[idx:idx + len(ids)] = np.asarray(ids, dtype=np.uint16)
            idx += len(ids)
        assert idx == total, f"token count mismatch for {split}: wrote {idx}, expected {total}"
        mm.flush()
        del mm
        info[split] = {"path": path, "tokens": int(total), "rebuilt": True}
    return info


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

# Module-level memmap cache, keyed by absolute path. memmaps are cheap to keep
# open and avoid re-syscalling for every batch.
_MEMMAPS: dict[str, np.memmap] = {}


def _get_memmap(path: str) -> np.memmap:
    abspath = os.path.abspath(path)
    if abspath not in _MEMMAPS:
        _MEMMAPS[abspath] = np.memmap(abspath, dtype=np.uint16, mode="r")
    return _MEMMAPS[abspath]


def get_batch(data_dir: str | os.PathLike, split: str, batch_size: int, block_size: int,
              device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` random `block_size+1` windows; return (x, y) int64."""
    mm = _get_memmap(_split_bin_path(str(data_dir), split))
    n = len(mm)
    if n < block_size + 1:
        raise RuntimeError(
            f"{split}.bin has only {n} tokens; need at least block_size+1={block_size + 1}. "
            "Increase subset_{split} in the config or pick a larger corpus."
        )
    # Random start indices into [0, n - block_size - 1].
    ix = torch.randint(0, n - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(mm[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(mm[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y
