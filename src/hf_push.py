"""Push a trained nano-llm checkpoint + tokenizer to the Hugging Face Hub.

Raw-artifact upload (no `transformers` dependency): the latest checkpoint is
uploaded as `pytorch_model.pt`, alongside `tokenizer.json` and an auto-generated
`README.md` model card. Files can be loaded back with this repo's
`src/sample.py:load_model_and_tokenizer`.

Auth: the write token is read from the `HF_TOKEN` env var (set from Kaggle
Secrets in the notebook), falling back to `cfg["huggingface"]["token"]`.
"""

from __future__ import annotations

import os

from .utils import find_latest_checkpoint, load_checkpoint


def _resolve_token(hf_cfg: dict) -> str:
    token = os.environ.get("HF_TOKEN") or hf_cfg.get("token")
    if not token:
        raise RuntimeError(
            "no Hugging Face token found. Set the HF_TOKEN environment variable "
            "(e.g. from Kaggle Secrets) or add huggingface.token to your config."
        )
    return token


def _count_params(model_state: dict) -> int:
    """Sum unique tensors (dedup tied weights by storage pointer)."""
    seen: set[int] = set()
    total = 0
    for t in model_state.values():
        ptr = t.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += t.numel()
    return total


def _build_model_card(cfg: dict, repo_id: str, step: int, n_params: int) -> str:
    m = cfg.get("model", {})
    data = cfg.get("data", {})
    dataset = data.get("dataset", "roneneldan/TinyStories")
    n_params_m = n_params / 1e6
    return f"""---
license: mit
library_name: pytorch
tags:
- pytorch
- tinystories
- nano-llm
- from-scratch
- text-generation
datasets:
- {dataset}
---

# {repo_id}

A from-scratch decoder-only transformer (~{n_params_m:.1f}M params) trained on
[{dataset}](https://huggingface.co/datasets/{dataset}). No `transformers` model
code: the architecture is hand-written in
[nano-llm](https://github.com/ferhat00/nano-llm) (`src/model.py`) — RoPE,
RMSNorm, SwiGLU MLP, tied embeddings, `F.scaled_dot_product_attention`.

## Model details

| field | value |
|-------|-------|
| parameters | {n_params:,} (~{n_params_m:.1f}M) |
| n_layer | {m.get('n_layer')} |
| n_head | {m.get('n_head')} |
| n_embd | {m.get('n_embd')} |
| block_size | {m.get('block_size')} |
| rope_base | {m.get('rope_base', 10000.0)} |
| training step | {step} |

## Files

- `pytorch_model.pt` — checkpoint (`model` state_dict + training `cfg`).
- `tokenizer.json` — byte-level BPE tokenizer (HF `tokenizers` format).

## Usage

```python
# from a clone of https://github.com/ferhat00/nano-llm
from huggingface_hub import hf_hub_download
from src.sample import load_model_and_tokenizer, generate

ckpt = hf_hub_download("{repo_id}", "pytorch_model.pt")
tok = hf_hub_download("{repo_id}", "tokenizer.json")
model, tokenizer, cfg = load_model_and_tokenizer(ckpt, tok, device="cpu")
print(generate(model, tokenizer, prompt="Once upon a time", max_new_tokens=200))
```
"""


def push_to_hub(cfg: dict, out_dir: str, tokenizer_path: str) -> str:
    """Upload the latest checkpoint + tokenizer + model card. Returns the repo URL."""
    hf_cfg = cfg.get("huggingface") or {}
    repo_id = hf_cfg.get("repo_id")
    if not repo_id or "<" in str(repo_id):
        raise RuntimeError(
            "huggingface.repo_id is not set. Set it in the config or pass "
            "--huggingface.repo_id=<your-username>/nano-llm-tinystories"
        )
    private = bool(hf_cfg.get("private", False))
    token = _resolve_token(hf_cfg)

    ckpt_path = cfg.get("resume_from") or find_latest_checkpoint(out_dir)
    if not ckpt_path:
        raise FileNotFoundError(f"no checkpoint found in {out_dir!r}; train first.")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"tokenizer not found at {tokenizer_path!r}; run the tokenizer stage first.")

    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    step = int(ckpt.get("step", 0)) + 1
    ckpt_cfg = ckpt.get("cfg", cfg)
    n_params = _count_params(ckpt["model"])

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"[push] target repo: {repo_id} (private={private})", flush=True)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    print(f"[push] uploading checkpoint {ckpt_path} -> pytorch_model.pt", flush=True)
    api.upload_file(
        path_or_fileobj=ckpt_path,
        path_in_repo="pytorch_model.pt",
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"[push] uploading tokenizer {tokenizer_path} -> tokenizer.json", flush=True)
    api.upload_file(
        path_or_fileobj=tokenizer_path,
        path_in_repo="tokenizer.json",
        repo_id=repo_id,
        repo_type="model",
    )

    card = _build_model_card(ckpt_cfg, repo_id, step, n_params)
    print("[push] uploading README.md (model card)", flush=True)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"[push] done -> {url}", flush=True)
    return url
