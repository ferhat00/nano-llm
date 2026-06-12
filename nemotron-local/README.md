# Nemotron Nano 4B on a local RTX 3060 12 GB

A zero-edit-runnable **local** notebook that runs **NVIDIA Nemotron-3-Nano-4B** — a *dense*
**Mamba-2 + Transformer hybrid** (~3.97 B params) — on a single consumer **RTX 3060 12 GB**
(Ampere, compute capability 8.6). It covers environment audit, model load, single-/multi-turn
inference, reasoning ON/OFF, streaming, batch, a benchmark suite, and a **local
OpenAI-compatible API server**.

> This is the **local sibling** of `../nemotron-kaggle/` (which targets a Kaggle T4). Both are
> independent of the from-scratch `nano-llm` training pipeline in the parent repo.

## Files
- `nemotron_nano_4b_local_rtx3060.ipynb` — the notebook (run this locally).
- `build_notebook_local.py` — regenerates the notebook (`python build_notebook_local.py`).
  Edit here if you want to change cells reproducibly.
- `README.md` — this file.

## Why the RTX 3060 is a fine (and easy) host

| | Kaggle T4 (Turing, cc 7.5) | **RTX 3060 (Ampere, cc 8.6)** |
|---|---|---|
| VRAM | 16 GB | **12 GB** |
| Native bf16 | No (emulated) | **Yes** |
| FlashAttention | No | **Yes** (enabled by default here) |
| `mamba-ssm` fused kernels | Unreliable | **Supported** (Ampere+) |
| vLLM | Avoid (wants sm_80+) | Works (sm_86) |

The 4 B model is *dense*, and only its few attention layers grow a KV cache, so even a 16k
context stays cheap. The default **Q8_0** GGUF (~4.5 GB weights, ~5–6 GB total at 16k) leaves
comfortable headroom in 12 GB.

## Quick start (local)
1. Have an NVIDIA driver + CUDA-capable PyTorch installed and the RTX 3060 visible to
   `nvidia-smi`.
2. Open `nemotron_nano_4b_local_rtx3060.ipynb` in Jupyter / VS Code.
3. *(Optional)* `export HF_TOKEN=...` only if you point it at a gated repo — the default model
   is open, so this is not needed.
4. **Run All.** First run installs `llama-cpp-python[server]` (2–4 min) and downloads the GGUF
   (~4.5 GB at Q8_0). Warm re-runs reuse `./hf-cache` and `./models`.

Tune everything from the **`Config`** dataclass near the top — repo IDs, quant (`Q8_0` →
`Q4_K_M`), `n_ctx`, `max_new_tokens`, `temperature`, `thinking`, `thinking_budget`, and the API
`host`/`port`.

## Parameter choices baked in (and why)
- **Backend: llama.cpp (GGUF)** — own Mamba-2 kernels, one pip wheel, smallest footprint;
  matches the Kaggle notebook.
- **Quant: `Q8_0` (INT8)** — near-lossless vs fp16, ~4.5 GB, plenty of headroom on 12 GB.
  Switch to `Q4_K_M` (~2.3 GB) if you want maximum headroom/speed at a small quality cost.
- **Context: `n_ctx = 16384`** — comfortable for long docs/RAG given the small hybrid KV cache.
- **`flash_attn = True`** — Ampere supports it (the T4 notebook leaves it off).
- **Scope: chat + reasoning toggle, benchmark, local API server.** (No QLoRA fine-tuning — the
  4 B *can* be QLoRA-tuned in 12 GB with `bitsandbytes`/`peft`, but it's out of scope here.)

## Local API server
Cell 11 starts llama.cpp's **OpenAI-compatible** server at
`http://127.0.0.1:8000/v1`. Point any OpenAI client (or `curl`) at it:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nemotron-nano-4b","messages":[{"role":"user","content":"Hi"}]}'
```

**VRAM caveat:** the server runs in a child process and loads its **own** copy of the weights.
Free the in-notebook model first (run Cell 12) — two Q8_0 copies would exceed 12 GB.

## Reasoning mode
- `Config.thinking = True/False` toggles the `<think>` block via the chat template's
  `enable_thinking` flag (Cell 06 shows the measurable token-count difference).
- `Config.thinking_budget = N` is a **best-effort** cap, passed only if the template supports it.
- In the chat loop: `/thinking on|off`, `/budget N|off`, `/reset`, `/exit`.

## Alternate model
"Nemotron Nano 4B" also refers to the newer dense **`Llama-3.1-Nemotron-Nano-4B-v1.1`**
(~4.51 B, Llama-based). It's a drop-in alternate — see the commented block in `Config`. Its
reasoning toggle differs: put **`"detailed thinking on"` / `"detailed thinking off"`** in the
**system** message instead of using `enable_thinking`.

## Benchmark results — targets vs. observed

These are *expected* ranges for **Q8_0 on an RTX 3060 12 GB**. This notebook was authored
**without** an RTX 3060, so the Observed column is intentionally blank — run `benchmark()`
(Cell 10) and paste your numbers. No measured values are fabricated.

| Metric | Target (RTX 3060, Q8_0 GGUF) | Observed |
|---|---|---|
| VRAM after model load (16k ctx) | ~5–6 GB | _run locally_ |
| Time-to-first-token (short prompt) | < 1.5 s | _run locally_ |
| Sustained throughput | ~30–50 tok/s | _run locally_ |
| Cold start (warm cache) | < 60 s | _run locally_ |
| Peak VRAM | < 9 GB (≥ 3 GB headroom) | _run locally_ |

## Regenerating the notebook
```bash
cd nemotron-local
python build_notebook_local.py     # rewrites nemotron_nano_4b_local_rtx3060.ipynb
```
Requires `nbformat` locally (only for regeneration, not for running).
