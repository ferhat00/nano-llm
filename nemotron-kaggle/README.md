# Nemotron Nano 4B on a Kaggle T4

A zero-edit-runnable Kaggle notebook that runs **NVIDIA Nemotron Nano 4B** — a *dense*
**Mamba-2 + Transformer hybrid** (~3.97 B params) — on a single **Tesla T4** (16 GB, Turing,
compute capability 7.5). It covers environment audit, model load, single-/multi-turn inference,
reasoning ON/OFF, streaming, batch, and a benchmark suite.

> This subproject is **independent of the `nano-llm` repo** it lives in (that repo trains a
> from-scratch transformer). It is self-contained here.

## Files
- `nemotron_nano_4b_kaggle.ipynb` — the notebook (run this on Kaggle).
- `build_notebook.py` — regenerates the notebook (`python build_notebook.py`). Optional; edit
  here if you want to change cells reproducibly.
- `README.md` — this file.

## Quick start (Kaggle)
1. Upload `nemotron_nano_4b_kaggle.ipynb` to a new Kaggle notebook (or *File → Import Notebook*).
2. **Settings → Accelerator → GPU T4**, and **Internet → ON** (required for `pip` + downloads).
3. *(Optional)* If you ever point it at a gated repo, add **`HF_TOKEN`** under *Add-ons → Secrets*.
   The default model is open, so this is not needed out of the box.
4. **Run All.** First run downloads the GGUF (~3 GB) and installs `llama-cpp-python` (2–4 min).
   Warm re-runs reuse the cache under `/kaggle/working/`.

Tune everything from the **`Config`** dataclass near the top — repo IDs, quant (`Q4_K_M` →
`Q8_0`), `n_ctx`, `max_new_tokens`, `temperature`, `thinking`, `thinking_budget`.

## Backend choice — why llama.cpp (GGUF) is primary

| Backend | Mamba-2 | T4 (Turing) | Install | Verdict |
|---|---|---|---|---|
| **llama-cpp-python (GGUF)** | **Yes** — own SSM kernels, no `mamba-ssm` dep | **Yes** (CUDA-12 wheel covers sm_75) | one `pip` | **Primary** |
| transformers + mamba-ssm | Yes | Risky — fused kernels target Ampere+; no native bf16 | source build | Fallback (best-effort) |
| vLLM | Yes | No — recent vLLM limits pre-Ampere; kernels want ≥ sm_80 | medium | Avoid on T4 |
| Unsloth | Train/FT tool (infer via GGUF) | n/a for inference | n/a | Not for inference |
| TensorRT-LLM | Partial | Technically yes, practically no | very high (long build) | Avoid on Kaggle |

**The decisive fact:** llama.cpp carries its **own** Mamba-2 CUDA kernels and does **not** depend
on the `mamba-ssm` package, whose fused kernels are unreliable on Turing. That is what makes a
hybrid SSM model run cleanly on a free T4. INT4 (Q4_K_M, ~3 GB) leaves wide headroom inside 16 GB.

**Loader fallback chain:** primary GGUF → alternate GGUF quant (same backend) →
`transformers` fp16 (best-effort) → clear manual `pip` command if all fail.

## What "Mamba-2 hybrid" means (for non-experts)

A normal Transformer is **all attention**: every new token re-examines every previous token, so
its memory (the *KV cache*) and cost grow with sequence length. **Mamba-2** layers are
**state-space models** — they stream through the text keeping a **fixed-size running summary**
(a recurrent state) instead of re-reading everything. This model **interleaves** the two:
mostly Mamba layers with a few attention layers (~5:1) plus MLPs.

Why it matters here:
- **It runs long contexts cheaply.** Only the few attention layers grow a KV cache; the Mamba
  layers do not. A 4 B model therefore handles very long inputs inside 16 GB.
- **It needs special kernels.** Generic `transformers` is not enough on a T4 — you need a backend
  with working SSM kernels. llama.cpp has them; that is the whole reason for the backend choice.
- **State is not carried between calls.** In these backends the Mamba running summary resets each
  call, so multi-turn chat re-feeds the full history every turn (the notebook does this).

## Two corrections to the original brief
- **The 4 B is dense, not MoE.** Only the larger `30B-A3B` sibling uses Mixture-of-Experts. The
  4 B is a plain dense hybrid (Mamba-2 + attention + MLP).
- **Repo names may differ.** A `...-4B-Instruct` repo may not exist; the real family is
  `NVIDIA-Nemotron-3-Nano-4B-{BF16,FP8,GGUF}`. The notebook **resolves repo IDs at runtime** and
  falls back across candidates, so a one-line `Config` edit adapts to whatever is published.

## Benchmark results — targets vs. observed

The targets below are *expected* T4 ranges. **This notebook was authored without a T4**, so the
Observed column is intentionally blank — run `benchmark()` (Cell 10) on Kaggle and paste the
numbers. I have not fabricated measured values.

| Metric | Target (INT4 GGUF on T4) | Observed |
|---|---|---|
| VRAM after model load | < 6 GB | _run on Kaggle_ |
| Time-to-first-token (short prompt) | < 2 s | _run on Kaggle_ |
| Sustained throughput | > 25 tokens/s | _run on Kaggle_ |
| Cold start (warm cache) | < 90 s | _run on Kaggle_ |
| Peak VRAM | < 14 GB (2 GB headroom) | _run on Kaggle_ |

## Reasoning mode
- `Config.thinking = True/False` toggles the `<think>` reasoning block via the chat template's
  `enable_thinking` flag. Cell 06 shows the measurable token-count difference on the bat-and-ball
  puzzle.
- `Config.thinking_budget = N` is a **best-effort** cap: the notebook passes a budget kwarg to the
  template if the model's template supports one. If it does not, thinking is simply ON/OFF.
- In the chat loop: `/thinking on|off`, `/budget N|off`, `/reset`, `/exit`.

## Known issues / limitations
1. **No T4-measured benchmark numbers shipped** — placeholders only; populate on first run.
2. **Repo/gating resolved at runtime.** If every candidate 404s, edit `CFG.gguf_candidates` /
   `CFG.tokenizer_candidates`.
3. **No FP8 on Turing.** The FP8 variant cannot run on a T4; BF16 weights are emulated as fp16.
4. **transformers fallback is best-effort.** It may require `mamba-ssm`, which is unreliable on
   Turing — treat it as a safety net, not a guaranteed path.
5. **VRAM metering.** `torch.cuda.*` cannot see llama.cpp allocations; the notebook uses
   `nvidia-smi` as the authoritative meter (and `torch.cuda.memory_summary()` for the fallback).
6. **Stateless SSM across calls** — history is re-fed each turn (see the explainer above).
7. **`chat_loop()` is left commented** so *Run All* does not block on `input()`; uncomment to chat.

## Regenerating the notebook
```bash
cd nemotron-kaggle
python build_notebook.py        # rewrites nemotron_nano_4b_kaggle.ipynb
```
Requires `nbformat` locally (only for regeneration, not for running on Kaggle).
