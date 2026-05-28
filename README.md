# nano-llm

A from-scratch decoder-only transformer, trained on TinyStories. Designed to
run end-to-end on **Kaggle 2×T4** (16 GB each) within the 12 h session limit
and the 30 h/week quota. All real logic lives in this repo; the Kaggle notebook
just clones, installs, and runs `orchestrate.py`.

## Quickstart

```bash
git clone https://github.com/ferhat00/nano-llm.git
cd nano-llm
pip install -r requirements.txt

# Smoke first (CPU-friendly, 2-3 min). DO NOT skip this.
python orchestrate.py --config configs/smoke.yaml

# Real run (~30M params; designed for Kaggle 2xT4)
python orchestrate.py --config configs/small.yaml
```

The pipeline is a single entry point. `orchestrate.py` runs five stages in
order — **tokenizer → data → train → eval → sample** — each idempotent unless
you pass `--force`.

## What's in here

```
orchestrate.py            single entry point
configs/
  smoke.yaml              tiny model + tiny data subset, runs in 2-3 min
  small.yaml              ~30M-param model on full TinyStories
src/
  model.py                decoder transformer: RoPE, RMSNorm, SwiGLU, SDPA, tied head
  train.py                AMP fp16 loop, grad accum, cosine+warmup, atomic resume, optional DDP
  tokenizer.py            byte-level BPE (or reuse GPT-2)
  data.py                 TinyStories -> uint16 memmap; random-window batching
  sample.py               temperature / top-k / top-p generation
  utils.py                config loader, seeding, checkpoint I/O
notebooks/
  kaggle_runner.ipynb     3-cell launcher: clone + install + run
requirements.txt          pinned (torch 2.4.1, tokenizers 0.20.3, datasets 3.0.2, ...)
```

## CLI

```bash
python orchestrate.py --config <yaml>
                      [--force]                   # ignore stage idempotency
                      [--stage tokenizer|data|train|eval|sample|all]
                      [--resume-from PATH]        # override config's resume_from
                      [--<key>.<subkey>=VAL ...]  # arbitrary config overrides

# Examples
python orchestrate.py --config configs/smoke.yaml --train.max_steps=10
python orchestrate.py --config configs/small.yaml --model.n_layer=6
python orchestrate.py --config configs/smoke.yaml --stage sample
```

## Smoke test first — every time

The smoke config (`configs/smoke.yaml`) trains a 2-layer 64-dim model on a
1000-doc TinyStories subset for 30 steps on CPU. It exists so you can prove
the whole pipeline works in a few minutes before spending Kaggle GPU quota.

What it validates:

1. `python orchestrate.py --config configs/smoke.yaml` — runs all 5 stages, prints a sample.
2. `Ctrl+C` mid-train, re-run — you should see `[resume] continuing from step N`.
3. `python orchestrate.py --config configs/smoke.yaml --stage sample` — loads latest ckpt, samples.
4. `python orchestrate.py --config configs/smoke.yaml --force` — redoes everything.

If any of those fail locally, fix before launching `small.yaml` on Kaggle.

## Kaggle workflow

Open `notebooks/kaggle_runner.ipynb` on Kaggle, set:
- **Accelerator**: GPU T4 x2
- **Internet**: On

Run cells top-to-bottom. Cell 1 clones + installs. Cell 2 (optional) attaches
previous-session checkpoints. Cell 3a runs the smoke test on the GPU; cell 3b
runs the real training.

### Cross-session checkpoint pattern (12 h cliffs)

A Kaggle session is wiped at the 12 h mark and `/kaggle/working` goes with it,
so a 20 000-step run almost certainly spans multiple sessions. The pattern:

1. **In-session**: `out_dir` defaults to `/kaggle/working/checkpoints` (`small.yaml`).
   Checkpoints save every 500 steps; only the last 3 are kept.
2. **End of session**: download `/kaggle/working/checkpoints` (or use Kaggle's
   "Save Version" → Output to persist it), then **publish that folder as a
   Kaggle Dataset** (e.g. `nano-llm-checkpoints`).
3. **Next session**: attach that dataset under Add Input, then either
   - copy it into `/kaggle/working/checkpoints/` (see cell 2 in the notebook), or
   - point `resume_from` at the specific ckpt:
     `python orchestrate.py --config configs/small.yaml --resume_from=/kaggle/input/nano-llm-checkpoints/ckpt_step10000.pt`

Auto-resume picks the highest-step `ckpt_step*.pt` in `out_dir` on startup, so
the first option needs no flag changes.

## Design notes (the things that matter)

- **Attention** uses `torch.nn.functional.scaled_dot_product_attention` with
  `is_causal=True`. T4 (Turing) is **not** supported by FlashAttention-2;
  SDPA picks the memory-efficient backend automatically.
- **Precision**: `fp16` AMP with `GradScaler`. T4 has no real bf16 — using bf16
  would silently fall back to fp32 and burn 2× memory.
- **DDP** uses `torch.multiprocessing.spawn` (not `torchrun`), because
  `torchrun` doesn't launch cleanly from inside a Kaggle notebook.
  Single-GPU and CPU paths are first-class — DDP only activates when
  `train.distributed: true` **and** more than 1 GPU is visible.
- **Tokenizer**: byte-level BPE, vocab=8192 by default. The `tokenizer.type:
  gpt2` flag reuses GPT-2's tokenizer via `tokenizers.from_pretrained("gpt2")`
  if you'd rather not train one.
- **Data**: pre-tokenized once into `uint16` memmap files (`.bin`). vocab must
  fit in `uint16` (<= 65535).
- **Checkpoints** are written atomically (`tmp` → `os.replace`) so a SIGKILL
  mid-save can't corrupt them. Resume restores model + optimizer + GradScaler
  + RNG state + step.

## Pinned deps

```
torch==2.4.1
numpy==1.26.4
tokenizers==0.20.3
datasets==3.0.2
pyyaml==6.0.2
tqdm==4.66.5
```

## License

MIT (see `LICENSE`).
