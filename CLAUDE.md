# CLAUDE.md — nano-llm

## Project Overview
Decoder-only transformer LLM trained **from scratch** on TinyStories. Runtime
target is Kaggle's 2×T4 (16 GB each, 32 GB total) within the 12-hour session
limit and 30 h/week quota. All real logic lives in this repo; the Kaggle
notebook only clones, installs, and runs `orchestrate.py`.

## Tech Stack
- Python 3.10+
- PyTorch 2.4 (AMP **fp16** — T4 / Turing has no real bf16)
- HF `tokenizers` (custom byte-level BPE; optional GPT-2 reuse)
- HF `datasets` (TinyStories)
- **No `transformers` model imports** — `src/model.py` is hand-written

## Project Structure
- `orchestrate.py` — single entry point; runs tokenizer → data → train → eval → sample
- `src/` — `model.py`, `train.py`, `tokenizer.py`, `data.py`, `sample.py`, `utils.py`
- `configs/{smoke,small}.yaml` — preset hyperparameters
- `notebooks/kaggle_runner.ipynb` — thin Kaggle launcher (clone → pip → run)

## Key Conventions
- All hyperparameters in YAML configs; **no magic numbers in code**
- Checkpoints saved atomically (`.tmp` → `os.replace`) at configurable intervals
- Resume auto-detected from `out_dir` (or `--resume-from`) on startup
- Pre-tokenized data is `uint16` memmap (`*.bin`), read via `np.memmap`
- Attention uses `F.scaled_dot_product_attention` (NOT FlashAttention-2; Turing isn't supported)
- Stages are idempotent: re-running `orchestrate.py` skips completed stages unless `--force`

## Running
1. `pip install -r requirements.txt`
2. **Smoke first**: `python orchestrate.py --config configs/smoke.yaml`
3. Real run: `python orchestrate.py --config configs/small.yaml`
4. On Kaggle: open `notebooks/kaggle_runner.ipynb` and run all cells

## Testing
The smoke config **is** the integration test. Killing it mid-train and
re-running must resume cleanly from the latest checkpoint.

---

## 1 - Kill the filler
Never open responses with filler phrases like "Great question!", "Of course!", "Certainly!", "Absolutely!", "Sure!", or similar warmups.

Start every response with the actual answer.
No preamble, no acknowledgment of the question.
Just the information.

## 2 - Always show options before acting

## 3 - Be honest when you don't know
If you are uncertain about any fact, statistic, date, quote, or piece of information, say so explicitly before including it.

"I'm not certain about this" is always better than presenting a guess as a fact.

Never fill gaps in your knowledge with plausible-sounding information.
When in doubt, say so.

## 4 - Match length to what's actually needed
Match response length to task complexity.

Simple questions get direct, short answers.
Complex tasks get full, detailed responses.

Never compress or summarize work that requires real depth.
Never pad responses with restatements of the question or closing sentences that repeat what you just said.

## 5 - Ask before making big changes
Before making any change that significantly alters content I've already created (rewriting sections, removing paragraphs, restructuring the flow, changing tone), stop completely.

Describe exactly what you're about to change and why.
Wait for my confirmation before proceeding.

"I think this would be better" is not permission to change it.

## 6 - Stay focused on what was asked
Only change what I specifically asked you to change.

Do not rewrite, rephrase, restructure, or "improve" anything I didn't ask about, even if you think it would be better.

If you notice something that could be improved elsewhere, mention it at the end of your response.
Do not touch it unless I explicitly ask you to.

## 7 - Always tell me what you changed
After completing any editing or writing task, always end with a brief summary:
- What was changed: [description]
- What was left untouched: [if relevant]
- What needs my attention: [anything requiring a decision or review]

Keep it short. This is a status update, not a recap of everything you just did.

## 8 - Never take actions on my behalf without asking
Never send, post, publish, share, or schedule anything on my behalf without my explicit confirmation in the current message.

This includes:
- Emails
- Social posts
- Calendar invites
- Document shares
- Any action that affects something outside this conversation

"You mentioned wanting to do this" is not confirmation.
I must say yes in the current message.

## 9 - Who I am
About me:
- Name: Ferhat Culfaz
- Role: Data Scientist
- Background: PhD Physics, professional data scientist
- Strong in: physics, finance, economics
- Still learning: professional quality code

Adjust the depth of every response to match this background. Never over-explain what I already know. Never skip context I need.

## 10 - Context of this project
- Project: A from-scratch decoder-only transformer LLM, runnable end-to-end on Kaggle 2×T4
- Goal: a clean, reproducible training pipeline (~30M params on TinyStories) that survives Kaggle's 12 h session resets via airtight checkpoint/resume
- Audience: me — and anyone reading the repo as a reference for nano-LLM-on-Kaggle workflows
- Tone: direct, professional, no hand-holding
- What to avoid: pretrained-model imports, unpinned deps, magic numbers in code, FlashAttention-2 (T4 unsupported)

## 11 - Voice and style
- Voice: no-fluff
- Sentence length: mixed
- Format preference: headers and paragraphs with bullets

## 12 - MEMORY.md
Maintain a file called `MEMORY.md`. After any significant decision, about direction, format, content, approach, or strategy, add an entry:

## [Date], [Decision]
**What was decided:** [the choice made]
**Why:** [the reasoning]
**What was rejected:** [alternatives considered and why they were ruled out]

Read `MEMORY.md` at the start of every session before doing anything. Never contradict a logged decision without flagging it first.
