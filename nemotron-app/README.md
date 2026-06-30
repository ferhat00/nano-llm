# nemotron-app — local Streamlit chat for Nemotron Nano 4B

A local web app for running small LLMs on your RTX 3060 (12 GB), starting with
**NVIDIA Nemotron-3-Nano-4B**. It is the GUI sibling of
[`../nemotron-local/`](../nemotron-local/): the notebook proves the model runs, this
app gives it a chat UI with these extras:

- **Model picker** — a sidebar dropdown switches between a curated set of 3–9B models
  (Qwen3-4B, Phi-4-mini, Gemma 3 4B, Ministral 3, GLM-4-9B, Granite, SmolLM3, …) that
  fit 12 GB. Only one model is held in VRAM at a time; non-local models are
  auto-downloaded by llama-server on first use.
- **Thinking mode** — toggle the model's `<think>…</think>` reasoning on/off (for
  models that support it).
- **Document RAG** — attach PDFs / txt / md / docx, ask questions over them (vector
  search with citations).
- **Web search** — ground answers in live DuckDuckGo results (no API key).
- **Coding help** — a coding-oriented mode with fenced, syntax-highlighted code.

It **reuses the `llama-server.exe` binary the notebook already downloaded** plus the
local Nemotron GGUF. Other models are fetched on demand via llama-server's `-hf`
auto-download into its model cache.

## How it works

```
Streamlit (app.py)
   │  sidebar toggles: thinking · RAG · web · mode · sampling
   ▼
nemo_app/  ── llm.py ──────► llama-server.exe  (OpenAI /v1/chat/completions, :8000)
            ── rag.py ──────► Chroma (persistent) + bge-small embeddings (CPU)
            ── websearch.py ► DuckDuckGo (ddgs)
            ── prompts.py ──► assembles [system]+history+context+user
            ── state.py ────► @st.cache_resource: server / embedder / store
```

- The app launches **one `llama-server.exe`** for the selected model and talks to its
  OpenAI-compatible `/v1/chat/completions`. Switching models in the dropdown stops that
  subprocess and starts a new one, so there is only ever **one copy of the weights** in
  VRAM. It reuses an already-running server only if it is serving the same model
  (verified via the server's properties endpoint).
- Prompt templating is **server-side**: `--jinja` makes llama-server apply each GGUF's
  own chat template and stop tokens, and `--reasoning-format` surfaces `<think>`
  reasoning separately; the thinking toggle is passed per request via
  `chat_template_kwargs`. This is what lets one code path serve many model families.
- Tools are driven by **explicit sidebar toggles**, not agentic tool-calling.
- Everything tunable lives in [`config.yaml`](config.yaml) (no magic numbers in code).

## Prerequisites

1. The `nemotron-local` notebook has been run once so these exist (paths set in
   `config.yaml`):
   - `../nemotron-local/llama-bin/bin/llama-server.exe` — a **recent build**; it must
     support `-hf`, `--jinja`, and `--reasoning-format`.
   - `../nemotron-local/models/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf` (the default model).
2. The RTX 3060 visible to `nvidia-smi`, recent NVIDIA driver.
3. Internet on first run (for the `bge-small` embedding model ~130 MB, web searches, and
   the one-time auto-download of any non-local model you select — several GB each).

## Setup & run

From this folder, using the repo's existing virtual env (`../local_nemotron`):

```powershell
# Windows PowerShell
..\local_nemotron\Scripts\python.exe -m pip install -r requirements.txt
..\local_nemotron\Scripts\python.exe -m streamlit run app.py
```

The app opens at `http://localhost:8501`. First launch loads the model into the GPU
(a few seconds) and shows **"Model server connected"** in the sidebar.

Optional secrets: copy `.env.example` to `.env` and set `HF_TOKEN` (only if the
model repo is gated) or `TAVILY_API_KEY` (only if you switch the web provider to
Tavily in `config.yaml`). Both are optional; the defaults need neither.

## Using it

- **Model** (sidebar dropdown): pick a model; switching unloads the current one and
  loads the selected one (only one in VRAM at a time). A non-local model downloads on
  first selection (watch the VRAM meter / `data/server.log`). Switching clears the chat,
  since templates and special tokens differ across models.
- **Thinking mode** (sidebar toggle): on → answers include a collapsible *Thinking*
  block; off → direct answers. Disabled for models that don't expose a thinking switch.
- **Documents (RAG)**: upload files → **Index** → toggle *Use my documents* → ask.
  Retrieved sources show under each answer as `[filename p.N]`. *Clear docs* empties
  the store. The store persists on disk under `data/chroma/`.
- **Web search**: toggle *Search the web*; answers cite `[n]` with links. Enable
  *Fetch full pages* for fuller (slower) context.
- **Coding**: switch Mode to *Coding* for code-first responses.
- **GPU / VRAM**: a live VRAM meter (sidebar) shows board-wide usage via
  `nvidia-smi`. **♻️ Free VRAM** stops the model server and releases its VRAM while
  keeping the app open (restart it with **▶ Start server**); **🛑 Shut down** frees
  VRAM and exits the app process. The model's VRAM lives in the `llama-server.exe`
  subprocess, so this is the reliable way to reclaim it without closing the terminal.

## Releasing VRAM

The weights are held by the `llama-server.exe` **subprocess**, not the Python
process, so freeing VRAM means stopping that subprocess:

- **Clean exit** (Ctrl-C in the terminal): an `atexit` handler stops a server the
  app started.
- **Abrupt close** (closing the terminal window, Task-Manager kill): the server is
  launched inside a Windows **Job Object** flagged kill-on-close, so it dies with
  the app even when `atexit` can't run — no orphaned VRAM.
- **Manual**: use **♻️ Free VRAM** / **🛑 Shut down** in the sidebar any time.
- **Externally-started server** (`reuse_existing` attached to the notebook's
  server): the app won't kill it — stop it where it was launched.

## Configuration

Edit [`config.yaml`](config.yaml) — shared server settings (binary path/port, the
runtime flags applied to every model: `n_gpu_layers`, `flash_attn`, KV-cache types,
`jinja`, `reasoning_format`), the `models:` catalog, sampling defaults, embedding
model/device, chunk size/overlap, retrieval `top_k`, web result count, system prompts,
and UI slider ranges.

To add or change a model, edit `models.available`: each entry is either a local file
(`gguf_path`) **or** an auto-download (`hf_repo` + `quant`, fetched via `-hf`), with its
own `n_ctx`, `supports_thinking`, and optional per-model `sampling`. Set `models.active`
to the model that loads at startup. Confirm a model's exact GGUF repo id and quant tag on
its Hugging Face model card before first use (prefer official or `bartowski` GGUFs — they
are ungated, so `-hf` needs no token).

## Scope notes

- **Code execution is intentionally out of scope** (security/sandboxing) — the app
  shows code, it does not run it.
- The Mamba recurrent state is not carried across calls; full chat history is
  re-fed each turn (same as the notebook).
- The default web provider is DuckDuckGo (keyless). Tavily is wired as an optional
  upgrade behind `web_search.provider`.
