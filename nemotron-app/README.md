# nemotron-app — local Streamlit chat for Nemotron Nano 4B

A local web app over **NVIDIA Nemotron-3-Nano-4B** running on your RTX 3060. It is
the GUI sibling of [`../nemotron-local/`](../nemotron-local/): the notebook proves
the model runs, this app gives it a chat UI with four extras:

- **Thinking mode** — toggle the model's `<think>…</think>` reasoning on/off.
- **Document RAG** — attach PDFs / txt / md / docx, ask questions over them (vector
  search with citations).
- **Web search** — ground answers in live DuckDuckGo results (no API key).
- **Coding help** — a coding-oriented mode with fenced, syntax-highlighted code.

It **reuses the artifacts the notebook already downloaded** — the
`llama-server.exe` binary, the GGUF weights, and the HF tokenizer cache — so there
is nothing extra to download for the model itself.

## How it works

```
Streamlit (app.py)
   │  sidebar toggles: thinking · RAG · web · mode · sampling
   ▼
nemo_app/  ── llm.py ──────► llama-server.exe  (OpenAI /v1, 127.0.0.1:8000)
            ── rag.py ──────► Chroma (persistent) + bge-small embeddings (CPU)
            ── websearch.py ► DuckDuckGo (ddgs)
            ── prompts.py ──► assembles [system]+history+context+user
            ── state.py ────► @st.cache_resource: server / tokenizer / embedder / store
```

- The app **detects a healthy llama-server** on `127.0.0.1:8000` and reuses it; if
  none is running it launches one from the notebook's binary + GGUF. Either way
  there is only ever **one copy of the weights** in VRAM.
- Thinking is controlled client-side: prompts are rendered with the HF tokenizer's
  `enable_thinking` switch and sent to `/v1/completions` (the path the notebook
  verified).
- Tools are driven by **explicit sidebar toggles**, not agentic tool-calling —
  deterministic and reliable on a 4B model.
- Everything tunable lives in [`config.yaml`](config.yaml) (no magic numbers in code).

## Prerequisites

1. The `nemotron-local` notebook has been run once so these exist (paths set in
   `config.yaml`):
   - `../nemotron-local/llama-bin/bin/llama-server.exe`
   - `../nemotron-local/models/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`
   - `../nemotron-local/hf-cache/hub/…` (tokenizer)
2. The RTX 3060 visible to `nvidia-smi`, recent NVIDIA driver.
3. Internet on first run only (to fetch the `bge-small` embedding model, ~130 MB,
   and any web searches).

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

- **Thinking mode** (sidebar toggle): on → answers include a collapsible *Thinking*
  block; off → direct answers.
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

Edit [`config.yaml`](config.yaml) — server paths/port, context length, sampling
defaults, embedding model/device, chunk size/overlap, retrieval `top_k`, web result
count, system prompts, and UI slider ranges. To use the lighter/heavier GGUF quant
or a different model, point `server.gguf_path` at it.

## Scope notes

- **Code execution is intentionally out of scope** (security/sandboxing) — the app
  shows code, it does not run it.
- The Mamba recurrent state is not carried across calls; full chat history is
  re-fed each turn (same as the notebook).
- The default web provider is DuckDuckGo (keyless). Tavily is wired as an optional
  upgrade behind `web_search.provider`.
