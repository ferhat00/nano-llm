"""Nemotron Nano 4B — local Streamlit chat app.

Thin entry point: builds the UI, reads sidebar controls, and wires the per-turn
flow (optional web search + RAG -> prompt assembly -> streamed generation). All
real logic lives in the `nemo_app` package.

Run from this directory:
    streamlit run app.py
"""
from __future__ import annotations

import dataclasses
import os
import sys
import threading
import time

import streamlit as st

# Make the local package importable regardless of launch directory.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from dotenv import load_dotenv  # noqa: E402

from nemo_app import gpu, llm, prompts, rag, websearch  # noqa: E402
from nemo_app.state import (  # noqa: E402
    get_chroma_collection,
    get_config,
    get_embedder,
    get_server,
    get_tokenizer,
)

load_dotenv(os.path.join(_APP_DIR, ".env"))

st.set_page_config(page_title="Nemotron Nano 4B", page_icon="🧠", layout="wide")
cfg = get_config()

# --------------------------------------------------------------------------- session state
_DEFAULTS = {
    "messages": [],          # [{"role","content", optional "think", optional "sources"}]
    "indexed_files": set(),  # filenames added to the vector store this session
    "server_paused": False,  # True after "Free VRAM": keep the server stopped
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# --------------------------------------------------------------------------- rendering helpers
def render_sources(sources: dict) -> None:
    """Render a Sources expander for a turn that used RAG and/or web context."""
    if not sources:
        return
    rag_src = sources.get("rag") or []
    web_src = sources.get("web") or []
    if not rag_src and not web_src:
        return
    with st.expander("📎 Sources", expanded=False):
        for tag in rag_src:
            st.markdown(f"- {tag}")
        for i, (title, url) in enumerate(web_src, start=1):
            label = title or url
            st.markdown(f"- [{i}] [{label}]({url})" if url else f"- [{i}] {label}")


def render_message(msg: dict) -> None:
    """Render one stored message (with think expander + sources for assistant turns)."""
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("think"):
            with st.expander("🧠 Thinking", expanded=False):
                st.markdown(msg["think"])
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources") or {})


# --------------------------------------------------------------------------- GPU / VRAM
@st.fragment(run_every=cfg.gpu.refresh_seconds)
def render_vram_meter() -> None:
    """Live total-VRAM bar; auto-refreshes in place without rerunning the app.

    Reads board-wide usage via nvidia-smi (the llama-server's footprint lives in a
    separate process, so torch.cuda can't see it). Per-process VRAM is unreliable
    on Windows WDDM, so total used/free is the trustworthy signal.
    """
    stat = gpu.read_vram(cfg.gpu.device_index)
    if stat is None:
        st.caption("VRAM: stats unavailable (nvidia-smi not found or no GPU).")
        return
    frac = stat.used_mib / stat.total_mib if stat.total_mib else 0.0
    st.progress(
        min(max(frac, 0.0), 1.0),
        text=f"VRAM {stat.used_mib:,} / {stat.total_mib:,} MiB ({stat.free_mib:,} free)",
    )


def _shutdown_process() -> None:
    """Exit the whole Streamlit process after a short delay.

    The delay lets Streamlit flush the goodbye message to the browser before the
    process dies. os._exit skips atexit, but VRAM is already freed by the explicit
    shutdown_server call (and the Job Object would catch it regardless).
    """
    def _exit() -> None:
        time.sleep(1.0)
        os._exit(0)

    threading.Thread(target=_exit, daemon=True).start()


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🧠 Nemotron Nano 4B")

    # Server status / lifecycle. When paused (after "Free VRAM") we deliberately
    # skip get_server so the model is NOT reloaded on every rerun.
    server_ok = False
    handle = None
    if st.session_state.server_paused:
        st.info("Model server stopped — VRAM freed.")
        if st.button("▶ Start server", use_container_width=True):
            st.session_state.server_paused = False
            st.rerun()
    else:
        try:
            handle = get_server(cfg)
            server_ok = True
            where = "reused" if not handle.owned else "started"
            st.success(f"Model server connected ({where})")
        except Exception as exc:  # binary/model missing, startup timeout, etc.
            st.error(f"Model server unavailable:\n\n{exc}")

    # GPU / VRAM: live meter + release/shutdown controls.
    if cfg.gpu.show_vram:
        render_vram_meter()

    col_free, col_shut = st.columns(2)
    free_clicked = col_free.button(
        "♻️ Free VRAM", use_container_width=True,
        help="Stop the model server and release its VRAM. The app stays open; "
             "restart the server anytime.",
    )
    shutdown_clicked = col_shut.button(
        "🛑 Shut down", use_container_width=True,
        help="Free VRAM and exit the app process.",
    )

    if free_clicked:
        if handle is not None and handle.owned:
            llm.shutdown_server(handle)
            get_server.clear()
            if cfg.rag.embedding_device == "cuda":
                get_embedder.clear()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            st.session_state.server_paused = True
            st.toast("VRAM freed — server stopped. Click ▶ Start server to reload.")
            st.rerun()
        elif handle is not None and not handle.owned:
            st.warning("This server was started elsewhere (e.g. the notebook). "
                       "Stop it there to free its VRAM.")
        else:
            st.info("No running server to stop.")

    if shutdown_clicked:
        if handle is not None:
            status = llm.shutdown_server(handle)   # "terminated" or "external"
        else:
            status = "already_stopped"
        get_server.clear()
        if status == "external":
            st.warning("Shutting down. An externally-started server keeps running — "
                       "stop it where it was launched to free its VRAM.")
        else:
            st.warning("Shutting down — VRAM released. You can close this browser tab.")
        _shutdown_process()
        st.stop()

    st.divider()
    mode = st.radio("Mode", ["Chat", "Coding"], horizontal=True)
    thinking = st.toggle("Thinking mode", value=cfg.reasoning.thinking_default,
                         help="Toggle the model's <think> reasoning block.")

    with st.expander("Sampling", expanded=False):
        t_lo, t_hi = cfg.ui.temperature_range
        temperature = st.slider("Temperature", float(t_lo), float(t_hi),
                                float(cfg.sampling.temperature), 0.05)
        p_lo, p_hi = cfg.ui.top_p_range
        top_p = st.slider("top_p", float(p_lo), float(p_hi),
                          float(cfg.sampling.top_p), 0.01)
        m_lo, m_hi = cfg.ui.max_new_tokens_range
        max_new_tokens = st.slider("Max new tokens", int(m_lo), int(m_hi),
                                   int(cfg.sampling.max_new_tokens), 32)

    st.divider()
    st.subheader("📄 Documents (RAG)")
    rag_enabled = st.toggle("Use my documents", value=cfg.rag.enabled_default)
    uploaded = st.file_uploader(
        "Attach files", type=list(cfg.rag.supported_extensions),
        accept_multiple_files=True,
    )
    col_idx, col_clr = st.columns(2)
    index_clicked = col_idx.button("Index", use_container_width=True,
                                   disabled=not uploaded)
    clear_docs_clicked = col_clr.button("Clear docs", use_container_width=True)

    # RAG indexing actions (load embedder + store lazily, only when used).
    rag_ready = False
    collection = None
    embedder = None
    try:
        if rag_enabled or index_clicked or clear_docs_clicked:
            collection = get_chroma_collection(cfg.rag.persist_dir, cfg.rag.collection_name)
            embedder = get_embedder(cfg.rag.embedding_model, cfg.rag.embedding_device)
            rag_ready = True
    except Exception as exc:
        st.error(f"RAG unavailable: {exc}")

    if index_clicked and rag_ready and uploaded:
        uploads_dir = os.path.join(cfg.data_dir, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        added_total = 0
        with st.status("Indexing documents…", expanded=False) as status:
            for uf in uploaded:
                dest = os.path.join(uploads_dir, uf.name)
                with open(dest, "wb") as fh:
                    fh.write(uf.getbuffer())
                chunks = rag.chunk_file(dest, cfg.rag.chunk_size, cfg.rag.chunk_overlap)
                added = rag.add_documents(collection, embedder, chunks,
                                          cfg.rag.query_instruction)
                added_total += added
                st.session_state.indexed_files.add(uf.name)
                status.write(f"• {uf.name}: {added} chunks")
            status.update(label=f"Indexed {added_total} chunks", state="complete")

    if clear_docs_clicked and rag_ready:
        rag.clear(collection)
        st.session_state.indexed_files = set()
        st.toast("Cleared all indexed documents.")

    if rag_ready and collection is not None:
        st.caption(f"Indexed chunks: {rag.count(collection)}")
        if st.session_state.indexed_files:
            st.caption("Files: " + ", ".join(sorted(st.session_state.indexed_files)))

    st.divider()
    st.subheader("🌐 Web search")
    web_enabled = st.toggle("Search the web", value=cfg.web_search.enabled_default)
    web_count = st.slider("Results", 1, 10, int(cfg.web_search.result_count))
    web_fetch = st.toggle("Fetch full pages", value=cfg.web_search.fetch_pages,
                          help="Slower: download and strip each result page for fuller context.")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------- main chat area
st.caption("Local NVIDIA Nemotron-3-Nano-4B · thinking · RAG · web search · coding help")

for msg in st.session_state.messages:
    render_message(msg)

prompt = st.chat_input("Message Nemotron…", disabled=not server_ok)
if prompt:
    history_for_model = list(st.session_state.messages)  # prior turns only
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ---- gather ephemeral context (explicit toggles control this) ----
    rag_chunks = []
    web_results = []
    sources = {"rag": [], "web": []}

    if web_enabled:
        web_cfg = dataclasses.replace(cfg.web_search, result_count=web_count,
                                      fetch_pages=web_fetch)
        try:
            with st.spinner("Searching the web…"):
                web_results = websearch.search(web_cfg, prompt)
            sources["web"] = [(r.get("title", ""), r.get("url", "")) for r in web_results]
        except Exception as exc:
            st.warning(f"Web search failed: {exc}")

    if rag_enabled and rag_ready and collection is not None and rag.count(collection) > 0:
        try:
            with st.spinner("Searching your documents…"):
                rag_chunks = rag.retrieve(collection, embedder, prompt,
                                          cfg.rag.top_k, cfg.rag.query_instruction)
            sources["rag"] = [
                (f"{c.source} p.{c.page}" if c.page else c.source) for c in rag_chunks
            ]
        except Exception as exc:
            st.warning(f"Document retrieval failed: {exc}")

    # ---- build the prompt and stream the answer ----
    messages = prompts.assemble_messages(cfg, mode, history_for_model, prompt,
                                         rag_chunks=rag_chunks, web_results=web_results)
    sampling = dataclasses.replace(cfg.sampling, temperature=temperature,
                                   top_p=top_p, max_new_tokens=max_new_tokens)
    tokenizer = get_tokenizer(cfg.tokenizer.repo_id, cfg.tokenizer.cache_dir)
    prompt_str = llm.build_prompt(tokenizer, messages, thinking=thinking,
                                  thinking_budget=cfg.reasoning.thinking_budget)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        acc = ""
        try:
            for piece in llm.stream_completion(handle, prompt_str, sampling):
                acc += piece
                placeholder.markdown(acc + " ▌")
        except Exception as exc:
            placeholder.error(f"Generation failed: {exc}")
        # When thinking is on, the template opened "<think>" in the prompt, so the
        # completion holds only the reasoning + closing "</think>". Prepend the
        # opening tag so split_think captures the reasoning (and truncations).
        raw = ("<think>\n" + acc) if thinking else acc
        think, answer = llm.split_think(raw)
        if not answer:
            answer = ("_(Response was truncated during reasoning — raise “Max new "
                      "tokens”.)_" if think else acc)
        placeholder.empty()
        if think:
            with st.expander("🧠 Thinking", expanded=False):
                st.markdown(think)
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "think": think, "sources": sources,
    })
