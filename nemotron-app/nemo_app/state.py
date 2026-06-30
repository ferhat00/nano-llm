"""Cached resource factories — the linchpin for surviving Streamlit reruns.

Streamlit re-executes the whole script on every interaction, so every heavy object
(the llama-server handle, the embedding model, the Chroma collection) is created
inside an `@st.cache_resource` function and reused. The llama-server is therefore a
single long-lived subprocess, not reloaded per rerun.

Args prefixed with `_` are not hashed by Streamlit's cache (used to pass the
unhashable/rich config object through without re-keying the cache).
"""
from __future__ import annotations

import os
from typing import Optional

import streamlit as st

from . import llm
from .config import AppConfig, load_config


@st.cache_resource(show_spinner=False)
def get_config() -> AppConfig:
    return load_config()


@st.cache_resource(show_spinner="Starting llama-server (loading the model)…")
def get_server(model_identity, _cfg: AppConfig, _model) -> llm.ServerHandle:
    """Detect-or-start the llama-server for the active model.

    `model_identity` (the hashable tuple from `ModelConfig.identity`) is the real cache
    key, so selecting a different model is a cache miss → a fresh launch, and a stale
    handle for the wrong model can never be served. `_cfg` / `_model` are passed through
    unhashed (underscore prefix) so the rich config objects don't re-key the cache.
    """
    return llm.ensure_server(_cfg, _model, log_dir=_cfg.data_dir)


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder(model_id: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, device=device)


@st.cache_resource(show_spinner="Opening vector store…")
def get_chroma_collection(persist_dir: str, collection_name: str):
    import chromadb

    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    # embedding_function=None: we always pass our own embeddings, so Chroma never
    # needs (or downloads) its default ONNX embedder. Cosine space matches the
    # normalised bge embeddings.
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
