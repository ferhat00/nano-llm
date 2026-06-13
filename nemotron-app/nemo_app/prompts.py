"""Prompt assembly: turn the per-turn state into a `messages` list for the model.

Context (RAG chunks, web results) is injected as ephemeral system messages placed
just before the current user turn, and is NOT persisted into chat history — so the
16k window stays clean and context is always freshly retrieved per question.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .config import AppConfig
from .rag import Chunk


def system_for_mode(cfg: AppConfig, mode: str) -> str:
    """Base system prompt for the selected mode."""
    if mode.lower() == "coding":
        return cfg.prompts.coding_system
    return cfg.prompts.chat_system


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


def build_rag_context_message(cfg: AppConfig, chunks: List[Chunk]) -> Optional[Dict]:
    """A system message embedding retrieved document excerpts, with citation tags."""
    if not chunks:
        return None
    blocks = []
    for c in chunks:
        tag = f"{c.source} p.{c.page}" if c.page else c.source
        blocks.append(f"[{tag}]\n{c.text}")
    context = _truncate("\n\n".join(blocks), cfg.rag.max_context_chars)
    return {"role": "system", "content": f"{cfg.prompts.rag_system}\n\n{context}"}


def build_web_context_message(cfg: AppConfig, results: List[Dict]) -> Optional[Dict]:
    """A system message embedding numbered web results, with citation tags."""
    if not results:
        return None
    blocks = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "").strip()
        url = r.get("url", "").strip()
        snippet = r.get("snippet", "").strip()
        blocks.append(f"[{i}] {title}\n{url}\n{snippet}")
    context = _truncate("\n\n".join(blocks), cfg.web_search.max_context_chars)
    return {"role": "system", "content": f"{cfg.prompts.web_system}\n\n{context}"}


def assemble_messages(
    cfg: AppConfig,
    mode: str,
    history: List[Dict],
    user_text: str,
    rag_chunks: Optional[List[Chunk]] = None,
    web_results: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Compose [system] + history + [ephemeral context] + [current user]."""
    messages: List[Dict] = [{"role": "system", "content": system_for_mode(cfg, mode)}]
    # history holds prior turns as {"role", "content"} (assistant content = visible answer).
    messages.extend({"role": m["role"], "content": m["content"]} for m in history)

    web_msg = build_web_context_message(cfg, web_results or [])
    if web_msg:
        messages.append(web_msg)
    rag_msg = build_rag_context_message(cfg, rag_chunks or [])
    if rag_msg:
        messages.append(rag_msg)

    messages.append({"role": "user", "content": user_text})
    return messages
