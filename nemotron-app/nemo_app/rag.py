"""RAG: document loading, hand-rolled chunking, embeddings, Chroma persistence.

Backend is Chroma (persistent on disk) so indexed documents survive app restarts
and carry metadata for citation. We always pass our own embeddings (bge-small via
sentence-transformers), so Chroma never invokes its default ONNX embedder.

Public surface used by the app:
    chunk_file(path, chunk_size, chunk_overlap) -> list[Chunk]
    add_documents(collection, embedder, chunks, query_instruction)   # upsert
    retrieve(collection, embedder, query, top_k, query_instruction) -> list[Chunk]
    count(collection) -> int
    clear(collection) -> None
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Chunk:
    text: str
    source: str        # original filename (basename)
    page: int          # 1-based page (PDF) or 0 when not paginated
    chunk_index: int   # running index within the file


# --------------------------------------------------------------------------- loaders
def _load_pdf(path: str) -> List[Tuple[str, int]]:
    """Return [(page_text, page_number)], 1-based page numbers."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    units: List[Tuple[str, int]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            units.append((text, i))
    return units


def _load_docx(path: str) -> List[Tuple[str, int]]:
    import docx  # python-docx

    document = docx.Document(path)
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    return [(text, 0)] if text.strip() else []


def _load_text(path: str) -> List[Tuple[str, int]]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    return [(text, 0)] if text.strip() else []


def load_file(path: str) -> List[Tuple[str, int]]:
    """Dispatch on extension -> list of (text, page) units."""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext == "pdf":
        return _load_pdf(path)
    if ext == "docx":
        return _load_docx(path)
    if ext in ("txt", "md", "markdown"):
        return _load_text(path)
    raise ValueError(f"Unsupported file type: .{ext}")


# --------------------------------------------------------------------------- chunking
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Recursive character splitter: split on the coarsest separator that fits.

    Greedily packs pieces up to chunk_size, then carries `chunk_overlap` characters
    of tail into the next chunk for context continuity.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # Pick the finest split that yields more than one piece.
    pieces = [text]
    for sep in _SEPARATORS:
        if sep == "":
            pieces = list(text)
            break
        if sep in text:
            pieces = [p for p in text.split(sep) if p != ""]
            pieces = [p + sep for p in pieces[:-1]] + [pieces[-1]]
            break

    chunks: List[str] = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current.strip():
                chunks.append(current.strip())
            if len(piece) > chunk_size:
                # A single piece is still too big -> recurse on it.
                chunks.extend(split_text(piece, chunk_size, chunk_overlap))
                current = ""
            else:
                overlap = current[-chunk_overlap:] if chunk_overlap else ""
                current = overlap + piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_file(path: str, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    """Load a file and split it into Chunks with source/page metadata."""
    source = os.path.basename(path)
    out: List[Chunk] = []
    idx = 0
    for text, page in load_file(path):
        for piece in split_text(text, chunk_size, chunk_overlap):
            out.append(Chunk(text=piece, source=source, page=page, chunk_index=idx))
            idx += 1
    return out


# --------------------------------------------------------------------------- embeddings + store
def _embed(embedder, texts: List[str]) -> List[List[float]]:
    """Normalised embeddings (cosine-ready) as plain lists for Chroma."""
    vecs = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.tolist()


def _chunk_id(c: Chunk) -> str:
    """Stable content-addressed id so re-indexing the same file upserts, not duplicates."""
    h = hashlib.md5(f"{c.source}|{c.page}|{c.chunk_index}|{c.text}".encode("utf-8")).hexdigest()
    return h


def add_documents(collection, embedder, chunks: List[Chunk], query_instruction: str = "") -> int:
    """Upsert chunks into the collection. Documents are embedded WITHOUT the query
    instruction (that prefix is for queries only). Returns the number added."""
    if not chunks:
        return 0
    ids = [_chunk_id(c) for c in chunks]
    docs = [c.text for c in chunks]
    metas = [{"source": c.source, "page": c.page, "chunk_index": c.chunk_index} for c in chunks]
    embeddings = _embed(embedder, docs)
    collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    return len(ids)


def retrieve(collection, embedder, query: str, top_k: int,
             query_instruction: str = "") -> List[Chunk]:
    """Return the top_k most similar chunks for a query."""
    if count(collection) == 0:
        return []
    q_emb = _embed(embedder, [query_instruction + query])
    res = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    out: List[Chunk] = []
    for text, meta in zip(docs, metas):
        meta = meta or {}
        out.append(Chunk(
            text=text,
            source=meta.get("source", "?"),
            page=int(meta.get("page", 0)),
            chunk_index=int(meta.get("chunk_index", 0)),
        ))
    return out


def count(collection) -> int:
    try:
        return collection.count()
    except Exception:
        return 0


def clear(collection) -> None:
    """Delete every item but keep the same collection object (cache-friendly)."""
    try:
        existing = collection.get(include=[])  # ids are always returned
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)
    except Exception:
        pass
