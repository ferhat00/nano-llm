"""Load and validate config.yaml into typed, hashable dataclasses.

Frozen dataclasses with tuple (not list) fields so the whole config can be used
as a Streamlit `@st.cache_resource` key if needed. Relative paths in the YAML are
resolved against the app directory (the folder containing config.yaml).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import yaml


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    binary_path: str
    gguf_path: str
    n_ctx: int
    n_gpu_layers: int
    flash_attn: bool
    startup_timeout_s: int
    reuse_existing: bool

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class TokenizerConfig:
    repo_id: str
    cache_dir: str


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    max_new_tokens: int
    stop: Tuple[str, ...]


@dataclass(frozen=True)
class ReasoningConfig:
    thinking_default: bool
    thinking_budget: Optional[int]


@dataclass(frozen=True)
class RagConfig:
    enabled_default: bool
    embedding_model: str
    embedding_device: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_context_chars: int
    persist_dir: str
    collection_name: str
    query_instruction: str
    supported_extensions: Tuple[str, ...]


@dataclass(frozen=True)
class WebSearchConfig:
    enabled_default: bool
    provider: str
    result_count: int
    fetch_pages: bool
    fetch_char_limit: int
    request_timeout_s: int
    total_time_budget_s: int
    max_context_chars: int


@dataclass(frozen=True)
class PromptsConfig:
    chat_system: str
    coding_system: str
    rag_system: str
    web_system: str


@dataclass(frozen=True)
class UiConfig:
    temperature_range: Tuple[float, float]
    top_p_range: Tuple[float, float]
    max_new_tokens_range: Tuple[int, int]


@dataclass(frozen=True)
class GpuConfig:
    show_vram: bool
    refresh_seconds: int
    device_index: int


@dataclass(frozen=True)
class AppConfig:
    server: ServerConfig
    tokenizer: TokenizerConfig
    sampling: SamplingConfig
    reasoning: ReasoningConfig
    rag: RagConfig
    web_search: WebSearchConfig
    prompts: PromptsConfig
    ui: UiConfig
    gpu: GpuConfig
    app_dir: str
    data_dir: str


def _resolve(base_dir: str, path: str) -> str:
    """Resolve `path` against `base_dir` if it is relative; normalise either way."""
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base_dir, path))


def default_config_path() -> str:
    """config.yaml sits next to the app, one level above this package."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def load_config(path: Optional[str] = None) -> AppConfig:
    """Read config.yaml and build a validated AppConfig with resolved paths."""
    cfg_path = path or default_config_path()
    app_dir = os.path.dirname(os.path.abspath(cfg_path))
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    server = ServerConfig(
        host=raw["server"]["host"],
        port=int(raw["server"]["port"]),
        binary_path=_resolve(app_dir, raw["server"]["binary_path"]),
        gguf_path=_resolve(app_dir, raw["server"]["gguf_path"]),
        n_ctx=int(raw["server"]["n_ctx"]),
        n_gpu_layers=int(raw["server"]["n_gpu_layers"]),
        flash_attn=bool(raw["server"]["flash_attn"]),
        startup_timeout_s=int(raw["server"]["startup_timeout_s"]),
        reuse_existing=bool(raw["server"]["reuse_existing"]),
    )
    tokenizer = TokenizerConfig(
        repo_id=raw["tokenizer"]["repo_id"],
        cache_dir=_resolve(app_dir, raw["tokenizer"]["cache_dir"]),
    )
    sampling = SamplingConfig(
        temperature=float(raw["sampling"]["temperature"]),
        top_p=float(raw["sampling"]["top_p"]),
        max_new_tokens=int(raw["sampling"]["max_new_tokens"]),
        stop=tuple(raw["sampling"]["stop"]),
    )
    reasoning = ReasoningConfig(
        thinking_default=bool(raw["reasoning"]["thinking_default"]),
        thinking_budget=raw["reasoning"]["thinking_budget"],
    )
    rag = RagConfig(
        enabled_default=bool(raw["rag"]["enabled_default"]),
        embedding_model=raw["rag"]["embedding_model"],
        embedding_device=raw["rag"]["embedding_device"],
        chunk_size=int(raw["rag"]["chunk_size"]),
        chunk_overlap=int(raw["rag"]["chunk_overlap"]),
        top_k=int(raw["rag"]["top_k"]),
        max_context_chars=int(raw["rag"]["max_context_chars"]),
        persist_dir=_resolve(app_dir, raw["rag"]["persist_dir"]),
        collection_name=raw["rag"]["collection_name"],
        query_instruction=raw["rag"]["query_instruction"],
        supported_extensions=tuple(str(e).lower().lstrip(".") for e in raw["rag"]["supported_extensions"]),
    )
    web_search = WebSearchConfig(
        enabled_default=bool(raw["web_search"]["enabled_default"]),
        provider=raw["web_search"]["provider"],
        result_count=int(raw["web_search"]["result_count"]),
        fetch_pages=bool(raw["web_search"]["fetch_pages"]),
        fetch_char_limit=int(raw["web_search"]["fetch_char_limit"]),
        request_timeout_s=int(raw["web_search"]["request_timeout_s"]),
        total_time_budget_s=int(raw["web_search"]["total_time_budget_s"]),
        max_context_chars=int(raw["web_search"]["max_context_chars"]),
    )
    prompts = PromptsConfig(
        chat_system=raw["prompts"]["chat_system"].strip(),
        coding_system=raw["prompts"]["coding_system"].strip(),
        rag_system=raw["prompts"]["rag_system"].strip(),
        web_system=raw["prompts"]["web_system"].strip(),
    )
    ui = UiConfig(
        temperature_range=tuple(float(x) for x in raw["ui"]["temperature_range"]),
        top_p_range=tuple(float(x) for x in raw["ui"]["top_p_range"]),
        max_new_tokens_range=tuple(int(x) for x in raw["ui"]["max_new_tokens_range"]),
    )
    gpu = GpuConfig(
        show_vram=bool(raw["gpu"]["show_vram"]),
        refresh_seconds=int(raw["gpu"]["refresh_seconds"]),
        device_index=int(raw["gpu"]["device_index"]),
    )

    return AppConfig(
        server=server,
        tokenizer=tokenizer,
        sampling=sampling,
        reasoning=reasoning,
        rag=rag,
        web_search=web_search,
        prompts=prompts,
        ui=ui,
        gpu=gpu,
        app_dir=app_dir,
        data_dir=os.path.join(app_dir, "data"),
    )
