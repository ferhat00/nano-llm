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
    startup_timeout_s: int
    reuse_existing: bool
    # Shared runtime flags applied to whichever model the server launches.
    flash_attn: bool
    n_gpu_layers: int
    cache_type_k: str
    cache_type_v: str
    jinja: bool             # apply each GGUF's embedded chat template (server-side)
    reasoning_format: str   # how the server surfaces <think> (e.g. "deepseek" | "none")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    max_new_tokens: int
    stop: Tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    """One selectable model. Weights are either a local GGUF (`gguf_path`) or
    auto-downloaded by llama-server from a Hugging Face repo (`hf_repo` + `quant`)."""
    name: str
    label: str
    gguf_path: Optional[str]
    hf_repo: Optional[str]
    quant: Optional[str]
    n_ctx: int
    supports_thinking: bool
    sampling: Optional[SamplingConfig]   # None -> fall back to AppConfig.sampling

    @property
    def identity(self) -> Tuple:
        """Hashable cache key — a change in any field here means relaunch the server."""
        return (self.name, self.gguf_path, self.hf_repo, self.quant, self.n_ctx)

    @property
    def hf_spec(self) -> Optional[str]:
        """The `-hf` argument (repo[:quant]) for auto-download, or None for a local file."""
        if not self.hf_repo:
            return None
        return f"{self.hf_repo}:{self.quant}" if self.quant else self.hf_repo


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
    models: Tuple[ModelConfig, ...]
    active_model_name: str
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


def _parse_model(entry: dict, app_dir: str) -> ModelConfig:
    """Build one ModelConfig, validating it names exactly one weight source."""
    gguf = entry.get("gguf_path")
    hf_repo = entry.get("hf_repo")
    quant = entry.get("quant")
    has_local = bool(gguf)
    has_hf = bool(hf_repo) and bool(quant)
    if has_local == has_hf:  # exactly one source required
        raise ValueError(
            f"model '{entry.get('name')}' must set either gguf_path OR (hf_repo + quant), "
            f"not both and not neither."
        )
    samp = entry.get("sampling")
    sampling = (
        SamplingConfig(
            temperature=float(samp["temperature"]),
            top_p=float(samp["top_p"]),
            max_new_tokens=int(samp["max_new_tokens"]),
            stop=tuple(samp.get("stop", ())),   # server-side templating handles stops
        )
        if samp else None
    )
    return ModelConfig(
        name=str(entry["name"]),
        label=str(entry["label"]),
        gguf_path=_resolve(app_dir, gguf) if gguf else None,
        hf_repo=hf_repo or None,
        quant=quant or None,
        n_ctx=int(entry["n_ctx"]),
        supports_thinking=bool(entry.get("supports_thinking", False)),
        sampling=sampling,
    )


def _parse_models(block: dict, app_dir: str) -> Tuple[Tuple[ModelConfig, ...], str]:
    """Parse the models block into (models, active_name), validating names."""
    available = block.get("available") or []
    if not available:
        raise ValueError("config 'models.available' is empty — at least one model is required.")
    models = tuple(_parse_model(e, app_dir) for e in available)
    names = [m.name for m in models]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate model name(s) in models.available: {dupes}")
    active = str(block["active"])
    if active not in names:
        raise ValueError(f"models.active='{active}' is not one of {names}")
    return models, active


def model_by_name(cfg: "AppConfig", name: str) -> ModelConfig:
    """Look up a model by its `name`; raises KeyError if absent."""
    for m in cfg.models:
        if m.name == name:
            return m
    raise KeyError(f"no model named '{name}' in config")


def load_config(path: Optional[str] = None) -> AppConfig:
    """Read config.yaml and build a validated AppConfig with resolved paths."""
    cfg_path = path or default_config_path()
    app_dir = os.path.dirname(os.path.abspath(cfg_path))
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    srv = raw["server"]
    server = ServerConfig(
        host=srv["host"],
        port=int(srv["port"]),
        binary_path=_resolve(app_dir, srv["binary_path"]),
        startup_timeout_s=int(srv["startup_timeout_s"]),
        reuse_existing=bool(srv["reuse_existing"]),
        flash_attn=bool(srv["flash_attn"]),
        n_gpu_layers=int(srv["n_gpu_layers"]),
        cache_type_k=str(srv["cache_type_k"]),
        cache_type_v=str(srv["cache_type_v"]),
        jinja=bool(srv["jinja"]),
        reasoning_format=str(srv["reasoning_format"]),
    )
    sampling = SamplingConfig(
        temperature=float(raw["sampling"]["temperature"]),
        top_p=float(raw["sampling"]["top_p"]),
        max_new_tokens=int(raw["sampling"]["max_new_tokens"]),
        stop=tuple(raw["sampling"]["stop"]),
    )
    models, active_model_name = _parse_models(raw["models"], app_dir)
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
        models=models,
        active_model_name=active_model_name,
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
