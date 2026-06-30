"""LLM backend: llama-server lifecycle + chat streaming.

Launches the prebuilt `llama-server.exe` for the selected model (a local GGUF or an
`-hf` auto-download) and streams `/v1/chat/completions`. Templating is server-side:
`--jinja` makes llama-server apply each GGUF's own embedded chat template and stop
tokens, and `--reasoning-format` surfaces `<think>` reasoning in a separate field, so
the app sends `messages` (not a pre-rendered prompt) and works across model families.
The reasoning toggle is passed per-request via `chat_template_kwargs`. The app talks
HTTP only — one process, one copy of the weights.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import requests

from .config import AppConfig, ModelConfig, SamplingConfig

log = logging.getLogger(__name__)

# Keep Windows Job Object handles alive for the process lifetime. Closing a job
# handle is what triggers KILL_ON_JOB_CLOSE, so these must NOT be garbage
# collected while the app runs.
_JOB_HANDLES: list = []


@dataclass
class ServerHandle:
    """Reference to a running llama-server. `owned` means we started it."""
    base_url: str
    proc: Optional[subprocess.Popen]
    owned: bool


# --------------------------------------------------------------------------- server lifecycle
def _health_ok(base_url: str, timeout: float = 2.0) -> bool:
    try:
        return requests.get(base_url + "/health", timeout=timeout).status_code == 200
    except Exception:
        return False


def _loaded_model_basename(base_url: str, timeout: float = 2.0) -> Optional[str]:
    """Basename of the model a running server has loaded, via /props (best-effort).

    The field name varies across llama.cpp builds, so a few keys are tried.
    """
    try:
        j = requests.get(base_url + "/props", timeout=timeout).json()
    except Exception:
        return None
    if not isinstance(j, dict):
        return None
    for key in ("model_path", "model"):
        v = j.get(key)
        if isinstance(v, str) and v:
            return os.path.basename(v)
    gen = j.get("default_generation_settings") or {}
    v = gen.get("model") if isinstance(gen, dict) else None
    if isinstance(v, str) and v:
        return os.path.basename(v)
    return None


def _server_matches(model: ModelConfig, loaded_basename: Optional[str]) -> bool:
    """Best-effort check that a reusable server is running the expected model.

    Local models match on exact GGUF basename. Auto-download (-hf) models can't know
    the cached filename ahead of time, so they match heuristically on the quant tag
    plus a distinctive fragment of the repo id. A false negative just forces a manual
    "Free VRAM", which is the safe direction — it never serves the wrong weights.
    """
    if not loaded_basename:
        return False
    name = loaded_basename.lower()
    if model.gguf_path:
        return os.path.basename(model.gguf_path).lower() == name
    if model.quant and model.quant.lower() not in name:
        return False
    stem = model.hf_repo.split("/")[-1].lower().replace("-gguf", "")
    head = stem.split("_")[-1].split("-")[0]   # e.g. "qwen3", "phi", "gemma", "smollm3"
    return bool(head) and head in name


def _terminate(proc: Optional[subprocess.Popen]) -> None:
    """Terminate a server process we own, escalating to kill if needed."""
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()


def _assign_to_kill_on_close_job(proc: subprocess.Popen) -> None:
    """Tie `proc`'s lifetime to ours via a Windows Job Object (best-effort).

    `atexit` only runs on a *clean* interpreter exit, so closing the terminal
    window or killing the app from Task Manager would orphan llama-server.exe and
    leak its VRAM. A Job Object flagged KILL_ON_JOB_CLOSE fixes this: when the app
    dies for any reason, the OS closes the (un-inheritable) job handle, the job
    closes, and every process in it — the server — is killed too. No-op on
    non-Windows; failures are logged and ignored (atexit still covers clean exits).
    Nested jobs are allowed on Windows 8+, so this works even if Streamlit already
    runs inside a job.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        ULONGLONG = ctypes.c_uint64

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ULONGLONG),
                ("WriteOperationCount", ULONGLONG),
                ("OtherOperationCount", ULONGLONG),
                ("ReadTransferCount", ULONGLONG),
                ("WriteTransferCount", ULONGLONG),
                ("OtherTransferCount", ULONGLONG),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        JobObjectExtendedLimitInformation = 9  # JOBOBJECTINFOCLASS value

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info),
        ):
            raise ctypes.WinError(ctypes.get_last_error())

        if not kernel32.AssignProcessToJobObject(job, int(proc._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

        _JOB_HANDLES.append(job)  # keep alive so the job isn't closed early
    except Exception as exc:  # best-effort; atexit still handles clean exits
        log.warning("Could not attach llama-server to a kill-on-close job: %s", exc)


def shutdown_server(handle: "ServerHandle") -> str:
    """Stop a server this app started; return 'terminated' or 'external'.

    Only owned servers are killed — an attached external server (e.g. one the
    notebook launched) is left alone, so its VRAM must be freed where it was
    started. `_terminate` is poll-guarded, so calling this twice is safe.
    """
    if handle.owned and handle.proc is not None:
        _terminate(handle.proc)
        return "terminated"
    return "external"


def ensure_server(cfg: AppConfig, model: ModelConfig,
                  log_dir: Optional[str] = None) -> ServerHandle:
    """Detect a healthy server for `model` on host:port and reuse it; else launch one.

    Reuse avoids a second copy of the weights when the notebook (or a previous app run)
    already serves the SAME model. If a healthy server is up but running a DIFFERENT
    model, we refuse to attach — serving the wrong weights would be a silent
    correctness bug — and raise so the caller can free it. A freshly launched server is
    terminated on interpreter exit via atexit.
    """
    base_url = cfg.server.base_url
    if cfg.server.reuse_existing and _health_ok(base_url):
        loaded = _loaded_model_basename(base_url)
        if _server_matches(model, loaded):
            return ServerHandle(base_url=base_url, proc=None, owned=False)
        raise RuntimeError(
            f"A llama-server is already running on {cfg.server.host}:{cfg.server.port} "
            f"with a different model ('{loaded or 'unknown'}'); the selected model is "
            f"'{model.label}'. Stop it first (sidebar ♻️ Free VRAM, or stop the "
            f"notebook's server) and retry."
        )

    exe = cfg.server.binary_path
    if not os.path.isfile(exe):
        raise FileNotFoundError(
            f"llama-server.exe not found at {exe}. Run the nemotron-local notebook's "
            f"setup (Cell 01) first, or fix server.binary_path in config.yaml."
        )

    # Weight source: a local GGUF file, or an -hf repo:quant auto-download.
    if model.gguf_path:
        if not os.path.isfile(model.gguf_path):
            raise FileNotFoundError(
                f"GGUF for model '{model.label}' not found at {model.gguf_path}. Run "
                f"the nemotron-local notebook's download (Cell 02), or fix the "
                f"gguf_path for '{model.name}' in config.yaml."
            )
        source = ["-m", model.gguf_path]
    else:
        source = ["-hf", model.hf_spec]   # llama-server downloads + caches on first use

    ngl = cfg.server.n_gpu_layers if cfg.server.n_gpu_layers >= 0 else 99
    cmd = [
        exe, *source,
        "-ngl", str(ngl),
        "-c", str(model.n_ctx),
        "-fa", "on" if cfg.server.flash_attn else "off",
        "-np", "1",
        "--host", cfg.server.host,
        "--port", str(cfg.server.port),
    ]
    if cfg.server.flash_attn:   # quantized KV cache requires flash attention
        cmd += ["-ctk", cfg.server.cache_type_k, "-ctv", cfg.server.cache_type_v]
    if cfg.server.jinja:
        cmd.append("--jinja")
    if cfg.server.reasoning_format and cfg.server.reasoning_format.lower() != "none":
        cmd += ["--reasoning-format", cfg.server.reasoning_format]

    log_dir = log_dir or cfg.data_dir
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "server.log")
    log_fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    _assign_to_kill_on_close_job(proc)   # orphan-proof against abrupt app exit
    atexit.register(_terminate, proc)    # clean-exit path

    # Poll /health until ready, failing fast if the process dies early. The timeout is
    # generous (config) because an -hf model downloads several GB on first launch.
    t0 = time.time()
    while time.time() - t0 < cfg.server.startup_timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited during startup (code {proc.returncode}) while "
                f"loading '{model.label}'; see {log_path}."
            )
        if _health_ok(base_url):
            return ServerHandle(base_url=base_url, proc=proc, owned=True)
        time.sleep(1.0)

    _terminate(proc)
    raise TimeoutError(
        f"llama-server did not become healthy within {cfg.server.startup_timeout_s}s "
        f"while loading '{model.label}'; see {log_path}."
    )


# --------------------------------------------------------------------------- reasoning split
def split_think(text: str):
    """Return (thinking, visible_answer).

    Note: this is deliberately more robust than the notebook's version. The
    Nemotron chat template opens the `<think>` tag *inside the prompt* when
    thinking is enabled, so a streamed completion usually contains only the
    CLOSING `</think>` (e.g. "...reasoning...</think>answer"). The callers may
    therefore prepend a synthetic "<think>\n" before splitting. This handles all
    four combinations: opening tag present/absent and closed/unclosed.
    """
    if "</think>" in text:
        head, _, tail = text.partition("</think>")
        head = head.split("<think>", 1)[-1]   # drop the opening tag if present
        return head.strip(), tail.strip()
    if "<think>" in text:                      # opened but not closed (truncated)
        return text.split("<think>", 1)[-1].strip(), ""
    return "", text.strip()


# --------------------------------------------------------------------------- streaming
def stream_chat(handle: ServerHandle, messages: list, sampling: SamplingConfig, *,
                thinking: bool, supports_thinking: bool) -> Iterator[Tuple[str, str]]:
    """Yield (content_delta, reasoning_delta) from /v1/chat/completions (SSE).

    Sends `messages`; the server applies the model's own chat template (`--jinja`) and,
    with `--reasoning-format` set, returns reasoning in `delta.reasoning_content`
    separately from the visible `delta.content`. For models that expose a thinking
    switch, `chat_template_kwargs.enable_thinking` toggles it per request. Models that
    instead emit inline `<think>...</think>` in content leave reasoning_delta empty and
    are handled by `split_think` in the caller.
    """
    payload = dict(
        messages=messages,
        max_tokens=sampling.max_new_tokens,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        stream=True,
    )
    if supports_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    with requests.post(f"{handle.base_url}/v1/chat/completions", json=payload,
                       stream=True, timeout=600) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", "ignore")
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or ""
            if content or reasoning:
                yield content, reasoning
