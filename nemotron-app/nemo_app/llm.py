"""LLM backend: llama-server lifecycle + prompt building + streaming.

Mirrors the proven patterns in
`nemotron-local/nemotron_nano_4b_local_rtx3060.ipynb` (Cells 03/04 and the
generation core): launch the prebuilt `llama-server.exe`, render prompts with the
HF tokenizer's `enable_thinking` switch, and stream raw `/v1/completions` so the
reasoning toggle works per-request. The app talks HTTP only — one process, one
copy of the weights.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional

import re
import requests

from .config import AppConfig, SamplingConfig

log = logging.getLogger(__name__)

# Stop strings for the Nemotron ChatML template (matches the notebook).
NEMO_STOPS: List[str] = ["<|im_end|>", "<|endoftext|>", "</s>"]

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


def ensure_server(cfg: AppConfig, log_dir: Optional[str] = None) -> ServerHandle:
    """Detect a healthy server on host:port and reuse it; otherwise launch one.

    Reuse avoids loading a second copy of the weights when the notebook (or a
    previous app run) already has the server up. A freshly launched server is
    terminated on interpreter exit via atexit.
    """
    base_url = cfg.server.base_url
    if cfg.server.reuse_existing and _health_ok(base_url):
        return ServerHandle(base_url=base_url, proc=None, owned=False)

    exe, gguf = cfg.server.binary_path, cfg.server.gguf_path
    if not os.path.isfile(exe):
        raise FileNotFoundError(
            f"llama-server.exe not found at {exe}. Run the nemotron-local notebook's "
            f"setup (Cell 01) first, or fix server.binary_path in config.yaml."
        )
    if not os.path.isfile(gguf):
        raise FileNotFoundError(
            f"GGUF model not found at {gguf}. Run the nemotron-local notebook's "
            f"download (Cell 02) first, or fix server.gguf_path in config.yaml."
        )

    ngl = cfg.server.n_gpu_layers if cfg.server.n_gpu_layers >= 0 else 999
    cmd = [
        exe, "-m", gguf,
        "-ngl", str(ngl),
        "-c", str(cfg.server.n_ctx),
        "-fa", "on" if cfg.server.flash_attn else "off",
        "--host", cfg.server.host,
        "--port", str(cfg.server.port),
    ]

    log_dir = log_dir or cfg.data_dir
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "server.log")
    log_fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    _assign_to_kill_on_close_job(proc)   # orphan-proof against abrupt app exit
    atexit.register(_terminate, proc)    # clean-exit path

    # Poll /health until ready, failing fast if the process dies early.
    t0 = time.time()
    while time.time() - t0 < cfg.server.startup_timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited during startup (code {proc.returncode}); "
                f"see {log_path}."
            )
        if _health_ok(base_url):
            return ServerHandle(base_url=base_url, proc=proc, owned=True)
        time.sleep(1.0)

    _terminate(proc)
    raise TimeoutError(
        f"llama-server did not become healthy within {cfg.server.startup_timeout_s}s; "
        f"see {log_path}."
    )


# --------------------------------------------------------------------------- prompt building
def supports_thinking_kwarg(tokenizer) -> bool:
    tmpl = getattr(tokenizer, "chat_template", None) or ""
    return "enable_thinking" in tmpl


def build_prompt(tokenizer, messages: list, thinking: bool = True,
                 thinking_budget: Optional[int] = None) -> str:
    """Messages -> prompt string via the native template (plain fallback if none).

    Lifted from the notebook so the app renders prompts identically. The
    `enable_thinking` kwarg is what toggles the model's `<think>...</think>` block.
    """
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        base = dict(tokenize=False, add_generation_prompt=True)
        if supports_thinking_kwarg(tokenizer):
            base["enable_thinking"] = bool(thinking)
            if thinking_budget is not None:           # best-effort across template variants
                for k in ("thinking_budget", "reasoning_budget", "max_thinking_tokens"):
                    try:
                        return tokenizer.apply_chat_template(messages, **{**base, k: thinking_budget})
                    except TypeError:
                        continue
        try:
            return tokenizer.apply_chat_template(messages, **base)
        except TypeError:
            base.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **base)
    # plain ChatML-ish fallback
    sys_txt = "\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages if m["role"] != "system")
    return (f"<|system|>\n{sys_txt}\n" if sys_txt else "") + convo + "<|assistant|>\n"


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
def stream_completion(handle: ServerHandle, prompt: str,
                      sampling: SamplingConfig) -> Iterator[str]:
    """Yield text deltas from llama-server's OpenAI-compatible /v1/completions.

    Sends the already-rendered raw prompt (which bakes in the thinking switch), so
    reasoning ON/OFF works per-call without relying on server-side templating.
    """
    payload = dict(
        prompt=prompt,
        max_tokens=sampling.max_new_tokens,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        stop=list(sampling.stop),
        stream=True,
    )
    with requests.post(f"{handle.base_url}/v1/completions", json=payload,
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
            txt = (obj.get("choices") or [{}])[0].get("text", "")
            if txt:
                yield txt
