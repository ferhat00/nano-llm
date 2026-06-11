"""Generator for nemotron_nano_4b_kaggle.ipynb.

Run:  python build_notebook.py
Produces a fully runnable Kaggle notebook (12 core cells + markdown + functional
tests) for NVIDIA Nemotron Nano 4B (dense Mamba-2/Transformer hybrid) on a T4.

Authoring note: cell sources are wrapped in raw triple-single-quoted strings
(r'''...'''), and cell code uses only \"\"\" for docstrings, so nothing collides.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(src):  C.append(nbf.v4.new_markdown_cell(src))
def code(src): C.append(nbf.v4.new_code_cell(src))

# ----------------------------------------------------------------------------- title
md(r'''# Nemotron Nano 4B on a Kaggle T4 — Inference, Reasoning & Benchmark

Runs **NVIDIA Nemotron Nano 4B** (a *dense* **Mamba-2 + Transformer hybrid**, ~3.97B params)
on a single **Tesla T4** (16 GB, Turing, compute capability 7.5) end-to-end:
environment audit → model load → inference → reasoning ON/OFF → streaming →
multi-turn chat → batch → benchmark → teardown.

**Backend strategy**
- **PRIMARY — `llama-cpp-python` (GGUF, INT4 Q4_K_M).** llama.cpp ships its *own* Mamba-2
  CUDA kernels (it does **not** depend on the `mamba-ssm` package, whose fused kernels do not
  run on Turing), installs from a prebuilt CUDA-12 wheel, and fits the T4 with wide headroom.
- **FALLBACK — `transformers` fp16** (best-effort; may be slow/unsupported on Turing). The
  loader also retries an alternate GGUF quant on the same backend before falling through.

**Two corrections to common briefs about this model**
1. The 4B is **dense**, *not* MoE — only the larger `30B-A3B` sibling is MoE.
2. Repo names like `...-4B-Instruct` may not exist; the real family is
   `NVIDIA-Nemotron-3-Nano-4B-{BF16,FP8,GGUF}`. Cell 02 **resolves repo IDs at runtime**, so a
   one-line edit in `Config` adapts to whatever NVIDIA published.

> **Prereqs:** Settings → Accelerator = **GPU T4**, Internet = **ON**. The model is published
> under an open NVIDIA license; an `HF_TOKEN` Kaggle secret is *optional* and only used if present.
''')

# ----------------------------------------------------------------------------- explainer
md(r'''## Why a Mamba-2 hybrid needs special handling

A standard Transformer is all attention; this model interleaves **Mamba-2 state-space (SSM)
layers** with a *few* attention layers and MLPs (roughly 5:1 Mamba:attention). Consequences:

- **Most inference stacks need kernel-level SSM support.** Plain `transformers` alone is not
  enough on a T4 — the official fused Mamba kernels (`mamba-ssm`, `causal-conv1d`) target
  Ampere+ and are unreliable on Turing. **llama.cpp** carries an independent SSM implementation
  that runs on the T4, which is why it is the primary backend.
- **The KV cache is small.** Only the handful of attention layers grow a KV cache with sequence
  length; the Mamba layers keep a **fixed-size recurrent state**. That is why a 4B model handles
  very long contexts comfortably inside 16 GB.
- **The SSM state is *not* persisted across separate generate() calls** in these backends. The
  multi-turn chat loop therefore re-feeds the full conversation history each turn (standard, but
  worth stating explicitly).
''')

# ----------------------------------------------------------------------------- Cell 00
md(r'''## Cell 00 — GPU & environment audit
**Purpose:** confirm we are on a T4 (cc 7.5) and read the CUDA version that governs the wheel
choice. **Key API:** `nvidia-smi`, `torch.version.cuda`, `torch.cuda.get_device_capability`.
**Est. runtime:** ~5 s · **VRAM Δ:** 0.''')
code(r'''import subprocess, sys, platform

def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except Exception as e:
        return f"(failed: {e})"

print("Python :", sys.version.split()[0], "|", platform.platform())
print("=" * 70)
print(_run(["nvidia-smi"]))
print("=" * 70)
try:
    import torch
    print("torch          :", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("cuda available :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device         :", torch.cuda.get_device_name(0))
        print("compute cap.   :", torch.cuda.get_device_capability(0), "(expect (7, 5) on a T4)")
except Exception as e:
    print("torch not importable yet:", e)
''')

# ----------------------------------------------------------------------------- Config
md(r'''## Configuration
All tunable parameters live here (`@dataclass Config`) — no magic numbers downstream. Override a
repo name, quant, context length, or generation setting in one place.''')
code(r'''from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Config:
    # --- model targeting (edit these if NVIDIA's repo names differ) -----------------
    quant: str = "Q4_K_M"                       # "Q4_K_M" (INT4, default) | "Q8_0" (INT8)
    gguf_candidates: Tuple[str, ...] = (
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
        "unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF",
        "lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF",
    )
    tokenizer_candidates: Tuple[str, ...] = (
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "unsloth/NVIDIA-Nemotron-3-Nano-4B",
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",    # template-compatible last resort
    )
    transformers_repo: str = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"   # fp16 fallback backend

    # --- runtime ------------------------------------------------------------------
    n_ctx: int = 8192                # plenty for the tests; raise for long-context work
    n_gpu_layers: int = -1           # -1 = offload every layer to the T4
    n_threads: Optional[int] = None  # None -> os.cpu_count()
    flash_attn: bool = False         # leave off on Turing for safety

    # --- generation ---------------------------------------------------------------
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95

    # --- reasoning ----------------------------------------------------------------
    thinking: bool = True
    thinking_budget: Optional[int] = None   # None = uncapped; int = best-effort cap

    # --- io / cache ---------------------------------------------------------------
    stream: bool = True
    hf_token_env: str = "HF_TOKEN"
    cache_dir: str = "/kaggle/working/hf-cache"
    model_dir: str = "/kaggle/working/models"

    # --- benchmark ----------------------------------------------------------------
    bench_max_new_tokens: int = 128
    bench_prompts: Tuple[str, ...] = (
        "Explain what a state-space model is in two sentences.",
        "Write a haiku about gradient descent.",
        "List three real-world uses of the Fibonacci sequence.",
    )

CFG = Config()
print(CFG)
''')

# ----------------------------------------------------------------------------- Cell 01
md(r'''## Cell 01 — Install the primary backend (CUDA-aware)
**Purpose:** install a **Turing-safe CUDA-12 `llama-cpp-python` wheel** chosen from the detected
CUDA version, plus support libs. **Key API:** `pip --extra-index-url`. **Est. runtime:** 2–4 min
· **VRAM Δ:** 0. Falls back to a source build, then prints the exact manual command if all fails.''')
code(r'''import os, sys, subprocess

def _pip(args):
    print(">>> pip", " ".join(args))
    return subprocess.run([sys.executable, "-m", "pip"] + args).returncode

def detect_cuda_tag(default="cu121"):
    try:
        import torch
        cu = torch.version.cuda          # e.g. "12.1"
    except Exception:
        cu = None
    if not cu:
        return default
    major, _, minor = cu.partition(".")
    tag = f"cu{major}{minor}"
    known = {"cu118", "cu121", "cu122", "cu123", "cu124", "cu125"}  # all cover sm_75 (Turing)
    if tag in known:
        return tag
    if major == "12":
        return "cu125"                   # nearest CUDA-12 wheel; avoid cu13x (drops pre-Turing)
    if major == "11":
        return "cu118"
    print(f"[warn] CUDA {cu}: no known Turing-safe wheel; defaulting to {default}")
    return default

CUDA_TAG = detect_cuda_tag()
INDEX = f"https://abetlen.github.io/llama-cpp-python/whl/{CUDA_TAG}"
MANUAL_CMD = f"pip install --upgrade llama-cpp-python --extra-index-url {INDEX}"
print("llama-cpp-python wheel index:", INDEX)

rc = _pip(["install", "-q", "--upgrade", "llama-cpp-python", "--extra-index-url", INDEX])
if rc != 0:
    print("\n[!] Prebuilt wheel failed; attempting CUDA source build (needs nvcc)...")
    os.environ["CMAKE_ARGS"] = "-DGGML_CUDA=on"
    rc = _pip(["install", "-q", "--upgrade", "--no-cache-dir", "llama-cpp-python"])
    if rc != 0:
        print("\n[X] Install failed. Run this manually, then re-run the cell:\n   ", MANUAL_CMD)

_pip(["install", "-q", "huggingface_hub", "hf_transfer", "transformers", "pandas", "tqdm"])
print("install step done.")
''')

# ----------------------------------------------------------------------------- Cell 02
md(r'''## Cell 02 — HF cache config, repo resolver & download
**Purpose:** redirect the HF cache to `/kaggle/working/` (the home dir is tiny), optionally load
an `HF_TOKEN` secret, **resolve the real repo IDs at runtime**, and download the GGUF + tokenizer
with progress bars. **Key API:** `huggingface_hub.list_repo_files`, `hf_hub_download`.
**Est. runtime:** 1–3 min cold / ~5 s warm · **VRAM Δ:** 0.''')
code(r'''import os
# Redirect caches BEFORE importing huggingface_hub.
os.environ.setdefault("HF_HOME", CFG.cache_dir)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(CFG.cache_dir, "hub"))
os.makedirs(CFG.cache_dir, exist_ok=True)
os.makedirs(CFG.model_dir, exist_ok=True)
try:
    import hf_transfer  # noqa: F401  (fast downloads)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except Exception:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

def load_hf_token(cfg) -> Optional[str]:
    """HF_TOKEN is OPTIONAL (model is open). Use env, else a Kaggle secret if present."""
    tok = os.environ.get(cfg.hf_token_env, "")
    if tok:
        print("[hf] token from environment"); return tok
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret(cfg.hf_token_env)
        os.environ[cfg.hf_token_env] = tok
        print("[hf] token from Kaggle secret"); return tok
    except Exception:
        print("[hf] no token (fine — model is open)"); return None

HF_TOKEN = load_hf_token(CFG)

from huggingface_hub import hf_hub_download, list_repo_files

def resolve_gguf(cfg, token, quant=None):
    """First candidate repo exposing a .gguf for the requested quant -> (repo, filename)."""
    want = (quant or cfg.quant).lower()
    for repo in cfg.gguf_candidates:
        try:
            files = list_repo_files(repo, token=token)
        except Exception as e:
            print(f"[resolve] {repo}: unavailable ({type(e).__name__})"); continue
        ggufs = [f for f in files if f.lower().endswith(".gguf")]
        match = [f for f in ggufs if want in f.lower()]
        pick = match or ggufs
        if pick:
            chosen = sorted(pick, key=len)[0]      # avoid multi-part split files
            print(f"[resolve] GGUF -> {repo} :: {chosen}")
            return repo, chosen
        print(f"[resolve] {repo}: no .gguf matching '{want}'")
    raise FileNotFoundError("No GGUF repo resolved — edit CFG.gguf_candidates.")

def resolve_tokenizer(cfg, token) -> Optional[str]:
    wanted = {"tokenizer.json", "tokenizer.model", "tokenizer_config.json"}
    for repo in cfg.tokenizer_candidates:
        try:
            if wanted & set(list_repo_files(repo, token=token)):
                print(f"[resolve] tokenizer -> {repo}"); return repo
        except Exception as e:
            print(f"[resolve] tokenizer {repo}: unavailable ({type(e).__name__})")
    print("[resolve] no tokenizer repo; will rely on GGUF metadata templating")
    return None

GGUF_REPO, GGUF_FILE = resolve_gguf(CFG, HF_TOKEN)
TOKENIZER_REPO = resolve_tokenizer(CFG, HF_TOKEN)

print(f"\nDownloading {GGUF_REPO}/{GGUF_FILE} ...")
GGUF_PATH = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE,
                            local_dir=CFG.model_dir, token=HF_TOKEN)
print("GGUF path :", GGUF_PATH)
print("GGUF size : %.2f GB" % (os.path.getsize(GGUF_PATH) / 1e9))
''')

# ----------------------------------------------------------------------------- utilities
md(r'''## VRAM utilities
**Important:** `torch.cuda.*` cannot see llama.cpp's GPU allocations (they bypass PyTorch's
allocator), so we use **`nvidia-smi`** as the authoritative VRAM meter for the GGUF backend.
`torch.cuda.memory_summary()` is still logged for the transformers fallback path.''')
code(r'''import gc, subprocess

def gpu_mem_mb(index: int = 0) -> float:
    """Authoritative GPU memory used (MiB) via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader",
             "-i", str(index)],
            capture_output=True, text=True).stdout.strip().splitlines()
        return float(out[0]) if out else float("nan")
    except Exception:
        return float("nan")

def free_vram() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

print("baseline VRAM:", gpu_mem_mb(), "MiB")
''')

# ----------------------------------------------------------------------------- Cell 03
md(r'''## Cell 03 — Model load + VRAM verification (with graceful fallback)
**Purpose:** load the GGUF on the T4 (`n_gpu_layers=-1`); on failure, retry an alternate quant,
then transformers fp16. **Key API:** `llama_cpp.Llama`. **Est. runtime:** 30–60 s
· **VRAM Δ:** +~3 GB (INT4). `load_model(config) -> (model, tokenizer)`.''')
code(r'''import os, time

class NemotronModel:
    """Backend-agnostic wrapper so generate()/benchmark() do not branch everywhere."""
    def __init__(self, backend: str, raw, tokenizer, meta: Optional[dict] = None):
        self.backend = backend            # "llama.cpp" | "transformers"
        self.raw = raw
        self.tokenizer = tokenizer
        self.meta = meta or {}

def _load_llama_cpp(path: str, cfg: Config):
    from llama_cpp import Llama
    kw = dict(model_path=path, n_gpu_layers=cfg.n_gpu_layers, n_ctx=cfg.n_ctx,
              n_threads=cfg.n_threads or os.cpu_count(), verbose=False)
    try:
        return Llama(flash_attn=cfg.flash_attn, **kw)
    except TypeError:
        return Llama(**kw)                # older llama-cpp-python without flash_attn kwarg

def _load_transformers(repo: str, cfg: Config, token: Optional[str]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, token=token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo, token=token, trust_remote_code=True,
        torch_dtype=torch.float16,        # Turing has no native bf16
        device_map="auto")
    return model, tok

def load_model(config: Config):
    """Primary GGUF/llama.cpp -> alternate GGUF quant -> transformers fp16 (best-effort)."""
    from transformers import AutoTokenizer
    tok = None
    if TOKENIZER_REPO:
        try:
            tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO, token=HF_TOKEN, trust_remote_code=True)
        except Exception as e:
            print("[load] tokenizer load failed (continuing):", e)

    before, t0 = gpu_mem_mb(), time.time()
    try:
        llm = _load_llama_cpp(GGUF_PATH, config)
        after = gpu_mem_mb()
        print(f"[load] PRIMARY llama.cpp OK in {time.time()-t0:.1f}s | "
              f"VRAM {before:.0f} -> {after:.0f} MiB (delta {after-before:.0f})")
        return NemotronModel("llama.cpp", llm, tok, {"path": GGUF_PATH}), tok
    except Exception as e:
        print("[load] PRIMARY llama.cpp failed:", repr(e))

    try:
        print("[load] FALLBACK A: alternate GGUF quant (Q8_0) on the same backend...")
        alt_repo, alt_file = resolve_gguf(config, HF_TOKEN, quant="Q8_0")
        alt_path = hf_hub_download(repo_id=alt_repo, filename=alt_file,
                                   local_dir=config.model_dir, token=HF_TOKEN)
        llm = _load_llama_cpp(alt_path, config)
        print("[load] alternate GGUF OK | VRAM", gpu_mem_mb(), "MiB")
        return NemotronModel("llama.cpp", llm, tok, {"path": alt_path}), tok
    except Exception as e:
        print("[load] FALLBACK A failed:", repr(e))

    try:
        print("[load] FALLBACK B: transformers fp16 (may be slow/unsupported on Turing)...")
        model, tok2 = _load_transformers(config.transformers_repo, config, HF_TOKEN)
        try:
            import torch
            if torch.cuda.is_available():
                print(torch.cuda.memory_summary())
        except Exception:
            pass
        return NemotronModel("transformers", model, tok2 or tok, {}), (tok2 or tok)
    except Exception as e:
        print("[load] FALLBACK B failed:", repr(e))
        print("\n[X] All backends failed. Retry the install manually:\n   ", MANUAL_CMD)
        raise RuntimeError("Could not load Nemotron on any backend") from e

MODEL, TOKENIZER = load_model(CFG)
print("active backend:", MODEL.backend)
''')

# ----------------------------------------------------------------------------- Cell 04
md(r'''## Cell 04 — Tokenizer & chat-template inspection
**Purpose:** render messages with the model's **native chat template** and expose the
`enable_thinking` switch + best-effort `thinking_budget`. **Key API:**
`tokenizer.apply_chat_template`. **Est. runtime:** ~5 s · **VRAM Δ:** 0.''')
code(r'''def supports_thinking_kwarg(tokenizer) -> bool:
    tmpl = getattr(tokenizer, "chat_template", None) or ""
    return "enable_thinking" in tmpl

def build_prompt(tokenizer, messages: list, thinking: bool = True,
                 thinking_budget: Optional[int] = None) -> str:
    """Messages -> prompt string via the native template (plain fallback if none)."""
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

if TOKENIZER is not None:
    print("chat_template present :", bool(getattr(TOKENIZER, "chat_template", None)))
    print("enable_thinking kwarg :", supports_thinking_kwarg(TOKENIZER))
    demo = [{"role": "user", "content": "Hello!"}]
    print("\n--- thinking ON  (first 400 chars) ---\n", build_prompt(TOKENIZER, demo, True)[:400])
    print("\n--- thinking OFF (first 400 chars) ---\n", build_prompt(TOKENIZER, demo, False)[:400])
else:
    print("No HF tokenizer resolved; llama.cpp will use the GGUF's embedded template.")
''')

# ----------------------------------------------------------------------------- generation core
md(r'''## Generation core — `generate()` and streaming
`generate(model, tokenizer, messages, config) -> str` honours `config.stream` and
`config.max_new_tokens`, and works on either backend.''')
code(r'''import sys, time

NEMO_STOPS = ["<|im_end|>", "<|endoftext|>", "</s>"]

def _llama_stream(llm, prompt, cfg, max_new):
    stream = llm.create_completion(
        prompt=prompt, max_tokens=max_new or cfg.max_new_tokens,
        temperature=cfg.temperature, top_p=cfg.top_p, stop=NEMO_STOPS, stream=True)
    for chunk in stream:
        yield chunk["choices"][0]["text"]

def _hf_stream(model, tok, prompt, cfg, max_new):
    from threading import Thread
    from transformers import TextIteratorStreamer
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kw = dict(**inputs, max_new_tokens=max_new or cfg.max_new_tokens,
              temperature=cfg.temperature, top_p=cfg.top_p,
              do_sample=cfg.temperature > 0, streamer=streamer)
    Thread(target=model.generate, kwargs=kw, daemon=True).start()
    for piece in streamer:
        yield piece

def stream_tokens(model, tokenizer, messages, cfg, max_new=None):
    prompt = build_prompt(tokenizer, messages, cfg.thinking, cfg.thinking_budget)
    if model.backend == "llama.cpp":
        yield from _llama_stream(model.raw, prompt, cfg, max_new)
    else:
        yield from _hf_stream(model.raw, model.tokenizer, prompt, cfg, max_new)

def generate(model, tokenizer, messages: list, config: Config) -> str:
    """Single-turn generation with streaming support toggle."""
    pieces = []
    for piece in stream_tokens(model, tokenizer, messages, config):
        pieces.append(piece)
        if config.stream:
            sys.stdout.write(piece); sys.stdout.flush()
    if config.stream:
        sys.stdout.write("\n")
    return "".join(pieces).strip()
''')

# ----------------------------------------------------------------------------- Cell 05
md(r'''## Cell 05 — Single-turn inference smoke test
**Purpose:** prove the full path works. **Key API:** `generate`. **Est. runtime:** 5–15 s
· **VRAM Δ:** + small (KV).''')
code(r'''ans = generate(MODEL, TOKENIZER,
               [{"role": "user", "content": "In one sentence, what is a language model?"}],
               Config(thinking=False, stream=True, max_new_tokens=120))
print("\n[returned]:", ans[:300])
''')

# ----------------------------------------------------------------------------- Cell 06
md(r'''## Cell 06 — Reasoning: thinking ON vs OFF
**Purpose:** show a measurable reasoning-token difference on the classic bat-and-ball puzzle.
**Key API:** `generate` + `<think>` parsing. **Est. runtime:** 20–40 s · **VRAM Δ:** + small.''')
code(r'''import re

def split_think(text: str):
    m = re.search(r"<think>(.*?)</think>(.*)", text, flags=re.S)
    return (m.group(1).strip(), m.group(2).strip()) if m else ("", text.strip())

def count_tokens(text: str) -> int:
    if TOKENIZER is not None and text:
        try:
            return len(TOKENIZER.encode(text))
        except Exception:
            pass
    return len(text.split())

PUZZLE = ("A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. "
          "How much does the ball cost? Show your reasoning.")
msgs = [{"role": "user", "content": PUZZLE}]

on  = generate(MODEL, TOKENIZER, msgs, Config(thinking=True,  stream=False, max_new_tokens=512))
off = generate(MODEL, TOKENIZER, msgs, Config(thinking=False, stream=False, max_new_tokens=512))
on_t, on_a   = split_think(on)
off_t, off_a = split_think(off)

print("THINKING ON  -> think=%4d tok | answer=%4d tok" % (count_tokens(on_t),  count_tokens(on_a)))
print("THINKING OFF -> think=%4d tok | answer=%4d tok" % (count_tokens(off_t), count_tokens(off_a)))
print("\n[ON  answer]", on_a[:280])
print("[OFF answer]", off_a[:280])
print("\nExpected: ON shows a populated <think> block (ball = $0.05); OFF answers directly.")
''')

# ----------------------------------------------------------------------------- Cell 07
md(r'''## Cell 07 — Streaming inference with TTFT
**Purpose:** live token stream + **time-to-first-token**. **Key API:** `create_completion(stream=True)`.
**Est. runtime:** 5–15 s · **VRAM Δ:** + small.''')
code(r'''def stream_with_ttft(model, tokenizer, messages, cfg, max_new=None) -> str:
    t0, ttft, n, out = time.time(), None, 0, []
    for piece in stream_tokens(model, tokenizer, messages, cfg, max_new):
        if ttft is None and piece.strip():
            ttft = time.time() - t0
        out.append(piece); n += 1
        sys.stdout.write(piece); sys.stdout.flush()
    dt = time.time() - t0
    print(f"\n\n[stream] TTFT={ttft or dt:.2f}s | {n} steps | "
          f"{n / max(dt, 1e-9):.1f} tok/s | {dt:.2f}s total")
    return "".join(out)

_ = stream_with_ttft(MODEL, TOKENIZER,
                     [{"role": "user", "content": "Count from 1 to 5, each with a fun fact."}],
                     Config(thinking=False))
''')

# ----------------------------------------------------------------------------- Cell 08
md(r'''## Cell 08 — Multi-turn chat loop (stateful history)
**Purpose:** interactive REPL with history and commands. **Key API:** `generate` + `input`.
Commands: `/reset`, `/thinking on|off`, `/budget N|off`, `/exit`.
**Reminder:** the Mamba state is *not* carried across calls, so the full history is re-fed each
turn. The call is left commented so *Run All* does not block on `input()`.''')
code(r'''def chat_loop(model, tokenizer, config: Config) -> None:
    """Interactive REPL with history and /reset, /thinking, /budget, /exit commands."""
    history, think, budget = [], config.thinking, config.thinking_budget
    print("Chat ready. Commands: /reset  /thinking on|off  /budget N|off  /exit")
    while True:
        try:
            user = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]"); return
        if not user:
            continue
        if user == "/exit":
            print("[exit]"); return
        if user == "/reset":
            history = []; print("[history cleared]"); continue
        if user.startswith("/thinking"):
            think = user.split()[-1].lower() == "on"; print(f"[thinking={think}]"); continue
        if user.startswith("/budget"):
            arg = user.split()[-1].lower()
            budget = None if arg == "off" else int(arg); print(f"[budget={budget}]"); continue
        history.append({"role": "user", "content": user})
        turn = Config(thinking=think, thinking_budget=budget, stream=True,
                      max_new_tokens=config.max_new_tokens)
        sys.stdout.write("Bot> ")
        reply = generate(model, tokenizer, history, turn)
        _, visible = split_think(reply)
        history.append({"role": "assistant", "content": visible or reply})

# chat_loop(MODEL, TOKENIZER, CFG)   # <- uncomment for an interactive session
print("chat_loop() defined — uncomment the line above to chat.")
''')

# ----------------------------------------------------------------------------- Cell 09
md(r'''## Cell 09 — Batch inference utility
**Purpose:** map a list of prompts to a list of responses with a progress bar. **Key API:**
`tqdm` + `generate`. **Est. runtime:** varies · **VRAM Δ:** + small.''')
code(r'''from tqdm.auto import tqdm

def batch_generate(model, tokenizer, prompts: list, config: Config) -> list:
    """List of prompts -> list of responses (non-streaming)."""
    bcfg = Config(thinking=config.thinking, stream=False, max_new_tokens=config.max_new_tokens)
    out = []
    for p in tqdm(prompts, desc="batch"):
        out.append(generate(model, tokenizer, [{"role": "user", "content": p}], bcfg))
    return out

_demo = batch_generate(MODEL, TOKENIZER, list(CFG.bench_prompts),
                       Config(thinking=False, max_new_tokens=120))
for q, r in zip(CFG.bench_prompts, _demo):
    print("Q:", q); print("A:", r[:200]); print("-" * 60)
''')

# ----------------------------------------------------------------------------- Cell 10
md(r'''## Cell 10 — Performance benchmark
**Purpose:** `benchmark(model, tokenizer, config) -> pd.DataFrame` with **TTFT, tps, tokens,
memory_mb**. **Key API:** `pandas`, `nvidia-smi`. **Est. runtime:** 1–2 min · **VRAM Δ:** + small.
Note: for llama.cpp each stream step is ~one token; for the transformers fallback `gen_tokens`
is a fragment count (approximate).''')
code(r'''import pandas as pd

def benchmark(model, tokenizer, config: Config) -> pd.DataFrame:
    """Run standard prompts; return a DataFrame of TTFT, tps, tokens, memory_mb."""
    bcfg = Config(thinking=False, stream=False, max_new_tokens=config.bench_max_new_tokens)
    rows = []
    for p in config.bench_prompts:
        t0, ttft, n = time.time(), None, 0
        for piece in stream_tokens(model, tokenizer, [{"role": "user", "content": p}],
                                   bcfg, config.bench_max_new_tokens):
            if ttft is None and piece.strip():
                ttft = time.time() - t0
            n += 1
        dt = time.time() - t0
        rows.append(dict(prompt=p[:38], ttft_s=round(ttft or dt, 3), gen_tokens=n,
                         total_s=round(dt, 3), tps=round(n / max(dt, 1e-9), 1),
                         memory_mb=round(gpu_mem_mb())))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nMEDIAN tps=%.1f | MEDIAN TTFT=%.2fs | peak mem=%d MiB"
          % (df["tps"].median(), df["ttft_s"].median(), int(df["memory_mb"].max())))
    return df

BENCH_DF = benchmark(MODEL, TOKENIZER, CFG)
''')

# ----------------------------------------------------------------------------- Functional tests
md(r'''## Functional test suite (Phase 5)
Five checks: factual, coding, reasoning ON, reasoning OFF (same prompt), and a ~4k-token
long-context retrieval. Outputs populate on the first Kaggle run.''')
code(r'''FUNCTIONAL = [
    ("1. factual",
     "What is the Mamba-2 architecture and how does it differ from Transformers?", False),
    ("2. coding",
     "Write a Python function that computes a prefix sum using numpy vectorisation.", False),
    ("3. reasoning ON",
     "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. "
     "How much does the ball cost? Show your reasoning.", True),
    ("4. reasoning OFF (same prompt)",
     "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. "
     "How much does the ball cost? Show your reasoning.", False),
]
for name, q, think in FUNCTIONAL:
    print("=" * 72, "\n#", name)
    out = generate(MODEL, TOKENIZER, [{"role": "user", "content": q}],
                   Config(thinking=think, stream=False, max_new_tokens=400))
    th, vis = split_think(out)
    if th:
        print(f"[<think> {count_tokens(th)} tokens]")
    print(vis[:700])
''')
code(r'''# 5. Long-context retrieval (~4k-token document with a planted needle)
needle = "NOTE: the vault passphrase is BLUE-HERON-42."
filler = ("TinyStories is a synthetic corpus of short, simple stories used to study how small "
          "language models acquire grammar and basic reasoning. ")
document = (filler * 70) + needle + (filler * 70)   # ~4k tokens
print("approx document tokens:", count_tokens(document))

retrieval = generate(
    MODEL, TOKENIZER,
    [{"role": "system", "content": "Answer using only the supplied document."},
     {"role": "user", "content": document +
      "\n\nQuestion: What is the vault passphrase? Reply with only the passphrase."}],
    Config(thinking=False, stream=False, max_new_tokens=32))
print("retrieved:", retrieval)
print("expected : BLUE-HERON-42")
''')

# ----------------------------------------------------------------------------- Cell 11
md(r'''## Cell 11 — Graceful teardown & VRAM release
**Purpose:** free the model and verify VRAM returns toward baseline. **Key API:**
`del`, `gc.collect`, `torch.cuda.empty_cache`. **Est. runtime:** ~2 s · **VRAM Δ:** − all.''')
code(r'''print("VRAM before teardown:", gpu_mem_mb(), "MiB")
try:
    if MODEL.backend == "llama.cpp" and hasattr(MODEL.raw, "close"):
        MODEL.raw.close()
    del MODEL
except Exception as e:
    print("teardown note:", e)
free_vram()
print("VRAM after  teardown:", gpu_mem_mb(), "MiB")
print("Done. Some VRAM may only fully release when the kernel restarts.")
''')

# ----------------------------------------------------------------------------- results md
md(r'''## Results — targets vs. observed

Fill the **Observed** column after running on a Kaggle T4 (paste from `BENCH_DF` / Cell 03).

| Metric | Target (T4, INT4 GGUF) | Observed |
|---|---|---|
| VRAM after model load | < 6 GB | `<run on Kaggle>` |
| Time-to-first-token (short prompt) | < 2 s | `<run on Kaggle>` |
| Sustained throughput | > 25 tok/s | `<run on Kaggle>` |
| Cold start (warm cache) | < 90 s | `<run on Kaggle>` |
| Peak VRAM | < 14 GB (2 GB headroom) | `<run on Kaggle>` |

**Known limitations**
1. These targets are *expected* ranges; this notebook was authored without a T4, so paste your
   real numbers above on first run.
2. Repo IDs/gating are resolved at runtime; if all candidates 404, edit `CFG.gguf_candidates`.
3. The FP8 variant will not run on Turing (no FP8 hardware); BF16 is emulated as fp16.
4. The transformers fallback may require `mamba-ssm`, which is unreliable on Turing — treat it as
   best-effort, not a guaranteed path.
5. The Mamba recurrent state is not carried across `generate()` calls; history is re-fed each turn.
''')

nb.cells = C
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
    "accelerator": "GPU",
}
out_path = "nemotron_nano_4b_kaggle.ipynb"
nbf.write(nb, out_path)
print(f"wrote {out_path} with {len(C)} cells")
