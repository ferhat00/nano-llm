"""Byte-level BPE tokenizer (HF tokenizers) with optional GPT-2 reuse.

Trains a small BPE on a stream of strings and serializes to a single
tokenizer.json file. A thin wrapper exposes a uniform encode/decode/vocab_size
API so the rest of the codebase doesn't care which backend is loaded.
"""

from __future__ import annotations

import os
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from tokenizers.decoders import ByteLevel as ByteLevelDec
from tokenizers.trainers import BpeTrainer


EOS_TOKEN = "<|endoftext|>"


class TokenizerWrapper:
    """Uniform interface around a HF tokenizers.Tokenizer."""

    def __init__(self, tok: Tokenizer, eos_id: int):
        self.tok = tok
        self.eos_id = eos_id

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    def save(self, path: str | os.PathLike) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.tok.save(str(path))


def _eos_id(tok: Tokenizer) -> int:
    tok_id = tok.token_to_id(EOS_TOKEN)
    if tok_id is None:
        # GPT-2 uses the same string for its EOS, so this normally resolves.
        # Fall back to id 0 only if absolutely no EOS is registered.
        raise ValueError(
            f"tokenizer has no {EOS_TOKEN!r}; pass a tokenizer that defines one."
        )
    return tok_id


def train_bpe(text_iter: Iterable[str], vocab_size: int, save_path: str | os.PathLike) -> TokenizerWrapper:
    """Train a byte-level BPE on an iterable of strings and save to disk."""
    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
    tok.decoder = ByteLevelDec()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS_TOKEN],
        initial_alphabet=ByteLevelPre.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(text_iter, trainer=trainer)
    wrapped = TokenizerWrapper(tok, eos_id=_eos_id(tok))
    wrapped.save(save_path)
    return wrapped


def load_tokenizer(path: str | os.PathLike) -> TokenizerWrapper:
    tok = Tokenizer.from_file(str(path))
    return TokenizerWrapper(tok, eos_id=_eos_id(tok))


def load_gpt2() -> TokenizerWrapper:
    """Reuse GPT-2's tokenizer (downloaded once from the HF hub by `tokenizers`)."""
    tok = Tokenizer.from_pretrained("gpt2")
    return TokenizerWrapper(tok, eos_id=_eos_id(tok))


def build_or_load(cfg_tokenizer: dict, text_iter_factory, save_path: str | os.PathLike,
                  force: bool = False) -> TokenizerWrapper:
    """Idempotent entry point used by orchestrate.py.

    - type=bpe  : train BPE if save_path missing (or --force), else load from save_path.
    - type=gpt2 : always reuse GPT-2 (no training).
    """
    ttype = cfg_tokenizer.get("type", "bpe")
    if ttype == "gpt2":
        return load_gpt2()
    if ttype != "bpe":
        raise ValueError(f"unknown tokenizer.type={ttype!r}")
    if os.path.exists(save_path) and not force:
        return load_tokenizer(save_path)
    vocab_size = int(cfg_tokenizer["vocab_size"])
    return train_bpe(text_iter_factory(), vocab_size=vocab_size, save_path=save_path)
