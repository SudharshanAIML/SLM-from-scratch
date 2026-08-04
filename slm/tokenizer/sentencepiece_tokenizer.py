from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import sentencepiece as spm


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self._sp = None
        if self.model_path is not None and self.model_path.exists():
            self.load(self.model_path)

    def train(self, input_texts: Iterable[str], model_path: str | Path, vocab_size: int = 50000, model_type: str = "bpe") -> None:
        output_path = Path(model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        training_file = output_path.with_suffix(".txt")
        training_file.write_text("\n".join(input_texts), encoding="utf-8")
        spm.SentencePieceTrainer.train(
            input=str(training_file),
            model_prefix=str(output_path.with_suffix("")),
            vocab_size=vocab_size,
            model_type=model_type,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=["<mask>"]
        )
        self.model_path = output_path.with_suffix(".model")
        self.load(self.model_path)

    def load(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._sp = spm.SentencePieceProcessor(model_file=str(self.model_path))

    def encode(self, text: str) -> list[int]:
        if self._sp is None:
            raise RuntimeError("Tokenizer model is not loaded")
        return self._sp.EncodeAsIds(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        if self._sp is None:
            raise RuntimeError("Tokenizer model is not loaded")
        return self._sp.DecodeIds(list(token_ids))

    def save(self, path: str | Path) -> None:
        if self._sp is None:
            raise RuntimeError("Tokenizer model is not loaded")
        self.model_path = Path(path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._sp.save(str(self.model_path))

    @classmethod
    def load_from_file(cls, path: str | Path) -> "SentencePieceTokenizer":
        tokenizer = cls()
        tokenizer.load(path)
        return tokenizer
