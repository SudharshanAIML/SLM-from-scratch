from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from slm.configs.tokenizer_config import TokenizerConfig


class SimpleTokenizer:
    """A minimal tokenizer that works without external dependencies."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: list[str] = []
        self._token_pattern = re.compile(r"\w+|[^\w\s]")
        self._build_special_tokens()

    def _build_special_tokens(self) -> None:
        for token in [self.config.unk_token, self.config.bos_token, self.config.eos_token]:
            self._add_token(token)

    def _add_token(self, token: str) -> None:
        if token not in self.vocab:
            if self.config.vocab_size and len(self.vocab) >= self.config.vocab_size:
                return
            self.vocab[token] = len(self.inverse_vocab)
            self.inverse_vocab.append(token)

    def _normalize(self, text: str) -> str:
        if self.config.lowercase:
            text = text.lower()
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return text

    def _tokenize_text(self, text: str) -> list[str]:
        normalized = self._normalize(text)
        tokens = self._token_pattern.findall(normalized)
        return tokens if tokens else [self.config.unk_token]

    def train_from_texts(self, texts: Iterable[str]) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self._tokenize_text(text))

        ordered_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        for token, _ in ordered_tokens:
            self._add_token(token)

    def encode(self, text: str) -> list[int]:
        tokens = self._tokenize_text(text)
        return [self.vocab.get(token, self.vocab[self.config.unk_token]) for token in tokens]

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def decode(self, token_ids: Sequence[int]) -> str:
        tokens = [self.inverse_vocab[token_id] for token_id in token_ids if 0 <= token_id < len(self.inverse_vocab)]
        return "".join(tokens)

    def save(self, path: str | Path) -> None:
        payload = {
            "config": self.config.to_dict(),
            "vocab": self.vocab,
            "inverse_vocab": self.inverse_vocab,
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SimpleTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(TokenizerConfig.from_dict(payload["config"]))
        tokenizer.vocab = payload["vocab"]
        tokenizer.inverse_vocab = payload["inverse_vocab"]
        return tokenizer
