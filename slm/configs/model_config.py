from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 1000
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 32
    intermediate_size: int = 512
    max_seq_len: int = 256
    dropout: float = 0.0
    tie_embeddings: bool = True
    norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    mlp_dropout: float = 0.0
    residual_dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
    @classmethod
    def small_v1(cls, vocab_size: int = 50000, max_seq_len: int = 2048) -> "ModelConfig":
        """Small model optimized for RTX 3060 with 12GB VRAM. ~90M params."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=512,
            num_layers=16,
            num_heads=8,
            num_kv_heads=2,
            head_dim=64,
            intermediate_size=2048,
            max_seq_len=max_seq_len,
            dropout=0.1,
            tie_embeddings=True,
            norm_eps=1e-5,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            residual_dropout=0.0,
        )

    @classmethod
    def medium_v1(cls, vocab_size: int = 50000, max_seq_len: int = 4096) -> "ModelConfig":
        """Medium model. ~251M params (requires more VRAM or gradient checkpointing)."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=896,
            num_layers=24,
            num_heads=14,
            num_kv_heads=4,
            head_dim=64,
            intermediate_size=2432,
            max_seq_len=max_seq_len,
            dropout=0.1,
            tie_embeddings=True,
            norm_eps=1e-5,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            residual_dropout=0.0,
        )