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
