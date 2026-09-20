from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Decoder-only transformer configuration.

    The defaults are the 259M reference architecture:

        embedding          32,768 x 1,024              =  33,554,432
        attention/layer    (Q 1024x1024) + (K,V 1024x256 x2) + (O 1024x1024)
                                                       =   2,621,440
        swiglu/layer       3 x 1,024 x 2,816           =   8,650,752
        rmsnorm/layer      2 x 1,024                   =       2,048
        ------------------------------------------------------------
        per layer                                      =  11,274,240
        x 20 layers                                    = 225,484,800
        + embedding (tied with lm_head, counted once)  =  33,554,432
        ============================================================
        analytical total                               = 259,039,232
        + final RMSNorm (1,024)                        = 259,040,256
    """

    vocab_size: int = 32_768
    hidden_size: int = 1024
    num_layers: int = 20
    num_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 2816
    max_seq_len: int = 2048
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    attention_dropout: float = 0.0
    mlp_dropout: float = 0.0
    residual_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads}) for GQA"
            )
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must equal num_heads * head_dim "
                f"({self.num_heads} * {self.head_dim} = {self.num_heads * self.head_dim})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE")
        if self.vocab_size > 65_535:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) exceeds the uint16 range used by the "
                "binary shard format"
            )

    @property
    def num_kv_groups(self) -> int:
        """How many query heads share each key/value head."""
        return self.num_heads // self.num_kv_heads

    def parameter_breakdown(self) -> dict[str, int]:
        """Analytical parameter count, mirroring the architecture derivation."""
        h, d = self.hidden_size, self.head_dim
        q_dim = self.num_heads * d
        kv_dim = self.num_kv_heads * d

        embedding = self.vocab_size * h
        attention = h * q_dim + 2 * (h * kv_dim) + q_dim * h
        swiglu = 3 * h * self.intermediate_size
        norms = 2 * h
        per_layer = attention + swiglu + norms
        stack = per_layer * self.num_layers
        final_norm = h

        total = embedding + stack + final_norm
        if not self.tie_embeddings:
            total += self.vocab_size * h

        return {
            "embedding": embedding,
            "attention_per_layer": attention,
            "swiglu_per_layer": swiglu,
            "norms_per_layer": norms,
            "per_layer": per_layer,
            "transformer_stack": stack,
            "final_norm": final_norm,
            "total": total,
        }

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
    def debug(cls, vocab_size: int = 256, max_seq_len: int = 64) -> "ModelConfig":
        """Tiny configuration for tests and CPU smoke runs (~0.1M params)."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            head_dim=16,
            intermediate_size=128,
            max_seq_len=max_seq_len,
        )
