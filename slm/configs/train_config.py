from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    log_every: int = 5
    save_every: int = 500
    eval_every: int = 500
    use_gradient_checkpointing: bool = False
    use_mixed_precision: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    @classmethod
    def rtx3060_v1(cls) -> "TrainConfig":
        """Config optimized for RTX 3060 12GB VRAM."""
        return cls(
            batch_size=4,
            gradient_accumulation_steps=4,
            max_steps=100000,
            learning_rate=5e-4,
            weight_decay=0.1,
            warmup_steps=1000,
            max_grad_norm=1.0,
            device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            checkpoint_dir="checkpoints",
            log_every=10,
            save_every=500,
            eval_every=1000,
            use_gradient_checkpointing=True,
            use_mixed_precision=False,
        )
