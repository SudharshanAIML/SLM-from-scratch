from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    batch_size: int = 8
    max_steps: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    log_every: int = 5

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
