from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    """Training hyperparameters.

    `device` and `precision` default to "auto" so the same config runs
    unchanged on a CUDA box and on a CPU-only machine.
    """

    # Batching
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    context_length: int = 2048

    # Schedule
    max_steps: int = 50_000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 2_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Runtime
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    precision: str = "auto"  # "auto" | "bf16" | "fp16" | "fp32"
    use_gradient_checkpointing: bool = False
    num_workers: int = 2
    seed: int = 42

    # IO
    checkpoint_dir: str = "checkpoints"
    log_every: int = 10
    save_every: int = 1_000
    eval_every: int = 1_000
    eval_steps: int = 50
    keep_last_n_checkpoints: int = 3

    def resolve_device(self) -> str:
        """Pick the runtime device, honouring an explicit override."""
        import torch

        if self.device != "auto":
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(
                    f"device={self.device!r} was requested but CUDA is not available"
                )
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def resolve_precision(self, device: str | None = None) -> str:
        """Pick the autocast precision for the resolved device.

        CPU always runs fp32: autocast on CPU is slower than it is useful here,
        and fp16 has no CPU kernel coverage for this model.
        """
        import torch

        device = device or self.resolve_device()

        if self.precision != "auto":
            if self.precision != "fp32" and not device.startswith("cuda"):
                return "fp32"
            return self.precision

        if not device.startswith("cuda"):
            return "fp32"
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"

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
