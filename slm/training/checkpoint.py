from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, path: str | Path | None = None) -> Path:
        checkpoint_path = Path(path) if path is not None else self.checkpoint_dir / f"model_step_{step}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def load(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str | Path) -> int:
        payload = torch.load(path, map_location="cpu")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        return int(payload.get("step", 0))
