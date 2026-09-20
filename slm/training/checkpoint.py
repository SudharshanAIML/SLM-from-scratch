from __future__ import annotations

import re
from pathlib import Path

import torch

STEP_PATTERN = re.compile(r"model_step_(\d+)\.pt$")


class CheckpointManager:
    """Save/load training state, keeping only the most recent checkpoints."""

    def __init__(self, checkpoint_dir: str | Path, keep_last_n: int = 3) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

    # ------------------------------------------------------------- discovery

    def list_checkpoints(self) -> list[Path]:
        """Checkpoints ordered by step number.

        Sorting by filename would put model_step_1000 before model_step_999,
        so the step is parsed out and compared numerically.
        """
        found: list[tuple[int, Path]] = []
        for path in self.checkpoint_dir.glob("model_step_*.pt"):
            match = STEP_PATTERN.search(path.name)
            if match:
                found.append((int(match.group(1)), path))
        return [path for _, path in sorted(found)]

    def latest(self) -> Path | None:
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None

    # ------------------------------------------------------------------ save

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        scaler: torch.amp.GradScaler | None = None,
        extra: dict | None = None,
    ) -> Path:
        path = self.checkpoint_dir / f"model_step_{step}.pt"
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "model_config": model.config.to_dict() if hasattr(model, "config") else None,
            "torch_rng_state": torch.get_rng_state(),
        }
        if extra:
            payload.update(extra)

        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(path)  # atomic: a killed job never leaves a half-written file

        self._prune()
        return path

    def _prune(self) -> None:
        if self.keep_last_n <= 0:
            return
        checkpoints = self.list_checkpoints()
        for stale in checkpoints[: -self.keep_last_n]:
            stale.unlink(missing_ok=True)

    # ------------------------------------------------------------------ load

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        path: str | Path | None = None,
        scaler: torch.amp.GradScaler | None = None,
        map_location: str = "cpu",
        strict_config: bool = True,
    ) -> int:
        checkpoint_path = Path(path) if path is not None else self.latest()
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")

        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        saved_config = payload.get("model_config")
        current_config = model.config.to_dict() if hasattr(model, "config") else None
        if strict_config and saved_config and current_config:
            differing = {
                key: (saved_config.get(key), current_config.get(key))
                for key in set(saved_config) | set(current_config)
                if saved_config.get(key) != current_config.get(key)
            }
            if differing:
                raise ValueError(
                    "Checkpoint architecture does not match the current model.\n"
                    + "\n".join(
                        f"  {k}: checkpoint={s!r} current={c!r}"
                        for k, (s, c) in sorted(differing.items())
                    )
                )

        model.load_state_dict(payload["model_state"])
        if optimizer is not None and payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        if scaler is not None and payload.get("scaler_state"):
            scaler.load_state_dict(payload["scaler_state"])
        if payload.get("torch_rng_state") is not None:
            torch.set_rng_state(payload["torch_rng_state"].cpu().to(torch.uint8))

        return int(payload.get("step", 0))
