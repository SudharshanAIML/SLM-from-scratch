from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TrainingLogger:
    """Append-only JSONL log plus a readable console line."""

    def __init__(self, log_dir: str | Path, echo: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "history.jsonl"
        self.echo = echo

    def log(self, payload: dict[str, Any]) -> None:
        record = dict(payload)
        record["timestamp"] = time.time()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if self.echo:
            print(self.format(record), flush=True)

    @staticmethod
    def format(record: dict[str, Any]) -> str:
        parts = [f"step {record.get('step', 0):>7,}"]
        if "loss" in record:
            parts.append(f"loss {record['loss']:.4f}")
        if "val_loss" in record:
            parts.append(f"val_loss {record['val_loss']:.4f}")
        if "perplexity" in record:
            parts.append(f"ppl {record['perplexity']:.2f}")
        if "lr" in record:
            parts.append(f"lr {record['lr']:.2e}")
        if "grad_norm" in record:
            parts.append(f"gnorm {record['grad_norm']:.2f}")
        if "tokens_per_sec" in record:
            parts.append(f"tok/s {record['tokens_per_sec']:,.0f}")
        return " | ".join(parts)

    def read_history(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
