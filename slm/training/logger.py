from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TrainingLogger:
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, Any]] = []

    def log(self, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload["timestamp"] = time.time()
        self.history.append(payload)
        with (self.log_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def read_history(self) -> list[dict[str, Any]]:
        history_path = self.log_dir / "history.jsonl"
        if not history_path.exists():
            return []
        return [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
