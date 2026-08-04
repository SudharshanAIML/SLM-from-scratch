from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class KVCache:
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    positions: list[int] = field(default_factory=list)

    def update(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k = k
        self.v = v
