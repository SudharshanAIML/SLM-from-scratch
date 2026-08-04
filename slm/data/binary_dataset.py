from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset


class BinaryShardDataset(IterableDataset):
    """Memory-mapped dataset that reads from binary shards."""

    def __init__(
        self,
        shard_dir: str | Path,
        context_length: int = 4096,
        dtype: str = "uint16",
        shuffle_shards: bool = True,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.context_length = context_length
        self.dtype = dtype
        self.shard_paths = sorted(self.shard_dir.glob("shard_*.bin"))

        if not self.shard_paths:
            raise FileNotFoundError(f"No binary shards found in {self.shard_dir}")

        self.shard_cache: dict[int, np.memmap] = {}
        if shuffle_shards:
            random.shuffle(self.shard_paths)

    def _get_shard(self, shard_idx: int) -> np.memmap:
        if shard_idx not in self.shard_cache:
            self.shard_cache[shard_idx] = np.memmap(self.shard_paths[shard_idx], dtype=self.dtype, mode="r")
        return self.shard_cache[shard_idx]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for shard_idx in range(len(self.shard_paths)):
            shard = self._get_shard(shard_idx)
            shard_len = len(shard)

            for offset in range(0, max(1, shard_len - self.context_length), self.context_length):
                end = min(offset + self.context_length + 1, shard_len)
                if end - offset < 2:
                    continue

                tokens = shard[offset:end]
                x = torch.tensor(tokens[:-1], dtype=torch.long)
                y = torch.tensor(tokens[1:], dtype=torch.long)
                yield x, y
