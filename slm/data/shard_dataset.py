from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from slm.data.shard_writer import SHARD_DTYPE, SHARD_GLOB


class BinaryShardDataset(IterableDataset):
    """Memory-mapped next-token dataset over packed uint16 shards.

    Every yielded pair is exactly `context_length` tokens:

        x = tokens[i     : i + L]
        y = tokens[i + 1 : i + L + 1]

    Short tail windows are dropped rather than padded, so all batches collate
    without a ragged-shape error.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        context_length: int = 2048,
        shuffle: bool = True,
        infinite: bool = False,
        seed: int = 42,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.context_length = context_length
        self.shuffle = shuffle
        self.infinite = infinite
        self.seed = seed

        self.shard_paths = sorted(self.shard_dir.glob(SHARD_GLOB))
        if not self.shard_paths:
            raise FileNotFoundError(
                f"No shards matching {SHARD_GLOB!r} in {self.shard_dir}. "
                "Run scripts/preprocess.py first."
            )

        metadata_path = self.shard_dir / "metadata.json"
        self.metadata: dict = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )

    @property
    def vocab_size(self) -> int | None:
        return self.metadata.get("vocab_size")

    @property
    def num_tokens(self) -> int:
        if "num_tokens" in self.metadata:
            return int(self.metadata["num_tokens"])
        return sum(p.stat().st_size // 2 for p in self.shard_paths)

    def __len__(self) -> int:
        """Number of full windows in one pass (not used when infinite)."""
        return max(0, self.num_tokens - 1) // self.context_length

    def _shards_for_worker(self) -> list[Path]:
        """Split shards across DataLoader workers so nothing is yielded twice."""
        info = get_worker_info()
        if info is None:
            return list(self.shard_paths)
        return list(self.shard_paths[info.id :: info.num_workers])

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        shards = self._shards_for_worker()
        if not shards:
            return

        rng = random.Random(self.seed + worker_id)
        epoch = 0

        while True:
            order = list(shards)
            if self.shuffle:
                rng.shuffle(order)

            for path in order:
                tokens = np.memmap(path, dtype=SHARD_DTYPE, mode="r")
                limit = len(tokens) - self.context_length - 1
                if limit < 0:
                    continue

                starts = list(range(0, limit + 1, self.context_length))
                if self.shuffle:
                    rng.shuffle(starts)

                for start in starts:
                    window = np.asarray(
                        tokens[start : start + self.context_length + 1], dtype=np.int64
                    )
                    yield (
                        torch.from_numpy(window[:-1]),
                        torch.from_numpy(window[1:]),
                    )

                del tokens

            epoch += 1
            if not self.infinite:
                return
            rng = random.Random(self.seed + worker_id + epoch * 1_000_003)
