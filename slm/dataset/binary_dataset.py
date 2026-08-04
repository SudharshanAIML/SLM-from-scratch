from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np


class BinaryDataset:
    def __init__(self, shard_dir: str | Path, dtype: str = "uint16") -> None:
        self.shard_dir = Path(shard_dir)
        self.dtype = dtype
        self.shards = sorted(self.shard_dir.glob("*.bin"))

    def __len__(self) -> int:
        return len(self.shards)

    def read_shard(self, shard_idx: int) -> np.ndarray:
        shard_path = self.shards[shard_idx]
        return np.memmap(shard_path, dtype=self.dtype, mode="r")

    def sample_window(self, shard_idx: int, offset: int, window: int) -> np.ndarray:
        shard = self.read_shard(shard_idx)
        if offset + window > len(shard):
            raise IndexError("Requested window exceeds shard size")
        return shard[offset : offset + window]


def pack_tokens(documents: Sequence[Sequence[int]], context_length: int, eos_token: int) -> List[np.ndarray]:
    packed: List[np.ndarray] = []
    current: List[int] = []
    for doc in documents:
        if current and len(current) + len(doc) + 1 > context_length:
            packed.append(np.array(current[:context_length], dtype=np.uint16))
            current = []
        current.extend(doc)
        current.append(eos_token)
    if current:
        packed.append(np.array(current[:context_length], dtype=np.uint16))
    return packed


def build_binary_shards(
    documents: Sequence[Sequence[int]],
    output_dir: str | Path,
    context_length: int = 4096,
    eos_token: int = 2,
    shard_size: int = 1_000_000,
    dtype: str = "uint16",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    packed = pack_tokens(documents, context_length=context_length, eos_token=eos_token)
    shards: List[Path] = []
    current_tokens: List[np.ndarray] = []
    current_len = 0

    for chunk in packed:
        current_tokens.append(chunk)
        current_len += len(chunk)
        if current_len >= shard_size:
            shard_path = output_dir / f"shard_{len(shards):03d}.bin"
            np.concatenate(current_tokens).astype(dtype).tofile(shard_path)
            shards.append(shard_path)
            current_tokens = []
            current_len = 0

    if current_tokens:
        shard_path = output_dir / f"shard_{len(shards):03d}.bin"
        np.concatenate(current_tokens).astype(dtype).tofile(shard_path)
        shards.append(shard_path)

    metadata = {
        "dtype": dtype,
        "context_length": context_length,
        "eos_token": eos_token,
        "num_shards": len(shards),
        "num_tokens": sum(len(chunk) for chunk in packed),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
