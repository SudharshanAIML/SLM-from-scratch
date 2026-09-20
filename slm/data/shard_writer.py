from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

SHARD_DTYPE = "uint16"
SHARD_GLOB = "shard_*.bin"


class BinaryShardWriter:
    """Stream token ids into fixed-size uint16 shards at constant memory.

    Tokens are buffered in a numpy array and flushed with `tofile`, so the
    writer never holds more than `buffer_size` tokens. Shards are plain
    little-endian uint16 arrays with no header, which is what makes them
    memory-mappable at train time.
    """

    def __init__(
        self,
        output_dir: str | Path,
        shard_size: int = 100_000_000,
        buffer_size: int = 1_000_000,
        vocab_size: int | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.vocab_size = vocab_size

        self._buffer = np.empty(buffer_size, dtype=SHARD_DTYPE)
        self._buffer_len = 0
        self._shard_index = 0
        self._tokens_in_shard = 0
        self._handle = None
        self.shards: list[Path] = []
        self.total_tokens = 0
        self._open_shard()

    def _open_shard(self) -> None:
        path = self.output_dir / f"shard_{self._shard_index:05d}.bin"
        # Plain binary write: no platform-specific flags, so this behaves
        # identically on Linux, macOS and Windows.
        self._handle = path.open("wb")
        self.shards.append(path)
        self._tokens_in_shard = 0
        self._shard_index += 1

    def write_document(self, token_ids: Sequence[int]) -> None:
        for token in token_ids:
            self.write(token)

    def write(self, token: int) -> None:
        token = int(token)
        if token < 0 or token > 65_535:
            raise ValueError(f"Token id {token} does not fit in uint16")
        if self.vocab_size is not None and token >= self.vocab_size:
            raise ValueError(
                f"Token id {token} is outside the declared vocab of {self.vocab_size}"
            )

        self._buffer[self._buffer_len] = token
        self._buffer_len += 1
        self._tokens_in_shard += 1
        self.total_tokens += 1

        if self._buffer_len == len(self._buffer):
            self.flush()
        if self._tokens_in_shard >= self.shard_size:
            self.flush()
            self._handle.close()
            self._open_shard()

    def flush(self) -> None:
        if self._buffer_len == 0:
            return
        if self._handle is None:
            raise RuntimeError("Shard writer is already closed")
        self._buffer[: self._buffer_len].tofile(self._handle)
        self._buffer_len = 0

    def close(self) -> None:
        self.flush()
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        # A trailing empty shard is possible when the token count lands exactly
        # on a shard boundary.
        if self.shards and self.shards[-1].stat().st_size == 0:
            self.shards.pop().unlink()

    def __enter__(self) -> "BinaryShardWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def verify_shard(path: str | Path, eos_id: int) -> dict[str, object]:
    """Checksum a shard and report its real document-separator count."""
    path = Path(path)
    data = np.memmap(path, dtype=SHARD_DTYPE, mode="r")
    return {
        "path": path.name,
        "token_count": int(len(data)),
        "eos_count": int(np.count_nonzero(np.asarray(data) == eos_id)),
        "sha256": hashlib.sha256(np.asarray(data).tobytes()).hexdigest(),
    }


def finalize_metadata(
    output_dir: str | Path,
    writer: "BinaryShardWriter",
    eos_id: int,
    vocab_size: int,
    num_documents: int,
    bos_id: int | None = None,
    shard_size: int = 100_000_000,
    metadata_extra: dict[str, object] | None = None,
    verify: bool = True,
) -> dict[str, object]:
    """Write metadata.json describing a finished shard directory.

    `eos_id` is recorded alongside the shards so a later reader can confirm the
    separator actually matches the tokenizer that produced them.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, object] = {
        "version": "2.0",
        "dtype": SHARD_DTYPE,
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "bos_id": bos_id,
        "num_documents": num_documents,
        "num_tokens": writer.total_tokens,
        "num_shards": len(writer.shards),
        "shard_size": shard_size,
        "packing": "continuous",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    if verify:
        metadata["verification"] = [verify_shard(p, eos_id) for p in writer.shards]

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def write_shards(
    documents: Iterable[Sequence[int]],
    output_dir: str | Path,
    eos_id: int,
    vocab_size: int,
    shard_size: int = 100_000_000,
    add_bos: bool = False,
    bos_id: int | None = None,
    metadata_extra: dict[str, object] | None = None,
    verify: bool = True,
) -> dict[str, object]:
    """Pack an iterable of tokenized documents into one shard directory.

    Convenience wrapper around BinaryShardWriter + finalize_metadata for
    single-split writes (tests, small corpora). The FineWeb pipeline drives the
    writer directly so it can fill train/ and val/ in one pass.
    """
    output_dir = Path(output_dir)
    num_documents = 0

    with BinaryShardWriter(
        output_dir, shard_size=shard_size, vocab_size=vocab_size
    ) as writer:
        for doc in documents:
            if not len(doc):
                continue
            num_documents += 1
            if add_bos:
                writer.write(bos_id if bos_id is not None else 2)
            writer.write_document(doc)
            writer.write(eos_id)

    return finalize_metadata(
        output_dir,
        writer,
        eos_id=eos_id,
        vocab_size=vocab_size,
        num_documents=num_documents,
        bos_id=bos_id if add_bos else None,
        shard_size=shard_size,
        metadata_extra=metadata_extra,
        verify=verify,
    )
