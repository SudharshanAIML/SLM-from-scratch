import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from slm.data.shard_dataset import BinaryShardDataset
from slm.data.shard_writer import BinaryShardWriter, verify_shard, write_shards

EOS = 3
BOS = 2


def make_shards(tmp_path, docs, **kwargs):
    kwargs.setdefault("shard_size", 10_000)
    return write_shards(docs, tmp_path, eos_id=EOS, vocab_size=1000, **kwargs)


def test_writer_round_trips_tokens(tmp_path):
    docs = [[10, 11, 12], [20, 21], [30]]
    meta = make_shards(tmp_path, docs)

    data = np.fromfile(tmp_path / "shard_00000.bin", dtype="uint16")
    assert data.tolist() == [10, 11, 12, EOS, 20, 21, EOS, 30, EOS]
    assert meta["num_documents"] == 3
    assert meta["num_tokens"] == 9


def test_eos_separator_is_the_tokenizer_eos(tmp_path):
    """Regression: the separator written must be the real EOS id.

    The previous pipeline wrote id 3 while its tokenizer mapped 3 to a common
    word, so documents were joined by an ordinary token and the recorded
    eos_count actually counted that word.
    """
    docs = [[100, 101], [102, 103]]
    meta = make_shards(tmp_path, docs)
    assert meta["eos_id"] == EOS
    assert meta["verification"][0]["eos_count"] == len(docs)

    data = np.fromfile(tmp_path / "shard_00000.bin", dtype="uint16")
    assert int(np.count_nonzero(data == EOS)) == len(docs)


def test_bos_is_prepended_when_requested(tmp_path):
    make_shards(tmp_path, [[50, 51]], add_bos=True, bos_id=BOS)
    data = np.fromfile(tmp_path / "shard_00000.bin", dtype="uint16")
    assert data.tolist() == [BOS, 50, 51, EOS]


def test_shard_rollover_and_no_trailing_empty_shard(tmp_path):
    with BinaryShardWriter(tmp_path, shard_size=10, buffer_size=4) as writer:
        for token in range(30):  # exactly 3 full shards
            writer.write(token)
    shards = sorted(tmp_path.glob("shard_*.bin"))
    assert len(shards) == 3
    assert all(p.stat().st_size == 20 for p in shards)
    assert np.concatenate([np.fromfile(p, dtype="uint16") for p in shards]).tolist() == list(range(30))


def test_out_of_range_tokens_are_rejected(tmp_path):
    with BinaryShardWriter(tmp_path, vocab_size=500) as writer:
        with pytest.raises(ValueError, match="uint16"):
            writer.write(70_000)
        with pytest.raises(ValueError, match="declared vocab"):
            writer.write(600)


def test_verify_shard_reports_checksum_and_counts(tmp_path):
    make_shards(tmp_path, [[1, 2], [4, 5]])
    result = verify_shard(tmp_path / "shard_00000.bin", eos_id=EOS)
    assert result["token_count"] == 6
    assert result["eos_count"] == 2
    assert len(result["sha256"]) == 64


def test_dataset_yields_shifted_pairs_of_fixed_length(tmp_path):
    make_shards(tmp_path, [list(range(10, 210))])
    dataset = BinaryShardDataset(tmp_path, context_length=16, shuffle=False)

    x, y = next(iter(dataset))
    assert x.shape == y.shape == (16,)
    assert x.dtype == torch.int64
    # y is x shifted by exactly one position.
    torch.testing.assert_close(x[1:], y[:-1])


def test_dataset_drops_short_tail_so_batches_collate(tmp_path):
    make_shards(tmp_path, [list(range(100))])  # 101 tokens with EOS
    dataset = BinaryShardDataset(tmp_path, context_length=32, shuffle=False)
    shapes = {tuple(x.shape) for x, _ in dataset}
    assert shapes == {(32,)}

    loader = DataLoader(dataset, batch_size=2, drop_last=True)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == batch_y.shape == (2, 32)


def test_dataset_reads_metadata(tmp_path):
    make_shards(tmp_path, [list(range(300))])
    dataset = BinaryShardDataset(tmp_path, context_length=8)
    assert dataset.vocab_size == 1000
    assert dataset.num_tokens == 301


def test_infinite_dataset_cycles(tmp_path):
    make_shards(tmp_path, [list(range(100))])
    finite = BinaryShardDataset(tmp_path, context_length=16, shuffle=False)
    infinite = BinaryShardDataset(
        tmp_path, context_length=16, shuffle=False, infinite=True
    )
    finite_count = sum(1 for _ in finite)

    iterator = iter(infinite)
    produced = [next(iterator) for _ in range(finite_count * 3)]
    assert len(produced) == finite_count * 3


def test_workers_partition_shards_without_duplication(tmp_path):
    make_shards(tmp_path, [list(range(60)) for _ in range(20)], shard_size=100)
    dataset = BinaryShardDataset(tmp_path, context_length=8, shuffle=False)
    assert len(dataset.shard_paths) > 1

    single = [tuple(x.tolist()) for x, _ in DataLoader(dataset, batch_size=None, num_workers=0)]
    multi = [tuple(x.tolist()) for x, _ in DataLoader(dataset, batch_size=None, num_workers=2)]
    assert sorted(single) == sorted(multi)


def test_missing_shards_raise_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocess"):
        BinaryShardDataset(tmp_path)
