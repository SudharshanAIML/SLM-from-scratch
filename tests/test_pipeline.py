"""End-to-end dataflow: tokenizer -> shards -> dataset -> train -> generate."""

import numpy as np
import torch

from slm.configs.model_config import ModelConfig
from slm.configs.train_config import TrainConfig
from slm.data.shard_dataset import BinaryShardDataset
from slm.data.shard_writer import write_shards
from slm.model.generation import generate
from slm.model.transformer import Transformer
from slm.tokenizer.bpe_tokenizer import BOS_ID, EOS_ID, UNK_ID
from slm.training.checkpoint import CheckpointManager
from slm.training.trainer import Trainer

from conftest import _corpus


def build_corpus_shards(tokenizer, directory):
    documents = [tokenizer.encode(text) for text in _corpus(600, seed=7)]
    metadata = write_shards(
        documents,
        directory,
        eos_id=tokenizer.eos_id,
        vocab_size=tokenizer.vocab_size,
        shard_size=50_000,
        add_bos=True,
        bos_id=tokenizer.bos_id,
    )
    return documents, metadata


def test_shard_separator_matches_the_tokenizer(tokenizer, tmp_path):
    """The id packed between documents must be the tokenizer's own EOS.

    Checked against the tokenizer object rather than a literal, so the two
    halves of the pipeline cannot drift apart.
    """
    documents, metadata = build_corpus_shards(tokenizer, tmp_path / "shards")

    assert metadata["eos_id"] == tokenizer.eos_id == EOS_ID
    assert tokenizer.id_to_piece(metadata["eos_id"]) == "<eos>"

    separators = sum(v["eos_count"] for v in metadata["verification"])
    assert separators == len(documents) == metadata["num_documents"]


def test_corpus_contains_no_unknown_tokens(tokenizer, tmp_path):
    """Byte fallback means real text never degrades into <unk>."""
    build_corpus_shards(tokenizer, tmp_path / "shards")
    tokens = np.concatenate(
        [np.fromfile(p, dtype="uint16") for p in sorted((tmp_path / "shards").glob("*.bin"))]
    )
    assert int(np.count_nonzero(tokens == UNK_ID)) == 0
    assert int(tokens.max()) < tokenizer.vocab_size


def test_documents_are_delimited_by_bos_and_eos(tokenizer, tmp_path):
    build_corpus_shards(tokenizer, tmp_path / "shards")
    tokens = np.fromfile(sorted((tmp_path / "shards").glob("*.bin"))[0], dtype="uint16")
    assert tokens[0] == BOS_ID
    # Every EOS is followed by the BOS of the next document.
    eos_positions = np.flatnonzero(tokens == EOS_ID)
    followers = eos_positions[eos_positions + 1 < len(tokens)] + 1
    assert np.all(tokens[followers] == BOS_ID)


def test_metadata_vocab_drives_the_model_embedding(tokenizer, tmp_path):
    """The model must size its embedding from the data, not a stale constant."""
    build_corpus_shards(tokenizer, tmp_path / "shards")
    dataset = BinaryShardDataset(tmp_path / "shards", context_length=32)

    assert dataset.vocab_size == tokenizer.vocab_size
    model = Transformer(ModelConfig(
        vocab_size=dataset.vocab_size, hidden_size=64, num_layers=2, num_heads=4,
        num_kv_heads=2, head_dim=16, intermediate_size=128, max_seq_len=32,
    ))
    assert model.embedding.num_embeddings == tokenizer.vocab_size

    # Every id the dataset can emit indexes a real embedding row.
    x, y = next(iter(dataset))
    assert int(max(x.max(), y.max())) < model.embedding.num_embeddings


def test_full_pipeline_trains_then_generates(tokenizer, tmp_path):
    torch.manual_seed(0)
    build_corpus_shards(tokenizer, tmp_path / "shards")

    dataset = BinaryShardDataset(
        tmp_path / "shards", context_length=32, shuffle=True, infinite=True, seed=0
    )
    model_config = ModelConfig(
        vocab_size=dataset.vocab_size, hidden_size=64, num_layers=2, num_heads=4,
        num_kv_heads=2, head_dim=16, intermediate_size=128, max_seq_len=32,
    )
    train_config = TrainConfig(
        batch_size=4, gradient_accumulation_steps=2, context_length=32,
        max_steps=30, warmup_steps=5, learning_rate=3e-3, device="cpu",
        num_workers=0, log_every=5, save_every=30, eval_every=10**9,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )

    trainer = Trainer(Transformer(model_config), train_config, dataset)
    trainer.train()

    losses = [r["loss"] for r in trainer.logger.read_history() if "loss" in r]
    assert losses[-1] < losses[0]

    # Reload from disk exactly as scripts/generate.py does.
    reloaded = Transformer(model_config)
    step = CheckpointManager(tmp_path / "ckpt").load(reloaded, None)
    assert step == 30

    text = generate(
        reloaded, tokenizer, "science and", max_new_tokens=16,
        temperature=0.8, seed=0, device="cpu",
    )
    assert isinstance(text, str) and text.startswith("science and")
