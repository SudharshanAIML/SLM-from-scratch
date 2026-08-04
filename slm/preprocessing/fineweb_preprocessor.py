from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

from datasets import load_dataset

from slm.configs.tokenizer_config import TokenizerConfig
from slm.dataset.binary_dataset import build_binary_shards
from slm.tokenizer.simple_tokenizer import SimpleTokenizer
from slm.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer


def iter_fineweb_documents(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    split: str = "train",
    streaming: bool = True,
    limit: int | None = None,
    min_token_count: int = 32,
    language: str | None = "en",
) -> Iterator[str]:
    ds = load_dataset(dataset_name, name="sample-10BT", split=split, streaming=streaming)
    count = 0
    for row in ds:
        if limit is not None and count >= limit:
            break

        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        row_language = row.get("language")
        if language is not None and row_language not in (None, language):
            continue

        if len(text.split()) < min_token_count:
            continue

        yield text.strip()
        count += 1


def build_tokenizer(texts: Sequence[str], output_dir: str | Path, tokenizer_type: str = "simple", vocab_size: int = 2000):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if tokenizer_type == "sentencepiece":
        tokenizer = SentencePieceTokenizer()
        tokenizer.train(texts, output_dir / "tokenizer", vocab_size=vocab_size)
        return tokenizer

    config = TokenizerConfig(vocab_size=vocab_size, lowercase=True)
    tokenizer = SimpleTokenizer(config)
    tokenizer.train_from_texts(texts)
    tokenizer.save(output_dir / "tokenizer.json")
    return tokenizer


def preprocess_fineweb_to_shards(
    output_dir: str | Path,
    tokenizer_type: str = "simple",
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    split: str = "train",
    limit: int | None = None,
    sample_for_tokenizer: int = 200,
    min_token_count: int = 32,
    context_length: int = 4096,
    eos_token: int = 3,
    shard_size: int = 1_000_000,
    vocab_size: int = 2000,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = list(
        iter_fineweb_documents(
            dataset_name=dataset_name,
            split=split,
            limit=limit,
            min_token_count=min_token_count,
        )
    )
    if not documents:
        raise ValueError("No documents were collected from FineWeb-Edu")

    tokenizer_texts = documents[:sample_for_tokenizer]
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer = build_tokenizer(tokenizer_texts, tokenizer_dir, tokenizer_type=tokenizer_type, vocab_size=vocab_size)

    tokenized_documents = [tokenizer.encode(text) for text in documents]
    metadata = build_binary_shards(
        tokenized_documents,
        output_dir,
        context_length=context_length,
        eos_token=eos_token,
        shard_size=shard_size,
    )
    metadata.update(
        {
            "dataset": dataset_name,
            "split": split,
            "tokenizer_type": tokenizer_type,
            "num_documents": len(documents),
            "vocab_size": vocab_size,
            "min_token_count": min_token_count,
        }
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
