from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from slm.data.shard_writer import BinaryShardWriter, finalize_metadata
from slm.tokenizer.bpe_tokenizer import BOS_ID, EOS_ID, BPETokenizer

DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_CONFIG = "sample-10BT"


def iter_documents(
    dataset_name: str = DEFAULT_DATASET,
    config_name: str = DEFAULT_CONFIG,
    split: str = "train",
    limit: int | None = None,
    min_words: int = 32,
    desc: str = "documents",
) -> Iterator[str]:
    """Stream FineWeb-Edu text, one document per yield.

    Streaming means the parquet files are never materialised on disk; only the
    `text` field is used.
    """
    from datasets import load_dataset

    stream = load_dataset(
        dataset_name, name=config_name, split=split, streaming=True
    )

    count = 0
    progress = tqdm(total=limit, desc=desc, unit="doc")
    for row in stream:
        if limit is not None and count >= limit:
            break
        text = row.get("text")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text or len(text.split()) < min_words:
            continue
        count += 1
        progress.update(1)
        yield text
    progress.close()


def train_tokenizer(
    output_dir: str | Path,
    vocab_size: int = 32_768,
    sample_documents: int = 100_000,
    dataset_name: str = DEFAULT_DATASET,
    config_name: str = DEFAULT_CONFIG,
    min_words: int = 32,
) -> BPETokenizer:
    """Train the BPE model on a document sample.

    The sample must be large enough to cover the corpus vocabulary: training on
    a few hundred documents produces a model that maps a large share of the
    real corpus to <unk>.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = iter_documents(
        dataset_name=dataset_name,
        config_name=config_name,
        limit=sample_documents,
        min_words=min_words,
        desc="tokenizer sample",
    )
    return BPETokenizer.train(
        texts, model_prefix=output_dir / "tokenizer", vocab_size=vocab_size
    )


def build_shards(
    output_dir: str | Path,
    tokenizer: BPETokenizer,
    limit: int | None = None,
    val_every: int = 200,
    shard_size: int = 100_000_000,
    add_bos: bool = True,
    dataset_name: str = DEFAULT_DATASET,
    config_name: str = DEFAULT_CONFIG,
    min_words: int = 32,
) -> dict[str, dict]:
    """Tokenize the corpus into train/ and val/ shard directories.

    Both splits are written in a single streaming pass, so memory stays flat
    no matter how many documents are processed. Every `val_every`-th document
    goes to validation, which keeps the split deterministic and
    document-aligned -- no train text leaks into val through a shared window.
    """
    output_dir = Path(output_dir)
    extra = {
        "dataset": dataset_name,
        "config": config_name,
        "tokenizer": "sentencepiece-bpe",
        "tokenizer_model": "tokenizer/tokenizer.model",
        "min_words": min_words,
    }

    writers = {
        "train": BinaryShardWriter(
            output_dir / "train", shard_size=shard_size, vocab_size=tokenizer.vocab_size
        ),
        "val": BinaryShardWriter(
            output_dir / "val", shard_size=shard_size, vocab_size=tokenizer.vocab_size
        ),
    }
    doc_counts = {"train": 0, "val": 0}

    try:
        for index, text in enumerate(
            iter_documents(
                dataset_name=dataset_name,
                config_name=config_name,
                limit=limit,
                min_words=min_words,
                desc="tokenizing",
            )
        ):
            ids = tokenizer.encode(text)
            if not ids:
                continue
            split = "val" if index % val_every == 0 else "train"
            writer = writers[split]
            if add_bos:
                writer.write(BOS_ID)
            writer.write_document(ids)
            writer.write(EOS_ID)
            doc_counts[split] += 1
    finally:
        for writer in writers.values():
            writer.close()

    metadata = {}
    for split, writer in writers.items():
        metadata[split] = finalize_metadata(
            output_dir / split,
            writer,
            eos_id=EOS_ID,
            bos_id=BOS_ID if add_bos else None,
            vocab_size=tokenizer.vocab_size,
            num_documents=doc_counts[split],
            shard_size=shard_size,
            metadata_extra=extra,
        )
    return metadata


def preprocess(
    output_dir: str | Path,
    vocab_size: int = 32_768,
    limit: int | None = 100_000,
    sample_documents: int = 100_000,
    val_every: int = 200,
    shard_size: int = 100_000_000,
    min_words: int = 32,
    dataset_name: str = DEFAULT_DATASET,
    config_name: str = DEFAULT_CONFIG,
) -> dict[str, dict]:
    """Full pipeline: train tokenizer, then tokenize the corpus into shards."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_model = output_dir / "tokenizer" / "tokenizer.model"
    if tokenizer_model.exists():
        print(f"Reusing existing tokenizer: {tokenizer_model}")
        tokenizer = BPETokenizer(tokenizer_model)
    else:
        tokenizer = train_tokenizer(
            output_dir / "tokenizer",
            vocab_size=vocab_size,
            sample_documents=sample_documents,
            dataset_name=dataset_name,
            config_name=config_name,
            min_words=min_words,
        )
    print(f"Tokenizer ready: {tokenizer.vocab_size:,} pieces")

    metadata = build_shards(
        output_dir,
        tokenizer,
        limit=limit,
        val_every=val_every,
        shard_size=shard_size,
        dataset_name=dataset_name,
        config_name=config_name,
        min_words=min_words,
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata
