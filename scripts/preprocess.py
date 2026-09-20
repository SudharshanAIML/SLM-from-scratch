from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from slm.preprocessing.fineweb import DEFAULT_CONFIG, DEFAULT_DATASET, preprocess


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream FineWeb-Edu into a BPE tokenizer and packed binary shards"
    )
    parser.add_argument("--output-dir", default="datasets/fineweb_edu")
    parser.add_argument("--vocab-size", type=int, default=32_768)
    parser.add_argument(
        "--limit", type=int, default=100_000, help="Documents to tokenize (0 = all)"
    )
    parser.add_argument(
        "--sample-documents",
        type=int,
        default=100_000,
        help="Documents used to train the BPE model",
    )
    parser.add_argument(
        "--val-every", type=int, default=200, help="Hold out every Nth document for val"
    )
    parser.add_argument("--min-words", type=int, default=32)
    parser.add_argument(
        "--shard-size", type=int, default=100_000_000, help="Tokens per shard file"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    metadata = preprocess(
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        limit=args.limit or None,
        sample_documents=args.sample_documents,
        val_every=args.val_every,
        shard_size=args.shard_size,
        min_words=args.min_words,
        dataset_name=args.dataset,
        config_name=args.config,
    )

    for split, meta in metadata.items():
        print(
            f"{split:>5}: {meta['num_documents']:>9,} docs  "
            f"{meta['num_tokens']:>13,} tokens  "
            f"{meta['num_shards']:>4} shards  "
            f"eos_id={meta['eos_id']}"
        )


if __name__ == "__main__":
    main()
