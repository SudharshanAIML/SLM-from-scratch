from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slm.preprocessing.fineweb_preprocessor import preprocess_fineweb_to_shards


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream FineWeb-Edu documents into packed binary shards")
    parser.add_argument("--output-dir", default="datasets/fineweb_edu/processed/train")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--sample-for-tokenizer", type=int, default=20)
    parser.add_argument("--min-token-count", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--vocab-size", type=int, default=2000)
    parser.add_argument("--tokenizer", choices=["simple", "sentencepiece"], default="simple")
    args = parser.parse_args()

    metadata = preprocess_fineweb_to_shards(
        output_dir=args.output_dir,
        tokenizer_type=args.tokenizer,
        limit=args.limit,
        sample_for_tokenizer=args.sample_for_tokenizer,
        min_token_count=args.min_token_count,
        context_length=args.context_length,
        eos_token=3,
        shard_size=args.shard_size,
        vocab_size=args.vocab_size,
    )
    print("Finished preprocessing demo shards")
    print(metadata)


if __name__ == "__main__":
    main()
