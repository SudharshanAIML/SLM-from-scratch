import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import random

import pytest

from slm.tokenizer.bpe_tokenizer import BPETokenizer

WORDS = [
    "science", "history", "language", "model", "training", "research", "data",
    "learning", "system", "network", "compute", "gradient", "attention",
    "education", "students", "teachers", "knowledge", "reading", "writing",
    "physics", "chemistry", "biology", "mathematics", "geometry", "algebra",
]


def _corpus(lines: int = 4000, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [
        " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 20))) + "."
        for _ in range(lines)
    ]


@pytest.fixture(scope="session")
def tokenizer(tmp_path_factory) -> BPETokenizer:
    """A small BPE model trained once per test session."""
    path = tmp_path_factory.mktemp("tokenizer")
    return BPETokenizer.train(_corpus(), model_prefix=path / "tokenizer", vocab_size=512)
