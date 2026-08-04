from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from slm.configs.tokenizer_config import TokenizerConfig


def clean_text(text: str, tokenizer_config: TokenizerConfig | None = None) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    if tokenizer_config and tokenizer_config.lowercase:
        cleaned = cleaned.lower()
    return cleaned.strip()


def load_texts_from_files(paths: Iterable[str | Path]) -> List[str]:
    texts: List[str] = []
    for path in paths:
        path = Path(path)
        if path.exists():
            texts.append(path.read_text(encoding="utf-8"))
    return texts
