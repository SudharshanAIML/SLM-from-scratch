from __future__ import annotations

import torch

from slm.model.generation import generate_text
from slm.model.transformer import Transformer
from slm.tokenizer.simple_tokenizer import SimpleTokenizer


def sample_text(model: Transformer, tokenizer: SimpleTokenizer, prompt: str, max_new_tokens: int = 16) -> str:
    return generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
