from __future__ import annotations

import torch

from slm.model.transformer import Transformer
from slm.tokenizer.simple_tokenizer import SimpleTokenizer


def generate_text(
    model: Transformer,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 16,
    temperature: float = 1.0,
    device: str = "cpu",
) -> str:
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist())
