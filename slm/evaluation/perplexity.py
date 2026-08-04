from __future__ import annotations

import torch
import torch.nn.functional as F

from slm.model.transformer import Transformer


def calculate_perplexity(model: Transformer, input_ids: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1))
    return float(torch.exp(loss).item())
