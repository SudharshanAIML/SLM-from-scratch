from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
    max_batches: int = 50,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Mean next-token loss and perplexity over up to `max_batches` batches.

    The loader yields (x, y) pairs that are already shifted by the dataset, so
    no further slicing is applied here.
    """
    was_training = model.training
    model.eval()

    device = torch.device(device)
    device_type = device.type
    total_loss = 0.0
    total_tokens = 0

    context = (
        torch.autocast(device_type=device_type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )

    for index, (x, y) in enumerate(loader):
        if index >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with context:
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )
        total_loss += loss.float().item()
        total_tokens += y.numel()

    if was_training:
        model.train()

    if total_tokens == 0:
        return {"loss": float("nan"), "perplexity": float("nan"), "tokens": 0}

    mean_loss = total_loss / total_tokens
    return {
        "loss": mean_loss,
        # Clamp before exp so a diverged run reports inf instead of raising.
        "perplexity": math.exp(min(mean_loss, 80.0)),
        "tokens": total_tokens,
    }
