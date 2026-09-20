from __future__ import annotations

import math


def cosine_lr_with_warmup(
    step: int,
    learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """Linear warmup, then cosine decay to `min_learning_rate`.

    `step` is a zero-based optimizer step. Without warmup the first updates of
    a freshly initialised model are large enough to push the loss into a bad
    basin it spends thousands of steps climbing out of.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps

    if step >= max_steps:
        return min_learning_rate

    span = max(1, max_steps - warmup_steps)
    progress = (step - warmup_steps) / span
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + coeff * (learning_rate - min_learning_rate)


def apply_lr(optimizer, lr: float) -> float:
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr
