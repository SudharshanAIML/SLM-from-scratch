from __future__ import annotations

import torch
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rotary(x: torch.Tensor, seq_len: int | None = None) -> torch.Tensor:
    if x.size(-1) % 2 != 0:
        raise ValueError("RoPE expects an even head dimension")
    dim = x.size(-1)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(seq_len or x.size(-2), dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0).to(x.device)
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0).to(x.device)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((x_even, x_odd), dim=-1).reshape_as(x)
    rotated = rotate_half(rotated)
    return x * cos + rotated * sin
