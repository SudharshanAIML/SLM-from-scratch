from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rotary(x: torch.Tensor, seq_len: int | None = None) -> torch.Tensor:
    if x.size(-1) % 2 != 0:
        raise ValueError("RoPE expects an even head dimension")

    original_dtype = x.dtype
    x_float = x.to(torch.float32)
    dim = x.size(-1)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32, device=x.device) / dim))
    positions = torch.arange(seq_len or x.size(-2), dtype=torch.float32, device=x.device)
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0).to(x.device)
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0).to(x.device)

    x_even = x_float[..., 0::2]
    x_odd = x_float[..., 1::2]
    x_even_rotated = x_even * cos - x_odd * sin
    x_odd_rotated = x_even * sin + x_odd * cos
    rotated = torch.stack((x_even_rotated, x_odd_rotated), dim=-1).flatten(-2)
    return rotated.to(original_dtype)
