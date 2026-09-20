from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension: [a, b] -> [-b, a]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embeddings to `x` of shape (batch, heads, seq, head_dim)."""
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """Precomputed rotary position embedding tables.

    The cos/sin tables depend only on (position, head_dim), so they are built
    once and sliced per forward pass instead of being rebuilt for every layer
    on every step. They are registered non-persistently so they never enter a
    checkpoint.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        theta: float = 10_000.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")

        self.head_dim = head_dim
        self.theta = theta
        self._cached_len = 0

        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._build_cache(max_seq_len, device=device)

    def _build_cache(self, seq_len: int, device: torch.device | None = None) -> None:
        device = device or self.inv_freq.device
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        # Duplicate so the table lines up with the rotate_half split.
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()
        self._cached_len = seq_len

    def forward(
        self, seq_len: int, offset: int = 0, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) for positions [offset, offset + seq_len).

        Shapes are (1, 1, seq_len, head_dim) so they broadcast against
        (batch, heads, seq_len, head_dim).
        """
        needed = offset + seq_len
        if needed > self._cached_len or self.cos_cached.device != (
            device or self.inv_freq.device
        ):
            self._build_cache(max(needed, self._cached_len), device=device)

        cos = self.cos_cached[offset:needed].view(1, 1, seq_len, self.head_dim)
        sin = self.sin_cached[offset:needed].view(1, 1, seq_len, self.head_dim)
        return cos, sin

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, theta={self.theta}, "
            f"cached_len={self._cached_len}"
        )
