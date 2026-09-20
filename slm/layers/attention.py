from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.configs.model_config import ModelConfig
from slm.layers.rope import apply_rope


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand each KV head to `n_rep` query heads without copying storage.

    Head `j` of the result reads KV head `j // n_rep`, which is the pairing the
    query-head layout expects.
    """
    if n_rep == 1:
        return x
    batch, kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch, kv_heads * n_rep, seq_len, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """Grouped-query attention with RoPE, SDPA, and no biases.

    For the 259M reference config (H=1024, 16 Q heads, 4 KV heads, D=64):

        q_proj  1024 x 1024 = 1,048,576
        k_proj  1024 x  256 =   262,144
        v_proj  1024 x  256 =   262,144
        o_proj  1024 x 1024 = 1,048,576
        ------------------------------
        total               = 2,621,440 per layer
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups
        self.attention_dropout = config.attention_dropout

        q_dim = config.num_heads * config.head_dim
        kv_dim = config.num_kv_heads * config.head_dim

        self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: "KVCache | None" = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Rotate only the incoming q/k. Cached keys were already rotated at
        # their own positions, so re-rotating them would corrupt the cache.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            past = cache.get(layer_idx)
            if past is not None:
                past_k, past_v = past
                k = torch.cat((past_k, k), dim=2)
                v = torch.cat((past_v, v), dim=2)
            cache.set(layer_idx, k, v)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        key_len = k.size(2)
        attn_mask: torch.Tensor | None = None
        is_causal = False
        if seq_len == key_len:
            # Full forward (training / prefill): the fused causal kernel applies.
            is_causal = True
        elif seq_len > 1:
            # Chunked prefill against an existing cache: build the offset mask
            # explicitly, since is_causal would align to the wrong corner.
            attn_mask = torch.ones(
                seq_len, key_len, dtype=torch.bool, device=x.device
            ).tril(diagonal=key_len - seq_len)
        # seq_len == 1 attends to every cached position: no mask needed.

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn)


class KVCache:
    """Per-layer key/value cache for incremental decoding."""

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.entries: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers
        self.offset = 0

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self.entries[layer_idx]

    def set(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.entries[layer_idx] = (k, v)

    def reset(self) -> None:
        self.entries = [None] * self.num_layers
        self.offset = 0

    def __len__(self) -> int:
        return self.offset
