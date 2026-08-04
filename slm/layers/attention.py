from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.configs.model_config import ModelConfig
from slm.layers.rope import apply_rotary


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def _reshape(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, kv_cache: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._reshape(q, self.num_heads)
        k = self._reshape(k, self.num_kv_heads)
        v = self._reshape(v, self.num_kv_heads)

        use_cache = kv_cache is not None
        if use_cache:
            past_k = kv_cache.get("k")
            past_v = kv_cache.get("v")
            if past_k is not None and past_v is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

        q = apply_rotary(q, seq_len=q.size(2))
        k = apply_rotary(k, seq_len=k.size(2))

        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=not use_cache,
            dropout_p=0.0,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        if use_cache:
            kv_cache["k"] = k
            kv_cache["v"] = v
        return self.out_proj(attn_output)
