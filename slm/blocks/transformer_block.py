from __future__ import annotations

import torch
import torch.nn as nn

from slm.configs.model_config import ModelConfig
from slm.layers.attention import GroupedQueryAttention, KVCache
from slm.layers.rmsnorm import RMSNorm
from slm.layers.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """Pre-norm decoder block.

        x -> RMSNorm -> GQA+RoPE -> +residual
          -> RMSNorm -> SwiGLU   -> +residual

    Parameters for the 259M reference config:
        attention 2,621,440 + swiglu 8,650,752 + 2 norms 2,048 = 11,274,240
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = SwiGLU(config)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        attn_out = self.attn(
            self.attn_norm(x), cos, sin, cache=cache, layer_idx=self.layer_idx
        )
        x = x + self.residual_dropout(attn_out)

        mlp_out = self.mlp(self.mlp_norm(x))
        x = x + self.residual_dropout(mlp_out)
        return x
