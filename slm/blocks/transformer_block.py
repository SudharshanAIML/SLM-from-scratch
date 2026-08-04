from __future__ import annotations

import torch
import torch.nn as nn

from slm.configs.model_config import ModelConfig
from slm.layers.attention import MultiHeadAttention
from slm.layers.mlp import MLP
from slm.layers.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = self.attn_dropout(x)
        x = residual + x
        x = self.residual_dropout(x)

        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.residual_dropout(x)
        return residual + x
