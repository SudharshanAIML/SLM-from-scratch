from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from slm.blocks.transformer_block import TransformerBlock
from slm.configs.model_config import ModelConfig
from slm.layers.rmsnorm import RMSNorm


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig, use_gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
