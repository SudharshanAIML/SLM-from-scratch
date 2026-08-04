from __future__ import annotations

import torch
import torch.nn as nn

from slm.configs.model_config import ModelConfig
from slm.layers.swiglu import SwiGLU


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.swiglu = SwiGLU(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.swiglu(x))
