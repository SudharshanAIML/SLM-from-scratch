from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.configs.model_config import ModelConfig


class SwiGLU(nn.Module):
    r"""SwiGLU feed-forward network.

                    x
                  /   \
            gate_proj  up_proj
                 |       |
               SiLU      |
                 \___ * _/
                     |
                 down_proj

        y = down(SiLU(gate(x)) * up(x))

    Three H x F matrices and no biases, so the parameter count is exactly
    `3 * hidden_size * intermediate_size` -- 3 * 1024 * 2816 = 8,650,752 for
    the 259M reference config.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden, inter = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.dropout = nn.Dropout(config.mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
