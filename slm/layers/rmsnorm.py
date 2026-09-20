from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    r"""Root-mean-square layer normalisation.

        RMS(x) = sqrt(mean(x_i^2) + eps)
        y      = x / RMS(x) * gamma

    The reduction runs in fp32 regardless of the autocast dtype: computing the
    mean of squares in bf16/fp16 loses enough precision to destabilise deep
    stacks. Only `gamma` carries parameters -- `hidden_size` of them.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self._norm(x.float()).type_as(x)
        return self.weight * normed

    def extra_repr(self) -> str:
        return f"dim={tuple(self.weight.shape)[0]}, eps={self.eps}"
