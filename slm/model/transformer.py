from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from slm.blocks.transformer_block import TransformerBlock
from slm.configs.model_config import ModelConfig
from slm.layers.attention import KVCache
from slm.layers.rmsnorm import RMSNorm
from slm.layers.rope import RotaryEmbedding


class Transformer(nn.Module):
    """Decoder-only transformer.

        token ids -> embedding -> [TransformerBlock] x N -> RMSNorm -> lm_head

    The LM head shares storage with the embedding when `tie_embeddings` is set,
    so the vocabulary projection is counted once.
    """

    def __init__(
        self, config: ModelConfig, use_gradient_checkpointing: bool = False
    ) -> None:
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)
        self._scale_residual_projections()

    # ---------------------------------------------------------------- weights

    def _init_weights(self, module: nn.Module) -> None:
        """Normal(0, 0.02) init.

        Torch's default `nn.Linear` init (kaiming-uniform on fan_in) produces
        activations that compound across a 20-layer residual stack and send the
        first training loss into the dozens instead of ~ln(vocab_size).
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def _scale_residual_projections(self) -> None:
        """Damp the layers that write into the residual stream.

        Each block adds two contributions, so scaling by 1/sqrt(2 * n_layers)
        keeps the residual variance roughly constant with depth.
        """
        std = 0.02 / math.sqrt(2 * self.config.num_layers)
        for block in self.layers:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=std)

    # ---------------------------------------------------------------- forward

    def forward(
        self, input_ids: torch.Tensor, cache: KVCache | None = None
    ) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        offset = cache.offset if cache is not None else 0

        if offset + seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence of {offset + seq_len} tokens exceeds max_seq_len "
                f"({self.config.max_seq_len})"
            )

        x = self.embedding(input_ids)
        cos, sin = self.rope(seq_len, offset=offset, device=x.device)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)

        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, cos, sin, None, use_reentrant=False)
            else:
                x = layer(x, cos, sin, cache=cache)

        if cache is not None:
            cache.offset += seq_len

        return self.lm_head(self.norm(x))

    def new_cache(self) -> KVCache:
        return KVCache(self.config.num_layers)

    # ------------------------------------------------------------- inspection

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count unique parameter tensors.

        `nn.Module.parameters()` already de-duplicates tied weights, so a tied
        lm_head is not double counted.
        """
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def parameter_breakdown(self) -> dict[str, int]:
        """Measured counts, grouped to match the architecture derivation."""
        groups = {
            "embedding": 0,
            "attention": 0,
            "swiglu": 0,
            "block_norms": 0,
            "final_norm": 0,
            "lm_head": 0,
        }
        seen: set[int] = set()
        for name, param in self.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))

            if name.startswith("embedding"):
                groups["embedding"] += param.numel()
            elif ".attn." in name:
                groups["attention"] += param.numel()
            elif ".mlp." in name:
                groups["swiglu"] += param.numel()
            elif name.startswith("layers"):
                groups["block_norms"] += param.numel()
            elif name.startswith("norm"):
                groups["final_norm"] += param.numel()
            elif name.startswith("lm_head"):
                groups["lm_head"] += param.numel()

        groups["total"] = sum(groups.values())
        return groups

    def extra_repr(self) -> str:
        return f"params={self.num_parameters():,}"
