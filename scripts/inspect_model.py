from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

import torch

from slm.configs.model_config import ModelConfig
from slm.model.transformer import Transformer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the parameter derivation for the model configuration"
    )
    parser.add_argument("--vocab-size", type=int, default=32_768)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--build", action="store_true", help="Also instantiate and count")
    args = parser.parse_args()

    config = ModelConfig(vocab_size=args.vocab_size, max_seq_len=args.context_length)
    breakdown = config.parameter_breakdown()
    h, d, f = config.hidden_size, config.head_dim, config.intermediate_size
    q_dim, kv_dim = config.num_heads * d, config.num_kv_heads * d

    print("MODEL CONFIGURATION")
    print(f"  layers   {config.num_layers}")
    print(f"  hidden   {h}")
    print(f"  q heads  {config.num_heads}")
    print(f"  kv heads {config.num_kv_heads}  (GQA group size {config.num_kv_groups})")
    print(f"  head dim {d}")
    print(f"  ffn      {f}")
    print(f"  vocab    {config.vocab_size:,}")
    print(f"  context  {config.max_seq_len:,}")

    print("\nPARAMETER CALCULATION")
    print(f"  embedding      {config.vocab_size:,} x {h:,} = {breakdown['embedding']:>13,}")
    print(f"\n  attention / layer")
    print(f"    q_proj       {h:,} x {q_dim:,} = {h * q_dim:>13,}")
    print(f"    k_proj       {h:,} x {kv_dim:,} = {h * kv_dim:>13,}")
    print(f"    v_proj       {h:,} x {kv_dim:,} = {h * kv_dim:>13,}")
    print(f"    o_proj       {q_dim:,} x {h:,} = {q_dim * h:>13,}")
    print(f"    subtotal                 = {breakdown['attention_per_layer']:>13,}")
    print(f"\n  swiglu / layer   3 x {h:,} x {f:,} = {breakdown['swiglu_per_layer']:>13,}")
    print(f"  rmsnorm / layer      2 x {h:,} = {breakdown['norms_per_layer']:>13,}")
    print(f"  ---------------------------------------------")
    print(f"  per layer                    = {breakdown['per_layer']:>13,}")
    print(f"  x {config.num_layers} layers                 = {breakdown['transformer_stack']:>13,}")
    print(f"  + embedding (tied lm_head)   = {breakdown['embedding']:>13,}")
    print(f"  + final rmsnorm              = {breakdown['final_norm']:>13,}")
    print(f"  =============================================")
    print(f"  TOTAL                        = {breakdown['total']:>13,}")
    print(f"  ({breakdown['total'] / 1e6:.3f}M)")

    if args.build:
        model = Transformer(config)
        measured = model.num_parameters()
        print(f"\nMEASURED (instantiated)        = {measured:>13,}")
        print(f"  matches analytical: {measured == breakdown['total']}")
        x = torch.randint(0, config.vocab_size, (1, 16))
        with torch.no_grad():
            print(f"  forward {tuple(x.shape)} -> {tuple(model(x).shape)}")


if __name__ == "__main__":
    main()
