from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import torch

from slm.configs.model_config import ModelConfig
from slm.model.generation import generate
from slm.model.transformer import Transformer
from slm.tokenizer.bpe_tokenizer import BPETokenizer
from slm.training.checkpoint import CheckpointManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint")
    parser.add_argument("--checkpoint", default=None, help="Path to a .pt checkpoint")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument(
        "--tokenizer", default="datasets/fineweb_edu/tokenizer/tokenizer.model"
    )
    parser.add_argument("--prompt", default="The history of science")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = BPETokenizer(args.tokenizer)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    manager = CheckpointManager(args.checkpoint_dir)
    if checkpoint_path is None:
        checkpoint_path = manager.latest()
        if checkpoint_path is None:
            raise SystemExit(f"No checkpoint found in {args.checkpoint_dir}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(payload["model_config"])
    model = Transformer(config)
    model.load_state_dict(payload["model_state"])
    model.to(device)

    print(f"Checkpoint: {checkpoint_path.name} (step {payload.get('step', 0):,})")
    print(f"Device:     {device}\n")
    print(
        generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
