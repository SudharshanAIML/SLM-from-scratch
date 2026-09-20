from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import torch

from slm.configs.model_config import ModelConfig
from slm.configs.train_config import TrainConfig
from slm.data.shard_dataset import BinaryShardDataset
from slm.model.transformer import Transformer
from slm.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 259M SLM on binary shards")
    parser.add_argument("--data-dir", default="datasets/fineweb_edu")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--keep-last-n-checkpoints", type=int, default=None)
    parser.add_argument(
        "--device", default=None, help="auto | cpu | cuda (default: auto)"
    )
    parser.add_argument(
        "--precision", default=None, help="auto | bf16 | fp16 | fp32 (default: auto)"
    )
    parser.add_argument(
        "--model",
        choices=["259m", "debug"],
        default="259m",
        help="259m = the reference architecture; debug = tiny model for CPU smoke runs",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train_config = TrainConfig()
    for name in (
        "batch_size",
        "gradient_accumulation_steps",
        "context_length",
        "max_steps",
        "learning_rate",
        "warmup_steps",
        "num_workers",
        "save_every",
        "eval_every",
        "eval_steps",
        "log_every",
        "seed",
        "keep_last_n_checkpoints",
        "device",
        "precision",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(train_config, name, value)
    train_config.checkpoint_dir = args.checkpoint_dir
    if args.gradient_checkpointing:
        train_config.use_gradient_checkpointing = True

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train" if (data_dir / "train").exists() else data_dir
    val_dir = data_dir / "val"

    train_dataset = BinaryShardDataset(
        train_dir,
        context_length=train_config.context_length,
        shuffle=True,
        infinite=True,
        seed=train_config.seed,
    )
    val_dataset = (
        BinaryShardDataset(
            val_dir,
            context_length=train_config.context_length,
            shuffle=False,
            infinite=False,
            seed=train_config.seed,
        )
        if val_dir.exists()
        else None
    )

    # The shards carry the vocabulary that produced them; trusting metadata
    # here is what keeps the embedding table and the token ids in agreement.
    vocab_size = train_dataset.vocab_size or ModelConfig.vocab_size
    builder = ModelConfig.debug if args.model == "debug" else ModelConfig
    model_config = builder(
        vocab_size=vocab_size, max_seq_len=train_config.context_length
    )
    model = Transformer(
        model_config, use_gradient_checkpointing=train_config.use_gradient_checkpointing
    )

    print(f"Model:   {model.num_parameters():,} parameters")
    print(f"Data:    {train_dataset.num_tokens:,} train tokens "
          f"across {len(train_dataset.shard_paths)} shards")
    if val_dataset:
        print(f"         {val_dataset.num_tokens:,} val tokens")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:     {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    trainer = Trainer(model, train_config, train_dataset, val_dataset)
    if args.resume:
        trainer.resume()
    trainer.train()


if __name__ == "__main__":
    main()
