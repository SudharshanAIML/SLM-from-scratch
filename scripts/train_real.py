from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from slm.configs.model_config import ModelConfig
from slm.configs.train_config import TrainConfig
from slm.data.binary_dataset import BinaryShardDataset
from slm.model.transformer import Transformer
from slm.training.trainer import Trainer


def load_metadata(metadata_path: str | Path) -> dict:
    """Load dataset metadata."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SLM on FineWeb-Edu binary shards")
    parser.add_argument("--data-dir", default="datasets/fineweb_edu/processed/train", help="Path to binary shards")
    parser.add_argument("--model-size", choices=["small", "medium"], default="small", help="Model size")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default from config)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum steps (default from config)")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--context-length", type=int, default=None, help="Context length (overrides dataset metadata)")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print(f"Please run: python3 scripts/preprocess_fineweb.py --output-dir {data_dir}")
        return

    # Load metadata
    metadata_path = data_dir / "metadata.json"
    metadata = load_metadata(metadata_path)
    print(f"Loaded dataset metadata: {metadata}")

    # Create dataset
    context_length = args.context_length or metadata.get("context_length", 2048)
    dtype = metadata.get("dtype", "uint16")
    dataset = BinaryShardDataset(
        shard_dir=data_dir,
        context_length=context_length,
        dtype=dtype,
        shuffle_shards=True,
    )

    # Create model config
    vocab_size = metadata.get("vocab_size", 50000)
    if args.model_size == "small":
        model_config = ModelConfig.small_v1(vocab_size=vocab_size, max_seq_len=context_length)
    else:
        model_config = ModelConfig.medium_v1(vocab_size=vocab_size, max_seq_len=context_length)

    print(f"Model config: {model_config}")

    # Create training config (RTX 3060 optimized)
    train_config = TrainConfig.rtx3060_v1()

    # Override with CLI args
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        train_config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.max_steps is not None:
        train_config.max_steps = args.max_steps
    if args.learning_rate is not None:
        train_config.learning_rate = args.learning_rate
    if args.gradient_checkpointing:
        train_config.use_gradient_checkpointing = True

    train_config.checkpoint_dir = args.checkpoint_dir

    print(f"Training config: {train_config}")

    # Create model
    model = Transformer(
        config=model_config,
        use_gradient_checkpointing=train_config.use_gradient_checkpointing,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Check VRAM
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # Create trainer and train
    trainer = Trainer(model, train_config)
    print(f"Starting training on {train_config.device}...")
    trainer.train_on_dataset(dataset)


if __name__ == "__main__":
    main()
