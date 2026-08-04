from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slm.configs.model_config import ModelConfig
from slm.configs.tokenizer_config import TokenizerConfig
from slm.configs.train_config import TrainConfig
from slm.model.transformer import Transformer
from slm.tokenizer.simple_tokenizer import SimpleTokenizer
from slm.training.trainer import Trainer


def main() -> None:
    tokenizer_config = TokenizerConfig(vocab_size=200)
    tokenizer = SimpleTokenizer(tokenizer_config)
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Small language models can be trained from scratch.",
        "Transformer blocks learn patterns from sequential data.",
        "FineWeb-style datasets are converted into binary shards before training.",
        "A production pipeline uses packing, EOS tokens, sharding, and memmapped data.",
    ]
    tokenizer.train_from_texts(texts)

    model_config = ModelConfig(
        vocab_size=len(tokenizer.vocab),
        hidden_size=192,
        num_layers=3,
        num_heads=6,
        num_kv_heads=2,
        head_dim=32,
        intermediate_size=512,
        max_seq_len=128,
    )
    train_config = TrainConfig(batch_size=2, max_steps=20, log_every=1, device="cpu", checkpoint_dir="checkpoints")

    model = Transformer(model_config)
    trainer = Trainer(model, train_config, tokenizer)
    trainer.train(texts, block_size=32)

    output_path = Path("checkpoints/model.pt")
    trainer.save(output_path)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
