from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from slm.configs.model_config import ModelConfig
from slm.configs.train_config import TrainConfig
from slm.data.text_dataset import TextDataset
from slm.model.transformer import Transformer
from slm.tokenizer.simple_tokenizer import SimpleTokenizer
from slm.training.checkpoint import CheckpointManager
from slm.training.logger import TrainingLogger


class Trainer:
    def __init__(
        self,
        model: Transformer,
        train_config: TrainConfig,
        tokenizer: SimpleTokenizer | None = None,
    ) -> None:
        self.model = model
        self.train_config = train_config
        self.tokenizer = tokenizer
        self.device = torch.device(train_config.device)
        self.model.to(self.device)
        self.checkpoint_manager = CheckpointManager(train_config.checkpoint_dir)
        self.logger = TrainingLogger(Path(train_config.checkpoint_dir) / "logs")
        self.global_step = 0

    def train_on_texts(self, texts: list[str], block_size: int) -> None:
        """Train on text dataset."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text training")

        dataset = TextDataset(texts, self.tokenizer, block_size)
        loader = DataLoader(dataset, batch_size=self.train_config.batch_size, shuffle=True)
        self._training_loop(loader)

    def train_on_dataset(self, dataset: IterableDataset) -> None:
        """Train on iterable dataset (e.g., BinaryShardDataset)."""
        loader = DataLoader(dataset, batch_size=self.train_config.batch_size)
        self._training_loop(loader)

    def _training_loop(self, loader: DataLoader) -> None:
        """Main training loop with gradient accumulation."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )

        # Try to resume from checkpoint
        start_step = self._try_resume(optimizer)
        self.global_step = start_step

        accumulated_loss = 0.0
        num_accumulated = 0

        for step, (x, y) in enumerate(loader):
            if step < start_step:
                continue

            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )

            loss_scaled = loss / self.train_config.gradient_accumulation_steps
            loss_scaled.backward()

            accumulated_loss += loss.item()
            num_accumulated += 1

            # Gradient accumulation
            if (step + 1) % self.train_config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.train_config.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()

                avg_loss = accumulated_loss / num_accumulated
                self.global_step += 1

                if self.global_step % self.train_config.log_every == 0:
                    print(f"step={self.global_step} loss={avg_loss:.4f}")
                    self.logger.log({"step": self.global_step, "loss": float(avg_loss)})

                if self.global_step % self.train_config.save_every == 0:
                    self.checkpoint_manager.save(self.model, optimizer, self.global_step)
                    print(f"Saved checkpoint at step {self.global_step}")

                accumulated_loss = 0.0
                num_accumulated = 0

            if self.global_step >= self.train_config.max_steps:
                break

        # Save final checkpoint
        self.checkpoint_manager.save(self.model, optimizer, self.global_step)
        print(f"Training complete. Final step: {self.global_step}")

    def _try_resume(self, optimizer: torch.optim.Optimizer) -> int:
        """Try to resume from latest checkpoint."""
        checkpoint_dir = Path(self.train_config.checkpoint_dir)
        if not checkpoint_dir.exists():
            return 0

        checkpoints = sorted(checkpoint_dir.glob("model_step_*.pt"))
        if not checkpoints:
            return 0

        latest = checkpoints[-1]
        try:
            step = self.checkpoint_manager.load(self.model, optimizer, latest)
            print(f"Resumed from checkpoint at step {step}")
            return step
        except Exception as e:
            print(f"Failed to resume from checkpoint: {e}")
            return 0
