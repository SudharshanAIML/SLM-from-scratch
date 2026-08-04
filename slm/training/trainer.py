from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from slm.configs.model_config import ModelConfig
from slm.configs.train_config import TrainConfig
from slm.data.text_dataset import TextDataset
from slm.model.transformer import Transformer
from slm.tokenizer.simple_tokenizer import SimpleTokenizer


class Trainer:
    def __init__(self, model: Transformer, train_config: TrainConfig, tokenizer: SimpleTokenizer) -> None:
        self.model = model
        self.train_config = train_config
        self.tokenizer = tokenizer
        self.device = torch.device(train_config.device)
        self.model.to(self.device)

    def train(self, texts: list[str], block_size: int) -> None:
        dataset = TextDataset(texts, self.tokenizer, block_size)
        loader = DataLoader(dataset, batch_size=self.train_config.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_config.learning_rate, weight_decay=self.train_config.weight_decay)

        for step, (x, y) in enumerate(loader):
            x = x.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad()
            logits = self.model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.max_grad_norm)
            optimizer.step()

            if step % self.train_config.log_every == 0:
                print(f"step={step} loss={loss.item():.4f}")

            if step >= self.train_config.max_steps - 1:
                break

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
