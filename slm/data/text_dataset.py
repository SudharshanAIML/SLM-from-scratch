from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from slm.tokenizer.simple_tokenizer import SimpleTokenizer


class TextDataset(Dataset):
    def __init__(self, texts: Sequence[str], tokenizer: SimpleTokenizer, block_size: int) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.ids = []
        for text in texts:
            self.ids.extend(tokenizer.encode(text))

        if len(self.ids) < block_size:
            self.ids = self.ids + [tokenizer.vocab[tokenizer.config.unk_token]] * (block_size - len(self.ids))

    def __len__(self) -> int:
        return max(1, len(self.ids) - self.block_size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index
        end = start + self.block_size
        token_ids = self.ids[start:end]
        x = torch.tensor(token_ids[:-1], dtype=torch.long)
        y = torch.tensor(token_ids[1:], dtype=torch.long)
        return x, y
