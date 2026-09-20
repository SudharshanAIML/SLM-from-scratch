from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from slm.configs.train_config import TrainConfig
from slm.evaluation.evaluator import evaluate
from slm.training.checkpoint import CheckpointManager
from slm.training.logger import TrainingLogger
from slm.training.scheduler import apply_lr, cosine_lr_with_warmup

AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on matmul weights only.

    Norm gains and biases are 1-D and should not be decayed; decaying them
    pulls the network toward a degenerate scale.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )


class Trainer:
    """Gradient-accumulating trainer with warmup+cosine LR and periodic eval.

    One iteration of the outer loop is exactly one optimizer step, so
    `global_step`, the LR schedule, checkpoint names and resume all count the
    same thing.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        train_dataset,
        val_dataset=None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.resolve_device())
        self.precision = config.resolve_precision(self.device.type)
        self.amp_dtype = AMP_DTYPES[self.precision]

        torch.manual_seed(config.seed)

        self.model = model.to(self.device)
        self.optimizer = build_optimizer(self.model, config)
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.precision == "fp16"
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader = self._make_loader(train_dataset)
        self.val_loader = self._make_loader(val_dataset) if val_dataset else None

        self.checkpoints = CheckpointManager(
            config.checkpoint_dir, keep_last_n=config.keep_last_n_checkpoints
        )
        self.logger = TrainingLogger(Path(config.checkpoint_dir) / "logs")
        self.global_step = 0

    def _make_loader(self, dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )

    def _autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    # ---------------------------------------------------------------- resume

    def resume(self, path: str | Path | None = None) -> int:
        """Restore weights, optimizer and step counter from a checkpoint.

        The data stream restarts with a fresh shard shuffle rather than being
        fast-forwarded: replaying millions of windows to reach an exact
        position costs more than it is worth for pretraining, and the shards
        are shuffled every epoch anyway.
        """
        self.global_step = self.checkpoints.load(
            self.model, self.optimizer, path=path, scaler=self.scaler,
            map_location=str(self.device),
        )
        print(f"Resumed from step {self.global_step:,}")
        return self.global_step

    # ----------------------------------------------------------------- train

    def train(self) -> None:
        config = self.config
        self.model.train()

        data_iter = iter(self.train_loader)
        tokens_per_step = (
            config.batch_size * config.gradient_accumulation_steps * config.context_length
        )

        print(
            f"Training on {self.device} ({self.precision}) | "
            f"{tokens_per_step:,} tokens/step | "
            f"steps {self.global_step:,} -> {config.max_steps:,}"
        )

        start_time = time.time()
        for step in range(self.global_step, config.max_steps):
            lr = apply_lr(
                self.optimizer,
                cosine_lr_with_warmup(
                    step,
                    config.learning_rate,
                    config.min_learning_rate,
                    config.warmup_steps,
                    config.max_steps,
                ),
            )

            self.optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0

            for _ in range(config.gradient_accumulation_steps):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    x, y = next(data_iter)

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with self._autocast():
                    logits = self.model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                    )

                accumulated += loss.item()
                self.scaler.scale(loss / config.gradient_accumulation_steps).backward()

            # Unscale before clipping so max_grad_norm means the same thing
            # whether or not fp16 loss scaling is active.
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_step = step + 1
            avg_loss = accumulated / config.gradient_accumulation_steps

            if self.global_step % config.log_every == 0:
                elapsed = time.time() - start_time
                self.logger.log(
                    {
                        "step": self.global_step,
                        "loss": avg_loss,
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                        "tokens_per_sec": tokens_per_step * config.log_every / max(elapsed, 1e-6),
                    }
                )
                start_time = time.time()

            if self.val_loader and self.global_step % config.eval_every == 0:
                metrics = evaluate(
                    self.model,
                    self.val_loader,
                    device=self.device,
                    max_batches=config.eval_steps,
                    autocast_dtype=self.amp_dtype,
                )
                self.logger.log(
                    {
                        "step": self.global_step,
                        "val_loss": metrics["loss"],
                        "perplexity": metrics["perplexity"],
                    }
                )
                start_time = time.time()

            if self.global_step % config.save_every == 0:
                path = self.checkpoints.save(
                    self.model, self.optimizer, self.global_step, scaler=self.scaler
                )
                print(f"  saved {path.name}")
                start_time = time.time()

        final = self.checkpoints.save(
            self.model, self.optimizer, self.global_step, scaler=self.scaler
        )
        print(f"Training complete at step {self.global_step:,} -> {final.name}")
