import pytest
import torch

from slm.configs.model_config import ModelConfig
from slm.configs.train_config import TrainConfig
from slm.data.shard_writer import write_shards
from slm.data.shard_dataset import BinaryShardDataset
from slm.model.transformer import Transformer
from slm.training.checkpoint import CheckpointManager
from slm.training.scheduler import apply_lr, cosine_lr_with_warmup
from slm.training.trainer import Trainer, build_optimizer

SCHEDULE = dict(learning_rate=1e-3, min_learning_rate=1e-4, warmup_steps=10, max_steps=100)


def test_warmup_ramps_linearly_to_peak():
    assert cosine_lr_with_warmup(0, **SCHEDULE) == pytest.approx(1e-4)
    assert cosine_lr_with_warmup(4, **SCHEDULE) == pytest.approx(5e-4)
    assert cosine_lr_with_warmup(9, **SCHEDULE) == pytest.approx(1e-3)


def test_cosine_decays_from_peak_to_floor():
    assert cosine_lr_with_warmup(10, **SCHEDULE) == pytest.approx(1e-3)
    midpoint = cosine_lr_with_warmup(55, **SCHEDULE)
    assert 1e-4 < midpoint < 1e-3
    assert cosine_lr_with_warmup(100, **SCHEDULE) == pytest.approx(1e-4)
    assert cosine_lr_with_warmup(500, **SCHEDULE) == pytest.approx(1e-4)


def test_schedule_is_monotonic_after_warmup():
    values = [cosine_lr_with_warmup(s, **SCHEDULE) for s in range(10, 101)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_apply_lr_updates_every_group():
    model = Transformer(ModelConfig.debug())
    optimizer = build_optimizer(model, TrainConfig())
    apply_lr(optimizer, 0.007)
    assert [g["lr"] for g in optimizer.param_groups] == [0.007, 0.007]


def test_optimizer_excludes_norms_from_weight_decay():
    """1-D parameters (RMSNorm gains) must not be decayed."""
    model = Transformer(ModelConfig.debug())
    optimizer = build_optimizer(model, TrainConfig(weight_decay=0.1))
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() == 1 for p in no_decay["params"])
    assert len(no_decay["params"]) == 2 * model.config.num_layers + 1  # +final norm


def test_checkpoints_are_ordered_numerically(tmp_path):
    """Regression: lexicographic sort puts model_step_1000 before _999."""
    manager = CheckpointManager(tmp_path, keep_last_n=0)
    for step in (1, 9, 10, 99, 100, 999, 1000, 10_000):
        (tmp_path / f"model_step_{step}.pt").touch()
    steps = [int(p.stem.split("_")[-1]) for p in manager.list_checkpoints()]
    assert steps == [1, 9, 10, 99, 100, 999, 1000, 10_000]
    assert manager.latest().name == "model_step_10000.pt"


def test_checkpoint_round_trip_restores_weights_and_step(tmp_path):
    config = ModelConfig.debug()
    model = Transformer(config)
    optimizer = build_optimizer(model, TrainConfig())
    manager = CheckpointManager(tmp_path)
    manager.save(model, optimizer, step=42)

    restored = Transformer(config)
    assert not torch.allclose(
        restored.embedding.weight, model.embedding.weight
    )
    step = manager.load(restored, build_optimizer(restored, TrainConfig()))
    assert step == 42
    torch.testing.assert_close(restored.embedding.weight, model.embedding.weight)


def test_rotation_keeps_only_the_most_recent(tmp_path):
    config = ModelConfig.debug()
    model = Transformer(config)
    optimizer = build_optimizer(model, TrainConfig())
    manager = CheckpointManager(tmp_path, keep_last_n=2)
    for step in (1, 2, 3, 4):
        manager.save(model, optimizer, step=step)
    assert [p.name for p in manager.list_checkpoints()] == [
        "model_step_3.pt",
        "model_step_4.pt",
    ]


def test_architecture_mismatch_is_rejected(tmp_path):
    manager = CheckpointManager(tmp_path)
    model = Transformer(ModelConfig.debug())
    manager.save(model, build_optimizer(model, TrainConfig()), step=1)

    different = ModelConfig.debug()
    different.num_layers = 3
    other = Transformer(different)
    with pytest.raises(ValueError, match="does not match"):
        manager.load(other, None)


def test_train_config_resolves_device_and_precision_without_cuda():
    config = TrainConfig()
    device = config.resolve_device()
    assert device in {"cpu", "cuda"}
    if device == "cpu":
        assert config.resolve_precision("cpu") == "fp32"
        # An explicit half-precision request still degrades safely on CPU.
        assert TrainConfig(precision="bf16").resolve_precision("cpu") == "fp32"


def test_requesting_cuda_without_cuda_fails_loudly():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        TrainConfig(device="cuda").resolve_device()


def test_end_to_end_training_reduces_loss(tmp_path):
    """A tiny model on a repeating corpus should learn it quickly."""
    torch.manual_seed(0)
    pattern = list(range(1, 33)) * 40
    write_shards([pattern], tmp_path / "shards", eos_id=3, vocab_size=64, shard_size=10**6)

    dataset = BinaryShardDataset(
        tmp_path / "shards", context_length=16, shuffle=False, infinite=True
    )
    config = ModelConfig(
        vocab_size=64, hidden_size=64, num_layers=2, num_heads=4,
        num_kv_heads=2, head_dim=16, intermediate_size=128, max_seq_len=16,
    )
    train_config = TrainConfig(
        batch_size=4, gradient_accumulation_steps=2, context_length=16,
        max_steps=40, warmup_steps=5, learning_rate=3e-3, min_learning_rate=1e-4,
        device="cpu", num_workers=0, log_every=5, save_every=1000,
        eval_every=10**9, checkpoint_dir=str(tmp_path / "ckpt"),
    )

    trainer = Trainer(Transformer(config), train_config, dataset)
    trainer.train()

    history = trainer.logger.read_history()
    losses = [r["loss"] for r in history if "loss" in r]
    assert trainer.global_step == 40
    assert (tmp_path / "ckpt" / "model_step_40.pt").exists()
    assert losses and losses[-1] < losses[0]


def test_resume_continues_from_saved_step(tmp_path):
    torch.manual_seed(0)
    write_shards([list(range(1, 33)) * 20], tmp_path / "shards",
                 eos_id=3, vocab_size=64, shard_size=10**6)
    dataset = BinaryShardDataset(
        tmp_path / "shards", context_length=16, shuffle=False, infinite=True
    )
    config = ModelConfig(
        vocab_size=64, hidden_size=32, num_layers=2, num_heads=2,
        num_kv_heads=1, head_dim=16, intermediate_size=64, max_seq_len=16,
    )
    common = dict(
        batch_size=2, gradient_accumulation_steps=1, context_length=16,
        device="cpu", num_workers=0, log_every=10**9, eval_every=10**9,
        warmup_steps=2, checkpoint_dir=str(tmp_path / "ckpt"),
    )

    first = Trainer(Transformer(config), TrainConfig(max_steps=10, save_every=10, **common), dataset)
    first.train()

    second = Trainer(Transformer(config), TrainConfig(max_steps=20, save_every=10, **common), dataset)
    assert second.resume() == 10
    second.train()
    assert second.global_step == 20
