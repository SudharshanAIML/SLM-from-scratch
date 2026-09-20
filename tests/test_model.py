import math

import pytest
import torch

from slm.configs.model_config import ModelConfig
from slm.model.transformer import Transformer

REFERENCE_TOTAL = 259_039_232  # spec total, excluding the final RMSNorm


def test_reference_config_matches_spec():
    config = ModelConfig()
    assert (config.num_layers, config.hidden_size) == (20, 1024)
    assert (config.num_heads, config.num_kv_heads, config.head_dim) == (16, 4, 64)
    assert config.intermediate_size == 2816
    assert config.vocab_size == 32_768
    assert config.max_seq_len == 2048
    assert config.hidden_size == config.num_heads * config.head_dim


def test_analytical_breakdown_matches_derivation():
    parts = ModelConfig().parameter_breakdown()
    assert parts["embedding"] == 32_768 * 1024 == 33_554_432
    assert parts["attention_per_layer"] == 2_621_440
    assert parts["swiglu_per_layer"] == 3 * 1024 * 2816 == 8_650_752
    assert parts["norms_per_layer"] == 2_048
    assert parts["per_layer"] == 11_274_240
    assert parts["transformer_stack"] == 11_274_240 * 20 == 225_484_800
    assert parts["total"] - parts["final_norm"] == REFERENCE_TOTAL


def test_built_model_parameter_count_is_exact():
    """The instantiated model must hit the spec number, not approximate it."""
    config = ModelConfig()
    model = Transformer(config)
    measured = model.num_parameters()
    assert measured == config.parameter_breakdown()["total"]
    assert measured - config.hidden_size == REFERENCE_TOTAL


def test_tied_lm_head_shares_storage_and_is_counted_once():
    model = Transformer(ModelConfig.debug())
    assert model.lm_head.weight is model.embedding.weight
    assert model.parameter_breakdown()["lm_head"] == 0


def test_untied_embeddings_add_a_second_matrix():
    tied = ModelConfig.debug()
    untied = ModelConfig.debug()
    untied.tie_embeddings = False
    extra = untied.vocab_size * untied.hidden_size
    assert (
        untied.parameter_breakdown()["total"]
        == tied.parameter_breakdown()["total"] + extra
    )
    assert Transformer(untied).num_parameters() == untied.parameter_breakdown()["total"]


def test_debug_preset_accepts_data_driven_vocab_and_context():
    """scripts/train.py --model debug builds this from dataset metadata."""
    config = ModelConfig.debug(vocab_size=1024, max_seq_len=128)
    assert config.vocab_size == 1024
    assert config.max_seq_len == 128
    model = Transformer(config)
    assert model.num_parameters() == config.parameter_breakdown()["total"]
    assert model.num_parameters() < 1_000_000


def test_initial_loss_is_near_uniform():
    """A correctly initialised LM starts at about ln(vocab_size)."""
    torch.manual_seed(0)
    config = ModelConfig(vocab_size=4096, num_layers=6, max_seq_len=128)
    model = Transformer(config)
    model.eval()

    x = torch.randint(0, config.vocab_size, (4, 65))
    with torch.no_grad():
        logits = model(x[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size), x[:, 1:].reshape(-1)
        )
    assert loss.item() == pytest.approx(math.log(config.vocab_size), rel=0.1)


def test_forward_shape_and_dtype():
    config = ModelConfig.debug()
    model = Transformer(config)
    out = model(torch.randint(0, config.vocab_size, (3, 17)))
    assert out.shape == (3, 17, config.vocab_size)
    assert out.dtype == torch.float32


def test_gradients_reach_every_parameter():
    config = ModelConfig.debug()
    model = Transformer(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    model(x).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == []


def test_gradient_checkpointing_matches_plain_forward():
    torch.manual_seed(0)
    config = ModelConfig.debug()
    plain = Transformer(config)
    checkpointed = Transformer(config, use_gradient_checkpointing=True)
    checkpointed.load_state_dict(plain.state_dict())
    plain.train(), checkpointed.train()

    x = torch.randint(0, config.vocab_size, (2, 16))
    torch.testing.assert_close(plain(x), checkpointed(x))


def test_sequence_longer_than_context_is_rejected():
    config = ModelConfig.debug()
    model = Transformer(config)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(torch.zeros(1, config.max_seq_len + 1, dtype=torch.long))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"num_heads": 15}, "divisible"),
        ({"hidden_size": 512}, "must equal"),
        ({"head_dim": 63, "hidden_size": 63 * 16}, "even"),
        ({"vocab_size": 70_000}, "uint16"),
    ],
)
def test_invalid_configs_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        ModelConfig(**overrides)
