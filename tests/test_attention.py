import torch

from slm.configs.model_config import ModelConfig
from slm.layers.attention import repeat_kv
from slm.model.transformer import Transformer


def test_repeat_kv_maps_head_j_to_kv_head_j_over_n():
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).view(2, 3, 4, 5)
    out = repeat_kv(x, 4)
    assert out.shape == (2, 12, 4, 5)
    for head in range(12):
        torch.testing.assert_close(out[:, head], x[:, head // 4])


def test_repeat_kv_matches_repeat_interleave():
    x = torch.randn(2, 4, 8, 16)
    torch.testing.assert_close(repeat_kv(x, 4), x.repeat_interleave(4, dim=1))


def test_attention_is_causal():
    """Changing token t must not alter any output before position t."""
    torch.manual_seed(0)
    config = ModelConfig.debug()
    model = Transformer(config).eval()

    x = torch.randint(0, config.vocab_size, (1, 24))
    with torch.no_grad():
        baseline = model(x)

    edited = x.clone()
    edited[0, 12] = (edited[0, 12] + 7) % config.vocab_size
    with torch.no_grad():
        changed = model(edited)

    torch.testing.assert_close(baseline[:, :12], changed[:, :12])
    assert not torch.allclose(baseline[:, 12], changed[:, 12])


def test_kv_cache_decoding_matches_full_forward():
    """Token-by-token decoding with the cache must equal a single full pass.

    This is the property the previous implementation violated: it re-applied
    RoPE to already-rotated cached keys and rotated each new query at
    position 0.
    """
    torch.manual_seed(0)
    config = ModelConfig.debug()
    model = Transformer(config).eval()

    x = torch.randint(0, config.vocab_size, (1, 20))
    with torch.no_grad():
        reference = model(x)

        cache = model.new_cache()
        stepwise = [model(x[:, :1], cache=cache)]
        for position in range(1, x.size(1)):
            stepwise.append(model(x[:, position : position + 1], cache=cache))
        stepwise = torch.cat(stepwise, dim=1)

    assert cache.offset == x.size(1)
    torch.testing.assert_close(reference, stepwise, rtol=1e-4, atol=1e-4)


def test_cache_prefill_then_decode_matches_full_forward():
    """Prefill a prompt in one pass, then decode: still equals the full pass."""
    torch.manual_seed(0)
    config = ModelConfig.debug()
    model = Transformer(config).eval()

    x = torch.randint(0, config.vocab_size, (1, 20))
    with torch.no_grad():
        reference = model(x)

        cache = model.new_cache()
        prefill = model(x[:, :12], cache=cache)
        rest = [model(x[:, i : i + 1], cache=cache) for i in range(12, x.size(1))]

    torch.testing.assert_close(reference[:, :12], prefill, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        reference[:, 12:], torch.cat(rest, dim=1), rtol=1e-4, atol=1e-4
    )


def test_chunked_prefill_against_cache_matches_full_forward():
    """A multi-token chunk appended to a non-empty cache stays causal."""
    torch.manual_seed(0)
    config = ModelConfig.debug()
    model = Transformer(config).eval()

    x = torch.randint(0, config.vocab_size, (1, 24))
    with torch.no_grad():
        reference = model(x)
        cache = model.new_cache()
        model(x[:, :10], cache=cache)
        chunk = model(x[:, 10:24], cache=cache)

    torch.testing.assert_close(reference[:, 10:], chunk, rtol=1e-4, atol=1e-4)


def test_cache_reset_clears_state():
    config = ModelConfig.debug()
    model = Transformer(config).eval()
    x = torch.randint(0, config.vocab_size, (1, 8))
    cache = model.new_cache()
    with torch.no_grad():
        first = model(x, cache=cache)
        cache.reset()
        assert cache.offset == 0
        second = model(x, cache=cache)
    torch.testing.assert_close(first, second)


def test_gqa_uses_fewer_kv_parameters_than_mha():
    config = ModelConfig()
    attn = Transformer(ModelConfig.debug()).layers[0].attn
    assert attn.k_proj.out_features == attn.num_kv_heads * attn.head_dim
    assert attn.q_proj.out_features == attn.num_heads * attn.head_dim
    # Reference config: 4 KV heads instead of 16 shrinks k/v to a quarter.
    assert config.num_kv_groups == 4
