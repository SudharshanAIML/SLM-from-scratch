import math

import pytest
import torch

from slm.layers.rope import RotaryEmbedding, apply_rope, rotate_half


def naive_rope(x, offset=0, theta=10_000.0):
    """Straightforward reference implementation, rebuilt per call."""
    *_, seq_len, dim = x.shape
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    pos = torch.arange(offset, offset + seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return apply_rope(x, emb.cos(), emb.sin())


def test_rotate_half_swaps_halves_with_sign():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    torch.testing.assert_close(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_cached_tables_match_naive_implementation():
    rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
    x = torch.randn(2, 4, 10, 16)
    cos, sin = rope(seq_len=10)
    torch.testing.assert_close(apply_rope(x, cos, sin), naive_rope(x))


def test_offset_positions_match_naive_implementation():
    rope = RotaryEmbedding(head_dim=16, max_seq_len=64)
    x = torch.randn(1, 2, 5, 16)
    cos, sin = rope(seq_len=5, offset=20)
    torch.testing.assert_close(apply_rope(x, cos, sin), naive_rope(x, offset=20))


def test_rope_preserves_vector_norm():
    """Rotation is orthogonal, so it cannot change magnitudes."""
    rope = RotaryEmbedding(head_dim=32, max_seq_len=16)
    x = torch.randn(2, 3, 16, 32)
    cos, sin = rope(seq_len=16)
    torch.testing.assert_close(
        apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), rtol=1e-5, atol=1e-5
    )


def test_attention_score_depends_only_on_relative_distance():
    """The defining RoPE property: <R(q,m), R(k,n)> is a function of m-n."""
    rope = RotaryEmbedding(head_dim=32, max_seq_len=128)
    q = torch.randn(1, 1, 1, 32)
    k = torch.randn(1, 1, 1, 32)

    def score(m, n):
        cq, sq = rope(seq_len=1, offset=m)
        ck, sk = rope(seq_len=1, offset=n)
        return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum()

    torch.testing.assert_close(score(5, 2), score(50, 47), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(score(9, 1), score(31, 23), rtol=1e-4, atol=1e-4)
    assert not torch.allclose(score(5, 2), score(5, 4), rtol=1e-3, atol=1e-3)


def test_cache_extends_beyond_initial_length():
    rope = RotaryEmbedding(head_dim=8, max_seq_len=4)
    cos, sin = rope(seq_len=40)
    assert cos.shape == (1, 1, 40, 8)
    x = torch.randn(1, 1, 40, 8)
    torch.testing.assert_close(apply_rope(x, cos, sin), naive_rope(x))


def test_odd_head_dim_is_rejected():
    with pytest.raises(ValueError, match="even"):
        RotaryEmbedding(head_dim=15)


def test_tables_are_not_persisted_in_state_dict():
    """Position tables are derived, so they must not bloat checkpoints."""
    rope = RotaryEmbedding(head_dim=16, max_seq_len=2048)
    assert rope.state_dict() == {}
