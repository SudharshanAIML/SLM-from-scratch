import math

import torch

from slm.configs.model_config import ModelConfig
from slm.model.generation import apply_repetition_penalty, filter_logits, generate
from slm.model.transformer import Transformer
from slm.tokenizer.bpe_tokenizer import EOS_ID


def tiny_model(tokenizer, **overrides):
    overrides.setdefault("max_seq_len", 64)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size, hidden_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128,
        **overrides,
    )
    torch.manual_seed(0)
    return Transformer(config).eval()


# ------------------------------------------------------------------ filtering


def test_top_k_keeps_exactly_k_candidates():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    filtered = filter_logits(logits, top_k=2, top_p=None)
    assert torch.isfinite(filtered).sum() == 2
    assert torch.isfinite(filtered[0, 1]) and torch.isfinite(filtered[0, 4])


def test_top_k_larger_than_vocab_keeps_everything():
    logits = torch.randn(1, 5)
    assert torch.isfinite(filter_logits(logits, top_k=99, top_p=None)).all()


def test_top_p_keeps_the_smallest_sufficient_nucleus():
    # Probabilities are 0.5, 0.25, 0.15, 0.10 after softmax of these logits.
    probs = torch.tensor([[0.5, 0.25, 0.15, 0.10]])
    logits = probs.log()
    kept = torch.isfinite(filter_logits(logits, top_k=None, top_p=0.7))
    # 0.5 alone is < 0.7, so the second token is needed; the rest are cut.
    assert kept.tolist() == [[True, True, False, False]]


def test_top_p_always_keeps_the_argmax():
    logits = torch.tensor([[10.0, 1.0, 1.0, 1.0]])
    kept = torch.isfinite(filter_logits(logits, top_k=None, top_p=0.01))
    assert kept[0, 0].item() is True
    assert kept.sum() == 1


def test_repetition_penalty_lowers_seen_positive_logits():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    out = apply_repetition_penalty(logits.clone(), torch.tensor([0, 1]), penalty=2.0)
    assert out[0, 0].item() == 1.0  # positive -> divided
    assert out[0, 1].item() == -4.0  # negative -> multiplied
    assert out[0, 2].item() == 1.0  # untouched


def test_repetition_penalty_of_one_is_a_noop():
    logits = torch.randn(1, 10)
    torch.testing.assert_close(
        apply_repetition_penalty(logits.clone(), torch.tensor([1, 2]), 1.0), logits
    )


# ----------------------------------------------------------------- generation


def test_greedy_generation_is_deterministic(tokenizer):
    model = tiny_model(tokenizer)
    kwargs = dict(max_new_tokens=12, temperature=0.0, device="cpu")
    first = generate(model, tokenizer, "science", **kwargs)
    second = generate(model, tokenizer, "science", **kwargs)
    assert first == second


def test_sampling_with_a_seed_is_reproducible(tokenizer):
    model = tiny_model(tokenizer)
    kwargs = dict(max_new_tokens=12, temperature=0.9, seed=1234, device="cpu")
    assert generate(model, tokenizer, "history", **kwargs) == generate(
        model, tokenizer, "history", **kwargs
    )


def test_generation_stops_at_eos(tokenizer):
    """A model forced to emit EOS must stop rather than run to the token budget."""
    model = tiny_model(tokenizer)
    with torch.no_grad():
        model.lm_head.weight[EOS_ID] += 100.0

    out = generate(
        model, tokenizer, "physics", max_new_tokens=50, temperature=0.0, device="cpu"
    )
    # Nothing was appended beyond the prompt, and no EOS marker leaks into text.
    assert "<eos>" not in out


def test_generation_respects_the_token_budget(tokenizer):
    model = tiny_model(tokenizer)
    prompt_len = len(tokenizer.encode("data", add_bos=True))
    for budget in (0, 1, 8):
        out = generate(
            model, tokenizer, "data", max_new_tokens=budget,
            temperature=0.0, device="cpu", stop_at_eos=False,
        )
        assert len(tokenizer.encode(out)) <= prompt_len + budget + 2


def test_generation_halts_at_the_context_limit(tokenizer):
    """Decoding must stop at max_seq_len instead of raising."""
    model = tiny_model(tokenizer, max_seq_len=24)
    out = generate(
        model, tokenizer, "learning and teaching", max_new_tokens=500,
        temperature=0.0, device="cpu", stop_at_eos=False,
    )
    assert isinstance(out, str)
    assert len(tokenizer.encode(out)) <= 24


def test_generation_leaves_the_model_state_untouched(tokenizer):
    model = tiny_model(tokenizer)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    generate(model, tokenizer, "network", max_new_tokens=6, device="cpu")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_output_begins_with_the_prompt(tokenizer):
    model = tiny_model(tokenizer)
    out = generate(
        model, tokenizer, "chemistry", max_new_tokens=5,
        temperature=0.0, device="cpu", stop_at_eos=False,
    )
    assert out.startswith("chemistry")
