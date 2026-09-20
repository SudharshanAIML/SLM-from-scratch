from __future__ import annotations

import torch
import torch.nn.functional as F

from slm.model.transformer import Transformer
from slm.tokenizer.bpe_tokenizer import BOS_ID, EOS_ID, BPETokenizer


def apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Divide the logits of already-emitted tokens (multiply if negative)."""
    if penalty == 1.0 or generated.numel() == 0:
        return logits
    for token in set(generated.flatten().tolist()):
        score = logits[0, token]
        logits[0, token] = score / penalty if score > 0 else score * penalty
    return logits


def filter_logits(
    logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None
) -> torch.Tensor:
    """Mask logits outside the top-k / top-p nucleus."""
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        ordered, indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        remove = cumulative - torch.softmax(ordered, dim=-1) >= top_p
        ordered = ordered.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, indices, ordered)

    return logits


@torch.no_grad()
def generate(
    model: Transformer,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    repetition_penalty: float = 1.0,
    device: torch.device | str | None = None,
    add_bos: bool = True,
    stop_at_eos: bool = True,
    seed: int | None = None,
) -> str:
    """Autoregressively continue `prompt`.

    Decoding runs through the KV cache: the prompt is processed once, then each
    new token attends to the cached keys/values instead of re-running the whole
    prefix. That makes generation linear in length rather than quadratic.
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    ids = tokenizer.encode(prompt, add_bos=add_bos)
    if not ids:
        ids = [BOS_ID]

    max_seq_len = model.config.max_seq_len
    if len(ids) >= max_seq_len:
        ids = ids[-(max_seq_len - 1) :]

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    cache = model.new_cache()
    generated: list[int] = []

    # Prefill the cache with the prompt, then decode one token at a time.
    logits = model(input_ids, cache=cache)[:, -1, :]

    for _ in range(max_new_tokens):
        if cache.offset >= max_seq_len:
            break

        step_logits = logits.float().clone()
        if generated:
            step_logits = apply_repetition_penalty(
                step_logits,
                torch.tensor(generated, device=device),
                repetition_penalty,
            )

        if temperature <= 0.0:
            next_token = step_logits.argmax(dim=-1, keepdim=True)
        else:
            step_logits = filter_logits(step_logits / temperature, top_k, top_p)
            probs = torch.softmax(step_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        token_id = int(next_token.item())
        if stop_at_eos and token_id == EOS_ID:
            break

        generated.append(token_id)
        logits = model(next_token, cache=cache)[:, -1, :]

    return tokenizer.decode(ids + generated)
