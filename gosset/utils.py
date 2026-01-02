from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch


def shannon_entropy_from_logits(logits: torch.Tensor, *, base: str = "e") -> torch.Tensor:
    """Compute Shannon entropy of a categorical distribution parameterized by logits.

    Args:
        logits: Tensor [..., V]
        base: "e" for nats, "2" for bits.

    Returns:
        Entropy tensor with shape logits.shape[:-1]
    """
    # log_softmax is stable
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    ent = -(probs * log_probs).sum(dim=-1)
    if base == "2":
        ent = ent / math.log(2.0)
    return ent


def seed_everything(seed: int) -> torch.Generator:
    """Seed torch and return a generator."""
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def safe_token_text(s: str) -> str:
    """Make a token string safe-ish for display in tables."""
    # Keep original for rendering; for labels we show repr-like escapes.
    return s.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def apply_repetition_penalty(logits: torch.Tensor, generated_token_ids: List[int], penalty: float) -> torch.Tensor:
    """Apply repetition penalty as in HF generate. penalty=1.0 means no-op."""
    if penalty is None or penalty == 1.0 or not generated_token_ids:
        return logits
    # logits is 1D [V]
    logits = logits.clone()
    for token_id in set(generated_token_ids):
        score = logits[token_id]
        # If score < 0 then multiply by penalty, else divide
        logits[token_id] = score * penalty if score < 0 else score / penalty
    return logits


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering on logits. top_p=1.0 is no-op.

    Returns modified logits with filtered tokens set to -inf.
    """
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)

    # Remove tokens with cumulative prob above threshold
    mask = cumprobs > top_p
    # Keep at least one token
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))

    # Scatter back
    new_logits = logits.clone()
    new_logits[sorted_indices] = sorted_logits
    return new_logits


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Top-k filtering on logits. top_k=0 disables."""
    if top_k is None or top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k)
    min_value = values[-1]
    return torch.where(logits < min_value, torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype), logits)


def min_p_filtering(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Min-p filtering (keep tokens with prob >= min_p * max_prob). min_p=0 disables."""
    if min_p is None or min_p <= 0.0:
        return logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    max_prob = probs.max()
    keep = probs >= (min_p * max_prob)
    # Ensure at least one token
    if keep.sum() == 0:
        keep[probs.argmax()] = True
    return logits.masked_fill(~keep, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repetition_penalty: float,
    generated_token_ids: List[int],
    generator: Optional[torch.Generator] = None,
) -> Tuple[int, torch.Tensor]:
    """Sample next token id from logits and return (token_id, probs_used)."""
    if temperature is None:
        temperature = 1.0
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = logits / temperature
    logits = apply_repetition_penalty(logits, generated_token_ids, repetition_penalty)
    logits = top_k_filtering(logits, top_k)
    logits = top_p_filtering(logits, top_p)
    logits = min_p_filtering(logits, min_p)

    probs = torch.nn.functional.softmax(logits, dim=-1)

    token_id = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
    return token_id, probs
