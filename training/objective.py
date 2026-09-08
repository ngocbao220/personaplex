"""Supervised objective for the fixed PersonaPlex stream layout."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _weighted_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    flat_weights = weights.reshape(-1).to(losses.dtype)
    return (losses * flat_weights).sum(), flat_weights.sum()


def weighted_personaplex_loss(
    text_logits: torch.Tensor,
    audio_logits: torch.Tensor,
    codes: torch.Tensor,
    weights: torch.Tensor,
    text_mask: torch.Tensor | None = None,
    audio_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute masked CE for agent text and the first eight audio streams.

    `audio_logits` may include all 16 codebooks produced by the LM. The last
    eight belong to user conditioning and are intentionally excluded.
    """
    if codes.ndim != 3 or codes.shape[1] != 17 or weights.shape != codes.shape:
        raise ValueError("codes and weights must have shape [B, 17, T]")
    if text_logits.shape[:3] != (codes.shape[0], 1, codes.shape[2]):
        raise ValueError("text logits do not match codes")
    if audio_logits.shape[:3] != (codes.shape[0], 16, codes.shape[2]):
        raise ValueError("audio logits must contain 16 streams")

    text_weights = weights[:, :1]
    audio_weights = weights[:, 1:9]
    if text_mask is not None:
        text_weights = text_weights * text_mask.to(text_weights.dtype)
    if audio_mask is not None:
        audio_weights = audio_weights * audio_mask[:, :8].to(audio_weights.dtype)
    text_total, text_count = _weighted_cross_entropy(text_logits, codes[:, :1], text_weights)
    audio_total, audio_count = _weighted_cross_entropy(audio_logits[:, :8], codes[:, 1:9], audio_weights)
    count = text_count + audio_count
    if count.item() == 0:
        raise ValueError("loss mask excludes every supervised token")
    return (text_total + audio_total) / count
