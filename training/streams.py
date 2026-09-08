"""PersonaPlex training-stream and loss ownership."""

from __future__ import annotations

import torch

from .contract import TrainingExample


def build_codes(example: TrainingExample) -> torch.Tensor:
    """Return `[agent text, agent audio x8, user audio x8]` codes."""
    if example.agent_text.ndim != 1:
        raise ValueError("agent_text must have shape [T]")
    if example.agent_audio.ndim != 2 or example.user_audio.ndim != 2:
        raise ValueError("audio codes must have shape [8, T]")
    if example.agent_audio.shape[0] != 8 or example.user_audio.shape[0] != 8:
        raise ValueError("Mimi streams require exactly eight codebooks per speaker")
    frames = example.agent_text.shape[0]
    if example.agent_audio.shape[1] != frames or example.user_audio.shape[1] != frames:
        raise ValueError("all streams must have the same frame count")
    return torch.cat((example.agent_text[None], example.agent_audio, example.user_audio), dim=0)


def build_loss_weights(codes: torch.Tensor, prompt_frames: int, text_padding_id: int) -> torch.Tensor:
    """Mask prompts/user conditions and apply PersonaPlex token weighting."""
    if codes.ndim != 2 or codes.shape[0] != 17:
        raise ValueError("codes must have shape [17, T]")
    if not 0 <= prompt_frames <= codes.shape[1]:
        raise ValueError("prompt_frames is outside the sequence")
    weights = torch.zeros_like(codes, dtype=torch.float32)
    weights[0] = torch.where(codes[0] == text_padding_id, 0.3, 1.0)
    weights[1] = 1.0
    weights[2:9] = 0.02
    weights[:, :prompt_frames] = 0.0
    return weights
