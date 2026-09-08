"""Turn reviewed stereo conversations into PersonaPlex training examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import sphn
import torch

from .contract import ConversationRecord, TimedText, TrainingExample, resolve_record_paths, validate_record
from .streams import build_codes


def align_text_tokens(
    segments: Iterable[tuple[float, float, Sequence[int]]], *, frame_rate: float, total_frames: int, padding_id: int
) -> torch.Tensor:
    """Put each agent token on a unique frame within its aligned word span."""
    stream = torch.full((total_frames,), padding_id, dtype=torch.long)
    for start, end, tokens in segments:
        first = max(0, round(start * frame_rate))
        stop = min(total_frames, max(first + 1, round(end * frame_rate)))
        if len(tokens) > stop - first:
            raise ValueError("too many text tokens for aligned timestamp window")
        for frame, token in zip(range(first, stop), tokens):
            stream[frame] = token
    return stream


def _read_mono(path: str, sample_rate: int, start: float = 0.0, end: float | None = None, channel: int | None = None) -> np.ndarray:
    pcm, source_rate = sphn.read(path)
    pcm = sphn.resample(pcm, src_sample_rate=source_rate, dst_sample_rate=sample_rate)
    if pcm.ndim != 2:
        raise ValueError(f"{path} did not decode to [channels, samples]")
    if channel is None:
        channel = 0
    if not 0 <= channel < pcm.shape[0]:
        raise ValueError(f"channel {channel} is unavailable in {path}")
    first = round(start * sample_rate)
    last = round(end * sample_rate) if end is not None else pcm.shape[1]
    if not 0 <= first < last <= pcm.shape[1]:
        raise ValueError(f"invalid prompt range for {path}")
    return pcm[channel : channel + 1, first:last]


def _encode(mimi, pcm: np.ndarray, device: str) -> torch.Tensor:
    with torch.inference_mode():
        codes = mimi.encode(torch.from_numpy(pcm).to(device=device, dtype=torch.float32)[None])
    if codes.shape[1] != 8:
        raise ValueError(f"expected eight Mimi codebooks, got {codes.shape[1]}")
    return codes[0].cpu().long()


def _token_segments(transcript: Sequence[TimedText], tokenizer) -> list[tuple[float, float, Sequence[int]]]:
    return [(segment.start, segment.end, tokenizer.encode(segment.text)) for segment in transcript]


def _system_prefix(record: ConversationRecord, agent_channel: int, tokenizer, mimi, device: str) -> tuple[torch.Tensor, int]:
    """Build voice then text prompt with paper-compatible silent/sine channels."""
    from moshi.models.lm import SILENCE_TOKENS, SINE_TOKENS

    prompt = record.voice_prompts[agent_channel]
    voice = _encode(mimi, _read_mono(prompt.path, mimi.sample_rate, prompt.start, prompt.end, prompt.channel), device)
    padding_id = 3
    role_tokens = tokenizer.encode(f"<system> {record.approved_role_prompts[agent_channel]} <system>")
    spacer_frames = round(0.5 * mimi.frame_rate)
    voice_text = torch.full((voice.shape[1],), padding_id, dtype=torch.long)
    voice_user = torch.as_tensor(SINE_TOKENS, dtype=torch.long)[:, None].expand(-1, voice.shape[1])
    spacer_text = torch.full((spacer_frames,), padding_id, dtype=torch.long)
    spacer_agent = torch.as_tensor(SILENCE_TOKENS, dtype=torch.long)[:, None].expand(-1, spacer_frames)
    spacer_user = torch.as_tensor(SINE_TOKENS, dtype=torch.long)[:, None].expand(-1, spacer_frames)
    role_text = torch.as_tensor(role_tokens, dtype=torch.long)
    role_agent = torch.as_tensor(SILENCE_TOKENS, dtype=torch.long)[:, None].expand(-1, len(role_tokens))
    role_user = torch.as_tensor(SINE_TOKENS, dtype=torch.long)[:, None].expand(-1, len(role_tokens))
    text_stream = torch.cat((voice_text, spacer_text, role_text, spacer_text))
    agent_stream = torch.cat((voice, spacer_agent, role_agent, spacer_agent), dim=1)
    user_stream = torch.cat((voice_user, spacer_user, role_user, spacer_user), dim=1)
    return torch.cat((text_stream[None], agent_stream, user_stream), dim=0), voice.shape[1] + 2 * spacer_frames + len(role_tokens)


def prepare_record(record: ConversationRecord, tokenizer, mimi, device: str) -> list[dict[str, object]]:
    """Materialize both A→B and B→A examples, preserving causal stream order."""
    pcm, source_rate = sphn.read(record.stereo_wav)
    pcm = sphn.resample(pcm, src_sample_rate=source_rate, dst_sample_rate=mimi.sample_rate)
    if pcm.ndim != 2 or pcm.shape[0] != 2:
        raise ValueError(f"{record.stereo_wav} must be exactly two-channel audio")
    duration = pcm.shape[1] / mimi.sample_rate
    if any(segment.end > duration for timeline in record.transcripts for segment in timeline):
        raise ValueError(f"transcript exceeds audio duration for {record.id}")
    speaker_codes = [_encode(mimi, pcm[index : index + 1], device) for index in range(2)]
    if speaker_codes[0].shape[1] != speaker_codes[1].shape[1]:
        raise ValueError("stereo channels encoded to unequal frame counts")
    examples = []
    for agent_channel, user_channel in ((0, 1), (1, 0)):
        frames = speaker_codes[agent_channel].shape[1]
        text = align_text_tokens(
            _token_segments(record.transcripts[agent_channel], tokenizer), frame_rate=mimi.frame_rate,
            total_frames=frames, padding_id=3,
        )
        prefix, prompt_frames = _system_prefix(record, agent_channel, tokenizer, mimi, device)
        dialogue = build_codes(TrainingExample(
            id=f"{record.id}:{user_channel}-to-{agent_channel}", user_channel=user_channel, agent_channel=agent_channel,
            agent_text=text, agent_audio=speaker_codes[agent_channel], user_audio=speaker_codes[user_channel], prompt_frames=prompt_frames,
        ))
        examples.append({
            "id": f"{record.id}:{user_channel}-to-{agent_channel}", "codes": torch.cat((prefix, dialogue), dim=1),
            "prompt_frames": prompt_frames, "split_group": record.split_group, "split": record.split,
            "agent_speaker_id": record.speaker_ids[agent_channel], "user_speaker_id": record.speaker_ids[user_channel],
            "language": record.language, "role_prompt": record.approved_role_prompts[agent_channel],
        })
    return examples


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare reviewed stereo dialogues for PersonaPlex fine-tuning.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mimi-weight", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    from moshi.models.loaders import get_mimi
    import sentencepiece

    args.output_dir.mkdir(parents=True, exist_ok=False)
    mimi = get_mimi(args.mimi_weight, args.device)
    tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)
    index_path = args.output_dir / "index.jsonl"
    with args.manifest.open() as source, index_path.open("w") as index:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = resolve_record_paths(validate_record(json.loads(line)), args.manifest.parent)
            for example in prepare_record(record, tokenizer, mimi, args.device):
                name = hashlib.sha256(str(example["id"]).encode()).hexdigest()[:16] + ".pt"
                destination = args.output_dir / name
                torch.save(example, destination)
                index.write(json.dumps({"id": example["id"], "path": name, "split_group": example["split_group"], "split": example["split"]}) + "\n")
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "manifest_sha256": _sha256(args.manifest), "mimi_weight": args.mimi_weight, "tokenizer": args.tokenizer,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
