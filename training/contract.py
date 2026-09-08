"""Strict, versionable input contract for reviewed duplex conversations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch


class ManifestError(ValueError):
    """A record cannot safely be used for supervised duplex training."""


@dataclass(frozen=True)
class TimedText:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class VoicePrompt:
    path: str
    start: float
    end: float
    channel: int | None = None


@dataclass(frozen=True)
class ConversationRecord:
    id: str
    stereo_wav: str
    speaker_ids: tuple[str, str]
    split_group: str
    split: str
    language: str
    approved_role_prompts: tuple[str, str]
    transcripts: tuple[tuple[TimedText, ...], tuple[TimedText, ...]]
    voice_prompts: tuple[VoicePrompt, VoicePrompt]


@dataclass(frozen=True)
class TrainingExample:
    """One directional, already tokenized dialogue sequence."""

    id: str
    user_channel: int
    agent_channel: int
    agent_text: torch.Tensor
    agent_audio: torch.Tensor
    user_audio: torch.Tensor
    prompt_frames: int


def _fail(message: str) -> None:
    raise ManifestError(message)


def _timed_text(items: Any, channel: int) -> tuple[TimedText, ...]:
    if not isinstance(items, list) or not items:
        _fail(f"transcripts[{channel}] must be a non-empty list")
    result = []
    previous_end = -1.0
    for item in items:
        try:
            entry = TimedText(float(item["start"]), float(item["end"]), str(item["text"]).strip())
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"invalid transcript entry in channel {channel}") from exc
        if not entry.text or entry.start < 0 or entry.end <= entry.start or entry.start < previous_end:
            _fail(f"invalid or unordered timestamp in transcripts[{channel}]")
        previous_end = entry.end
        result.append(entry)
    return tuple(result)


def _voice_prompt(value: Any, channel: int, stereo_wav: str, transcript: tuple[TimedText, ...]) -> VoicePrompt:
    try:
        prompt = VoicePrompt(
            path=str(value["path"]), start=float(value["start"]), end=float(value["end"]),
            channel=value.get("channel"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"invalid voice_prompts[{channel}]") from exc
    if not prompt.path or prompt.start < 0 or prompt.end <= prompt.start:
        _fail(f"invalid voice_prompts[{channel}]")
    if Path(prompt.path) == Path(stereo_wav) and prompt.channel == channel:
        for segment in transcript:
            if prompt.start < segment.end and segment.start < prompt.end:
                _fail(f"voice_prompts[{channel}] overlaps dialogue transcript")
    return prompt


def validate_record(raw: dict[str, Any]) -> ConversationRecord:
    """Parse one JSONL record and reject ambiguous or unsafe supervision."""
    required = ("id", "stereo_wav", "speaker_ids", "split_group", "split", "language", "approved_role_prompts",
                "role_prompt_provenance", "transcripts", "voice_prompts")
    missing = [key for key in required if key not in raw]
    if missing:
        _fail(f"missing required fields: {', '.join(missing)}")
    speakers = raw["speaker_ids"]
    if not isinstance(speakers, list) or len(speakers) != 2 or not all(isinstance(x, str) and x for x in speakers):
        _fail("speaker_ids must contain exactly two non-empty IDs")
    if speakers[0] == speakers[1]:
        _fail("speaker_ids must identify two distinct speakers")
    provenance = raw["role_prompt_provenance"]
    prompts = raw["approved_role_prompts"]
    if not isinstance(prompts, list) or len(prompts) != 2:
        _fail("approved_role_prompts must contain two prompts")
    if not isinstance(provenance, list) or len(provenance) != 2 or any(
        not isinstance(item, dict) or item.get("status") != "approved" or not item.get("version") for item in provenance
    ):
        _fail("each role prompt provenance must be approved and versioned")
    parsed_prompts = tuple(str(prompt).strip() for prompt in prompts)
    if not all(parsed_prompts):
        _fail("approved_role_prompts must not be empty")
    split = str(raw["split"])
    if split not in {"train", "validation", "test"}:
        _fail("split must be train, validation, or test")
    transcripts = raw["transcripts"]
    if not isinstance(transcripts, list) or len(transcripts) != 2:
        _fail("transcripts must contain exactly two channel timelines")
    parsed_transcripts = (_timed_text(transcripts[0], 0), _timed_text(transcripts[1], 1))
    prompts = raw["voice_prompts"]
    if not isinstance(prompts, list) or len(prompts) != 2:
        _fail("voice_prompts must contain exactly two entries")
    stereo_wav = str(raw["stereo_wav"])
    parsed_prompts = (
        _voice_prompt(prompts[0], 0, stereo_wav, parsed_transcripts[0]),
        _voice_prompt(prompts[1], 1, stereo_wav, parsed_transcripts[1]),
    )
    return ConversationRecord(
        id=str(raw["id"]), stereo_wav=stereo_wav, speaker_ids=(speakers[0], speakers[1]),
        split_group=str(raw["split_group"]), split=split, language=str(raw["language"]), approved_role_prompts=parsed_prompts,
        transcripts=parsed_transcripts, voice_prompts=parsed_prompts,
    )


def resolve_record_paths(record: ConversationRecord, manifest_dir: Path) -> ConversationRecord:
    """Make manifest-local asset references independent of the caller's cwd."""
    def resolve(path: str) -> str:
        candidate = Path(path)
        return str(candidate if candidate.is_absolute() else manifest_dir / candidate)

    resolved = replace(
        record,
        stereo_wav=resolve(record.stereo_wav),
        voice_prompts=tuple(replace(prompt, path=resolve(prompt.path)) for prompt in record.voice_prompts),
    )
    for channel, prompt in enumerate(resolved.voice_prompts):
        if Path(prompt.path) == Path(resolved.stereo_wav) and prompt.channel == channel:
            for segment in resolved.transcripts[channel]:
                if prompt.start < segment.end and segment.start < prompt.end:
                    _fail(f"voice_prompts[{channel}] overlaps dialogue transcript")
    return resolved
