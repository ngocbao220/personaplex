"""Bounded OtoSpeech download and conversion into the PersonaPlex manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import sphn


REPO_ID = "otoearth/otoSpeech-full-duplex-turn-104h"
REQUIRED_FILES = (
    "metadata.json",
    "speaker_1_annotation_a.srt",
    "speaker_2_annotation_a.srt",
    "speaker_1_audio.wav",
    "speaker_2_audio.wav",
)
_TIMESTAMP = re.compile(r"^(\d\d):(\d\d):(\d\d)[,.](\d\d\d)$")


def parse_srt(source: str) -> list[dict[str, object]]:
    """Parse the small SRT subset supplied by OtoSpeech into a timed timeline."""
    entries: list[dict[str, object]] = []
    for block in re.split(r"\r?\n\s*\r?\n", source.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3 or " --> " not in lines[1]:
            raise ValueError("invalid SRT entry")
        start, end = lines[1].split(" --> ", 1)
        entries.append({"start": _seconds(start), "end": _seconds(end), "text": " ".join(lines[2:])})
    if not entries:
        raise ValueError("SRT contained no entries")
    return entries


def _seconds(value: str) -> float:
    match = _TIMESTAMP.fullmatch(value.strip())
    if not match:
        raise ValueError(f"invalid SRT timestamp: {value}")
    hour, minute, second, millisecond = (int(part) for part in match.groups())
    return hour * 3600 + minute * 60 + second + millisecond / 1000


def sample_directories(files: Sequence[str], max_samples: int) -> list[str]:
    """Return deterministic sample roots whose required OtoSpeech assets exist."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    available = set(files)
    roots = sorted({path.rsplit("/", 1)[0] for path in available if path.endswith("/metadata.json")})
    complete = [root for root in roots if all(f"{root}/{name}" in available for name in REQUIRED_FILES)]
    return complete[:max_samples]


def download_snapshot(max_samples: int, repo_id: str = REPO_ID) -> Path:
    """Download only the exact files for at most ``max_samples`` conversations."""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:  # Keep local manifest tooling usable without HF installed.
        raise RuntimeError("install huggingface_hub to download OtoSpeech") from exc
    roots = sample_directories(HfApi().list_repo_files(repo_id, repo_type="dataset"), max_samples)
    if not roots:
        raise RuntimeError("no complete OtoSpeech samples were found")
    patterns = [f"{root}/{name}" for root in roots for name in REQUIRED_FILES]
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=patterns))


def prepare_from_directory(
    snapshot_dir: Path,
    output_dir: Path,
    max_samples: int,
    role_prompts: tuple[str, str],
    role_prompt_version: str,
    read_audio: Callable[[Path], tuple[np.ndarray, int]] = sphn.read,
    write_audio: Callable[[Path, np.ndarray, int], None] = sphn.write_wav,
) -> list[dict[str, object]]:
    """Create stereo WAVs, voice prompts, and a JSONL manifest from local files."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if not all(prompt.strip() for prompt in role_prompts) or not role_prompt_version.strip():
        raise ValueError("two non-empty reviewed role prompts and a role prompt version are required")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audio").mkdir()
    (output_dir / "prompts").mkdir()
    records = []
    for source in sorted(path for path in snapshot_dir.iterdir() if path.is_dir())[:max_samples]:
        paths = {name: source / name for name in REQUIRED_FILES}
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise ValueError(f"{source.name} is missing: {', '.join(missing)}")
        speaker_1, sample_rate = read_audio(paths["speaker_1_audio.wav"])
        speaker_2, second_rate = read_audio(paths["speaker_2_audio.wav"])
        if speaker_1.ndim == 2 and speaker_1.shape[0] == 1:
            speaker_1 = speaker_1[0]
        if speaker_2.ndim == 2 and speaker_2.shape[0] == 1:
            speaker_2 = speaker_2[0]
        if sample_rate != second_rate or speaker_1.shape != speaker_2.shape:
            raise ValueError(f"{source.name} speaker streams must have matching sample rates and lengths")
        if speaker_1.ndim != 1 or speaker_2.ndim != 1:
            raise ValueError(f"{source.name} speaker streams must be mono")
        first = parse_srt(paths["speaker_1_annotation_a.srt"].read_text())
        second = parse_srt(paths["speaker_2_annotation_a.srt"].read_text())
        metadata = json.loads(paths["metadata.json"].read_text())
        identifier = source.name
        stereo_path = output_dir / "audio" / f"{identifier}.wav"
        write_audio(stereo_path, np.stack((speaker_1, speaker_2)), sample_rate)
        prompt_frames = min(speaker_1.size, sample_rate * 3)
        prompt_1 = output_dir / "prompts" / f"{identifier}-speaker-1.wav"
        prompt_2 = output_dir / "prompts" / f"{identifier}-speaker-2.wav"
        write_audio(prompt_1, speaker_1[:prompt_frames][None], sample_rate)
        write_audio(prompt_2, speaker_2[:prompt_frames][None], sample_rate)
        records.append({
            "id": identifier, "stereo_wav": f"audio/{identifier}.wav",
            "speaker_ids": [f"{identifier}:speaker-1", f"{identifier}:speaker-2"],
            "split_group": identifier, "split": "train", "language": metadata.get("language", "en"),
            "approved_role_prompts": list(role_prompts),
            "role_prompt_provenance": [{"status": "approved", "version": role_prompt_version}] * 2,
            "transcripts": [first, second],
            "voice_prompts": [
                {"path": f"prompts/{identifier}-speaker-1.wav", "start": 0.0, "end": prompt_frames / sample_rate},
                {"path": f"prompts/{identifier}-speaker-2.wav", "start": 0.0, "end": prompt_frames / sample_rate},
            ],
        })
    if not records:
        raise ValueError("no OtoSpeech sample directories were found")
    with (output_dir / "manifest.jsonl").open("w") as manifest:
        for record in records:
            manifest.write(json.dumps(record) + "\n")
    return records


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download a bounded OtoSpeech subset and create a PersonaPlex manifest.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-samples", required=True, type=int)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--speaker-1-role-prompt", required=True)
    parser.add_argument("--speaker-2-role-prompt", required=True)
    parser.add_argument("--role-prompt-version", required=True)
    args = parser.parse_args(argv)
    snapshot_dir = download_snapshot(args.max_samples, args.repo_id)
    records = prepare_from_directory(
        snapshot_dir, args.output_dir, args.max_samples,
        (args.speaker_1_role_prompt, args.speaker_2_role_prompt), args.role_prompt_version,
    )
    print(json.dumps({"manifest": str(args.output_dir / "manifest.jsonl"), "samples": len(records)}))


if __name__ == "__main__":
    main()
