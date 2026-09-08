"""Preflight validation without downloading or loading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import sphn

from .contract import ManifestError, resolve_record_paths, validate_record


def validate_manifest(path: Path) -> int:
    """Return the number of safe records or raise with line-specific context."""
    count = 0
    group_splits: dict[str, str] = {}
    speaker_splits: dict[str, str] = {}
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = resolve_record_paths(validate_record(json.loads(line)), path.parent)
                pcm, sample_rate = sphn.read(record.stereo_wav)
                if pcm.ndim != 2 or pcm.shape[0] != 2:
                    raise ManifestError("stereo_wav must decode to exactly two channels")
                duration = pcm.shape[1] / sample_rate
                if any(segment.end > duration for timeline in record.transcripts for segment in timeline):
                    raise ManifestError("transcript exceeds stereo audio duration")
                for voice_prompt in record.voice_prompts:
                    if not Path(voice_prompt.path).is_file():
                        raise ManifestError(f"voice prompt is missing: {voice_prompt.path}")
                prior_group_split = group_splits.setdefault(record.split_group, record.split)
                if prior_group_split != record.split:
                    raise ManifestError("split_group appears in more than one split")
                for speaker in record.speaker_ids:
                    prior_speaker_split = speaker_splits.setdefault(speaker, record.split)
                    if prior_speaker_split != record.split:
                        raise ManifestError("speaker identity appears in more than one split")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise ManifestError(f"{path}:{line_number}: {exc}") from exc
            count += 1
    if not count:
        raise ManifestError("manifest contained no records")
    return count


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate a PersonaPlex stereo training manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps({"valid_records": validate_manifest(args.manifest)}))


if __name__ == "__main__":
    main()
