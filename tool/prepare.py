"""Prepare reviewed, stereo conversation WAVs for a future PersonaPlex trainer."""

import argparse
import json
import subprocess
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


INPUT_SAMPLE_RATE = 16_000
Transcript = list[dict[str, object]]
Transcriber = Callable[[bytes, int, str], Transcript]


@dataclass(frozen=True)
class PreparationResult:
    prepared: int
    rejected: int


def _json_lines(path: Path) -> Iterable[dict[str, object]]:
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{number}: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path}:{number}.")
            yield value


def _roles_by_id(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Roles file does not exist: {path}")
    records: dict[str, dict[str, object]] = {}
    for row in _json_lines(path):
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Every roles row requires a non-empty string id.")
        if sample_id in records:
            raise ValueError(f"Duplicate role metadata id: {sample_id}")
        records[sample_id] = row
    return records


def _validated_roles(row: dict[str, object]) -> tuple[list[str], list[str]]:
    speaker_ids = row.get("speaker_ids")
    prompts = row.get("role_prompts")
    if not isinstance(speaker_ids, list) or len(speaker_ids) != 2 or not all(isinstance(item, str) and item for item in speaker_ids):
        raise ValueError("Role metadata requires two non-empty speaker_ids.")
    if speaker_ids[0] == speaker_ids[1]:
        raise ValueError("Role metadata requires two distinct speaker_ids.")
    if not isinstance(prompts, list) or len(prompts) != 2:
        raise ValueError("Role metadata requires exactly two role_prompts.")
    texts: list[str] = []
    for prompt in prompts:
        if not isinstance(prompt, dict) or prompt.get("status") != "approved":
            raise ValueError("Every role prompt must have status 'approved'.")
        text = prompt.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Every approved role prompt requires non-empty text.")
        texts.append(text.strip())
    return speaker_ids, texts


def _read_stereo_wav(path: Path) -> tuple[bytes, int, int, int]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.getnframes()
        compression = source.getcomptype()
        pcm = source.readframes(frames)
    if channels != 2:
        raise ValueError("PersonaPlex preparation requires exactly stereo WAV audio.")
    if compression != "NONE":
        raise ValueError("WAV input must be uncompressed PCM.")
    if width != 2:
        raise ValueError("WAV input must be 16-bit PCM.")
    if sample_rate != INPUT_SAMPLE_RATE:
        raise ValueError("WAV input must use a 16 kHz sample rate.")
    if sample_rate < 1 or frames < 1:
        raise ValueError("WAV input must contain audio frames.")
    return pcm, sample_rate, channels, frames


def _channel_pcm(stereo_pcm: bytes, channel: int) -> bytes:
    samples = memoryview(stereo_pcm).cast("h")
    mono = bytearray(len(stereo_pcm) // 2)
    for frame, value in enumerate(samples[channel::2]):
        struct.pack_into("<h", mono, frame * 2, value)
    return bytes(mono)


def _write_wav(path: Path, pcm: bytes, sample_rate: int, channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


def _validate_transcript(value: object, duration: float) -> Transcript:
    if not isinstance(value, list):
        raise ValueError("ASR must return a list of timestamped segments.")
    segments: Transcript = []
    for segment in value:
        if not isinstance(segment, dict):
            raise ValueError("ASR returned an invalid transcript segment.")
        text, start, end = segment.get("text"), segment.get("start"), segment.get("end")
        if not isinstance(text, str) or not text.strip() or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError("ASR transcript segments require text, start, and end.")
        if start < 0 or end <= start or end > duration + 0.05:
            raise ValueError("ASR transcript timestamps fall outside the audio duration.")
        segments.append({"text": text.strip(), "start": float(start), "end": float(end)})
    return segments


def _default_transcriber(model_name: str) -> Transcriber:
    try:
        import numpy as np
        import torch
        import whisper_timestamped as whisper
    except ImportError as error:
        raise RuntimeError(
            "ASR requires whisper-timestamped and its dependencies. Install with "
            "pip install whisper-timestamped."
        ) from error
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)

    def transcribe(pcm: bytes, sample_rate: int, language: str) -> Transcript:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        result = whisper.transcribe(model, samples, language=None if language == "auto" else language, verbose=None)
        return [
            {"text": segment["text"], "start": segment["start"], "end": segment["end"]}
            for segment in result.get("segments", [])
            if isinstance(segment.get("text"), str) and isinstance(segment.get("start"), (int, float)) and isinstance(segment.get("end"), (int, float))
        ]

    return transcribe


def normalize_directory(source_dir: Path, output_dir: Path) -> int:
    """Convert source WAV files to the strict 16 kHz PCM input contract.

    Direct child WAVs are preferred. If none exist, nested WAVs are used as a
    compatibility fallback for archive layouts such as ``otospeech/1/*.wav``.
    """
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {source_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    sources = sorted(path for path in source_dir.glob("*.wav") if path.is_file())
    if not sources:
        sources = sorted(path for path in source_dir.rglob("*.wav") if path.is_file())
    if not sources:
        raise ValueError(f"No WAV files found in {source_dir}")
    names = [path.name for path in sources]
    if len(names) != len(set(names)):
        raise ValueError("Nested source WAVs must have unique filenames.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
                "-map", "0:a:0", "-ac", "2", "-ar", str(INPUT_SAMPLE_RATE),
                "-c:a", "pcm_s16le", str(output_dir / source.name),
            ],
            check=True,
        )
    return len(sources)


def prepare_directory(raw_dir: Path, output_dir: Path, *, roles_path: Path, role_prompt_version: str, language: str = "auto", voice_prompt_seconds: float = 3.0, split: str = "train", max_samples: int | None = None, transcribe: Transcriber | None = None) -> PreparationResult:
    """Prepare a directory of stereo WAV files, preserving their duplex timeline."""
    if language not in {"auto", "en", "vi"}:
        raise ValueError("language must be one of: auto, en, vi.")
    if not role_prompt_version.strip():
        raise ValueError("role_prompt_version must be non-empty.")
    if voice_prompt_seconds <= 0:
        raise ValueError("voice_prompt_seconds must be positive.")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive.")
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Raw directory does not exist: {raw_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    roles = _roles_by_id(roles_path)
    sources = sorted(path for path in raw_dir.glob("*.wav") if path.is_file() and not path.is_symlink())
    if not sources:
        raise ValueError(f"No WAV files found in {raw_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    transcriber = transcribe
    prepared = rejected = 0
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as manifest, (output_dir / "rejected.jsonl").open("w", encoding="utf-8") as rejections, (output_dir / "review.jsonl").open("w", encoding="utf-8") as review:
        for source in sources:
            if max_samples is not None and prepared >= max_samples:
                break
            sample_id = source.stem
            try:
                metadata = roles.get(sample_id)
                if metadata is None:
                    raise ValueError(f"No role metadata for id '{sample_id}'.")
                speaker_ids, role_prompts = _validated_roles(metadata)
                pcm, source_rate, _, _ = _read_stereo_wav(source)
                stereo_pcm = pcm
                frame_count = len(stereo_pcm) // 4
                duration = frame_count / INPUT_SAMPLE_RATE
                prompt_frames = min(round(voice_prompt_seconds * INPUT_SAMPLE_RATE), frame_count)
                if prompt_frames < 1:
                    raise ValueError("Voice prompt would be empty.")
                if transcriber is None:
                    transcriber = _default_transcriber("large-v3")
                channels = [_channel_pcm(stereo_pcm, index) for index in range(2)]
                transcripts = [_validate_transcript(transcriber(channel, INPUT_SAMPLE_RATE, language), duration) for channel in channels]
                stereo_path = Path("audio") / f"{sample_id}.wav"
                transcript_paths = [Path("transcripts") / f"{sample_id}-speaker-{index + 1}.json" for index in range(2)]
                prompt_paths = [Path("prompts") / f"{sample_id}-speaker-{index + 1}.wav" for index in range(2)]
                _write_wav(output_dir / stereo_path, stereo_pcm, INPUT_SAMPLE_RATE, 2)
                for index in range(2):
                    (output_dir / transcript_paths[index]).parent.mkdir(parents=True, exist_ok=True)
                    (output_dir / transcript_paths[index]).write_text(json.dumps({"speaker_id": speaker_ids[index], "language": language, "segments": transcripts[index]}, ensure_ascii=False) + "\n", encoding="utf-8")
                    _write_wav(output_dir / prompt_paths[index], channels[index][: prompt_frames * 2], INPUT_SAMPLE_RATE, 1)
                row = {"id": sample_id, "stereo_wav": str(stereo_path), "speaker_ids": speaker_ids, "split_group": sample_id, "split": split, "language": language, "approved_role_prompts": role_prompts, "role_prompt_provenance": role_prompt_version, "transcripts": [{"speaker_id": speaker_ids[index], "json": str(transcript_paths[index])} for index in range(2)], "voice_prompts": [{"speaker_id": speaker_ids[index], "wav": str(prompt_paths[index])} for index in range(2)]}
                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                review.write(json.dumps({"id": sample_id, "status": "accepted", "duration_sec": duration, "transcripts": row["transcripts"]}, ensure_ascii=False) + "\n")
                prepared += 1
            except Exception as error:
                rejected += 1
                rejected_row = {"id": sample_id, "source": str(source), "error": str(error)}
                rejections.write(json.dumps(rejected_row, ensure_ascii=False) + "\n")
                review.write(json.dumps({"id": sample_id, "status": "rejected", "error": str(error)}, ensure_ascii=False) + "\n")
    return PreparationResult(prepared, rejected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare", help="prepare stereo WAV data")
    prepare.add_argument("raw_dir", type=Path, help="directory containing source stereo WAV files")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--roles", type=Path, required=True, help="JSONL role metadata keyed by WAV stem")
    prepare.add_argument("--role-prompt-version", required=True)
    prepare.add_argument("--asr-model", default="large-v3")
    prepare.add_argument("--language", choices=("auto", "en", "vi"), default="auto")
    prepare.add_argument("--voice-prompt-seconds", type=float, default=3.0)
    prepare.add_argument("--split", default="train")
    prepare.add_argument("--max-samples", type=int)
    normalize = subcommands.add_parser("normalize-audio", help="convert source WAVs to stereo PCM16 at 16 kHz")
    normalize.add_argument("source_dir", type=Path)
    normalize.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "normalize-audio":
        count = normalize_directory(args.source_dir, args.output_dir)
        print(f"Normalized {count} WAV files: {args.output_dir}")
        return
    if args.command == "prepare":
        transcriber: Transcriber | None = None
        # Delay model allocation until the first valid, reviewed sample.
        def lazy_transcriber(pcm: bytes, rate: int, language: str) -> Transcript:
            nonlocal transcriber
            if transcriber is None:
                transcriber = _default_transcriber(args.asr_model)
            return transcriber(pcm, rate, language)
        result = prepare_directory(args.raw_dir, args.output_dir, roles_path=args.roles, role_prompt_version=args.role_prompt_version, language=args.language, voice_prompt_seconds=args.voice_prompt_seconds, split=args.split, max_samples=args.max_samples, transcribe=lazy_transcriber)
        print(f"Prepared {result.prepared} samples; rejected {result.rejected}. Manifest: {args.output_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
