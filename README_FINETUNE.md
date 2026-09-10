# PersonaPlex fine-tuning

> Status: data preparation is implemented. Model fine-tuning is not yet
> implemented in this repository.

This document is the evolving implementation guide for adapting PersonaPlex.
It intentionally separates commands that exist today from planned work, so a
data-preparation run is never mistaken for a training run.

## Current capability: prepare reviewed duplex data

The available command accepts one stereo WAV conversation per sample and emits
reviewable artifacts for a future trainer:

If a source dataset stores float or 48 kHz WAVs (for example, a flat Kaggle
layout `otospeech/stereo_1.wav`), normalize it first. The command prefers WAVs
directly inside the source directory and falls back to nested WAVs when no
direct files exist:

```bash
python -m tool normalize-audio otospeech --output-dir data/raw
```

This requires `ffmpeg` and writes stereo PCM16 at 16 kHz. The output directory
must be new or empty.

```bash
pip install whisper-timestamped

python -m tool prepare data/raw \
  --output-dir data/prepared-manifest \
  --roles data/roles.jsonl \
  --role-prompt-version roles-v1 \
  --asr-model small \
  --language auto \
  --voice-prompt-seconds 3 \
  --split train \
  --max-samples 5
```

`--max-samples` is recommended for initial validation. It limits accepted
samples; rejected input remains visible in the review artifacts.

### Input contract

Place source files directly in `data/raw/`. Each `<id>.wav` must be:

- 16 kHz, uncompressed, 16-bit PCM WAV with exactly two channels.
- One complete shared conversation timeline; preserve silence, overlap,
  interruptions, and backchannels.
- Channel 0 = speaker A and channel 1 = speaker B consistently for the whole
  recording.

Provide matching, human-reviewed metadata in `data/roles.jsonl`:

```json
{"id":"call-001","speaker_ids":["agent-a","user-b"],"role_prompts":[{"text":"You are a helpful support agent.","status":"approved"},{"text":"You are a customer seeking support.","status":"approved"}]}
```

The row `id` is the WAV filename without `.wav`. `speaker_ids` must be two
different non-empty strings. Prompt and speaker index follow the audio channel:
index 0 for channel 0 and index 1 for channel 1. Both role prompts must have
`"status": "approved"`; role generation/review happens outside this tool.

### Output contract

The output directory must be new or empty. For each accepted sample, the tool
writes:

```text
data/prepared-manifest/
├── audio/<id>.wav                         # stereo PCM16 at 16 kHz
├── transcripts/<id>-speaker-{1,2}.json    # timestamped channel ASR
├── prompts/<id>-speaker-{1,2}.wav         # independent mono voice prompts
├── manifest.jsonl                         # accepted samples only
├── review.jsonl                           # accepted and rejected status
└── rejected.jsonl                         # rejection reasons
```

Inspect `review.jsonl` and the channel transcripts before treating a manifest
as training data. The manifest retains both speaker role/voice pairs; a future
trainer can then construct either conversation direction without guessing a
global agent channel.

## Planned, not runnable yet

The following work is deliberately not provided by the current repository:

- PersonaPlex token preparation with Mimi and hybrid system-prompt masking.
- Bidirectional agent/user example construction from a prepared manifest.
- LoRA parameter selection, training loop, checkpointing, evaluation, or
  multi-GPU/FSDP launch.

When each item is implemented, add its exact command, required model assets,
expected artifacts, and a verification procedure here. Until then, do not use
the preparation manifest as evidence that fine-tuning has started or completed.
