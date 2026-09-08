# PersonaPlex

[![Weights](https://img.shields.io/badge/🤗-Weights-yellow)](https://huggingface.co/nvidia/personaplex-7b-v1)
[![Paper](https://img.shields.io/badge/📄-Paper-blue)](https://arxiv.org/abs/2602.06053)
[![Demo](https://img.shields.io/badge/🎮-Demo-green)](https://research.nvidia.com/labs/adlr/personaplex/)

PersonaPlex is a real-time, full-duplex speech-to-speech model with text-based role control and audio-based voice conditioning. It is based on [Moshi](https://arxiv.org/abs/2410.00037) and provides:

- a WebSocket server and browser client for live conversations;
- offline WAV-to-WAV inference; and
- a reproducible, single-process fine-tuning workflow for reviewed two-speaker stereo conversations.

<p align="center">
  <img src="assets/architecture_diagram.png" alt="PersonaPlex architecture">
</p>

## How it works

At every 12.5 Hz frame, PersonaPlex consumes three streams: user audio, agent text, and agent audio. Mimi encodes each audio stream into eight codebooks. The model generates the agent text and audio streams while receiving user audio, so it can listen while speaking and react to interruption or overlap.

PersonaPlex conditions a conversation with a hybrid system prompt:

1. An agent voice sample establishes the output voice.
2. A text prompt establishes the agent's role.

The released checkpoint is `nvidia/personaplex-7b-v1`. Accept its model license before downloading it.

## Quick start: inference

### Requirements

- Python 3.10 or later
- [Opus development headers](https://github.com/xiph/opus)
- A Hugging Face token with access to the model

On Ubuntu or Debian:

```bash
sudo apt install libopus-dev
python -m pip install --force-reinstall ./moshi
export HF_TOKEN=<your_hugging_face_token>
```

For Blackwell GPUs, install a CUDA 13 PyTorch build as described in [NVIDIA/personaplex#2](https://github.com/NVIDIA/personaplex/issues/2).

The forced local install is intentional: the training workflow needs this checkout's `LMModel.forward_train`, which is not present in every separately installed `moshi` package.

### Run the live server

```bash
SSL_DIR=$(mktemp -d)
python -m moshi.server --ssl "$SSL_DIR"
```

Open `https://localhost:8998`. If GPU memory is insufficient, install `accelerate` and add `--cpu-offload`:

```bash
python -m pip install accelerate
SSL_DIR=$(mktemp -d)
python -m moshi.server --ssl "$SSL_DIR" --cpu-offload
```

### Run offline inference

This writes generated agent audio with the same duration as the input user WAV, plus decoded agent text tokens.

```bash
python -m moshi.offline \
  --voice-prompt NATF2.pt \
  --input-wav assets/test/input_assistant.wav \
  --seed 42424242 \
  --output-wav output.wav \
  --output-text output.json
```

Use `--text-prompt` to set a role. If `--voice-prompt-dir` is omitted, the bundled voices are retrieved from the model repository.

## Fine-tune from reviewed stereo data

The `training` package is separate from inference. It starts from a compatible PersonaPlex checkpoint, keeps Mimi and the tokenizer fixed, and produces a `model.safetensors` file that the existing server and offline commands can load through `--moshi-weight`.

### Dataset contract

Input is JSONL. Relative asset paths are resolved from the manifest directory. Every record must include:

- exactly one two-channel conversation WAV, with one synchronized speaker per channel;
- distinct `speaker_ids`, a `train`, `validation`, or `test` split, and a stable `split_group`;
- timestamped transcript segments for both channels;
- an approved, versioned role prompt and a non-overlapping voice-prompt clip for each speaker.

A speaker cannot appear in more than one split. The validator rejects malformed timestamps, missing prompt clips, cross-split speaker leakage, and prompts that overlap their own dialogue transcript.

```json
{
  "id": "call-001",
  "stereo_wav": "audio/call-001.wav",
  "speaker_ids": ["speaker-a", "speaker-b"],
  "split_group": "call-001",
  "split": "train",
  "language": "en",
  "approved_role_prompts": [
    "You are a support agent.",
    "You are a customer."
  ],
  "role_prompt_provenance": [
    {"status": "approved", "version": "roles-v1"},
    {"status": "approved", "version": "roles-v1"}
  ],
  "transcripts": [
    [{"start": 0.0, "end": 0.4, "text": "Hello."}],
    [{"start": 0.5, "end": 0.9, "text": "Hi."}]
  ],
  "voice_prompts": [
    {"path": "prompts/speaker-a.wav", "start": 0.0, "end": 3.0},
    {"path": "prompts/speaker-b.wav", "start": 0.0, "end": 3.0}
  ]
}
```

### Prepare, train, and evaluate

Run all commands from the repository root after installing `./moshi`.

```bash
python -m training validate --manifest data/manifest.jsonl

python -m training prepare \
  --manifest data/manifest.jsonl \
  --output-dir prepared/run-001 \
  --mimi-weight /models/tokenizer-e351c8d8-checkpoint125.safetensors \
  --tokenizer /models/tokenizer_spm_32k_3.model

python -m training train \
  --prepared-index prepared/run-001/index.jsonl \
  --moshi-weight /models/model.safetensors \
  --output-dir training-runs/run-001 \
  --steps 1000

python -m training evaluate \
  --prepared-index prepared/run-001/index.jsonl \
  --moshi-weight training-runs/run-001/model.safetensors \
  --output training-runs/run-001/validation.json
```

`prepare` creates both directions of each conversation: A→B and B→A. In either direction, only the selected agent text and agent audio are supervised. User audio is teacher-forced conditioning; the hybrid system prompt is also excluded from loss. The objective weights semantic agent audio at `1.0`, non-semantic agent codebooks at `0.02`, and padded agent text at `0.3`.

Prepared shards, checkpoints, training state, and metrics are written under the supplied output directories. `prepared/` and `training-runs/` are ignored by Git. `training-state.pt` is only for resuming a trusted local run; distribute `model.safetensors` for inference.

### Current training scope

The included trainer is intentionally a bounded, single-process recipe. It supports resumable training, maximum sequence lengths, held-out teacher-forced loss, and compatible checkpoint export. It does not configure DDP, FSDP, dataset download, automatic role-prompt generation, or human role-adherence evaluation; those remain external infrastructure and review responsibilities.

## Repository layout

| Path | Purpose |
| --- | --- |
| `moshi/` | PersonaPlex model, Mimi codec, server, and offline inference |
| `client/` | Browser client |
| `training/` | Manifest validation, preparation, training, and evaluation CLI |
| `tests/` | Training-contract and local-model integration tests |

## Verification

```bash
python -m unittest discover -s tests -v
python -m py_compile training/*.py
```

The model license governs released weights. Repository code is MIT licensed.

## Citation

```bibtex
@misc{roy2026personaplexvoicerolecontrol,
  title={PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models},
  author={Rajarshi Roy and Jonathan Raiman and Sang-gil Lee and Teodor-Dumitru Ene and Robert Kirby and Sungwon Kim and Jaehyeon Kim and Bryan Catanzaro},
  year={2026},
  eprint={2602.06053},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2602.06053}
}
```
