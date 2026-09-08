"""Prepared-shard loading and bounded sequence selection."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def read_index(index_path: Path, split: str | None = None) -> list[Path]:
    paths = []
    with index_path.open() as source:
        for line in source:
            if line.strip():
                entry = json.loads(line)
                if split is None or entry.get("split") == split:
                    paths.append(index_path.parent / entry["path"])
    if not paths:
        raise ValueError(f"prepared index has no examples: {index_path}")
    return paths


def load_example(path: Path, max_frames: int | None = None) -> dict[str, object]:
    example = torch.load(path, map_location="cpu", weights_only=True)
    codes = example.get("codes")
    if not isinstance(codes, torch.Tensor) or codes.ndim != 2 or codes.shape[0] != 17:
        raise ValueError(f"invalid prepared codes: {path}")
    prompt_frames = int(example["prompt_frames"])
    if max_frames is not None and codes.shape[1] > max_frames:
        if prompt_frames >= max_frames:
            raise ValueError(f"prompt is longer than max_frames: {path}")
        dialogue_frames = max_frames - prompt_frames
        first = prompt_frames
        example["codes"] = torch.cat((codes[:, :prompt_frames], codes[:, first:first + dialogue_frames]), dim=1)
    return example
