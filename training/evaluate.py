"""Teacher-forced held-out loss for an inference-compatible checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from .dataset import load_example, read_index
from .objective import weighted_personaplex_loss
from .streams import build_loss_weights


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PersonaPlex checkpoint on prepared held-out shards.")
    parser.add_argument("--prepared-index", required=True, type=Path)
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--moshi-weight", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=2048)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    from moshi.models.lm import LMModel
    if not hasattr(LMModel, "forward_train"):
        raise RuntimeError("installed moshi package lacks PersonaPlex training support; reinstall this checkout with `pip install --force-reinstall moshi/.`")
    from moshi.models.loaders import get_moshi_lm

    model = get_moshi_lm(args.moshi_weight, device=args.device)
    model.eval()
    losses = []
    with torch.inference_mode():
        for shard in read_index(args.prepared_index, args.split):
            example = load_example(shard, args.max_frames)
            codes = example["codes"].unsqueeze(0).to(args.device)
            weights = build_loss_weights(codes[0], int(example["prompt_frames"]), model.text_padding_token_id).unsqueeze(0).to(args.device)
            output = model.forward_train(codes)
            losses.append(weighted_personaplex_loss(output.text_logits, output.logits, codes, weights, output.text_mask, output.mask).item())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"examples": len(losses), "mean_loss": sum(losses) / len(losses)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
