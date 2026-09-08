"""Single-process, resumable full fine-tuning for prepared PersonaPlex data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import torch

from .dataset import load_example, read_index
from .objective import weighted_personaplex_loss
from .streams import build_loss_weights


def _save(model, optimizer, step: int, output_dir: Path) -> None:
    from safetensors.torch import save_model

    save_model(model, str(output_dir / "model.safetensors"))
    torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()}, output_dir / "training-state.pt")


def _restore(model, optimizer, state_path: Path, device: str) -> int:
    state = torch.load(state_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"].cpu())
    return int(state["step"])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune PersonaPlex from prepared token shards.")
    parser.add_argument("--prepared-index", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--moshi-weight", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--max-frames", type=int, default=2048)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--seed", type=int, default=42424242)
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.max_frames <= 0 or args.save_every <= 0:
        parser.error("steps, max-frames, and save-every must be positive")
    if args.output_dir.exists() and not args.resume_state:
        parser.error("output-dir already exists; pass --resume-state to continue a trusted run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    from moshi.models.lm import LMModel
    if not hasattr(LMModel, "forward_train"):
        raise RuntimeError("installed moshi package lacks PersonaPlex training support; reinstall this checkout with `pip install --force-reinstall moshi/.`")
    from moshi.models.loaders import get_moshi_lm

    model = get_moshi_lm(args.moshi_weight, device=args.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    step = _restore(model, optimizer, args.resume_state, args.device) if args.resume_state else 0
    shards = read_index(args.prepared_index, args.split)
    metrics_path = args.output_dir / "metrics.jsonl"
    with metrics_path.open("a") as metrics:
        while step < args.steps:
            shard = shards[step % len(shards)]
            example = load_example(shard, args.max_frames)
            codes = example["codes"].unsqueeze(0).to(args.device)
            weights = build_loss_weights(codes[0], int(example["prompt_frames"]), model.text_padding_token_id).unsqueeze(0).to(args.device)
            output = model.forward_train(codes)
            loss = weighted_personaplex_loss(output.text_logits, output.logits, codes, weights, output.text_mask, output.mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            metrics.write(json.dumps({"step": step, "loss": loss.item(), "example_id": example["id"]}) + "\n")
            metrics.flush()
            if step % args.save_every == 0 or step == args.steps:
                _save(model, optimizer, step, args.output_dir)


if __name__ == "__main__":
    main()
