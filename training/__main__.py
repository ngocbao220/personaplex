"""Convenience dispatcher for training subcommands."""

from __future__ import annotations

import argparse

from . import evaluate, otospeech, prepare, train, validate


def main() -> None:
    parser = argparse.ArgumentParser(description="PersonaPlex fine-tuning tools")
    parser.add_argument("command", choices=("validate", "otospeech", "prepare", "train", "evaluate"))
    args, remainder = parser.parse_known_args()
    {"validate": validate, "otospeech": otospeech, "prepare": prepare, "train": train, "evaluate": evaluate}[args.command].main(remainder)


if __name__ == "__main__":
    main()
