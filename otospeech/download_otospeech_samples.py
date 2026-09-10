#!/usr/bin/env python3
"""Download a bounded, resumable subset of the OtoSpeech full-duplex dataset."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DATASET_ID = "otoearth/otoSpeech-full-duplex-turn-104h"
MANIFEST_NAME = "otospeech_samples.jsonl"
SOURCE_FILES = ("combined_audio.wav", "metadata.json")


def destination_filenames(folder_number: int) -> tuple[str, str]:
    return (f"stereo_{folder_number}.wav", f"metadata_{folder_number}.json")


def discover_existing_samples(root: Path) -> dict[int, str]:
    """Return completed numeric sample folders as {folder_number: task_id}."""
    samples: dict[int, str] = {}
    for child in root.iterdir():
        if not child.is_dir() or not child.name.isdecimal() or int(child.name) < 1:
            continue
        folder_number = int(child.name)
        stereo_name, metadata_name = destination_filenames(folder_number)
        if not (child / stereo_name).is_file() or not (child / metadata_name).is_file():
            continue
        try:
            metadata = json.loads((child / metadata_name).read_text(encoding="utf-8"))
            task_id = str(metadata["task_id"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        samples[folder_number] = task_id
    return samples


def select_source_ids(
    source_ids: list[str], existing_task_ids: set[str], needed: int
) -> list[str]:
    """Choose the next unseen source folders in the Hub manifest order."""
    selected = [source_id for source_id in source_ids if source_id not in existing_task_ids]
    if len(selected) < needed:
        raise ValueError(
            f"requested {needed} additional samples, but only {len(selected)} available"
        )
    return selected[:needed]


def source_ids_from_siblings(siblings: list[Any]) -> list[str]:
    """Return complete source folders in the Hub manifest order."""
    files_by_source: dict[str, set[str]] = {}
    source_order: list[str] = []
    for sibling in siblings:
        path = sibling.get("rfilename") if isinstance(sibling, dict) else None
        if not isinstance(path, str):
            continue
        parts = path.split("/")
        if len(parts) != 2 or parts[1] not in SOURCE_FILES:
            continue
        source_id, filename = parts
        if source_id not in files_by_source:
            files_by_source[source_id] = set()
            source_order.append(source_id)
        files_by_source[source_id].add(filename)

    source_ids = [
        source_id
        for source_id in source_order
        if files_by_source[source_id] == set(SOURCE_FILES)
    ]
    if not source_ids:
        raise RuntimeError("no complete OtoSpeech source folders were found")
    return source_ids


def fetch_dataset_sources() -> tuple[str, list[str]]:
    """Fetch the revision and source folders having exactly the two required files."""
    url = f"https://huggingface.co/api/datasets/{DATASET_ID}"
    request = urllib.request.Request(url, headers={"User-Agent": "otospeech-sample-downloader"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload: dict[str, Any] = json.load(response)
    except urllib.error.URLError as error:
        raise RuntimeError(f"cannot reach the Hugging Face dataset manifest: {error.reason}") from error

    revision = payload.get("sha")
    siblings = payload.get("siblings")
    if not isinstance(revision, str) or not isinstance(siblings, list):
        raise RuntimeError("Hugging Face returned an unexpected dataset manifest")
    return revision, source_ids_from_siblings(siblings)


def write_manifest(root: Path, samples: dict[int, str], revision: str) -> None:
    manifest_path = root / MANIFEST_NAME
    rows = [
        {
            "folder": folder_number,
            "task_id": task_id,
            "files": list(destination_filenames(folder_number)),
            "dataset": DATASET_ID,
            "revision": revision,
        }
        for folder_number, task_id in sorted(samples.items())
    ]
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=root, prefix=f".{MANIFEST_NAME}.", delete=False
    ) as temporary_file:
        for row in rows:
            temporary_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, manifest_path)


def existing_manifest_revision(root: Path) -> str | None:
    try:
        first_row = (root / MANIFEST_NAME).read_text(encoding="utf-8").splitlines()[0]
        revision = json.loads(first_row).get("revision")
    except (IndexError, OSError, ValueError, AttributeError):
        return None
    return revision if isinstance(revision, str) else None


def download_pair(root: Path, target_folder: int, source_id: str, revision: str) -> None:
    """Download both files to staging, then publish them together into one folder."""
    with tempfile.TemporaryDirectory(dir=root, prefix=".otospeech-download-") as staging_dir:
        staging = Path(staging_dir)
        subprocess.run(
            [
                "hf",
                "download",
                DATASET_ID,
                *(f"{source_id}/{filename}" for filename in SOURCE_FILES),
                "--repo-type",
                "dataset",
                "--revision",
                revision,
                "--local-dir",
                str(staging),
                "--max-workers",
                "2",
            ],
            check=True,
        )
        source_dir = staging / source_id
        if not all((source_dir / filename).is_file() for filename in SOURCE_FILES):
            raise RuntimeError(f"downloaded source {source_id} is incomplete")

        destination = root / str(target_folder)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing folder: {destination}")
        destination.mkdir()
        for source_name, destination_name in zip(
            SOURCE_FILES, destination_filenames(target_folder), strict=True
        ):
            os.replace(source_dir / source_name, destination / destination_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Total completed sample folders to keep (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Folder holding numbered samples (default: current directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the sources that would be downloaded without changing files.",
    )
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    root = args.output_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {root}")

    existing = discover_existing_samples(root)
    if len(existing) > args.num_samples:
        print(
            f"Already have {len(existing)} completed samples; target is {args.num_samples}. Nothing to do."
        )
        return 0
    if len(existing) == args.num_samples:
        revision = existing_manifest_revision(root)
        if revision:
            write_manifest(root, existing, revision)
        print(f"Already have {len(existing)} completed samples; target is {args.num_samples}. Nothing to do.")
        return 0

    revision, source_ids = fetch_dataset_sources()
    needed = args.num_samples - len(existing)
    selected = select_source_ids(source_ids, set(existing.values()), needed)
    if args.dry_run:
        print(f"Have {len(existing)} completed samples; would add {needed} to reach {args.num_samples}.")
        for offset, source_id in enumerate(selected, start=max(existing, default=0) + 1):
            print(f"  folder {offset}: task_id {source_id}")
        return 0

    for target_folder, source_id in enumerate(selected, start=max(existing, default=0) + 1):
        print(f"Downloading task_id {source_id} into folder {target_folder}...")
        download_pair(root, target_folder, source_id, revision)
        existing[target_folder] = source_id
        write_manifest(root, existing, revision)

    print(f"Ready: {len(existing)} completed samples in {root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
