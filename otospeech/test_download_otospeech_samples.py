import json
import tempfile
import unittest
from pathlib import Path

from download_otospeech_samples import (
    destination_filenames,
    discover_existing_samples,
    select_source_ids,
    source_ids_from_siblings,
)


def write_sample(root: Path, folder: str, task_id: str) -> None:
    sample_dir = root / folder
    sample_dir.mkdir()
    stereo_name, metadata_name = destination_filenames(int(folder))
    (sample_dir / stereo_name).write_bytes(b"wav")
    (sample_dir / metadata_name).write_text(
        json.dumps({"task_id": task_id}), encoding="utf-8"
    )


class ExistingSamplesTest(unittest.TestCase):
    def test_uses_folder_number_in_destination_filenames(self) -> None:
        self.assertEqual(
            destination_filenames(12),
            ("stereo_12.wav", "metadata_12.json"),
        )

    def test_discovers_valid_numeric_folders_and_ignores_unrelated_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            write_sample(root, "1", "1")
            write_sample(root, "2", "10")
            (root / "notes").mkdir()
            (root / "3").mkdir()

            samples = discover_existing_samples(root)

            self.assertEqual(
                samples,
                {1: "1", 2: "10"},
            )


class SourceSelectionTest(unittest.TestCase):
    def test_keeps_manifest_order_and_requires_both_requested_files(self) -> None:
        source_ids = source_ids_from_siblings(
            [
                {"rfilename": "1/combined_audio.wav"},
                {"rfilename": "1/metadata.json"},
                {"rfilename": "10/metadata.json"},
                {"rfilename": "100/metadata.json"},
                {"rfilename": "100/combined_audio.wav"},
                {"rfilename": "100/speaker_1_audio.wav"},
            ]
        )

        self.assertEqual(source_ids, ["1", "100"])

    def test_selects_manifest_order_and_skips_existing_task_ids(self) -> None:
        selected = select_source_ids(
            source_ids=["1", "10", "100", "101", "102"],
            existing_task_ids={"1", "10", "100"},
            needed=2,
        )

        self.assertEqual(selected, ["101", "102"])

    def test_rejects_a_target_larger_than_available_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 1 available"):
            select_source_ids(
                source_ids=["1", "10"],
                existing_task_ids={"1"},
                needed=2,
            )


if __name__ == "__main__":
    unittest.main()
