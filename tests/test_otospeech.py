import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import sphn
import torch

from training.otospeech import parse_srt, prepare_from_directory
from training.validate import validate_manifest


class OtoSpeechPreparationTests(unittest.TestCase):
    def test_parses_srt_and_builds_a_bounded_stereo_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "snapshot" / "call-001"
            root.mkdir(parents=True)
            for name in ("speaker_1_audio.wav", "speaker_2_audio.wav"):
                torch.save(torch.zeros(1), root / name)  # Replaced by the audio reader in this small unit test.
            (root / "speaker_1_annotation_a.srt").write_text("1\n00:00:00,000 --> 00:00:00,500\nHello\n")
            (root / "speaker_2_annotation_a.srt").write_text("1\n00:00:00,500 --> 00:00:01,000\nHi\n")
            (root / "metadata.json").write_text(json.dumps({"language": "en"}))

            records = prepare_from_directory(
                root.parent, Path(temporary) / "output", max_samples=1,
                role_prompts=("You are a caller.", "You are a responder."), role_prompt_version="review-v1",
                read_audio=lambda path: (np.zeros(8000), 8000),
                write_audio=lambda path, audio, rate: path.write_bytes(b"wav"),
            )

            self.assertEqual(len(records), 1)
            manifest = json.loads((Path(temporary) / "output" / "manifest.jsonl").read_text())
            self.assertEqual(manifest["stereo_wav"], "audio/call-001.wav")
            self.assertEqual(manifest["speaker_ids"], ["call-001:speaker-1", "call-001:speaker-2"])
            self.assertEqual(manifest["transcripts"][0][0]["text"], "Hello")
            self.assertEqual(manifest["voice_prompts"][0]["path"], "prompts/call-001-speaker-1.wav")

    def test_rejects_malformed_srt_timestamps(self):
        with self.assertRaisesRegex(ValueError, "invalid SRT timestamp"):
            parse_srt("1\nnot-a-timestamp --> 00:00:00,500\nHello\n")

    def test_writes_a_manifest_accepted_by_the_training_validator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "snapshot" / "call-001"
            root.mkdir(parents=True)
            for name in ("speaker_1_audio.wav", "speaker_2_audio.wav"):
                sphn.write_wav(root / name, np.zeros((1, 8000), dtype=np.float32), 8000)
            (root / "speaker_1_annotation_a.srt").write_text("1\n00:00:00,000 --> 00:00:00,500\nHello\n")
            (root / "speaker_2_annotation_a.srt").write_text("1\n00:00:00,500 --> 00:00:01,000\nHi\n")
            (root / "metadata.json").write_text("{}")

            prepare_from_directory(
                root.parent, Path(temporary) / "output", max_samples=1,
                role_prompts=("You are a caller.", "You are a responder."), role_prompt_version="review-v1",
            )

            self.assertEqual(validate_manifest(Path(temporary) / "output" / "manifest.jsonl"), 1)


if __name__ == "__main__":
    unittest.main()
