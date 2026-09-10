import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from tool.prepare import normalize_directory, prepare_directory


def write_wav(path: Path, *, channels: int, sample_rate: int = 16_000, frames: int = 16_000):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\0\0" * channels * frames)


class PrepareDirectoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.roles = self.root / "roles.jsonl"
        self.output = self.root / "prepared"

    def tearDown(self):
        self.temp.cleanup()

    def write_roles(self, record):
        self.roles.write_text(json.dumps(record) + "\n", encoding="utf-8")

    def test_prepares_stereo_wav_and_review_artifacts(self):
        write_wav(self.raw / "call-001.wav", channels=2)
        self.write_roles(
            {
                "id": "call-001",
                "speaker_ids": ["agent-a", "user-b"],
                "role_prompts": [
                    {"text": "Helpful support agent.", "status": "approved"},
                    {"text": "Customer seeking support.", "status": "approved"},
                ],
            }
        )

        def asr(channel_wav, sample_rate, language):
            self.assertEqual(sample_rate, 16_000)
            self.assertEqual(len(channel_wav), 32_000)
            return [{"text": "hello", "start": 0.0, "end": 0.5}]

        result = prepare_directory(
            self.raw,
            self.output,
            roles_path=self.roles,
            role_prompt_version="roles-v1",
            language="auto",
            voice_prompt_seconds=0.5,
            split="train",
            transcribe=asr,
        )

        self.assertEqual((result.prepared, result.rejected), (1, 0))
        manifest = json.loads((self.output / "manifest.jsonl").read_text())
        self.assertEqual(manifest["speaker_ids"], ["agent-a", "user-b"])
        self.assertEqual(manifest["language"], "auto")
        self.assertEqual(manifest["role_prompt_provenance"], "roles-v1")
        self.assertEqual(len(manifest["transcripts"]), 2)
        self.assertEqual(len(manifest["voice_prompts"]), 2)
        with wave.open(str(self.output / manifest["stereo_wav"]), "rb") as audio:
            self.assertEqual((audio.getnchannels(), audio.getframerate(), audio.getnframes()), (2, 16_000, 16_000))
        with wave.open(str(self.output / manifest["voice_prompts"][0]["wav"]), "rb") as prompt:
            self.assertEqual((prompt.getnchannels(), prompt.getframerate(), prompt.getnframes()), (1, 16_000, 8_000))
        transcript = json.loads((self.output / manifest["transcripts"][1]["json"]).read_text())
        self.assertEqual(transcript["speaker_id"], "user-b")
        self.assertEqual(transcript["segments"][0]["text"], "hello")
        self.assertEqual(json.loads((self.output / "review.jsonl").read_text())["status"], "accepted")

    def test_rejects_mono_audio_and_missing_approved_roles(self):
        write_wav(self.raw / "mono.wav", channels=1)
        write_wav(self.raw / "unreviewed.wav", channels=2)
        self.roles.write_text(
            "\n".join(
                [
                    json.dumps({"id": "mono", "speaker_ids": ["mono-a", "mono-b"], "role_prompts": [{"text": "A", "status": "approved"}, {"text": "B", "status": "approved"}]}),
                    json.dumps({"id": "unreviewed", "speaker_ids": ["a", "b"], "role_prompts": [{"text": "A", "status": "draft"}, {"text": "B", "status": "approved"}]}),
                ]
            ) + "\n",
            encoding="utf-8",
        )

        result = prepare_directory(
            self.raw, self.output, roles_path=self.roles, role_prompt_version="roles-v1", transcribe=lambda *_: []
        )

        self.assertEqual((result.prepared, result.rejected), (0, 2))
        rejected = [json.loads(line) for line in (self.output / "rejected.jsonl").read_text().splitlines()]
        self.assertTrue(any("stereo" in row["error"] for row in rejected))
        self.assertTrue(any("approved" in row["error"] for row in rejected))

    def test_max_samples_limits_accepted_records(self):
        write_wav(self.raw / "a.wav", channels=2)
        write_wav(self.raw / "b.wav", channels=2)
        self.roles.write_text(
            "\n".join(
                json.dumps({"id": item, "speaker_ids": [item + "-1", item + "-2"], "role_prompts": [{"text": "one", "status": "approved"}, {"text": "two", "status": "approved"}]})
                for item in ("a", "b")
            ) + "\n",
            encoding="utf-8",
        )

        result = prepare_directory(self.raw, self.output, roles_path=self.roles, role_prompt_version="v1", max_samples=1, transcribe=lambda *_: [])

        self.assertEqual((result.prepared, result.rejected), (1, 0))
        self.assertEqual(len((self.output / "manifest.jsonl").read_text().splitlines()), 1)

    def test_forwards_vietnamese_language_to_asr_and_manifest(self):
        write_wav(self.raw / "viet.wav", channels=2)
        self.write_roles(
            {"id": "viet", "speaker_ids": ["speaker-a", "speaker-b"], "role_prompts": [{"text": "Một", "status": "approved"}, {"text": "Hai", "status": "approved"}]}
        )
        received_languages = []

        def asr(_, __, language):
            received_languages.append(language)
            return []

        prepare_directory(self.raw, self.output, roles_path=self.roles, role_prompt_version="v1", language="vi", transcribe=asr)

        self.assertEqual(received_languages, ["vi", "vi"])
        self.assertEqual(json.loads((self.output / "manifest.jsonl").read_text())["language"], "vi")

    def test_rejects_input_that_is_not_16_khz(self):
        write_wav(self.raw / "wrong-rate.wav", channels=2, sample_rate=8_000)
        self.write_roles(
            {"id": "wrong-rate", "speaker_ids": ["a", "b"], "role_prompts": [{"text": "A", "status": "approved"}, {"text": "B", "status": "approved"}]}
        )

        result = prepare_directory(self.raw, self.output, roles_path=self.roles, role_prompt_version="v1", transcribe=lambda *_: [])

        self.assertEqual((result.prepared, result.rejected), (0, 1))
        rejection = json.loads((self.output / "rejected.jsonl").read_text())
        self.assertIn("16 kHz", rejection["error"])

    @patch("tool.prepare.subprocess.run")
    def test_normalize_accepts_flat_otospeech_layout(self, run):
        source = self.root / "otospeech"
        source.mkdir()
        (source / "stereo_1.wav").write_bytes(b"source")

        result = normalize_directory(source, self.root / "raw")

        self.assertEqual(result, 1)
        command = run.call_args.args[0]
        self.assertIn(str(source / "stereo_1.wav"), command)
        self.assertIn(str(self.root / "raw" / "stereo_1.wav"), command)
        self.assertIn("16000", command)
        self.assertIn("pcm_s16le", command)
