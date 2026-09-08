import unittest

import torch

from training.contract import ManifestError, TrainingExample, validate_record
from training.objective import weighted_personaplex_loss
from training.prepare import align_text_tokens
from training.streams import build_codes, build_loss_weights


class ManifestValidationTests(unittest.TestCase):
    def valid_record(self):
        return {
            "id": "conversation-1",
            "stereo_wav": "audio/conversation-1.wav",
            "speaker_ids": ["speaker-a", "speaker-b"],
            "split_group": "conversation-1",
            "split": "train",
            "language": "en",
            "approved_role_prompts": ["You are a helpful librarian.", "You are a curious patron."],
            "role_prompt_provenance": [
                {"status": "approved", "version": "v1"},
                {"status": "approved", "version": "v1"},
            ],
            "transcripts": [
                [{"start": 0.0, "end": 0.2, "text": "hello"}],
                [{"start": 0.3, "end": 0.5, "text": "hi"}],
            ],
            "voice_prompts": [
                {"path": "prompts/a.wav", "start": 0.0, "end": 1.0},
                {"path": "prompts/b.wav", "start": 0.0, "end": 1.0},
            ],
        }

    def test_accepts_reviewed_two_speaker_record(self):
        record = validate_record(self.valid_record())
        self.assertEqual(record.id, "conversation-1")
        self.assertEqual(record.voice_prompts[1].path, "prompts/b.wav")

    def test_rejects_unreviewed_role_prompt(self):
        record = self.valid_record()
        record["role_prompt_provenance"][0]["status"] = "generated"
        with self.assertRaisesRegex(ManifestError, "approved"):
            validate_record(record)

    def test_rejects_overlapping_in_conversation_voice_prompt(self):
        record = self.valid_record()
        record["voice_prompts"][0] = {
            "path": "audio/conversation-1.wav",
            "start": 0.1,
            "end": 0.4,
            "channel": 0,
        }
        with self.assertRaisesRegex(ManifestError, "overlaps"):
            validate_record(record)


class StreamLayoutTests(unittest.TestCase):
    def test_builds_personaplex_stream_order_for_one_direction(self):
        example = TrainingExample(
            id="conversation-1:b-to-a",
            user_channel=1,
            agent_channel=0,
            agent_text=torch.tensor([10, 3, 11]),
            agent_audio=torch.arange(24).reshape(8, 3),
            user_audio=torch.arange(24, 48).reshape(8, 3),
            prompt_frames=2,
        )

        codes = build_codes(example)

        self.assertEqual(tuple(codes.shape), (17, 3))
        self.assertTrue(torch.equal(codes[0], torch.tensor([10, 3, 11])))
        self.assertTrue(torch.equal(codes[1:9], example.agent_audio))
        self.assertTrue(torch.equal(codes[9:17], example.user_audio))

    def test_loss_masks_prompt_and_user_stream(self):
        codes = torch.zeros((17, 4), dtype=torch.long)
        codes[0, 2] = 3
        weights = build_loss_weights(codes, prompt_frames=2, text_padding_id=3)

        self.assertTrue(torch.equal(weights[:, :2], torch.zeros((17, 2))))
        self.assertTrue(torch.equal(weights[9:17], torch.zeros((8, 4))))
        self.assertAlmostEqual(weights[0, 2].item(), 0.3)
        self.assertAlmostEqual(weights[1, 2].item(), 1.0)
        self.assertAlmostEqual(weights[2, 2].item(), 0.02)


class ObjectiveTests(unittest.TestCase):
    def test_ignores_user_stream_loss(self):
        text_logits = torch.tensor([[[[0.0, 4.0, 0.0, 0.0]]]])
        audio_logits = torch.zeros((1, 16, 1, 4))
        audio_logits[:, 0, 0, 1] = 4.0
        codes = torch.ones((1, 17, 1), dtype=torch.long)
        weights = torch.zeros((1, 17, 1))
        weights[:, 0] = 1.0
        weights[:, 1] = 1.0

        loss = weighted_personaplex_loss(text_logits, audio_logits, codes, weights)

        self.assertLess(loss.item(), 0.1)


class TextAlignmentTests(unittest.TestCase):
    def test_places_tokens_inside_their_timestamp_window(self):
        stream = align_text_tokens(
            [(0.0, 0.4, [10, 11]), (0.4, 0.8, [12])], frame_rate=10, total_frames=8, padding_id=3
        )

        self.assertTrue(torch.equal(stream, torch.tensor([10, 11, 3, 3, 12, 3, 3, 3])))

    def test_rejects_more_tokens_than_frames(self):
        with self.assertRaisesRegex(ValueError, "too many text tokens"):
            align_text_tokens([(0.0, 0.1, [10, 11])], frame_rate=10, total_frames=1, padding_id=3)


if __name__ == "__main__":
    unittest.main()
