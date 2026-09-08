import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "moshi"))

from moshi.models.lm import LMModel
from training.objective import weighted_personaplex_loss
from training.streams import build_loss_weights


class PersonaPlexModelIntegrationTests(unittest.TestCase):
    def test_local_model_backward_accepts_prepared_stream_layout(self):
        model = LMModel(
            delays=[0] * 17, n_q=16, dep_q=16, card=8, text_card=16,
            dim=16, num_heads=4, num_layers=1, hidden_scale=2,
            depformer_dim=16, depformer_num_heads=4, depformer_num_layers=1,
            depformer_dim_feedforward=32, context=16, depformer_context=8,
        )
        codes = torch.randint(0, 8, (1, 17, 5))
        codes[:, 0] = torch.randint(0, 16, (1, 5))

        output = model.forward_train(codes)
        weights = build_loss_weights(codes[0], prompt_frames=1, text_padding_id=model.text_padding_token_id).unsqueeze(0)
        loss = weighted_personaplex_loss(output.text_logits, output.logits, codes, weights, output.text_mask, output.mask)
        loss.backward()

        self.assertEqual(tuple(output.logits.shape), (1, 16, 5, 8))
        self.assertIsNotNone(model.text_linear.weight.grad)


if __name__ == "__main__":
    unittest.main()
