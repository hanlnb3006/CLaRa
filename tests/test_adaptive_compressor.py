import unittest

import torch

from openrlhf.models.modeling_clara import CLaRa


class AdaptiveCompressorTest(unittest.TestCase):
    def _model(self):
        model = CLaRa.__new__(CLaRa)
        model.adaptive_compressor = True
        model.adaptive_compressor_top_k = 3
        model.adaptive_compressor_strength = 0.25
        model.adaptive_compressor_temperature = 0.10
        model.generation_top_k = 2
        return model

    def test_accepts_flattened_query_memory_tokens(self):
        model = self._model()
        selected_docs = torch.randn(4, 16, 4096)
        query_reps = torch.randn(2, 16 * 4096)

        out = model._apply_adaptive_compressor(selected_docs, query_reps)

        self.assertEqual(out.shape, selected_docs.shape)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.equal(out, selected_docs))

    def test_accepts_query_memory_token_sequence(self):
        model = self._model()
        selected_docs = torch.randn(4, 16, 32)
        query_reps = torch.randn(2, 16, 32)

        out = model._apply_adaptive_compressor(selected_docs, query_reps)

        self.assertEqual(out.shape, selected_docs.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_mismatched_query_shape_falls_back_to_input(self):
        model = self._model()
        selected_docs = torch.randn(4, 16, 32)
        query_reps = torch.randn(2, 33)

        out = model._apply_adaptive_compressor(selected_docs, query_reps)

        self.assertTrue(torch.equal(out, selected_docs))


if __name__ == "__main__":
    unittest.main()
    def test_gate_builds_and_is_trainable(self):
        model = self._model()
        model.adaptive_compressor_trainable = True
        model.adaptive_compressor_lora_r = 8
        model.adaptive_compressor_hidden = 0
        model._setup_adaptive_compressor_gate()
        self.assertIsNotNone(model.adaptive_gate_lora)
        params = [p for p in model.adaptive_gate_lora.parameters() if p.requires_grad]
        self.assertGreater(len(params), 0)
        self.assertLess(sum(p.numel() for p in params), 200_000)

    def test_gate_backward_changes_weights(self):
        model = self._model()
        model.adaptive_compressor_trainable = True
        model.adaptive_compressor_lora_r = 8
        model._setup_adaptive_compressor_gate()

        before = [p.detach().clone() for p in model.adaptive_gate_lora.parameters()]
        selected_docs = torch.randn(4, 16, 32)
        query_reps = torch.randn(2, 16 * 32)
        out = model._apply_adaptive_compressor(selected_docs, query_reps)
        loss = out.float().pow(2).mean()
        loss.backward()

        changed = 0
        for p_prev, p_now in zip(before, model.adaptive_gate_lora.parameters()):
            if not torch.allclose(p_prev, p_now.detach()):
                changed += 1
        self.assertEqual(changed, len(before))
