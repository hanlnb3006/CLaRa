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
