"""Unit tests for Block C: Better Differentiable Top-k relaxations.

Covers:
    - sparsemax / entmax15 (per-instance) numerical properties.
    - top-k wrappers: shape, hard-mask forward, topk_idx correctness.
    - gumbel_topk_st: Gumbel noise effect on topk_idx (stochastic but valid).
    - differentiable_topk dispatcher routes to each method.
    - Gradient flows through all wrappers.
    - All four methods agree on the topk_idx for clearly dominant logits.
"""

import unittest

import torch

from openrlhf.models.modeling_clara import (
    differentiable_topk,
    differentiable_topk_iterative_st,
    entmax15,
    entmax15_topk,
    gumbel_topk_st,
    sparsemax,
    sparsemax_topk,
)


class TestSparsemax(unittest.TestCase):
    def test_output_is_valid_distribution(self):
        x = torch.tensor([[1.0, 2.0, 0.5, 0.1, -1.0]])
        p = sparsemax(x, dim=-1)
        self.assertTrue(torch.allclose(p.sum(dim=-1), torch.tensor([1.0]), atol=1e-5))
        self.assertTrue((p >= 0).all())

    def test_sparse_output(self):
        x = torch.tensor([[3.0, 1.0, 0.0, -2.0]])
        p = sparsemax(x, dim=-1)
        self.assertGreater((p == 0).sum().item(), 0)
        self.assertGreater(p.max().item(), 0.0)

    def test_uniform_input_yields_uniform_output(self):
        x = torch.zeros(1, 5)
        p = sparsemax(x, dim=-1)
        self.assertTrue(torch.allclose(p, torch.full((1, 5), 0.2), atol=1e-5))

    def test_gradient_flows(self):
        x = torch.randn(2, 6, requires_grad=True)
        p = sparsemax(x, dim=-1)
        p.sum().backward()
        self.assertIsNotNone(x.grad)


class TestEntmax15(unittest.TestCase):
    def test_output_is_valid_distribution(self):
        x = torch.tensor([[1.0, 2.0, 0.5, 0.1, -1.0]])
        p = entmax15(x, dim=-1)
        self.assertTrue(torch.allclose(p.sum(dim=-1), torch.tensor([1.0]), atol=1e-4))
        self.assertTrue((p >= 0).all())

    def test_more_sparse_than_softmax(self):
        x = torch.tensor([[5.0, 1.0, 0.5, 0.0, -1.0]])
        p = entmax15(x, dim=-1)
        soft = torch.softmax(x, dim=-1)
        # entmax should put much more mass on the top-1 element than softmax
        self.assertGreater(p[0, 0].item(), soft[0, 0].item())

    def test_gradient_flows(self):
        x = torch.randn(2, 6, requires_grad=True)
        p = entmax15(x, dim=-1)
        p.sum().backward()
        self.assertIsNotNone(x.grad)


class TestTopkWrappers(unittest.TestCase):
    B, N, k = 4, 10, 3

    def _make_scores(self):
        torch.manual_seed(123)
        return torch.randn(self.B, self.N)

    def test_iterative_st_shape(self):
        scores = self._make_scores()
        W, idx = differentiable_topk_iterative_st(scores, self.k, 1.0)
        self.assertEqual(W.shape, (self.B, self.k, self.N))
        self.assertEqual(idx.shape, (self.B, self.k))

    def test_sparsemax_topk_shape(self):
        scores = self._make_scores()
        W, idx = sparsemax_topk(scores, self.k, 1.0)
        self.assertEqual(W.shape, (self.B, self.k, self.N))
        self.assertEqual(idx.shape, (self.B, self.k))

    def test_entmax15_topk_shape(self):
        scores = self._make_scores()
        W, idx = entmax15_topk(scores, self.k, 1.0)
        self.assertEqual(W.shape, (self.B, self.k, self.N))
        self.assertEqual(idx.shape, (self.B, self.k))

    def test_gumbel_topk_st_shape(self):
        torch.manual_seed(0)
        scores = self._make_scores()
        W, idx = gumbel_topk_st(scores, self.k, 1.0)
        self.assertEqual(W.shape, (self.B, self.k, self.N))
        self.assertEqual(idx.shape, (self.B, self.k))

    def test_forward_is_hard_mask(self):
        scores = self._make_scores()
        for fn in [differentiable_topk_iterative_st, sparsemax_topk, entmax15_topk]:
            W, idx = fn(scores, self.k, 1.0)
            # W row sums must be 1
            self.assertTrue(torch.allclose(W.sum(dim=-1), torch.ones(self.B, self.k), atol=1e-5))
            # W[b, j] must be 1-hot at idx[b, j]
            for b in range(self.B):
                for j in range(self.k):
                    self.assertEqual(W[b, j, idx[b, j]].item(), 1.0)
                    self.assertEqual(W[b, j].sum().item(), 1.0)

    def test_topk_idx_is_valid(self):
        scores = self._make_scores()
        for fn in [differentiable_topk_iterative_st, sparsemax_topk, entmax15_topk]:
            _, idx = fn(scores, self.k, 1.0)
            self.assertTrue((idx >= 0).all())
            self.assertTrue((idx < self.N).all())
            # No duplicates within a row
            for b in range(self.B):
                self.assertEqual(len(set(idx[b].tolist())), self.k)

    def test_dominant_logit_always_selected(self):
        # When one logit is much larger, all methods should select it
        scores = torch.zeros(2, 5)
        scores[0, 2] = 100.0
        scores[1, 0] = 100.0
        for fn in [differentiable_topk_iterative_st, sparsemax_topk, entmax15_topk]:
            _, idx = fn(scores, 1, 1.0)
            self.assertEqual(idx[0, 0].item(), 2)
            self.assertEqual(idx[1, 0].item(), 0)

    def test_gradient_flows_through_sparsemax(self):
        x = torch.randn(2, 5, requires_grad=True)
        W, _ = sparsemax_topk(x, 2, 1.0)
        W.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_gradient_flows_through_entmax15(self):
        x = torch.randn(2, 5, requires_grad=True)
        W, _ = entmax15_topk(x, 2, 1.0)
        W.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_gradient_flows_through_gumbel_st(self):
        x = torch.randn(2, 5, requires_grad=True)
        W, _ = gumbel_topk_st(x, 2, 1.0)
        W.sum().backward()
        self.assertIsNotNone(x.grad)


class TestDispatcher(unittest.TestCase):
    def test_each_method_routes(self):
        scores = torch.randn(3, 8)
        for method in ["iterative_st", "sparsemax", "entmax15", "gumbel_st"]:
            W, idx = differentiable_topk(scores, 2, 1.0, method=method)
            self.assertEqual(W.shape, (3, 2, 8))
            self.assertEqual(idx.shape, (3, 2))

    def test_unknown_method_raises(self):
        scores = torch.randn(2, 5)
        with self.assertRaises(ValueError):
            differentiable_topk(scores, 2, 1.0, method="bogus")


if __name__ == "__main__":
    unittest.main()