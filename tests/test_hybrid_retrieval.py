import importlib.util
import pathlib
import unittest

try:
    import torch

    module_path = pathlib.Path(__file__).resolve().parents[1] / "openrlhf" / "models" / "hybrid_retrieval.py"
    spec = importlib.util.spec_from_file_location("hybrid_retrieval_testmod", module_path)
    hybrid_retrieval = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(hybrid_retrieval)

    compute_bm25_scores = hybrid_retrieval.compute_bm25_scores
    compute_hybrid_alpha = hybrid_retrieval.compute_hybrid_alpha
    fuse_retrieval_scores = hybrid_retrieval.fuse_retrieval_scores
except ModuleNotFoundError as exc:
    torch = None
    compute_bm25_scores = None
    compute_hybrid_alpha = None
    fuse_retrieval_scores = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(torch is None, f"Missing optional dependency: {IMPORT_ERROR}")
class HybridRetrievalTest(unittest.TestCase):
    def test_bm25_prefers_exact_query_terms(self):
        scores = compute_bm25_scores(
            questions=["When was Timothy McVeigh executed?"],
            documents=[[
                "Timothy McVeigh was executed by lethal injection on June 11, 2001.",
                "Storms can produce heavy rain, wind, thunder, and lightning.",
            ]],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(scores.shape, (1, 2))
        self.assertGreater(scores[0, 0].item(), scores[0, 1].item())

    def test_fixed_fusion_can_change_top_document(self):
        latent_scores = torch.tensor([[0.35, 0.30]], dtype=torch.float32)

        fused, bm25_scores, alpha = fuse_retrieval_scores(
            latent_scores,
            questions=["When was Timothy McVeigh executed on 2001-06-11?"],
            documents=[[
                "Storms can produce heavy rain, wind, thunder, and lightning.",
                "Timothy McVeigh was executed by lethal injection on June 11, 2001.",
            ]],
            enabled=True,
            fixed_alpha=0.25,
            candidate_top_m=2,
        )

        self.assertIsNotNone(bm25_scores)
        self.assertTrue(torch.allclose(alpha, torch.tensor([0.25]), atol=1e-6))
        self.assertGreater(fused[0, 1].item(), fused[0, 0].item())

    def test_adaptive_alpha_moves_toward_bm25_for_specific_queries(self):
        bm25_scores = torch.tensor([[0.0, 4.0]], dtype=torch.float32)

        alpha = compute_hybrid_alpha(
            ["When was Timothy McVeigh executed on 2001-06-11?"],
            bm25_scores,
            documents=[["A weather report.", "Timothy McVeigh was executed on 2001-06-11."]],
            adaptive=True,
            alpha_min=0.75,
            alpha_max=0.95,
        )

        self.assertGreaterEqual(alpha.item(), 0.75 - 1e-6)
        self.assertLess(alpha.item(), 0.95)

    def test_adaptive_alpha_stays_near_latent_for_ambiguous_queries(self):
        bm25_scores = torch.tensor([[0.0, 4.0, 3.0]], dtype=torch.float32)

        alpha = compute_hybrid_alpha(
            ["Who is the spouse of the Green performer?"],
            bm25_scores,
            documents=[[
                "Green is an album by Steve Hillage.",
                "Kevin Green is married to Chele.",
                "Kenric Green is married to Sonequa Martin-Green.",
            ]],
            adaptive=True,
            alpha_min=0.75,
            alpha_max=0.95,
        )

        self.assertGreater(alpha.item(), 0.90)

    def test_candidate_top_m_prevents_bm25_from_promoting_far_candidates(self):
        latent_scores = torch.tensor([[0.9, 0.8, -0.9]], dtype=torch.float32)

        fused, bm25_scores, _ = fuse_retrieval_scores(
            latent_scores,
            questions=["When was Timothy McVeigh executed?"],
            documents=[[
                "Timothy McVeigh was discussed in a biographical article.",
                "A federal prison article mentions execution procedures.",
                "Timothy McVeigh was executed by lethal injection on June 11, 2001.",
            ]],
            enabled=True,
            fixed_alpha=0.25,
            candidate_top_m=2,
        )

        self.assertIsNotNone(bm25_scores)
        self.assertLess(fused[0, 2].item(), fused[0, 1].item())

    def test_ambiguous_query_falls_back_to_latent_ranking(self):
        latent_scores = torch.tensor([[0.9, 0.2, 0.1]], dtype=torch.float32)

        fused, _, _ = fuse_retrieval_scores(
            latent_scores,
            questions=["Who is the spouse of the Green performer?"],
            documents=[[
                "Green is an album by Steve Hillage.",
                "Kevin Green is married to Chele.",
                "Kenric Green is married to Sonequa Martin-Green.",
            ]],
            enabled=True,
            fixed_alpha=0.25,
            candidate_top_m=3,
        )

        latent_norm = ((latent_scores + 1.0) * 0.5).clamp(0.0, 1.0)
        self.assertTrue(torch.allclose(fused, latent_norm, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
