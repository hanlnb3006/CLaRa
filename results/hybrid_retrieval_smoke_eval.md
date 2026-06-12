# Hybrid Retrieval Smoke Evaluation

This deterministic smoke evaluation checks whether Hybrid Retrieval + adaptive fusion changes retrieval rankings in the intended direction.
It is not a replacement for full EM/F1 benchmark evaluation on HotpotQA, 2Wiki, MuSiQue, or NQ.

## Summary

| Method | Recall@1 | Recall@2 | MRR |
|---|---:|---:|---:|
| latent_only | 0.400 | 1.000 | 0.700 |
| hybrid_fixed_alpha_0.90_m2 | 1.000 | 1.000 | 1.000 |
| hybrid_adaptive_alpha_0.75_0.95_m2 | 1.000 | 1.000 | 1.000 |

## Per-case Results

### entity_date_exact_match

Question: When was Timothy McVeigh executed on 2001-06-11?

Relevant document index: [1]

| Method | Top-1 | Ranking | Recall@1 | MRR | Alpha |
|---|---:|---|---:|---:|---:|
| latent_only | 0 | [0, 1, 2] | 0.0 | 0.500 | n/a |
| hybrid_fixed_alpha_0.90_m2 | 1 | [1, 0, 2] | 1.0 | 1.000 | 0.900 |
| hybrid_adaptive_alpha_0.75_0.95_m2 | 1 | [1, 0, 2] | 1.0 | 1.000 | 0.910 |

Normalized BM25 scores: [0.0, 1.0, 0.0]

### legal_rare_term_exact_match

Question: Which regulation defines HIPAA covered entity under 45 CFR 160.103?

Relevant document index: [1]

| Method | Top-1 | Ranking | Recall@1 | MRR | Alpha |
|---|---:|---|---:|---:|---:|
| latent_only | 0 | [0, 1, 2] | 0.0 | 0.500 | n/a |
| hybrid_fixed_alpha_0.90_m2 | 1 | [1, 0, 2] | 1.0 | 1.000 | 0.900 |
| hybrid_adaptive_alpha_0.75_0.95_m2 | 1 | [1, 0, 2] | 1.0 | 1.000 | 0.879 |

Normalized BM25 scores: [0.0, 1.0, 0.0]

### medical_abbreviation_exact_match

Question: What does EGFR indicate in chronic kidney disease staging?

Relevant document index: [1]

| Method | Top-1 | Ranking | Recall@1 | MRR | Alpha |
|---|---:|---|---:|---:|---:|
| latent_only | 0 | [0, 1, 2] | 0.0 | 0.500 | n/a |
| hybrid_fixed_alpha_0.90_m2 | 1 | [1, 0, 2] | 1.0 | 1.000 | 0.900 |
| hybrid_adaptive_alpha_0.75_0.95_m2 | 1 | [1, 0, 2] | 1.0 | 1.000 | 0.926 |

Normalized BM25 scores: [0.333333, 1.0, 0.0]

### semantic_match_should_not_break

Question: What is a major risk of long-term high blood pressure?

Relevant document index: [1]

| Method | Top-1 | Ranking | Recall@1 | MRR | Alpha |
|---|---:|---|---:|---:|---:|
| latent_only | 1 | [1, 2, 0] | 1.0 | 1.000 | n/a |
| hybrid_fixed_alpha_0.90_m2 | 1 | [1, 2, 0] | 1.0 | 1.000 | 0.900 |
| hybrid_adaptive_alpha_0.75_0.95_m2 | 1 | [1, 2, 0] | 1.0 | 1.000 | 0.923 |

Normalized BM25 scores: [0.0, 1.0, 0.0]

### lexical_and_dense_agree

Question: What condition is treated by metformin?

Relevant document index: [0]

| Method | Top-1 | Ranking | Recall@1 | MRR | Alpha |
|---|---:|---|---:|---:|---:|
| latent_only | 0 | [0, 1, 2] | 1.0 | 1.000 | n/a |
| hybrid_fixed_alpha_0.90_m2 | 0 | [0, 2, 1] | 1.0 | 1.000 | 0.900 |
| hybrid_adaptive_alpha_0.75_0.95_m2 | 0 | [0, 2, 1] | 1.0 | 1.000 | 0.950 |

Normalized BM25 scores: [1.0, 0.0, 0.0]
