#
# Lightweight, deterministic evidence for Hybrid Retrieval + adaptive fusion.
# This is not a replacement for full EM/F1 benchmark evaluation.
#

import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
HYBRID_MODULE = ROOT / "openrlhf" / "models" / "hybrid_retrieval.py"
RESULTS_DIR = ROOT / "results"


def load_hybrid_module():
    spec = importlib.util.spec_from_file_location("hybrid_retrieval_smoke", HYBRID_MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CASES = [
    {
        "id": "entity_date_exact_match",
        "question": "When was Timothy McVeigh executed on 2001-06-11?",
        "documents": [
            "Storms can produce heavy rain, wind, thunder, and lightning.",
            "Timothy McVeigh was executed by lethal injection on June 11, 2001.",
            "The federal execution chamber is located in Terre Haute, Indiana.",
        ],
        "relevant": [1],
        "latent_scores": [0.85, 0.15, 0.10],
    },
    {
        "id": "legal_rare_term_exact_match",
        "question": "Which regulation defines HIPAA covered entity under 45 CFR 160.103?",
        "documents": [
            "Patient privacy law often discusses confidentiality and insurance records.",
            "45 CFR 160.103 defines covered entity for HIPAA administrative simplification rules.",
            "Corporate compliance programs usually include audit trails and employee training.",
        ],
        "relevant": [1],
        "latent_scores": [0.78, 0.18, 0.08],
    },
    {
        "id": "medical_abbreviation_exact_match",
        "question": "What does EGFR indicate in chronic kidney disease staging?",
        "documents": [
            "Cancer staging systems often describe tumor size and lymph node involvement.",
            "Estimated glomerular filtration rate, or eGFR, is used to stage chronic kidney disease.",
            "Creatinine can be measured in blood tests during routine clinical assessment.",
        ],
        "relevant": [1],
        "latent_scores": [0.70, 0.30, 0.20],
    },
    {
        "id": "semantic_match_should_not_break",
        "question": "What is a major risk of long-term high blood pressure?",
        "documents": [
            "A budget deficit happens when spending exceeds revenue over a fiscal period.",
            "Persistently elevated arterial pressure can damage blood vessels and increase stroke risk.",
            "Regular exercise can improve general cardiovascular fitness and endurance.",
        ],
        "relevant": [1],
        "latent_scores": [0.10, 0.82, 0.35],
    },
    {
        "id": "lexical_and_dense_agree",
        "question": "What condition is treated by metformin?",
        "documents": [
            "Metformin is commonly prescribed to treat type 2 diabetes.",
            "Ibuprofen is often used for fever, pain, and inflammation.",
            "Atorvastatin is used to manage elevated cholesterol levels.",
        ],
        "relevant": [0],
        "latent_scores": [0.80, 0.20, 0.15],
    },
]


def rank_indices(scores):
    return torch.argsort(scores, descending=True).tolist()


def reciprocal_rank(ranking, relevant):
    relevant = set(relevant)
    for idx, doc_idx in enumerate(ranking, start=1):
        if doc_idx in relevant:
            return 1.0 / idx
    return 0.0


def recall_at_k(ranking, relevant, k):
    relevant = set(relevant)
    retrieved = set(ranking[:k])
    return 1.0 if relevant & retrieved else 0.0


def summarize(rows, method):
    return {
        "recall@1": sum(row[method]["recall@1"] for row in rows) / len(rows),
        "recall@2": sum(row[method]["recall@2"] for row in rows) / len(rows),
        "mrr": sum(row[method]["mrr"] for row in rows) / len(rows),
    }


def evaluate_method(case, scores):
    ranking = rank_indices(scores)
    return {
        "scores": [round(float(score), 6) for score in scores.tolist()],
        "ranking": ranking,
        "top1": ranking[0],
        "recall@1": recall_at_k(ranking, case["relevant"], 1),
        "recall@2": recall_at_k(ranking, case["relevant"], 2),
        "mrr": reciprocal_rank(ranking, case["relevant"]),
    }


def main():
    hybrid = load_hybrid_module()
    rows = []

    for case in CASES:
        latent_scores = torch.tensor([case["latent_scores"]], dtype=torch.float32)
        questions = [case["question"]]
        documents = [case["documents"]]

        fixed_scores, fixed_bm25, fixed_alpha = hybrid.fuse_retrieval_scores(
            latent_scores,
            questions,
            documents,
            enabled=True,
            fixed_alpha=0.75,
            adaptive=False,
        )
        adaptive_scores, adaptive_bm25, adaptive_alpha = hybrid.fuse_retrieval_scores(
            latent_scores,
            questions,
            documents,
            enabled=True,
            fixed_alpha=0.75,
            adaptive=True,
            alpha_min=0.45,
            alpha_max=0.90,
        )

        row = {
            "id": case["id"],
            "question": case["question"],
            "relevant": case["relevant"],
            "latent_only": evaluate_method(case, latent_scores[0]),
            "hybrid_fixed_alpha_0.75": evaluate_method(case, fixed_scores[0]),
            "hybrid_adaptive_alpha_0.45_0.90": evaluate_method(case, adaptive_scores[0]),
            "bm25_normalized": [round(float(score), 6) for score in fixed_bm25[0].tolist()],
            "fixed_alpha": round(float(fixed_alpha[0].item()), 6),
            "adaptive_alpha": round(float(adaptive_alpha[0].item()), 6),
        }
        rows.append(row)

    summary = {
        "num_cases": len(CASES),
        "methods": {
            "latent_only": summarize(rows, "latent_only"),
            "hybrid_fixed_alpha_0.75": summarize(rows, "hybrid_fixed_alpha_0.75"),
            "hybrid_adaptive_alpha_0.45_0.90": summarize(rows, "hybrid_adaptive_alpha_0.45_0.90"),
        },
    }

    payload = {
        "note": (
            "Controlled smoke evaluation for Hybrid Retrieval + adaptive fusion. "
            "It verifies retrieval score fusion behavior on deterministic cases, "
            "but it is not a substitute for full dataset EM/F1/Recall benchmarking."
        ),
        "summary": summary,
        "cases": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "hybrid_retrieval_smoke_eval.json"
    md_path = RESULTS_DIR / "hybrid_retrieval_smoke_eval.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Hybrid Retrieval Smoke Evaluation",
        "",
        "This deterministic smoke evaluation checks whether Hybrid Retrieval + adaptive fusion changes retrieval rankings in the intended direction.",
        "It is not a replacement for full EM/F1 benchmark evaluation on HotpotQA, 2Wiki, MuSiQue, or NQ.",
        "",
        "## Summary",
        "",
        "| Method | Recall@1 | Recall@2 | MRR |",
        "|---|---:|---:|---:|",
    ]
    for method, metrics in summary["methods"].items():
        lines.append(
            f"| {method} | {metrics['recall@1']:.3f} | {metrics['recall@2']:.3f} | {metrics['mrr']:.3f} |"
        )

    lines.extend(["", "## Per-case Results", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['id']}",
                "",
                f"Question: {row['question']}",
                "",
                f"Relevant document index: {row['relevant']}",
                "",
                "| Method | Top-1 | Ranking | Recall@1 | MRR | Alpha |",
                "|---|---:|---|---:|---:|---:|",
                (
                    "| latent_only | "
                    f"{row['latent_only']['top1']} | {row['latent_only']['ranking']} | "
                    f"{row['latent_only']['recall@1']:.1f} | {row['latent_only']['mrr']:.3f} | n/a |"
                ),
                (
                    "| hybrid_fixed_alpha_0.75 | "
                    f"{row['hybrid_fixed_alpha_0.75']['top1']} | {row['hybrid_fixed_alpha_0.75']['ranking']} | "
                    f"{row['hybrid_fixed_alpha_0.75']['recall@1']:.1f} | "
                    f"{row['hybrid_fixed_alpha_0.75']['mrr']:.3f} | {row['fixed_alpha']:.3f} |"
                ),
                (
                    "| hybrid_adaptive_alpha_0.45_0.90 | "
                    f"{row['hybrid_adaptive_alpha_0.45_0.90']['top1']} | "
                    f"{row['hybrid_adaptive_alpha_0.45_0.90']['ranking']} | "
                    f"{row['hybrid_adaptive_alpha_0.45_0.90']['recall@1']:.1f} | "
                    f"{row['hybrid_adaptive_alpha_0.45_0.90']['mrr']:.3f} | {row['adaptive_alpha']:.3f} |"
                ),
                "",
                f"Normalized BM25 scores: {row['bm25_normalized']}",
                "",
            ]
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nWrote: results/hybrid_retrieval_smoke_eval.json")
    print("Wrote: results/hybrid_retrieval_smoke_eval.md")


if __name__ == "__main__":
    main()
