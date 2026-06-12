#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import math
import re
from collections import Counter
from typing import List, Optional, Tuple

import torch


LEXICAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were",
    "what", "when", "where", "which", "who", "whom", "why", "with", "how", "did",
    "does", "do", "can", "could", "would", "should", "about", "into", "than",
}


def lexical_tokens(text: str) -> List[str]:
    """Tokenize text for lightweight lexical retrieval."""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_./-]*", text or "")
    return [tok.lower() for tok in tokens if tok.lower() not in LEXICAL_STOPWORDS]


def specific_token_ratio(text: str) -> float:
    """Estimate whether a query depends on exact lexical matching."""
    raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_./-]*", text or "")
    content_tokens = [tok for tok in raw_tokens if tok.lower() not in LEXICAL_STOPWORDS]
    if not content_tokens:
        return 0.0

    def is_specific(token: str) -> bool:
        has_digit = any(ch.isdigit() for ch in token)
        has_symbol = any(ch in token for ch in "-/. _")
        is_acronym = token.isupper() and 2 <= len(token) <= 8
        return (
            has_digit
            or has_symbol
            or is_acronym
            or len(token) >= 12
        )

    return sum(1 for token in content_tokens if is_specific(token)) / len(content_tokens)

def query_anchor_strength(question: str, doc_list: List[str]) -> float:
    """Estimate whether lexical matching is anchored by rare, query-specific terms.

    High values mean the query contains a few distinctive tokens that appear in only a
    small subset of candidate documents, making lexical evidence more trustworthy.
    """
    query_terms = lexical_tokens(question)
    if not query_terms or not doc_list:
        return 0.0

    tokenized_docs = [lexical_tokens(doc) for doc in doc_list]
    num_docs = len(tokenized_docs)
    if num_docs == 0:
        return 0.0

    doc_freq = Counter()
    for doc_terms in tokenized_docs:
        doc_freq.update(set(doc_terms))

    content_terms = [term for term in query_terms if len(term) >= 4]
    if not content_terms:
        return 0.0

    anchored = []
    for term in content_terms:
        df = doc_freq.get(term, 0)
        if df == 0:
            continue
        rarity = 1.0 - min(1.0, (df - 1) / max(num_docs - 1, 1))
        anchored.append(rarity)

    if not anchored:
        return 0.0

    top_anchors = sorted(anchored, reverse=True)[:3]
    return sum(top_anchors) / len(top_anchors)


def normalize_bm25_scores(scores: torch.Tensor) -> torch.Tensor:
    """Normalize BM25 scores per query while preserving zero rows."""
    scores = scores.float()
    min_vals = scores.min(dim=-1, keepdim=True).values
    max_vals = scores.max(dim=-1, keepdim=True).values
    span = max_vals - min_vals
    return torch.where(span > 1e-6, (scores - min_vals) / span, torch.zeros_like(scores))


def compute_bm25_scores(
    questions: List[str],
    documents: List[List[str]],
    device: torch.device,
    dtype: torch.dtype,
    k1: float = 1.2,
    b: float = 0.75,
) -> torch.Tensor:
    """Compute per-query BM25 scores over the provided candidate documents."""
    rows = []
    for question, doc_list in zip(questions, documents):
        query_terms = lexical_tokens(question)
        tokenized_docs = [lexical_tokens(doc) for doc in doc_list]
        num_docs = len(tokenized_docs)

        if num_docs == 0:
            rows.append([])
            continue
        if not query_terms:
            rows.append([0.0] * num_docs)
            continue

        doc_freq = Counter()
        for doc_terms in tokenized_docs:
            doc_freq.update(set(doc_terms))

        avg_doc_len = max(sum(len(doc_terms) for doc_terms in tokenized_docs) / num_docs, 1.0)
        query_counts = Counter(query_terms)
        doc_scores = []

        for doc_terms in tokenized_docs:
            term_freq = Counter(doc_terms)
            doc_len = max(len(doc_terms), 1)
            score = 0.0

            for term, query_tf in query_counts.items():
                tf = term_freq.get(term, 0)
                if tf == 0:
                    continue

                df = doc_freq.get(term, 0)
                idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1.0 - b + b * doc_len / avg_doc_len)
                score += query_tf * idf * (tf * (k1 + 1.0) / max(denom, 1e-6))

            doc_scores.append(score)

        rows.append(doc_scores)

    return torch.tensor(rows, device=device, dtype=dtype)


def compute_hybrid_alpha(
    questions: List[str],
    bm25_scores: torch.Tensor,
    documents: Optional[List[List[str]]] = None,
    *,
    fixed_alpha: float = 0.90,
    adaptive: bool = False,
    alpha_min: float = 0.75,
    alpha_max: float = 0.95,
) -> torch.Tensor:
    """Return per-query latent-score weights for fixed or adaptive fusion."""
    device = bm25_scores.device
    dtype = bm25_scores.dtype

    if not adaptive:
        return torch.full((len(questions), 1), fixed_alpha, device=device, dtype=dtype)

    alpha_min = max(0.0, min(alpha_min, alpha_max))
    alpha_max = min(1.0, max(alpha_min, alpha_max))
    bm25_norm = normalize_bm25_scores(bm25_scores)
    signals = []

    for row_idx, question in enumerate(questions):
        specificity = specific_token_ratio(question)
        anchor_strength = query_anchor_strength(question, documents[row_idx]) if documents else 0.0

        if bm25_norm.size(1) > 1:
            top_vals = torch.topk(bm25_norm[row_idx], k=2).values
            bm25_confidence = float((top_vals[0] - top_vals[1]).clamp(min=0.0, max=1.0).item())
        else:
            bm25_confidence = 0.0

        short_specific_query = 1.0 if len(lexical_tokens(question)) <= 6 and specificity >= 0.4 else 0.0
        lexical_signal = min(
            1.0,
            0.45 * specificity
            + 0.35 * anchor_strength
            + 0.10 * bm25_confidence * anchor_strength
            + 0.10 * short_specific_query,
        )

        # Bridge-style and ambiguous questions should stay close to latent retrieval.
        if anchor_strength < 0.20:
            lexical_signal *= 0.20
        elif anchor_strength < 0.35:
            lexical_signal *= 0.50

        signal = lexical_signal
        signals.append(signal)

    lexical_signal = torch.tensor(signals, device=device, dtype=dtype).unsqueeze(-1)
    return alpha_max - lexical_signal * (alpha_max - alpha_min)


def fuse_retrieval_scores(
    latent_scores: torch.Tensor,
    questions: Optional[List[str]],
    documents: Optional[List[List[str]]],
    *,
    enabled: bool,
    fixed_alpha: float = 0.90,
    adaptive: bool = False,
    alpha_min: float = 0.75,
    alpha_max: float = 0.95,
    bm25_k1: float = 1.2,
    bm25_b: float = 0.75,
    candidate_top_m: int = 5,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fuse latent cosine scores with BM25 scores for hybrid retrieval."""
    if not enabled or not questions or documents is None:
        return latent_scores, None, None

    bm25_scores = compute_bm25_scores(
        questions=questions,
        documents=documents,
        device=latent_scores.device,
        dtype=latent_scores.dtype,
        k1=bm25_k1,
        b=bm25_b,
    )
    if bm25_scores.numel() == 0 or bm25_scores.shape != latent_scores.shape:
        return latent_scores, None, None

    latent_norm = ((latent_scores.float() + 1.0) * 0.5).clamp(0.0, 1.0).to(latent_scores.dtype)
    bm25_norm = normalize_bm25_scores(bm25_scores).to(latent_scores.dtype)
    alpha = compute_hybrid_alpha(
        questions,
        bm25_scores,
        documents=documents,
        fixed_alpha=fixed_alpha,
        adaptive=adaptive,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    ).to(latent_scores.dtype)
    fused_scores = alpha * latent_norm + (1.0 - alpha) * bm25_norm

    if candidate_top_m is not None and 0 < candidate_top_m < latent_scores.size(-1):
        top_m = min(candidate_top_m, latent_scores.size(-1))
        candidate_idx = latent_scores.topk(k=top_m, dim=-1).indices
        candidate_mask = torch.zeros_like(latent_scores, dtype=torch.bool)
        candidate_mask.scatter_(dim=-1, index=candidate_idx, value=True)
        fused_scores = torch.where(candidate_mask, fused_scores, latent_norm)

    return fused_scores.to(latent_scores.dtype), bm25_norm, alpha.squeeze(-1)
