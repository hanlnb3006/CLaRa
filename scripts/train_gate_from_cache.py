#!/usr/bin/env python3
"""Train the Adaptive Compressor gate from a precomputed embedding cache.

This script does NOT load Mistral 7B. It pulls (query_reps, doc_embeddings,
pos_index) tuples from a .pt cache produced by:
    scripts/encode_train_documents_colab.py

Because no Mistral forward is performed, the script runs comfortably on CPU
or any small GPU. The trained gate is saved as adaptive_gate.pth alongside
a copy of the base CLaRa checkpoint so `evaluate.py` can load it directly.

Usage:

    python scripts/train_gate_from_cache.py \
        --cache_path /content/CLaRa/checkpoints/CLaRa-7B-E2E/compression-16-gate/train_cache.pt \
        --ckpt_path /content/CLaRa/checkpoints/CLaRa-7B-E2E/compression-16 \
        --output_dir /content/CLaRa/checkpoints/CLaRa-7B-E2E/compression-16-gate \
        --epochs 3 \
        --learning_rate 5e-4 \
        --lora_r 8 \
        --temperature 0.10 \
        --strength 0.5 \
        --adaptive_top_k 3
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train gate from cached embeddings")
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Source CLaRa checkpoint; copied into output_dir minus the gate file")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=0,
                        help="Inner dim of the gate MLP (0 -> hidden_size // 8)")
    parser.add_argument("--retrieval_top_k", type=int, default=2,
                        help="Top-k docs passed to the gate, mirrors generation_top_k")
    parser.add_argument("--adaptive_top_k", type=int, default=3,
                        help="Top-k memory tokens per doc to keep after gating")
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    return parser.parse_args()


def build_gate(in_features: int, r: int, inner_dim: int) -> nn.ModuleDict:
    r = max(1, r)
    inner_dim = max(1, inner_dim or (in_features // 8))
    gate = nn.ModuleDict({
        "down": nn.Linear(in_features, r, bias=False),
        "up": nn.Linear(r, inner_dim, bias=False),
        "head": nn.Linear(inner_dim, 1, bias=False),
    })
    for module in gate.values():
        nn.init.normal_(module.weight, std=0.02)
    return gate


def copy_checkpoint_files(src_dir: str, dst_dir: str, skip: List[str]) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    skip_set = set(skip)
    for entry in os.listdir(src_dir):
        if entry in skip_set:
            continue
        src = os.path.join(src_dir, entry)
        dst = os.path.join(dst_dir, entry)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"Loading cache: {args.cache_path}")
    blob = torch.load(args.cache_path, map_location="cpu")
    samples: List[Dict] = blob["samples"]
    meta = blob["meta"]
    hidden_size: int = meta["hidden_size"]
    if not samples:
        raise RuntimeError("Cache contains 0 samples. Run encode script first.")
    print(f"Cache meta: hidden_size={hidden_size} dtype={meta.get('cache_dtype')} samples={len(samples)}")

    # Reshape logic relies on the model's contract:
    #   query_reps : (1, num_mem_tokens * hidden_size)
    #   doc_embeddings : (N, num_mem_tokens, hidden_size)
    src_query = samples[0]["query_reps"]
    src_doc = samples[0]["doc_embeddings"]
    num_mem_tokens = src_doc.size(1)
    print(f"Inferred num_mem_tokens={num_mem_tokens} (query_reps shape={tuple(src_query.shape)})")
    assert src_doc.size(2) == hidden_size, "cache doc last dim != hidden_size"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate = build_gate(hidden_size, args.lora_r, args.hidden_dim).to(device)
    print(
        f"Adaptive Compressor trainable gate: r={args.lora_r}, "
        f"inner_dim={args.hidden_dim or hidden_size // 8}, "
        f"params={sum(p.numel() for p in gate.parameters())}"
    )

    optimizer = torch.optim.AdamW(
        [p for p in gate.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    retrieval_top_k = args.retrieval_top_k
    adaptive_top_k = args.adaptive_top_k

    losses = []
    gate.train()
    for epoch in range(args.epochs):
        for idx, sample in enumerate(samples):
            query_flat = sample["query_reps"].to(device).float()  # (1, num_mem*hidden)
            doc_embs = sample["doc_embeddings"].to(device).float()  # (N, num_mem, hidden)
            pos_index = sample["pos_index"]

            query_mem = query_flat.view(1, num_mem_tokens, hidden_size)  # (1, num_mem, hidden)
            query_latent = query_mem.mean(dim=1)  # (1, hidden)

            num_docs = doc_embs.size(0)
            if num_docs < retrieval_top_k:
                continue

            doc_means = doc_embs.mean(dim=1)  # (N, hidden)
            scores = torch.einsum(
                "bh,nh->bn",
                F.normalize(query_latent, dim=-1),
                F.normalize(doc_means, dim=-1),
            ).squeeze(0)
            top_idx = torch.topk(scores, k=retrieval_top_k).indices
            selected_docs = doc_embs[top_idx].unsqueeze(0)  # (1, K, num_mem, hidden)

            B, K, T, H = selected_docs.shape
            flat = selected_docs.reshape(B * K * T, H)
            gate_scores = gate["head"](gate["up"](gate["down"](flat))).reshape(B, K, T)

            mode = "topk" if adaptive_top_k > 0 else "softmax"
            if mode == "topk":
                keep = min(adaptive_top_k, T)
                top_idx_t = gate_scores.topk(k=keep, dim=-1).indices
                token_mask = torch.zeros_like(gate_scores)
                token_mask.scatter_(dim=-1, index=top_idx_t, value=1.0)
                weights = token_mask
            else:
                temperature = max(args.temperature, 1e-4)
                weights = F.softmax(gate_scores / temperature, dim=-1)

            strength = max(0.0, min(args.strength, 1.0))
            weights = (1.0 - strength) + strength * weights

            gated_docs = selected_docs * weights.unsqueeze(-1)
            gated_pool = gated_docs.mean(dim=2)  # (1, K, hidden)
            score_after = torch.einsum(
                "bh,bkh->bk",
                F.normalize(query_latent, dim=-1),
                F.normalize(gated_pool, dim=-1),
            ).squeeze(0)

            target_val = pos_index // max(1, num_docs // retrieval_top_k)
            target_val = max(0, min(retrieval_top_k - 1, target_val))
            target = torch.tensor([target_val], device=device)
            loss = F.cross_entropy(score_after.unsqueeze(0), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (idx + 1) % args.log_every == 0:
                avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
                print(f"epoch={epoch} step={idx + 1} loss={loss.item():.4f} avg={avg:.4f}")

    gate_path = os.path.join(args.output_dir, "adaptive_gate.pth")
    torch.save(
        {k: v.detach().cpu() for k, v in gate.state_dict().items()},
        gate_path,
    )
    print(f"Saved trained gate to {gate_path}")

    copy_checkpoint_files(
        args.ckpt_path,
        args.output_dir,
        skip=["adaptive_gate.pth"],
    )
    print(f"Copied base checkpoint from {args.ckpt_path} into {args.output_dir}")


if __name__ == "__main__":
    main()