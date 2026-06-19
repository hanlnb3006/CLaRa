"""
Trainable gate for the Adaptive Compressor on Colab T4 (single GPU).

This script is a lightweight training loop that only updates the small
LoRA-r=8 gate on top of memory-token scores. The underlying CLaRa
checkpoint stays frozen. The objective mimics the stage-2 QA loss by
rewarding higher cosine similarity between the gold-document memory
tokens and the query latent state.

Usage (Colab T4):

    python scripts/train_adaptive_compressor_gate_colab_t4.py \
        --ckpt_path /content/CLaRa/checkpoints/CLaRa-7B-E2E/compression-16 \
        --decoder_model_name mistralai/Mistral-7B-Instruct-v0.2 \
        --compr_base_model_name mistralai/Mistral-7B-Instruct-v0.2 \
        --quantization int4 \
        --train_path /content/CLaRa/evaluation/evaluation_data/stage2/musique/train.jsonl \
        --output_dir /content/CLaRa/checkpoints/CLaRa-7B-E2E/compression-16-gate \
        --max_samples 200 \
        --epochs 1 \
        --learning_rate 5e-4 \
        --micro_batch_size 1
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
import torch.nn.functional as F

from openrlhf.models.modeling_clara import CLaRaConfig, CLaRa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Adaptive Compressor gate")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--decoder_model_name", type=str, required=True)
    parser.add_argument("--compr_base_model_name", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--quantization", type=str, default="int4")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.10)
    return parser.parse_args()


def load_train_records(path: str, max_samples: int) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            records.append(item)
            if len(records) >= max_samples:
                break
    return records


def freeze_decoder(model: CLaRa) -> None:
    for p in model.decoder.parameters():
        p.requires_grad = False
    if model.compr is not None:
        for p in model.compr.parameters():
            p.requires_grad = False


def select_trainable_params(model: CLaRa):
    return [p for p in model.adaptive_gate_lora.parameters() if p.requires_grad]


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    config = CLaRaConfig(
        decoder_model_name=args.decoder_model_name,
        compr_base_model_name=args.compr_base_model_name,
        compr_rate=16,
        doc_max_length=256,
        training_stage="stage2",
        generation_top_k=2,
        pure_inference=True,
        load_adapters=True,
        adaptive_compressor=True,
        adaptive_compressor_trainable=True,
        adaptive_compressor_lora_r=args.lora_r,
        adaptive_compressor_hidden=args.hidden_dim,
        adaptive_compressor_top_k=args.top_k,
        adaptive_compressor_strength=args.strength,
        adaptive_compressor_temperature=args.temperature,
    )

    print("Loading CLaRa checkpoint (this may take a few minutes)...")
    overrides = {
        "training_stage": "stage2",
        "generation_top_k": 2,
        "pure_inference": True,
        "load_adapters": True,
        "decoder_model_name": args.decoder_model_name,
        "compr_base_model_name": args.compr_base_model_name,
        "quantization": args.quantization,
        "adaptive_compressor": True,
        "adaptive_compressor_trainable": True,
        "adaptive_compressor_lora_r": args.lora_r,
        "adaptive_compressor_hidden": args.hidden_dim,
        "adaptive_compressor_top_k": args.top_k,
        "adaptive_compressor_strength": args.strength,
        "adaptive_compressor_temperature": args.temperature,
    }
    model = CLaRa.from_pretrained(args.ckpt_path, **overrides)
    model._setup_adaptive_compressor_gate()
    freeze_decoder(model)
    model.eval()

    if model.adaptive_gate_lora is None:
        raise RuntimeError("Gate failed to initialise. Check adaptive_compressor_trainable flag.")

    optimizer = torch.optim.AdamW(
        select_trainable_params(model),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    records = load_train_records(args.train_path, args.max_samples)
    print(f"Loaded {len(records)} training records")

    device = next(model.decoder.parameters()).device

    losses = []
    for epoch in range(args.epochs):
        for idx, record in enumerate(records):
            question = record["question"]
            documents = record["docs"]
            pos_index = record.get("pos_index", 0)

            q_tok = model._prepare_encoder_inputs([question], max_length=model.doc_max_length)
            query_reps = model._compr_query_reasoner_stage2(
                q_tok["input_ids"].to(device),
                q_tok["attention_mask"].to(device),
            )

            docs_input = model._prepare_encoder_inputs(documents, max_length=model.doc_max_length)
            doc_embeddings, _ = model.compress(
                docs_input["input_ids"].to(device),
                docs_input["attention_mask"].to(device),
            )
            doc_embeddings = doc_embeddings.view(
                doc_embeddings.size(0), -1, model.hidden_size
            )

            top_k = model.generation_top_k
            num_docs = doc_embeddings.size(0)
            if num_docs < top_k:
                continue

            scores = torch.einsum(
                "bd,nd->bn",
                F.normalize(query_reps.float(), dim=-1),
                F.normalize(doc_embeddings.float(), dim=-1),
            ).mean(dim=-1, keepdim=True).squeeze(-1)
            top_idx = torch.topk(scores, k=top_k).indices
            selected_docs = doc_embeddings[top_idx].unsqueeze(0)

            query_batched = query_reps.unsqueeze(0)
            selected_docs.requires_grad_(False)

            gated = model._apply_adaptive_compressor(selected_docs, query_batched)
            gated_norm = F.normalize(gated.float(), dim=-1)
            token_pool = gated_norm.mean(dim=2)
            score_after = torch.einsum(
                "bd,bnd->bn",
                F.normalize(query_reps.float(), dim=-1),
                token_pool,
            )
            target = torch.tensor([pos_index // max(1, num_docs // top_k)], device=device)
            target = target.clamp(max=score_after.size(1) - 1)
            loss = F.cross_entropy(score_after, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (idx + 1) % 25 == 0:
                avg = sum(losses[-25:]) / max(1, len(losses[-25:]))
                print(f"epoch={epoch} step={idx + 1} avg_loss={avg:.4f}")

    model.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned checkpoint to {args.output_dir}")


if __name__ == "__main__":
    train(parse_args())
