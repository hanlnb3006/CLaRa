"""Trainable gate for the Adaptive Compressor on Colab T4 (single GPU).

This script is a lightweight training loop that only updates the small
LoRA-r=8 gate on top of memory-token scores.
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
    parser.add_argument("--train_doc_max_length", type=int, default=128)
    parser.add_argument("--compression_batch_size", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def load_train_records(path: str, max_samples: int):
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


def freeze_decoder(model):
    for p in model.decoder.parameters():
        p.requires_grad = False
    if model.compr is not None:
        for p in model.compr.parameters():
            p.requires_grad = False


def select_trainable_params(model):
    return [p for p in model.adaptive_gate_lora.parameters() if p.requires_grad]


def enable_gradient_checkpointing(model):
    if hasattr(model.decoder, "gradient_checkpointing_enable"):
        try:
            model.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.decoder.gradient_checkpointing_enable()
    if hasattr(model.decoder, "config") and getattr(model.decoder.config, "use_cache", None) is True:
        model.decoder.config.use_cache = False
    try:
        if hasattr(model.decoder, "enable_input_require_grads"):
            model.decoder.enable_input_require_grads()
    except Exception:
        pass


def compress_documents_in_chunks(model, documents, max_length, chunk_size, device):
    all_embeddings = []
    for start in range(0, len(documents), max(chunk_size, 1)):
        chunk = documents[start:start + chunk_size]
        if not chunk:
            continue
        inp = model._prepare_encoder_inputs(chunk, max_length=max_length)
        emb, _ = model.compress(inp["input_ids"].to(device), inp["attention_mask"].to(device))
        emb = emb.view(emb.size(0), -1, model.hidden_size)
        all_embeddings.append(emb)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not all_embeddings:
        return torch.empty(0)
    return torch.cat(all_embeddings, dim=0)


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    config = CLaRaConfig(
        decoder_model_name=args.decoder_model_name,
        compr_base_model_name=args.compr_base_model_name,
        compr_rate=16,
        doc_max_length=args.train_doc_max_length,
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
    enable_gradient_checkpointing(model)
    if model.adaptive_gate_lora is not None:
        model.adaptive_gate_lora.train()

    if model.adaptive_gate_lora is None:
        raise RuntimeError("Gate failed to initialise.")

    optimizer = torch.optim.AdamW(select_trainable_params(model), lr=args.learning_rate, weight_decay=0.0)

    records = load_train_records(args.train_path, args.max_samples)
    print(f"Loaded {len(records)} training records")

    device = next(model.decoder.parameters()).device
    losses = []
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        for idx, record in enumerate(records):
            try:
                question = record["question"]
                documents = record["docs"]
                pos_index = record.get("pos_index", 0)

                q_tok = model._prepare_encoder_inputs([question], max_length=model.doc_max_length)
                query_reps = model._compr_query_reasoner_stage2(
                    q_tok["input_ids"].to(device),
                    q_tok["attention_mask"].to(device),
                )

                doc_embeddings = compress_documents_in_chunks(
                    model, documents, model.doc_max_length, args.compression_batch_size, device
                )
                if doc_embeddings.numel() == 0:
                    continue

                top_k = model.generation_top_k
                num_docs = doc_embeddings.size(0)
                if num_docs < top_k:
                    continue

                scores = torch.einsum(
                    "bd,nd->bn",
                    F.normalize(query_reps.float(), dim=-1),
                    F.normalize(doc_embeddings.detach().float(), dim=-1),
                ).mean(dim=-1, keepdim=True).squeeze(-1)
                top_idx = torch.topk(scores, k=top_k).indices
                selected_docs = doc_embeddings[top_idx].unsqueeze(0)

                query_batched = query_reps.unsqueeze(0)
                gated = model._apply_adaptive_compressor(selected_docs, query_batched)
                gated_norm = F.normalize(gated.float(), dim=-1)
                token_pool = gated_norm.mean(dim=2)
                score_after = torch.einsum(
                    "bd,bnd->bn",
                    F.normalize(query_reps.float(), dim=-1).detach(),
                    token_pool,
                )
                target = torch.tensor([pos_index // max(1, num_docs // top_k)], device=device)
                target = target.clamp(max=score_after.size(1) - 1)
                loss = F.cross_entropy(score_after, target)
                loss_value = float(loss.item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss_value)
                if (idx + 1) % args.log_every == 0:
                    avg = sum(losses[-args.log_every:]) / max(1, len(losses[-args.log_every:]))
                    print(f"epoch={epoch} step={idx + 1} loss={loss_value:.4f} avg_loss={avg:.4f}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as oom:
                print(f"[OOM] step={idx + 1} skipping sample. {oom!r}")
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

    model.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned checkpoint to {args.output_dir}")


if __name__ == "__main__":
    train(parse_args())