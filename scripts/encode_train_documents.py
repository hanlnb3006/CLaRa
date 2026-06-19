#!/usr/bin/env python3
"""Precompute question + document memory-token embeddings for gate training.

This script runs only the ENCODER half of CLaRa (Mistral 7B int4 forward).
It does NOT backpropagate anything, so the Colab T4 (15 GB) is enough as long
as we keep compressor_batch_size small. The output cache is a single .pt file
consumed by `scripts/train_gate_from_cache.py`, which trains the small gate
without ever re-loading Mistral.

Usage on Colab:

    python scripts/encode_train_documents_colab.py \
        --ckpt_path /content/CLaRa/checkpoints/CLaRa-7B-E2E/compression-16 \
        --decoder_model_name mistralai/Mistral-7B-Instruct-v0.2 \
        --compr_base_model_name mistralai/Mistral-7B-Instruct-v0.2 \
        --quantization int4 \
        --train_path /content/CLaRa/evaluation/evaluation_data/stage2/musique/train.jsonl \
        --cache_path /content/CLaRa/checkpoints/CLaRa-7B-E2E/compression-16-gate/train_cache.pt \
        --max_samples 200 \
        --doc_max_length 128 \
        --compression_batch_size 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import torch

from openrlhf.models.modeling_clara import CLaRaConfig, CLaRa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode train docs for adaptive-compressor gate")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--decoder_model_name", type=str, required=True)
    parser.add_argument("--compr_base_model_name", type=str, required=True)
    parser.add_argument("--quantization", type=str, default="int4")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--doc_max_length", type=int, default=128)
    parser.add_argument("--compression_batch_size", type=int, default=4)
    parser.add_argument("--dtype_cache", choices=["fp16", "bf16", "fp32"], default="fp16",
                        help="dtype used to store cached embeddings")
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


def compress_documents_in_chunks(
    model: CLaRa,
    documents: List[str],
    max_length: int,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode documents chunk-by-chunk so activation memory stays bounded on T4."""
    chunks = []
    for start in range(0, len(documents), max(chunk_size, 1)):
        chunk = documents[start:start + chunk_size]
        if not chunk:
            continue
        inp = model._prepare_encoder_inputs(chunk, max_length=max_length)
        emb, _ = model.compress(
            inp["input_ids"].to(device),
            inp["attention_mask"].to(device),
        )
        emb = emb.view(emb.size(0), -1, model.hidden_size)
        chunks.append(emb)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks, dim=0)


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.cache_path), exist_ok=True)

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    cache_dtype = dtype_map[args.dtype_cache]

    print("Loading CLaRa checkpoint (encoder-only path)...")
    overrides = {
        "training_stage": "stage2",
        "generation_top_k": 2,
        "pure_inference": True,
        "load_adapters": True,
        "decoder_model_name": args.decoder_model_name,
        "compr_base_model_name": args.compr_base_model_name,
        "quantization": args.quantization,
    }
    model = CLaRa.from_pretrained(args.ckpt_path, **overrides)
    model.eval()
    if hasattr(model.decoder, "config") and getattr(model.decoder.config, "use_cache", None) is True:
        model.decoder.config.use_cache = False

    device = next(model.decoder.parameters()).device

    records = load_train_records(args.train_path, args.max_samples)
    print(f"Loaded {len(records)} training records to encode")

    cache_samples: List[Dict] = []
    t0 = time.time()
    written = 0
    with torch.inference_mode():
        for idx, record in enumerate(records):
            try:
                question = record["question"]
                documents = record["docs"]
                pos_index = record.get("pos_index", 0)
                if not isinstance(pos_index, int):
                    pos_index = int(pos_index)
                if not documents or pos_index >= len(documents):
                    print(f"[skip] idx={idx} bad pos_index={pos_index} n_docs={len(documents)}")
                    continue

                q_tok = model._prepare_encoder_inputs([question], max_length=model.doc_max_length)
                query_reps = model._compr_query_reasoner_stage2(
                    q_tok["input_ids"].to(device),
                    q_tok["attention_mask"].to(device),
                )

                doc_embeddings = compress_documents_in_chunks(
                    model,
                    documents,
                    model.doc_max_length,
                    args.compression_batch_size,
                    device,
                )
                if doc_embeddings.numel() == 0 or doc_embeddings.size(0) < model.generation_top_k:
                    print(f"[skip] idx={idx} not enough docs ({doc_embeddings.size(0)})")
                    continue

                cache_samples.append({
                    "query_reps": query_reps.detach().to(cache_dtype).cpu(),
                    "doc_embeddings": doc_embeddings.detach().to(cache_dtype).cpu(),
                    "pos_index": pos_index,
                })
                written += 1
                if (idx + 1) % 10 == 0 or (idx + 1) == len(records):
                    elapsed = time.time() - t0
                    print(
                        f"encoded {idx + 1}/{len(records)} kept={written} "
                        f"elapsed={elapsed:.1f}s avg_per_sample={elapsed / max(1, idx + 1):.2f}s"
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as oom:
                print(f"[OOM] idx={idx} skipping sample. {oom!r}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

    blob = {
        "samples": cache_samples,
        "meta": {
            "hidden_size": int(model.hidden_size),
            "doc_max_length": int(args.doc_max_length),
            "compression_batch_size": int(args.compression_batch_size),
            "cache_dtype": args.dtype_cache,
        },
    }
    torch.save(blob, args.cache_path)
    mb = sum(s["doc_embeddings"].numel() * cache_samples[0]["doc_embeddings"].element_size()
             + s["query_reps"].numel() * cache_samples[0]["query_reps"].element_size()
             for s in cache_samples) / 1e6 if cache_samples else 0
    print(f"Saved {len(cache_samples)} samples to {args.cache_path} (~{mb:.1f} MB)")


if __name__ == "__main__":
    main()