#!/bin/bash
# Smoke eval for Block C: 1 method, 1 dataset, 2 samples.
# Use to verify the new topk_method flag works end-to-end.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
cd "$PROJECT_ROOT"
unset PYTHONPATH

SAVE_PATH=${SAVE_PATH:-$PROJECT_ROOT/checkpoints/CLaRa-7B-E2E/compression-16}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints/CLaRa-7B-E2E}
SAVE_MODEL_NAME=${SAVE_MODEL_NAME:-${SAVE_PATH##*/}}
EVAL_DATA_ROOT=${EVAL_DATA_ROOT:-$PROJECT_ROOT/evaluation/evaluation_data}
DATASETS=${DATASETS:-musique}
GENERATION_TOP_K=${GENERATION_TOP_K:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_PROCESSES=${NUM_PROCESSES:-1}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-2}
DECODER_MODEL_NAME=${DECODER_MODEL_NAME:-mistralai/Mistral-7B-Instruct-v0.2}
COMPR_BASE_MODEL_NAME=${COMPR_BASE_MODEL_NAME:-mistralai/Mistral-7B-Instruct-v0.2}
QUANTIZATION=${QUANTIZATION:-int4}
DEVICE_MAP=${DEVICE_MAP:-auto}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-}
TOPK_METHOD=${TOPK_METHOD:-sparsemax}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}

OPTIONAL_ARGS=""
[ -n "$MAX_EVAL_SAMPLES" ]      && OPTIONAL_ARGS="$OPTIONAL_ARGS --max_eval_samples $MAX_EVAL_SAMPLES"
[ -n "$DECODER_MODEL_NAME" ]    && OPTIONAL_ARGS="$OPTIONAL_ARGS --decoder_model_name $DECODER_MODEL_NAME"
[ -n "$COMPR_BASE_MODEL_NAME" ] && OPTIONAL_ARGS="$OPTIONAL_ARGS --compr_base_model_name $COMPR_BASE_MODEL_NAME"
[ -n "$QUANTIZATION" ]          && OPTIONAL_ARGS="$OPTIONAL_ARGS --quantization $QUANTIZATION"
[ -n "$DEVICE_MAP" ]            && OPTIONAL_ARGS="$OPTIONAL_ARGS --device_map $DEVICE_MAP"
[ -n "$ATTN_IMPLEMENTATION" ]   && OPTIONAL_ARGS="$OPTIONAL_ARGS --attn_implementation $ATTN_IMPLEMENTATION"
[ -n "$TOPK_METHOD" ]           && OPTIONAL_ARGS="$OPTIONAL_ARGS --topk_method $TOPK_METHOD"

export PYTHONPATH="$PROJECT_ROOT:$SAVE_PATH:$PYTHONPATH"

COMMON_ARGS="evaluation/evaluate.py
    --model_path $SAVE_MODEL_NAME
    --checkpoint_root $CHECKPOINT_ROOT
    --eval_data_root $EVAL_DATA_ROOT
    --stage stage2
    --dataset $DATASETS
    --generation_top_k $GENERATION_TOP_K
    --batch_size $BATCH_SIZE
    --gold_retrieval
    $OPTIONAL_ARGS"

echo "Running topk smoke: method=$TOPK_METHOD max_eval=$MAX_EVAL_SAMPLES dataset=$DATASETS"
accelerate launch
    --num_processes=$NUM_PROCESSES
    --num_machines=1
    --mixed_precision=$MIXED_PRECISION
    "$COMMON_ARGS"