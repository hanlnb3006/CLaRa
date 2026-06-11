#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}

cd "$PROJECT_ROOT"
unset PYTHONPATH

SAVE_PATH=${SAVE_PATH:-/mnt/task_wrapper/user_output/artifacts/data/train_checkpoint/xx}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/mnt/task_wrapper/user_output/artifacts/data/train_checkpoint}
EVAL_DATA_ROOT=${EVAL_DATA_ROOT:-$PROJECT_ROOT/evaluation/evaluation_data}
SAVE_MODEL_NAME=${SAVE_MODEL_NAME:-${SAVE_PATH##*/}}
DATASETS=${DATASETS:-musique,hotpotqa,2wiki,nq}
GENERATION_TOP_K=${GENERATION_TOP_K:-5}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_PROCESSES=${NUM_PROCESSES:-8}
GOLD_RETRIEVAL_FLAG=${GOLD_RETRIEVAL_FLAG:---gold_retrieval}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-}
DECODER_MODEL_NAME=${DECODER_MODEL_NAME:-}
COMPR_BASE_MODEL_NAME=${COMPR_BASE_MODEL_NAME:-}
COMPR_MODEL_NAME=${COMPR_MODEL_NAME:-}
QUANTIZATION=${QUANTIZATION:-}
DEVICE_MAP=${DEVICE_MAP:-}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-}

OPTIONAL_ARGS=""
if [ -n "$MAX_EVAL_SAMPLES" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --max_eval_samples $MAX_EVAL_SAMPLES"
fi
if [ -n "$DECODER_MODEL_NAME" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --decoder_model_name $DECODER_MODEL_NAME"
fi
if [ -n "$COMPR_BASE_MODEL_NAME" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --compr_base_model_name $COMPR_BASE_MODEL_NAME"
fi
if [ -n "$COMPR_MODEL_NAME" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --compr_model_name $COMPR_MODEL_NAME"
fi
if [ -n "$QUANTIZATION" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --quantization $QUANTIZATION"
fi
if [ -n "$DEVICE_MAP" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --device_map $DEVICE_MAP"
fi
if [ -n "$ATTN_IMPLEMENTATION" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --attn_implementation $ATTN_IMPLEMENTATION"
fi

export PYTHONPATH="$PROJECT_ROOT:$SAVE_PATH:$PYTHONPATH"

COMMON_ARGS="evaluation/evaluate.py \
    --model_path $SAVE_MODEL_NAME \
    --checkpoint_root $CHECKPOINT_ROOT \
    --eval_data_root $EVAL_DATA_ROOT \
    --stage stage2 \
    --dataset $DATASETS \
    --generation_top_k $GENERATION_TOP_K \
    --batch_size $BATCH_SIZE \
    $GOLD_RETRIEVAL_FLAG \
    $OPTIONAL_ARGS"

echo "Running baseline CLaRa latent retrieval..."
accelerate launch \
    --num_processes=$NUM_PROCESSES \
    --num_machines=1 \
    --mixed_precision=bf16 \
    $COMMON_ARGS

echo "Running fixed hybrid retrieval..."
accelerate launch \
    --num_processes=$NUM_PROCESSES \
    --num_machines=1 \
    --mixed_precision=bf16 \
    $COMMON_ARGS \
    --hybrid_retrieval \
    --hybrid_alpha 0.75

echo "Running adaptive hybrid retrieval..."
accelerate launch \
    --num_processes=$NUM_PROCESSES \
    --num_machines=1 \
    --mixed_precision=bf16 \
    $COMMON_ARGS \
    --hybrid_retrieval \
    --hybrid_adaptive_fusion \
    --hybrid_alpha_min 0.45 \
    --hybrid_alpha_max 0.90

echo "Hybrid retrieval ablation completed."
