# Ablation script for Better Differentiable Top-k relaxations.
# Runs baseline (iterative_st) and three alternatives: sparsemax, entmax15, gumbel_st.
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
GOLD_RETRIEVAL_FLAG=${GOLD_RETRIEVAL_FLAG:---gold_retrieval}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-20}
DECODER_MODEL_NAME=${DECODER_MODEL_NAME:-mistralai/Mistral-7B-Instruct-v0.2}
COMPR_BASE_MODEL_NAME=${COMPR_BASE_MODEL_NAME:-mistralai/Mistral-7B-Instruct-v0.2}
QUANTIZATION=${QUANTIZATION:-int4}
DEVICE_MAP=${DEVICE_MAP:-auto}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
TOPK_METHOD=${TOPK_METHOD:-}

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
if [ -n "$QUANTIZATION" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --quantization $QUANTIZATION"
fi
if [ -n "$DEVICE_MAP" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --device_map $DEVICE_MAP"
fi
if [ -n "$ATTN_IMPLEMENTATION" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --attn_implementation $ATTN_IMPLEMENTATION"
fi
if [ -n "$TOPK_METHOD" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --topk_method $TOPK_METHOD"
fi

export PYTHONPATH="$PROJECT_ROOT:$SAVE_PATH:$PYTHONPATH"

COMMON_ARGS="evaluation/evaluate.py
    --model_path $SAVE_MODEL_NAME
    --checkpoint_root $CHECKPOINT_ROOT
    --eval_data_root $EVAL_DATA_ROOT
    --stage stage2
    --dataset $DATASETS
    --generation_top_k $GENERATION_TOP_K
    --batch_size $BATCH_SIZE
    $GOLD_RETRIEVAL_FLAG
    $OPTIONAL_ARGS"

run_method() {
    local label=$1
    local method=$2
    echo "==> Running topk_method=$label"
    local args=()
    if [ -n "$method" ]; then
        args=(--topk_method "$method")
    fi
    accelerate launch
        --num_processes=$NUM_PROCESSES
        --num_machines=1
        --mixed_precision=$MIXED_PRECISION
        $COMMON_ARGS
        "${args[@]}"
}

run_method "iterative_st" "iterative_st"
run_method "sparsemax" "sparsemax"
run_method "entmax15" "entmax15"
run_method "gumbel_st" "gumbel_st"

echo "Topk relaxation ablation completed."