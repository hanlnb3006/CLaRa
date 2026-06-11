#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}

export PROJECT_ROOT
export SAVE_PATH=${SAVE_PATH:-$PROJECT_ROOT/checkpoints/CLaRa-7B-E2E/compression-16}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints/CLaRa-7B-E2E}
export SAVE_MODEL_NAME=${SAVE_MODEL_NAME:-compression-16}
export EVAL_DATA_ROOT=${EVAL_DATA_ROOT:-$PROJECT_ROOT/evaluation/evaluation_data}
export DATASETS=${DATASETS:-musique}
export GENERATION_TOP_K=${GENERATION_TOP_K:-2}
export BATCH_SIZE=${BATCH_SIZE:-1}
export NUM_PROCESSES=${NUM_PROCESSES:-1}
export MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-20}
export DECODER_MODEL_NAME=${DECODER_MODEL_NAME:-mistralai/Mistral-7B-Instruct-v0.2}
export COMPR_BASE_MODEL_NAME=${COMPR_BASE_MODEL_NAME:-mistralai/Mistral-7B-Instruct-v0.2}
export QUANTIZATION=${QUANTIZATION:-int4}
export DEVICE_MAP=${DEVICE_MAP:-auto}
export HYBRID_CANDIDATE_TOP_M=${HYBRID_CANDIDATE_TOP_M:-5}

bash "$SCRIPT_DIR/evaluation_hybrid_ablation.sh"
