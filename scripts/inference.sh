#!/bin/bash
# Launch vLLM servers + LAVA runner. Mirrors specreason/spec_reason_della.sh.
#
# Usage:
#   bash scripts/inference.sh aime 60 0       # dataset, problem_id, repeat_id

set -euo pipefail

DATASET=${1:-aime}
PROBLEM_ID=${2:-60}
REPEAT_ID=${3:-0}

DRAFT_MODEL=${DRAFT_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
STRONG_MODEL=${STRONG_MODEL:-Qwen/QwQ-32B}
DRAFT_PORT=${DRAFT_PORT:-30001}
STRONG_PORT=${STRONG_PORT:-30000}
PROBE_BANK=${PROBE_BANK:-probes/}
OUTPUT_DIR=${OUTPUT_DIR:-results/lava}

python scripts/run_lava.py \
    --dataset_name "$DATASET" \
    --problem_id "$PROBLEM_ID" \
    --repeat_id "$REPEAT_ID" \
    --probe_bank "$PROBE_BANK" \
    --hidden_backbone "$DRAFT_MODEL" \
    --draft_url "http://localhost:${DRAFT_PORT}/v1" \
    --strong_url "http://localhost:${STRONG_PORT}/v1" \
    --output_dir "$OUTPUT_DIR"
