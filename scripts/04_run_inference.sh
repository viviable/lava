#!/bin/bash
# Stage 4 — Run LAVA speculative inference on one problem.
#
# Talks to the two vLLM servers from stage 3 and the probe bank from stage 2.
# Writes per-problem <dataset>_<id>/<repeat>.pickle (+ .txt) under OUTPUT_DIR.
#
# Usage:
#   bash scripts/04_run_inference.sh aime 60 0     # dataset, problem_id, repeat_id

set -euo pipefail

DATASET=${1:-aime}             # aime | math | gpqa
PROBLEM_ID=${2:-60}
REPEAT_ID=${3:-0}

DRAFT_MODEL=${DRAFT_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
DRAFT_PORT=${DRAFT_PORT:-30001}
STRONG_PORT=${STRONG_PORT:-30000}
PROBE_BANK=${PROBE_BANK:-probes/}
HIDDEN_BACKBONE=${HIDDEN_BACKBONE:-$DRAFT_MODEL}
OUTPUT_DIR=${OUTPUT_DIR:-results/lava}
TOKEN_BUDGET=${TOKEN_BUDGET:-8192}

echo "[stage 4] LAVA inference: $DATASET #$PROBLEM_ID (repeat $REPEAT_ID)"
python scripts/run_lava.py \
    --dataset_name "$DATASET" \
    --problem_id "$PROBLEM_ID" \
    --repeat_id "$REPEAT_ID" \
    --probe_bank "$PROBE_BANK" \
    --hidden_backbone "$HIDDEN_BACKBONE" \
    --draft_url "http://localhost:${DRAFT_PORT}/v1" \
    --strong_url "http://localhost:${STRONG_PORT}/v1" \
    --token_budget "$TOKEN_BUDGET" \
    --output_dir "$OUTPUT_DIR"
