#!/bin/bash
# Step 1: Extract probe features from a JSONL of annotated reasoning steps.
#
# Usage:
#   bash scripts/extract_features.sh <input_jsonl> <output_dir>
# Example:
#   bash scripts/extract_features.sh data/raw/math_correctness.jsonl \
#                                    data/task_0_math_correctness
#
# Env overrides:
#   BACKBONE   HF model id for the frozen hidden-state backbone
#   AGG_MODE   concat | mean | min | last     (default: concat)
#   N_TAIL     tail tokens to aggregate       (default: 5)
#   LAYER_IDX  layer index, -1 = last         (default: -1)
#   GAMMA      confidence threshold (Eq. 10)  (default: 0.8)
#   TRAIN_SIZE / TEST_SIZE                    (default: 800 / 200)
#   DEVICE     cuda | cpu                     (default: cuda if available)
#   DTYPE      fp16 | bf16 | fp32             (default: fp16)

set -euo pipefail

INPUT_JSONL=${1:?usage: $0 <input_jsonl> <output_dir>}
OUTPUT_DIR=${2:?usage: $0 <input_jsonl> <output_dir>}

BACKBONE=${BACKBONE:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
AGG_MODE=${AGG_MODE:-concat}
N_TAIL=${N_TAIL:-5}
LAYER_IDX=${LAYER_IDX:--1}
GAMMA=${GAMMA:-0.8}
TRAIN_SIZE=${TRAIN_SIZE:-800}
TEST_SIZE=${TEST_SIZE:-200}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-fp16}

python scripts/extract_features.py \
    --input_jsonl "$INPUT_JSONL" \
    --output_dir  "$OUTPUT_DIR" \
    --backbone    "$BACKBONE" \
    --agg_mode    "$AGG_MODE" \
    --n_tail      "$N_TAIL" \
    --layer_idx   "$LAYER_IDX" \
    --gamma       "$GAMMA" \
    --train_size  "$TRAIN_SIZE" \
    --test_size   "$TEST_SIZE" \
    --device      "$DEVICE" \
    --dtype       "$DTYPE"
