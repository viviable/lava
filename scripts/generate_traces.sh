#!/bin/bash
# Step 0a: Generate raw CoT traces with a vLLM-served model.
#
# Produces a JSONL of {id, problem, completion, gold_answer} rows — the input
# scripts/build_probe_dataset.sh expects. This is the bootstrap step: it gives
# you the first traces before any probe exists (build_probe_dataset labels
# traces but doesn't create them, and run_lava needs a trained probe bank).
#
# Pipeline position:
#   generate_traces.sh -> build_probe_dataset.sh -> extract_features.sh
#                      -> train_probe.sh -> inference.sh
#
# Setup (one vLLM server is enough for plain CoT):
#   python -m vllm.entrypoints.openai.api_server \
#       --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
#       --port 30001 --enable-prefix-caching
#
# Usage:
#   bash scripts/generate_traces.sh <dataset_name> <output_jsonl>
#
# Examples:
#   bash scripts/generate_traces.sh hmmt data/raw/hmmt_traces.jsonl
#   LIMIT=30 NUM_SAMPLES=4 bash scripts/generate_traces.sh aime data/raw/aime_traces.jsonl
#
# <dataset_name>: gsm8k | aime | math500 | hmmt
#
# Env overrides:
#   BASE_URL      vLLM OpenAI endpoint     (default: http://localhost:30001/v1)
#   MODEL         override model name      (default: autodiscovered)
#   MAX_TOKENS    max tokens per trace     (default: 8192)
#   TEMPERATURE   sampling temperature     (default: 0.6)
#   TOP_P         nucleus top-p            (default: 0.95)
#   NUM_SAMPLES   traces per problem       (default: 1)
#   LIMIT         cap # problems           (default: all)
#   START         first problem index      (default: 0)
#   MAX_WORKERS   concurrent requests      (default: 4)
#
# Requires: pip install openai datasets

set -euo pipefail

DATASET=${1:?usage: $0 <dataset_name> <output_jsonl>}
OUTPUT=${2:?usage: $0 <dataset_name> <output_jsonl>}

BASE_URL=${BASE_URL:-http://localhost:30001/v1}
MAX_TOKENS=${MAX_TOKENS:-8192}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
NUM_SAMPLES=${NUM_SAMPLES:-1}
START=${START:-0}
MAX_WORKERS=${MAX_WORKERS:-4}

EXTRA=()
if [[ -n "${MODEL:-}" ]]; then
    EXTRA+=(--model "$MODEL")
fi
if [[ -n "${LIMIT:-}" ]]; then
    EXTRA+=(--limit "$LIMIT")
fi

python scripts/generate_traces.py \
    --dataset_name "$DATASET" \
    --output_jsonl "$OUTPUT" \
    --base_url     "$BASE_URL" \
    --max_tokens   "$MAX_TOKENS" \
    --temperature  "$TEMPERATURE" \
    --top_p        "$TOP_P" \
    --num_samples  "$NUM_SAMPLES" \
    --start        "$START" \
    --max_workers  "$MAX_WORKERS" \
    "${EXTRA[@]}"
