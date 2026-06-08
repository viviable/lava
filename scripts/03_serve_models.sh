#!/bin/bash
# Stage 3 — Launch the two vLLM servers (draft + strong) for inference.
#
# Starts both OpenAI-compatible vLLM servers in the background, writes logs to
# logs/, and records PIDs in logs/vllm_pids so you can stop them:
#     kill $(cat logs/vllm_pids)
#
# Wait until both logs print "Uvicorn running on ..." before running stage 4.
#
# Usage:
#   bash scripts/03_serve_models.sh
#   STRONG_TP=2 bash scripts/03_serve_models.sh

set -euo pipefail

DRAFT_MODEL=${DRAFT_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
STRONG_MODEL=${STRONG_MODEL:-Qwen/QwQ-32B}
DRAFT_PORT=${DRAFT_PORT:-30001}
STRONG_PORT=${STRONG_PORT:-30000}
STRONG_TP=${STRONG_TP:-2}      # tensor-parallel size for the strong model
LOG_DIR=${LOG_DIR:-logs}

mkdir -p "$LOG_DIR"
: > "$LOG_DIR/vllm_pids"

echo "[stage 3] starting draft server  $DRAFT_MODEL  (port $DRAFT_PORT)"
python -m vllm.entrypoints.openai.api_server \
    --model "$DRAFT_MODEL" \
    --port "$DRAFT_PORT" \
    --enable-prefix-caching \
    > "$LOG_DIR/draft_server.log" 2>&1 &
echo $! >> "$LOG_DIR/vllm_pids"

echo "[stage 3] starting strong server $STRONG_MODEL (port $STRONG_PORT, tp=$STRONG_TP)"
python -m vllm.entrypoints.openai.api_server \
    --model "$STRONG_MODEL" \
    --tensor-parallel-size "$STRONG_TP" \
    --port "$STRONG_PORT" \
    --enable-prefix-caching \
    > "$LOG_DIR/strong_server.log" 2>&1 &
echo $! >> "$LOG_DIR/vllm_pids"

echo "[stage 3] PIDs: $(cat "$LOG_DIR/vllm_pids" | tr '\n' ' ')"
echo "[stage 3] tail logs: tail -f $LOG_DIR/draft_server.log $LOG_DIR/strong_server.log"
echo "[stage 3] stop both: kill \$(cat $LOG_DIR/vllm_pids)"
