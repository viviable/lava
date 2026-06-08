#!/bin/bash
# Stage 1 — Feature extraction.
#
# Run a FROZEN HF backbone over labeled reasoning data (JSONL) and dump the
# per-step hidden-state features + labels that the probes train on (Sec 3.2.2).
# By default ALL layers are saved as (N, L+1, d_in) so stage 2 can search L*.
#
# Usage:
#   bash scripts/01_extract_features.sh
#   INPUT=data/my_steps.jsonl DEVICE=cuda bash scripts/01_extract_features.sh
#
# Override any default with an env var (shown below).

set -euo pipefail

MODEL=${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
INPUT=${INPUT:-data/math_steps.jsonl}
OUT_FEATURES=${OUT_FEATURES:-data/features.pt}
OUT_LABELS=${OUT_LABELS:-data/labels.pt}
MODE=${MODE:-concat}          # concat | pooling | min | last
N_TAIL=${N_TAIL:-5}
GAMMA=${GAMMA:-0.8}           # confidence-denoising threshold (Eq. 10)
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-float16}
# Set SINGLE_LAYER=1 to save only one layer (N, d_in) instead of all layers.
SINGLE_LAYER=${SINGLE_LAYER:-0}
LAYER_IDX=${LAYER_IDX:--1}

extra=()
if [ "$SINGLE_LAYER" = "1" ]; then
    extra+=(--single_layer --layer_idx "$LAYER_IDX")
fi

echo "[stage 1] extracting features: $INPUT -> $OUT_FEATURES / $OUT_LABELS"
python scripts/extract_features.py \
    --model "$MODEL" \
    --input "$INPUT" \
    --out_features "$OUT_FEATURES" \
    --out_labels "$OUT_LABELS" \
    --mode "$MODE" \
    --n_tail "$N_TAIL" \
    --gamma "$GAMMA" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    "${extra[@]}"
