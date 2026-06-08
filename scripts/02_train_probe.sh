#!/bin/bash
# Stage 2 — Train one concept probe.
#
# Trains a probe on the features from stage 1 and saves a ProbeBank directory.
# If the features are all-layer (N, L+1, d_in) the best layer L* is searched
# automatically; --d_model is auto-corrected from the feature shape.
#
# Usage:
#   bash scripts/02_train_probe.sh
#   CONCEPT=math_correctness PROBE_TYPE=mlp bash scripts/02_train_probe.sh

set -euo pipefail

FEATURES=${FEATURES:-data/features.pt}
LABELS=${LABELS:-data/labels.pt}
CONCEPT=${CONCEPT:-math_correctness}
PROBE_TYPE=${PROBE_TYPE:-mlp}              # mlp | linear
SUPERVISION=${SUPERVISION:-classification} # classification | preference
D_MODEL=${D_MODEL:-1536}                   # DeepSeek-R1-Distill-Qwen-1.5B hidden size
N_TAIL=${N_TAIL:-5}
THRESHOLD=${THRESHOLD:-0.5}
LR=${LR:-5e-3}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-64}
DEVICE=${DEVICE:-cpu}
OUTPUT=${OUTPUT:-probes/}
# Optional held-out eval: set TEST_FEATURES + TEST_LABELS to print test metrics.
TEST_FEATURES=${TEST_FEATURES:-}
TEST_LABELS=${TEST_LABELS:-}

extra=()
if [ "$SUPERVISION" = "classification" ]; then
    extra+=(--labels "$LABELS")
fi
if [ -n "$TEST_FEATURES" ] && [ -n "$TEST_LABELS" ]; then
    extra+=(--test_features "$TEST_FEATURES" --test_labels "$TEST_LABELS")
fi

echo "[stage 2] training '$CONCEPT' probe ($PROBE_TYPE) -> $OUTPUT"
python scripts/train_probe.py \
    --features "$FEATURES" \
    --concept "$CONCEPT" \
    --probe_type "$PROBE_TYPE" \
    --supervision "$SUPERVISION" \
    --d_model "$D_MODEL" \
    --n_tail "$N_TAIL" \
    --threshold "$THRESHOLD" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --output "$OUTPUT" \
    --verbose \
    "${extra[@]}"
