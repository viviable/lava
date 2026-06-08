#!/bin/bash
# Stage 5 — Continual-learning evaluation (Experiment A).
#
# Trains probes sequentially over the per-task feature dirs under DATA_DIR and
# reports ACC / BWT / FWT (BWT is always 0 for LAVA by construction). Each task
# dir holds train_features.pt / train_labels.pt / test_features.pt / test_labels.pt.
#
# Usage:
#   bash scripts/05_eval_continual.sh
#   DATA_DIR=data/tasks D_MODEL=1536 bash scripts/05_eval_continual.sh

set -euo pipefail

DATA_DIR=${DATA_DIR:-data/tasks}
D_MODEL=${D_MODEL:-1536}
N_TAIL=${N_TAIL:-5}
PROBE_TYPE=${PROBE_TYPE:-mlp}
EPOCHS=${EPOCHS:-100}
LR=${LR:-5e-3}
BATCH_SIZE=${BATCH_SIZE:-64}
THRESHOLD=${THRESHOLD:-0.5}
DEVICE=${DEVICE:-cpu}
OUTPUT=${OUTPUT:-results/continual.json}

echo "[stage 5] continual eval over $DATA_DIR -> $OUTPUT"
python scripts/eval_continual.py \
    --data_dir "$DATA_DIR" \
    --d_model "$D_MODEL" \
    --n_tail "$N_TAIL" \
    --probe_type "$PROBE_TYPE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --threshold "$THRESHOLD" \
    --device "$DEVICE" \
    --output "$OUTPUT" \
    --verbose
