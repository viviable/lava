#!/bin/bash
# Continual-learning eval (Experiment A): sequential concept learning, ACC/BWT/FWT.
#
# Usage:
#   bash scripts/eval_continual.sh <data_dir> [<output_json>]
# Example:
#   bash scripts/eval_continual.sh data/ results/continual.json
#
# <data_dir> must contain one subdirectory per task, each with
# {train,test}_features.pt and {train,test}_labels.pt (the layout
# extract_features.sh produces; preference tasks omit train_labels.pt).
#
# Env overrides:
#   D_MODEL       backbone hidden dim   (default: 1536)
#   PROBE_TYPE    mlp | linear          (default: mlp)
#   N_TAIL                              (default: 5)
#   EPOCHS / LR / BATCH_SIZE / THRESHOLD (defaults: 100 / 5e-3 / 64 / 0.5)
#   DEVICE        cuda | cpu            (default: cuda)
#   VERBOSE       non-empty -> --verbose

set -euo pipefail

DATA_DIR=${1:?usage: $0 <data_dir> [<output_json>]}
OUTPUT=${2:-results/continual.json}

D_MODEL=${D_MODEL:-1536}
PROBE_TYPE=${PROBE_TYPE:-mlp}
N_TAIL=${N_TAIL:-5}
EPOCHS=${EPOCHS:-100}
LR=${LR:-5e-3}
BATCH_SIZE=${BATCH_SIZE:-64}
THRESHOLD=${THRESHOLD:-0.5}
DEVICE=${DEVICE:-cuda}

EXTRA=()
if [[ -n "${VERBOSE:-}" ]]; then
    EXTRA+=(--verbose)
fi

python scripts/eval_continual.py \
    --data_dir   "$DATA_DIR" \
    --d_model    "$D_MODEL" \
    --probe_type "$PROBE_TYPE" \
    --n_tail     "$N_TAIL" \
    --epochs     "$EPOCHS" \
    --lr         "$LR" \
    --batch_size "$BATCH_SIZE" \
    --threshold  "$THRESHOLD" \
    --device     "$DEVICE" \
    --output     "$OUTPUT" \
    "${EXTRA[@]}"
