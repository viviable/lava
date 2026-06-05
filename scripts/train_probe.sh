#!/bin/bash
# Step 2: Train a single LAVA probe from extracted features.
#
# Usage:
#   bash scripts/train_probe.sh <features_dir> <concept_name> <probe_bank_dir>
# Example:
#   bash scripts/train_probe.sh data/task_0_math_correctness \
#                               math_correctness \
#                               probes/
#
# Expects <features_dir>/train_features.pt, train_labels.pt, test_features.pt,
# test_labels.pt (the layout produced by extract_features.sh).
#
# Env overrides:
#   D_MODEL       backbone hidden dim   (default: 1536, DeepSeek-R1-Distill-1.5B)
#   PROBE_TYPE    mlp | linear          (default: mlp)
#   SUPERVISION   classification | preference   (default: classification)
#   N_TAIL        tail tokens                   (default: 5)
#   THRESHOLD     accept threshold              (default: 0.5)
#   LR / EPOCHS / BATCH_SIZE / WEIGHT_DECAY     (defaults: 5e-3 / 100 / 64 / 1e-4)
#   DEVICE        cuda | cpu                    (default: cuda)
#   VERBOSE       any non-empty value -> --verbose

set -euo pipefail

FEATURES_DIR=${1:?usage: $0 <features_dir> <concept_name> <probe_bank_dir>}
CONCEPT=${2:?usage: $0 <features_dir> <concept_name> <probe_bank_dir>}
OUTPUT=${3:?usage: $0 <features_dir> <concept_name> <probe_bank_dir>}

D_MODEL=${D_MODEL:-1536}
PROBE_TYPE=${PROBE_TYPE:-mlp}
SUPERVISION=${SUPERVISION:-classification}
N_TAIL=${N_TAIL:-5}
THRESHOLD=${THRESHOLD:-0.5}
LR=${LR:-5e-3}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-64}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
DEVICE=${DEVICE:-cuda}

EXTRA=()
if [[ -n "${VERBOSE:-}" ]]; then
    EXTRA+=(--verbose)
fi

# Preference training: no train_labels.pt is produced upstream.
LABELS_FLAG=(--labels "$FEATURES_DIR/train_labels.pt")
if [[ "$SUPERVISION" == "preference" ]]; then
    LABELS_FLAG=()
fi

python scripts/train_probe.py \
    --features      "$FEATURES_DIR/train_features.pt" \
    "${LABELS_FLAG[@]}" \
    --test_features "$FEATURES_DIR/test_features.pt" \
    --test_labels   "$FEATURES_DIR/test_labels.pt" \
    --concept       "$CONCEPT" \
    --probe_type    "$PROBE_TYPE" \
    --supervision   "$SUPERVISION" \
    --d_model       "$D_MODEL" \
    --n_tail        "$N_TAIL" \
    --threshold     "$THRESHOLD" \
    --lr            "$LR" \
    --epochs        "$EPOCHS" \
    --batch_size    "$BATCH_SIZE" \
    --weight_decay  "$WEIGHT_DECAY" \
    --device        "$DEVICE" \
    --output        "$OUTPUT" \
    "${EXTRA[@]}"
