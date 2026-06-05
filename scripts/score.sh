#!/bin/bash
# Step 4: Score LAVA runs the way G-OPD does — last \boxed{} + math_verify.
#
# Usage:
#   bash scripts/score.sh                       # scores results/lava/
#   bash scripts/score.sh results/my_run        # custom results dir
#   bash scripts/score.sh results/lava gsm8k math500
#                                               # restrict to listed datasets
#
# Env overrides:
#   (none — pass datasets as positional args after the results dir)

set -euo pipefail

RESULTS_DIR=${1:-results/lava}
shift || true

if [[ $# -gt 0 ]]; then
    python scripts/score_runs.py --results_dir "$RESULTS_DIR" --datasets "$@"
else
    python scripts/score_runs.py --results_dir "$RESULTS_DIR"
fi
