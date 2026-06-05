#!/bin/bash
# Step 0: Build a probe-training JSONL from reasoning traces using Gemini.
#
# Pipeline (mirrors AngelaZZZ-611/reasoning_models_probing):
#   completion -> \n\n split -> collapse until transition marker
#              -> Gemini 2.0 Flash labels chunk True/False/None
#              -> None merged into next labelled chunk
#              -> JSONL ready for scripts/extract_features.sh
#
# Usage:
#   GEMINI_API_KEY=... bash scripts/build_probe_dataset.sh <input> <output_jsonl>
#
# <input> is either:
#   - a JSONL of {id, problem, completion, gold_answer} rows, or
#   - a LAVA results directory (auto-detected if --from-lava is set or path is a dir)
#
# Examples:
#   GEMINI_API_KEY=$KEY bash scripts/build_probe_dataset.sh \
#       data/raw/traces.jsonl data/raw/math_correctness.jsonl
#
#   FROM_LAVA=1 GEMINI_API_KEY=$KEY bash scripts/build_probe_dataset.sh \
#       results/lava data/raw/math_correctness.jsonl
#
# Env overrides:
#   GEMINI_MODEL   model id                 (default: gemini-2.0-flash)
#   CONFIDENCE     emitted confidence       (default: 1.0; set <0.8 to be dropped by gamma)
#   MAX_WORKERS    parallel Gemini reqs     (default: 4)
#   SLEEP          per-request sleep (s)    (default: 0.0)
#   LIMIT          cap # traces processed   (default: all)
#   FROM_LAVA      non-empty -> treat <input> as a LAVA results dir
#
# Requires: pip install google-generativeai

set -euo pipefail

INPUT=${1:?usage: $0 <input> <output_jsonl>}
OUTPUT=${2:?usage: $0 <input> <output_jsonl>}

GEMINI_MODEL=${GEMINI_MODEL:-gemini-2.0-flash}
CONFIDENCE=${CONFIDENCE:-1.0}
MAX_WORKERS=${MAX_WORKERS:-4}
SLEEP=${SLEEP:-0.0}

if [[ -n "${FROM_LAVA:-}" ]] || [[ -d "$INPUT" ]]; then
    SRC_FLAG=(--from_lava_results "$INPUT")
else
    SRC_FLAG=(--input_jsonl "$INPUT")
fi

EXTRA=()
if [[ -n "${LIMIT:-}" ]]; then
    EXTRA+=(--limit "$LIMIT")
fi

python scripts/build_probe_dataset.py \
    "${SRC_FLAG[@]}" \
    --output_jsonl "$OUTPUT" \
    --gemini_model "$GEMINI_MODEL" \
    --confidence   "$CONFIDENCE" \
    --max_workers  "$MAX_WORKERS" \
    --sleep        "$SLEEP" \
    "${EXTRA[@]}"
