"""
Score LAVA runs the way G-OPD (RUCBM/G-OPD) scores math evals.

For every ``<dataset>_<id>/<repeat>.pickle`` produced by ``run_lava.py``:

  1. Reconstruct the model's full response by concatenating ``step_str`` for
     all steps in the pickle.
  2. Extract the *last* ``\\boxed{...}`` substring (matched-brace aware).
  3. Read the gold answer from the sibling ``<repeat>.gold.json``.
  4. Compare with ``math_verify.verify(parse(...), parse(...))``.

Then per-dataset aggregates: pass@1 over repeats, mean output tokens, mean
wall-clock per problem, and draft-accept rate.

    python scripts/score_runs.py --results_dir results/lava

You can restrict scoring to a subset of datasets with ``--datasets math500 aime``.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pickle
from collections import defaultdict


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Boxed answer extraction (lifted from G-OPD/math_eval/eval_math.py).
# ---------------------------------------------------------------------------

def last_boxed_only_string(text: str) -> str | None:
    """Return the last \\boxed{...} / \\fbox{...} substring, brace-balanced."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    # Find the opening brace after the keyword.
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    j = i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[idx : j + 1]
        j += 1
    return None  # unbalanced — treat as no boxed answer


def remove_boxed(boxed: str) -> str:
    """Strip the leading \\boxed{ (or \\fbox{) and the trailing }."""
    for prefix in ("\\boxed{", "\\fbox{"):
        if boxed.startswith(prefix):
            return boxed[len(prefix) : -1]
    return boxed


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _make_grader():
    """Return a grader(pred: str, gold: str) -> bool.

    Prefers ``math_verify`` (the equivalence checker G-OPD uses); falls back to
    a normalized string compare so the script still runs without that dep.
    """
    try:
        from math_verify import parse, verify
    except ImportError:
        logging.warning("math_verify not installed — falling back to string match.")

        def _strgrade(pred: str, gold: str) -> bool:
            return pred.strip() == gold.strip()
        return _strgrade

    def _grade(pred: str, gold: str) -> bool:
        try:
            return bool(verify(parse("\\boxed{" + gold + "}"),
                               parse("\\boxed{" + pred + "}")))
        except Exception as e:  # math_verify can raise on malformed latex
            logging.debug(f"math_verify error pred={pred!r} gold={gold!r}: {e}")
            return False
    return _grade


def extract_prediction(metadata_list) -> str | None:
    """Concat per-step text, return the last boxed substring (sans wrapper)."""
    if not metadata_list:
        return None
    full = "".join(m.get("step_str") or "" for m in metadata_list)
    boxed = last_boxed_only_string(full)
    return None if boxed is None else remove_boxed(boxed)


# ---------------------------------------------------------------------------
# Result discovery / aggregation
# ---------------------------------------------------------------------------

def iter_runs(results_dir: str):
    """Yield (dataset, problem_id, repeat_id, pickle_path, gold_path)."""
    for pkl in sorted(glob.glob(os.path.join(results_dir, "*", "*.pickle"))):
        repeat_id = os.path.splitext(os.path.basename(pkl))[0]
        problem_dir = os.path.basename(os.path.dirname(pkl))
        if "_" not in problem_dir:
            continue
        dataset, _, pid = problem_dir.rpartition("_")
        gold_path = os.path.join(os.path.dirname(pkl), repeat_id + ".gold.json")
        yield dataset, pid, repeat_id, pkl, gold_path


def score(results_dir: str, datasets_filter: set[str] | None):
    grade = _make_grader()
    # Per-dataset accumulators.
    per_ds = defaultdict(lambda: {
        "n": 0, "correct": 0, "missing_gold": 0, "no_pred": 0,
        "tokens": 0, "wall": 0.0, "accepted": 0, "steps": 0,
    })

    for dataset, pid, rep, pkl, gold_path in iter_runs(results_dir):
        if datasets_filter and dataset not in datasets_filter:
            continue
        bucket = per_ds[dataset]
        bucket["n"] += 1

        with open(pkl, "rb") as f:
            metadata_list = pickle.load(f)

        # Aggregate run-level stats.
        bucket["tokens"] += sum(m.get("final_num_output_tokens", 0) for m in metadata_list)
        bucket["wall"]   += sum(m.get("step_time", 0.0) for m in metadata_list)
        bucket["accepted"] += sum(1 for m in metadata_list if m.get("accepted"))
        bucket["steps"]    += len(metadata_list)

        if not os.path.exists(gold_path):
            bucket["missing_gold"] += 1
            continue
        with open(gold_path) as f:
            gold = json.load(f)["answer"]

        pred = extract_prediction(metadata_list)
        if pred is None:
            bucket["no_pred"] += 1
            continue
        if grade(pred, gold):
            bucket["correct"] += 1

    # Report.
    if not per_ds:
        print("No runs found under", results_dir)
        return

    print(f"{'dataset':<10} {'n':>4} {'acc':>7} {'no_pred':>7} {'no_gold':>7} "
          f"{'avg_tok':>8} {'avg_wall':>9} {'accept':>7}")
    print("-" * 70)
    for ds, b in sorted(per_ds.items()):
        n = max(b["n"], 1)
        acc = b["correct"] / n
        avg_tok = b["tokens"] / n
        avg_wall = b["wall"] / n
        accept = (b["accepted"] / b["steps"]) if b["steps"] else 0.0
        print(f"{ds:<10} {b['n']:>4} {acc:>7.2%} {b['no_pred']:>7d} "
              f"{b['missing_gold']:>7d} {avg_tok:>8.0f} {avg_wall:>9.1f} {accept:>7.2%}")


def parse_args():
    p = argparse.ArgumentParser(description="Score LAVA runs (G-OPD-style).")
    p.add_argument("--results_dir", default="results/lava")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Restrict scoring to these dataset names.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    score(args.results_dir, set(args.datasets) if args.datasets else None)
