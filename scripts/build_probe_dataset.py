"""
Build a probe-training JSONL from CoT reasoning traces (step-wise + Gemini labels).

Mirrors the pipeline in AngelaZZZ-611/reasoning_models_probing:

    completion  -> \\n\\n split
                -> collapse pieces until a self-reflection transition marker
                -> Gemini 2.0 Flash labels each chunk: True / False / None
                -> None-labelled chunks merge into the next labelled one
                -> emit one {context, step_text, label, confidence} row per kept chunk

The output JSONL is the same schema scripts/extract_features.py consumes, so this
slots in as "Step 0" of the LAVA pipeline (see README "End-to-end pipeline").

Input row (--input_jsonl):
    {"id": ..., "problem": "...", "completion": "...", "gold_answer": "..."}

Or, reconstruct traces from existing LAVA runs:
    --from_lava_results results/lava
        Walks <dir>/**/*.pickle + sibling .gold.json, joins draft/strong text into
        a completion. The sidecar carries the gold answer the scorer already uses.

Confidence note:
    Gemini doesn't expose calibrated p(True). All emitted rows get --confidence
    (default 1.0); set ~0.9 if you want extract_features.py's gamma=0.8 filter to
    keep them, or 0.5 if you want it to drop them.

Usage:
    pip install google-generativeai
    GEMINI_API_KEY=... python scripts/build_probe_dataset.py \\
        --input_jsonl  data/raw/traces.jsonl \\
        --output_jsonl data/raw/math_correctness.jsonl \\
        --gemini_model gemini-2.0-flash \\
        --max_workers  4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


# --- Chunking (faithful to reasoning_models_probing/get_reasoning_chunks.py) -
# Their implementation uses spaCy Matcher over the same marker set; word-
# boundary regex produces identical boundaries on normal English completions
# without the spaCy dependency.
TRANSITION_PATTERNS = [
    r"\bwait\b",
    r"\balternatively\b",
    r"\bbut wait\b",
    r"\bverify\b",
    r"\blet me reconsider\b",
    r"\blet me (?:re)?check\b",
    r"\blet me try\b",
    r"\bactually\b",
    r"\bhmm+\b",
    r"\bhold on\b",
    r"\bon second thought\b",
    r"\binstead\b",
]
TRANSITION_RE = re.compile("|".join(TRANSITION_PATTERNS), re.IGNORECASE)


def chunk_completion(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for p in paragraphs:
        if current and TRANSITION_RE.search(p):
            chunks.append("\n\n".join(current))
            current = [p]
        else:
            current.append(p)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


# --- Gemini labelling --------------------------------------------------------
LABEL_PROMPT = """\
You are grading one chunk from a chain-of-thought reasoning trace.

PROBLEM:
{problem}

GROUND-TRUTH ANSWER:
{gold}

REASONING BEFORE THIS CHUNK:
{prior}

CHUNK TO EVALUATE:
{chunk}

Does this chunk contain an intermediate answer claim that matches the ground-truth answer?
Answer with exactly one of these three tokens (no punctuation, no explanation):

  True   the chunk states an intermediate answer that matches the ground truth
  False  the chunk states an intermediate answer that does NOT match the ground truth
  None   the chunk does not state any intermediate answer claim
"""


def _norm(reply: str | None) -> str:
    if not reply:
        return "None"
    tok = reply.strip().split()[0].rstrip(".,;:").capitalize()
    return tok if tok in {"True", "False", "None"} else "None"


def label_chunk(model, problem: str, gold: str, prior_text: str, chunk: str,
                retries: int = 3, sleep: float = 0.0) -> str:
    prior = prior_text if prior_text else "(none — this is the first chunk)"
    prompt = LABEL_PROMPT.format(problem=problem, gold=gold, prior=prior, chunk=chunk)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = model.generate_content(prompt)
            if sleep:
                time.sleep(sleep)
            return _norm(getattr(resp, "text", None))
        except Exception as e:                          # noqa: BLE001 - SDK errors vary
            last_err = e
            time.sleep(2 ** attempt)
    log.warning("Gemini failed after %d retries: %s", retries, last_err)
    return "None"


# --- Per-trace expansion -----------------------------------------------------
def build_rows(trace: dict, model, confidence: float, sleep: float = 0.0) -> list[dict]:
    chunks = chunk_completion(trace["completion"])
    rows: list[dict] = []
    pending: list[str] = []
    prior_text = ""
    for ch in chunks:
        merged = "\n\n".join(pending + [ch]) if pending else ch
        verdict = label_chunk(
            model, trace["problem"], str(trace["gold_answer"]),
            prior_text, merged, sleep=sleep,
        )
        if verdict == "None":
            pending.append(ch)
            continue
        ctx = trace["problem"] + (("\n\n" + prior_text) if prior_text else "")
        rows.append({
            "context":    ctx,
            "step_text":  merged,
            "label":      1 if verdict == "True" else 0,
            "confidence": confidence,
            "trace_id":   trace.get("id"),
        })
        prior_text = (prior_text + "\n\n" + merged) if prior_text else merged
        pending = []
    return rows


# --- I/O ---------------------------------------------------------------------
def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: {e}")
    return rows


def load_from_lava_results(root: str) -> list[dict]:
    out: list[dict] = []
    for pkl in sorted(Path(root).rglob("*.pickle")):
        gold_path = pkl.with_suffix(".gold.json")
        if not gold_path.exists():
            continue
        with open(pkl, "rb") as f:
            steps = pickle.load(f)
        with open(gold_path) as f:
            gold = json.load(f)
        completion = "\n\n".join(
            (s.get("strong_text") or s.get("draft_text") or "").strip()
            for s in steps
            if isinstance(s, dict) and (s.get("strong_text") or s.get("draft_text"))
        )
        if not completion:
            continue
        out.append({
            "id":          str(pkl.relative_to(root)),
            "problem":     gold.get("problem", ""),
            "completion":  completion,
            "gold_answer": gold.get("gold_answer", gold.get("answer", "")),
        })
    return out


# --- CLI ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Step-wise probe dataset builder (Gemini-labelled).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_jsonl", help="JSONL: {id, problem, completion, gold_answer}")
    src.add_argument("--from_lava_results", help="Reconstruct traces from a LAVA results dir")
    p.add_argument("--output_jsonl", required=True, help="Annotated-step JSONL to write")
    p.add_argument("--gemini_model", default="gemini-2.0-flash")
    p.add_argument("--confidence", type=float, default=1.0,
                   help="Confidence to emit per row (extract_features.py drops < gamma)")
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--sleep", type=float, default=0.0,
                   help="Per-request sleep (s); use to stay under rate limits")
    p.add_argument("--limit", type=int, help="Cap the number of traces processed")
    return p.parse_args()


def main():
    args = parse_args()

    if args.input_jsonl:
        traces = load_jsonl(args.input_jsonl)
        log.info("Loaded %d traces from %s", len(traces), args.input_jsonl)
    else:
        traces = load_from_lava_results(args.from_lava_results)
        log.info("Reconstructed %d traces from %s", len(traces), args.from_lava_results)

    if args.limit:
        traces = traces[: args.limit]
        log.info("Limited to first %d traces", len(traces))

    try:
        import google.generativeai as genai
    except ImportError as e:
        raise SystemExit(
            "google-generativeai not installed. Run: pip install google-generativeai"
        ) from e
    if "GEMINI_API_KEY" not in os.environ:
        raise SystemExit("GEMINI_API_KEY environment variable is not set.")
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(args.gemini_model)

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    n_in = n_chunks = n_kept = n_pos = 0
    with open(args.output_jsonl, "w") as out_f, \
         ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(build_rows, t, model, args.confidence, args.sleep): t
            for t in traces
        }
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                rows = fut.result()
            except Exception as e:                       # noqa: BLE001 - SDK errors vary
                log.warning("trace %s failed: %s", t.get("id"), e)
                continue
            n_in += 1
            n_chunks += len(chunk_completion(t["completion"]))
            for r in rows:
                out_f.write(json.dumps(r) + "\n")
                n_kept += 1
                n_pos += int(r["label"] == 1)
            if n_in % 10 == 0:
                log.info("traces=%d kept_rows=%d pos_rate=%.1f%%",
                         n_in, n_kept, 100.0 * n_pos / max(n_kept, 1))

    log.info("Done. traces=%d chunks=%d kept=%d pos=%d -> %s",
             n_in, n_chunks, n_kept, n_pos, args.output_jsonl)


if __name__ == "__main__":
    main()
