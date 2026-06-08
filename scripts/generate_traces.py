"""
Generate raw CoT reasoning traces (step "0a" of the LAVA probe pipeline).

Runs a vLLM-served model over a math benchmark (GSM8K / AIME 2024 / MATH-500 /
HMMT Feb 2025) and writes one ``{id, problem, completion, gold_answer}`` row per
generation — exactly the schema ``scripts/build_probe_dataset.py`` consumes via
``--input_jsonl``.

Why this exists (the missing bootstrap step):

    build_probe_dataset.py  labels traces but does not produce them.
    run_lava.py             can reconstruct traces, but needs a trained probe
                            bank to run — so the *first* concept cannot use
                            --from_lava_results.

So the very first dataset has to come from somewhere: generate traces here,
label them, extract features, train a probe, then run LAVA. After that you can
keep growing data with build_probe_dataset.py --from_lava_results.

Plain CoT needs only one model (no draft/strong split), so one vLLM server is
enough. We reuse run_lava.py's dataset table (DATASETS / load_problem) and
lava.speculative's DEFAULT_MATH_SYSTEM_PROMPT so the prompt matches inference.

Setup:

    python -m vllm.entrypoints.openai.api_server \\
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
        --port 30001 --enable-prefix-caching

    python scripts/generate_traces.py \\
        --dataset_name hmmt \\
        --output_jsonl data/raw/hmmt_traces.jsonl \\
        --base_url http://localhost:30001/v1 \\
        --max_tokens 8192 --num_samples 1

Then label and continue down the pipeline:

    python scripts/build_probe_dataset.py \\
        --input_jsonl  data/raw/hmmt_traces.jsonl \\
        --output_jsonl data/raw/math_correctness.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))  # scripts dir -> datasets_common

# Reuse the exact dataset table + gold-answer parsing the inference runner uses,
# so traces are generated over the same problems LAVA is later evaluated on.
# datasets_common has no torch dependency — plain CoT generation needs none.
from datasets_common import DATASETS, load_problems  # noqa: E402

# DEFAULT_MATH_SYSTEM_PROMPT is a bare string; inline it rather than importing
# lava.speculative, which would pull in torch just to generate text.
DEFAULT_MATH_SYSTEM_PROMPT = (
    "Solve the following math problem efficiently and clearly. Please reason "
    "step by step, separate logical reasoning steps with two newline characters "
    "(\\n\\n), and put your final answer within \\boxed{{}}.\nProblem: {problem}"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


# --- Generation --------------------------------------------------------------
def generate_completion(
    client,
    model_name: str,
    problem: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    retries: int = 3,
) -> str:
    """One full CoT completion for a problem (no per-step \\n\\n stop)."""
    user_msg = DEFAULT_MATH_SYSTEM_PROMPT.format(problem=problem)
    messages = [{"role": "user", "content": user_msg}]
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body={"add_generation_prompt": True},
            )
            msg = resp.choices[0].message
            text = msg.content or ""
            # If a reasoning parser is enabled, the chain-of-thought lands in
            # reasoning_content; fold it back in so the completion is the full CoT.
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning:
                text = f"{reasoning}\n\n{text}" if text else reasoning
            return text.strip()
        except Exception as e:                              # noqa: BLE001 - SDK errors vary
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"generation failed after {retries} retries: {last_err}")


# --- I/O ---------------------------------------------------------------------
def existing_ids(path: str) -> set[str]:
    """Ids already present in an output file, so reruns resume instead of dup."""
    ids: set[str] = set()
    if not os.path.exists(path):
        return ids
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(str(json.loads(line).get("id")))
            except json.JSONDecodeError:
                continue
    return ids


# --- CLI ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate CoT traces for probe-dataset construction.")
    p.add_argument("--dataset_name", choices=list(DATASETS.keys()), required=True)
    p.add_argument("--output_jsonl", required=True, help="Where to write {id, problem, completion, gold_answer}")

    p.add_argument("--base_url", default="http://localhost:30001/v1", help="vLLM OpenAI endpoint")
    p.add_argument("--model", default=None, help="Override autodiscovered vLLM model name")
    p.add_argument("--api_key", default="EMPTY")

    p.add_argument("--max_tokens", type=int, default=8192, help="Max tokens for the full trace")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)

    p.add_argument("--num_samples", type=int, default=1,
                   help="Traces per problem (use >1 with temperature>0 for more chunks)")
    p.add_argument("--limit", type=int, default=None, help="Cap # problems (from --start)")
    p.add_argument("--start", type=int, default=0, help="First problem index")
    p.add_argument("--max_workers", type=int, default=4, help="Concurrent generation requests")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("openai package required. Run: pip install openai") from e
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    model_name = args.model or client.models.list().data[0].id
    log.info("Using model %s at %s", model_name, args.base_url)

    problems = load_problems(args.dataset_name, args.limit, args.start)
    log.info("Loaded %d problems from %s", len(problems), args.dataset_name)

    # Build the task list (problem x sample), skipping ids already written.
    done = existing_ids(args.output_jsonl)
    if done:
        log.info("Resuming: %d ids already in %s", len(done), args.output_jsonl)
    tasks = []
    for pid, problem, gold in problems:
        for s in range(args.num_samples):
            tid = f"{args.dataset_name}_{pid}_{s}" if args.num_samples > 1 else f"{args.dataset_name}_{pid}"
            if tid in done:
                continue
            tasks.append((tid, pid, problem, gold))
    log.info("%d traces to generate (%d skipped)", len(tasks), len(done))
    if not tasks:
        return

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    n_ok = n_fail = 0
    # Append so reruns extend the file rather than clobber prior work.
    with open(args.output_jsonl, "a") as out_f, \
         ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                generate_completion, client, model_name, problem,
                args.max_tokens, args.temperature, args.top_p,
            ): (tid, pid, problem, gold)
            for (tid, pid, problem, gold) in tasks
        }
        for fut in as_completed(futures):
            tid, pid, problem, gold = futures[fut]
            try:
                completion = fut.result()
            except Exception as e:                          # noqa: BLE001
                log.warning("trace %s failed: %s", tid, e)
                n_fail += 1
                continue
            if not completion:
                log.warning("trace %s produced empty completion; skipping", tid)
                n_fail += 1
                continue
            out_f.write(json.dumps({
                "id":          tid,
                "problem":     problem,
                "completion":  completion,
                "gold_answer": gold,
            }) + "\n")
            out_f.flush()
            n_ok += 1
            if (n_ok + n_fail) % 10 == 0:
                log.info("progress: ok=%d fail=%d", n_ok, n_fail)

    log.info("Done. ok=%d fail=%d -> %s", n_ok, n_fail, args.output_jsonl)


if __name__ == "__main__":
    main()
