"""
Run LAVA speculative reasoning on GSM8K / AIME 2024 / MATH-500 / HMMT Feb 2025.

Setup (mirrors SpecReason):

    # Terminal 1 — draft model (e.g. DeepSeek-R1-Distill-Qwen-1.5B)
    python -m vllm.entrypoints.openai.api_server \\
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
        --port 30001 --enable-prefix-caching

    # Terminal 2 — strong model (e.g. QwQ-32B)
    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/QwQ-32B --tensor-parallel-size 2 \\
        --port 30000 --enable-prefix-caching

    # Terminal 3 — LAVA runner
    python scripts/run_lava.py \\
        --dataset_name aime --problem_id 0 --repeat_id 0 \\
        --probe_bank probes/ \\
        --draft_url http://localhost:30001/v1 \\
        --strong_url http://localhost:30000/v1 \\
        --hidden_backbone deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
        --output_dir results/lava

Per-problem output is a pickle of the full step-level metadata list (one dict
per step) plus a pretty-printed .txt for inspection — the same shape as
SpecReason's logs, with the LLM-as-judge score replaced by probe scores. A
sibling ``<problem_id>.<repeat>.gold.json`` records the dataset name, problem
text, and ground-truth answer for downstream scoring (see ``score_runs.py``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import pprint
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from lava import LAVAConfig, LAVAPipeline, ProbeBank
from lava.backbone import HFHiddenStateBackbone
from lava.speculative import (
    DEFAULT_MATH_SYSTEM_PROMPT,
    VLLMModelInterface,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Dataset loading — math reasoning benchmarks (GSM8K, AIME, MATH-500, HMMT).
# G-OPD (RUCBM/G-OPD) style: every example reduces to (problem, gold_answer),
# graded later via boxed-answer extraction + math_verify.
# ---------------------------------------------------------------------------

DATASETS = {
    "gsm8k":    {"hf": "openai/gsm8k",           "config": "main",      "split": "test"},
    "aime":     {"hf": "HuggingFaceH4/aime_2024","config": None,        "split": "train"},
    "math500":  {"hf": "HuggingFaceH4/MATH-500", "config": None,        "split": "test"},
    "hmmt":     {"hf": "MathArena/hmmt_feb_2025","config": None,        "split": "train"},
}


def _gsm8k_gold(answer_field: str) -> str:
    """GSM8K answers look like '... explanation ...\\n#### 18'. Keep the number."""
    return answer_field.split("####")[-1].strip()


def load_problem(dataset_name: str, problem_id: int):
    """Return (problem_text, gold_answer, system_prompt, prompt_kwargs)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("pip install datasets") from e

    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASETS)}.")
    spec = DATASETS[dataset_name]
    ds = load_dataset(spec["hf"], spec["config"]) if spec["config"] else load_dataset(spec["hf"])
    ds = ds[spec["split"]]

    if dataset_name == "gsm8k":
        problem = ds["question"][problem_id]
        gold = _gsm8k_gold(ds["answer"][problem_id])
    elif dataset_name == "aime":
        problem = ds["problem"][problem_id]
        gold = str(ds["answer"][problem_id])
    elif dataset_name == "math500":
        problem = ds["problem"][problem_id]
        gold = str(ds["answer"][problem_id])
    elif dataset_name == "hmmt":
        problem = ds["problem"][problem_id]
        gold = str(ds["answer"][problem_id])
    else:  # unreachable — guarded by DATASETS check above
        raise ValueError(dataset_name)

    return problem, gold, DEFAULT_MATH_SYSTEM_PROMPT, {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LAVA speculative reasoning runner.")
    p.add_argument("--dataset_name", choices=list(DATASETS.keys()), default="aime")
    p.add_argument("--problem_id", type=int, default=0)
    p.add_argument("--repeat_id", type=int, default=0)
    p.add_argument("--token_budget", type=int, default=8192)
    p.add_argument("--max_steps", type=int, default=128)
    p.add_argument("--max_tokens_per_step", type=int, default=512)

    p.add_argument("--probe_bank", required=True, help="Path to saved ProbeBank directory")
    p.add_argument("--hidden_backbone", default=None,
                   help="HF model id for hidden-state extraction. If unset, probes are skipped.")

    p.add_argument("--draft_url", default="http://localhost:30001/v1")
    p.add_argument("--draft_model", default=None, help="Override autodiscovered vLLM model name")
    p.add_argument("--strong_url", default="http://localhost:30000/v1")
    p.add_argument("--strong_model", default=None)

    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", default="results/lava")
    return p.parse_args()


def main():
    args = parse_args()

    # Skip if already done (matches SpecReason).
    out_base = os.path.join(args.output_dir, f"{args.dataset_name}_{args.problem_id}", str(args.repeat_id))
    if os.path.exists(out_base + ".pickle"):
        logging.info(f"Already resolved: {out_base}.pickle — exiting.")
        return

    problem, gold_answer, system_prompt, prompt_kwargs = load_problem(args.dataset_name, args.problem_id)
    logging.info(f"Loaded {args.dataset_name} #{args.problem_id} (gold={gold_answer!r})")

    # 1. Probe bank.
    logging.info(f"Loading probe bank from {args.probe_bank}")
    bank = ProbeBank.load(args.probe_bank, device=args.device)
    logging.info(f"  {bank.k} probes: {bank.concept_names()}")

    # 2. Hidden-state backbone (optional but recommended).
    hidden_backbone = None
    if args.hidden_backbone:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logging.info(f"Loading hidden backbone: {args.hidden_backbone}")
        tok = AutoTokenizer.from_pretrained(args.hidden_backbone)
        mdl = AutoModelForCausalLM.from_pretrained(args.hidden_backbone, torch_dtype=torch.float16)
        hidden_backbone = HFHiddenStateBackbone(mdl, tok, device=args.device)
    else:
        logging.warning("No --hidden_backbone provided; probe verification will be skipped.")

    # 3. vLLM clients.
    draft_iface = VLLMModelInterface(
        base_url=args.draft_url,
        model_name=args.draft_model,
        system_prompt=system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_kwargs=prompt_kwargs,
    )
    strong_iface = VLLMModelInterface(
        base_url=args.strong_url,
        model_name=args.strong_model,
        system_prompt=system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_kwargs=prompt_kwargs,
    )

    cfg = LAVAConfig(
        max_steps=args.max_steps,
        max_tokens_per_step=args.max_tokens_per_step,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )
    pipeline = LAVAPipeline(draft_iface, strong_iface, bank, hidden_backbone=hidden_backbone, config=cfg)

    # 4. Run.
    logging.info(f"Problem: {problem[:120]}{'…' if len(problem) > 120 else ''}")
    result = pipeline.run(problem, token_budget=args.token_budget)

    # 5. Build SpecReason-shaped metadata list.
    metadata_list = []
    for s in result.steps:
        used = sum(m["final_num_output_tokens"] for m in metadata_list) + (
            s.strong_tokens if not s.accepted else s.draft_tokens
        )
        scores = None if s.scores is None else [float(x) for x in s.scores.tolist()]
        metadata_list.append({
            "step_id": s.step_index,
            "step_str": s.text,
            "draft_step": s.draft_text,
            "num_output_tokens_draft": s.draft_tokens,
            "draft_latency": s.draft_latency,
            "probe_scores": scores,
            "concept_names": bank.concept_names(),
            "accepted": s.accepted,
            "verify_latency": s.verify_latency,
            "strong_step": None if s.accepted else s.text,
            "num_output_tokens_strong": s.strong_tokens,
            "strong_latency": s.strong_latency,
            "final_num_output_tokens": s.strong_tokens if not s.accepted else s.draft_tokens,
            "step_time": s.draft_latency + s.strong_latency + s.verify_latency,
            "finished": s.finished,
        })

    if metadata_list:
        used = sum(m["final_num_output_tokens"] for m in metadata_list)
        metadata_list[-1]["stop_reason"] = "finished" if used < args.token_budget else "budget"

    # 6. Persist.
    os.makedirs(os.path.dirname(out_base + ".pickle"), exist_ok=True)
    with open(out_base + ".pickle", "wb") as f:
        pickle.dump(metadata_list, f)
    with open(out_base + ".txt", "w") as f:
        pprint.pprint(metadata_list, stream=f)
    with open(out_base + ".gold.json", "w") as f:
        json.dump({
            "dataset": args.dataset_name,
            "problem_id": args.problem_id,
            "repeat_id": args.repeat_id,
            "problem": problem,
            "answer": gold_answer,
        }, f)

    logging.info(
        f"Done. steps={result.total_steps} "
        f"accept={result.draft_accept_rate:.1%} "
        f"draft_tok={result.total_draft_tokens} strong_tok={result.total_strong_tokens} "
        f"wall={result.total_time:.1f}s"
    )
    logging.info(f"Wrote {out_base}.pickle (+ .txt)")


if __name__ == "__main__":
    main()
