"""
Shared math-benchmark loading for the LAVA scripts (no torch dependency).

GSM8K / AIME 2024 / MATH-500 / HMMT Feb 2025 all reduce, G-OPD style, to
(problem_text, gold_answer); grading happens later via boxed-answer extraction
+ math_verify. Both the trace generator (scripts/generate_traces.py) and the
inference runner (scripts/run_lava.py) pull the table from here so the two stay
in lockstep — traces are generated over exactly the problems LAVA is evaluated on.
"""

from __future__ import annotations


DATASETS = {
    "gsm8k":    {"hf": "openai/gsm8k",            "config": "main", "split": "test"},
    "aime":     {"hf": "HuggingFaceH4/aime_2024", "config": None,   "split": "train"},
    "math500":  {"hf": "HuggingFaceH4/MATH-500",  "config": None,   "split": "test"},
    "hmmt":     {"hf": "MathArena/hmmt_feb_2025",  "config": None,   "split": "train"},
}


def _gsm8k_gold(answer_field: str) -> str:
    """GSM8K answers look like '... explanation ...\\n#### 18'. Keep the number."""
    return answer_field.split("####")[-1].strip()


def _columns(dataset_name: str, ds):
    """Return (problems, golds) lists for a loaded split."""
    if dataset_name == "gsm8k":
        return list(ds["question"]), [_gsm8k_gold(a) for a in ds["answer"]]
    # aime / math500 / hmmt all expose problem/answer columns
    return list(ds["problem"]), [str(a) for a in ds["answer"]]


def _load_split(dataset_name: str):
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASETS)}.")
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("pip install datasets") from e
    spec = DATASETS[dataset_name]
    ds = load_dataset(spec["hf"], spec["config"]) if spec["config"] else load_dataset(spec["hf"])
    return ds[spec["split"]]


def load_problem(dataset_name: str, problem_id: int) -> tuple[str, str]:
    """Return (problem_text, gold_answer) for a single index."""
    problems, golds = _columns(dataset_name, _load_split(dataset_name))
    return problems[problem_id], golds[problem_id]


def load_problems(
    dataset_name: str, limit: int | None = None, start: int = 0
) -> list[tuple[int, str, str]]:
    """Return [(problem_id, problem_text, gold_answer), ...] for a benchmark."""
    problems, golds = _columns(dataset_name, _load_split(dataset_name))
    end = len(problems) if limit is None else min(len(problems), start + limit)
    return [(i, problems[i], golds[i]) for i in range(start, end)]
