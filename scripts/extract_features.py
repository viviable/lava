"""
Extract probe features from a JSONL of annotated reasoning steps.

End-to-end implementation of §4 of the paper:

    raw annotated steps  --confidence filter (gamma=0.8)-->  stratified split
        -->  per-step backbone hidden-state extraction
        -->  feature aggregation (concat / mean / min / last)
        -->  saved {train,test}_{features,labels}.pt

Input format (one JSON object per line):

    {"context": "<prompt + prior steps>", "step_text": "<this step>",
     "label": 1, "confidence": 0.9}

`label` is 0/1; `confidence` is in [0, 1] (set to 1.0 if you trust the source).

Output (in --output_dir):

    train_features.pt   (N_train, d_in)
    train_labels.pt     (N_train,)
    test_features.pt    (N_test,  d_in)
    test_labels.pt      (N_test,)

where d_in = n_tail * d_model when --agg_mode is `concat`, else d_model.

Usage:

    python scripts/extract_features.py \\
        --input_jsonl data/raw/math_correctness.jsonl \\
        --backbone deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
        --output_dir data/task_0_math_correctness \\
        --agg_mode concat --n_tail 5 --layer_idx -1 \\
        --train_size 800 --test_size 200 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from lava.backbone import HFHiddenStateBackbone
from lava.data_pipeline import (
    AnnotatedStep,
    build_classification_dataset,
    confidence_filter,
    stratified_split,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description="Extract probe features from annotated steps.")
    p.add_argument("--input_jsonl", required=True, help="JSONL of annotated steps")
    p.add_argument("--backbone", required=True, help="HF model id for hidden-state extraction")
    p.add_argument("--output_dir", required=True, help="Where to save train/test .pt files")

    p.add_argument("--agg_mode", default="concat", choices=["concat", "mean", "min", "last"])
    p.add_argument("--n_tail", type=int, default=5)
    p.add_argument("--layer_idx", type=int, default=-1)
    p.add_argument("--gamma", type=float, default=0.8,
                   help="Annotator-confidence threshold (Eq. 10)")

    p.add_argument("--train_size", type=int, default=800)
    p.add_argument("--test_size",  type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    return p.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: bad JSON ({e})")
    return rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = load_jsonl(args.input_jsonl)
    logging.info(f"Loaded {len(rows)} annotated steps from {args.input_jsonl}")

    # 1. Load frozen HF backbone.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logging.info(f"Loading backbone {args.backbone} (dtype={args.dtype})")
    tok = AutoTokenizer.from_pretrained(args.backbone)
    mdl = AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=_torch_dtype(args.dtype))
    backbone = HFHiddenStateBackbone(mdl, tok, device=args.device, layer_idx=args.layer_idx)

    # 2. Extract per-step hidden states.
    steps: list[AnnotatedStep] = []
    for i, r in enumerate(rows):
        try:
            ctx = r["context"]; txt = r["step_text"]
            label = int(r["label"])
            conf = float(r.get("confidence", 1.0))
        except KeyError as e:
            raise ValueError(f"row {i}: missing field {e}")
        hidden = backbone.extract_step_hidden(ctx, txt)        # (L+1, T_step, d)
        steps.append(AnnotatedStep(ctx, txt, label, conf, hidden_states=hidden))
        if (i + 1) % 50 == 0:
            logging.info(f"  extracted {i + 1}/{len(rows)}")

    # 3. Confidence filter (gamma). Filtering before the split keeps both halves
    #    using only high-confidence data, matching Eq. 10 in the paper.
    filtered = confidence_filter(steps, gamma=args.gamma)
    logging.info(f"Confidence filter (gamma={args.gamma}): {len(steps)} -> {len(filtered)}")
    if not filtered:
        raise ValueError("No examples survived the confidence filter.")

    # 4. Stratified split.
    labels = [s.label for s in filtered]
    train_size = min(args.train_size, len(filtered) - 1)
    test_size  = min(args.test_size,  len(filtered) - train_size)
    if test_size <= 0:
        raise ValueError(
            f"Not enough data: need >= train_size + 1, got {len(filtered)} (train={train_size})."
        )
    train_steps, test_steps = stratified_split(
        filtered, labels, train_size=train_size, test_size=test_size, seed=args.seed
    )
    logging.info(f"Stratified split: train={len(train_steps)} test={len(test_steps)}")

    # 5. Build feature tensors and persist.
    def _build(split):
        return build_classification_dataset(
            split, mode=args.agg_mode, n_tail=args.n_tail, layer_idx=args.layer_idx,
            gamma=0.0,  # already filtered above; don't drop again
        )

    train_feats, train_labels = _build(train_steps)
    test_feats,  test_labels  = _build(test_steps)

    # Probes are fp32; backbone features may be fp16/bf16. Cast at the boundary.
    train_feats = train_feats.float()
    test_feats  = test_feats.float()

    torch.save(train_feats, os.path.join(args.output_dir, "train_features.pt"))
    torch.save(train_labels, os.path.join(args.output_dir, "train_labels.pt"))
    torch.save(test_feats,  os.path.join(args.output_dir, "test_features.pt"))
    torch.save(test_labels, os.path.join(args.output_dir, "test_labels.pt"))

    pos_rate = float(train_labels.float().mean())
    logging.info(
        f"Saved -> {args.output_dir} "
        f"(train: {tuple(train_feats.shape)}, test: {tuple(test_feats.shape)}, "
        f"train pos_rate={pos_rate:.2%})"
    )


if __name__ == "__main__":
    main()
