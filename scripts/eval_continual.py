"""
Continual learning evaluation (Experiment A: Sequential Concept Learning).

Expects a directory of pre-extracted per-task feature/label tensors:
    data/
        task_0_math_correctness/
            train_features.pt   (N, d_in)
            train_labels.pt     (N,)
            test_features.pt    (M, d_in)
            test_labels.pt      (M,)
        task_1_gsm8k_preference/
            train_features.pt   (N, 2, d_in)  -- preference, no labels file
            test_features.pt    (M, d_in)
            test_labels.pt      (M,)
        ...

Usage:
    python scripts/eval_continual.py \\
        --data_dir data/ \\
        --d_model 2048 \\
        --probe_type mlp \\
        --device cuda \\
        --output results/continual.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from lava import TrainConfig
from lava.continual import TaskData, run_experiment_a, cumulative_false_accept_bound


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--d_model", type=int, required=True)
    p.add_argument("--n_tail", type=int, default=5)
    p.add_argument("--probe_type", default="mlp", choices=["mlp", "linear"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="results/continual.json")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load_task(task_dir: str, device: str) -> TaskData:
    name = os.path.basename(task_dir)
    train_feats = torch.load(os.path.join(task_dir, "train_features.pt"), map_location=device)
    test_feats = torch.load(os.path.join(task_dir, "test_features.pt"), map_location=device)
    test_labels = torch.load(os.path.join(task_dir, "test_labels.pt"), map_location=device)

    label_path = os.path.join(task_dir, "train_labels.pt")
    if os.path.exists(label_path):
        train_labels = torch.load(label_path, map_location=device)
        supervision = "classification"
    else:
        train_labels = None
        supervision = "preference"

    return TaskData(
        name=name,
        supervision=supervision,
        train_features=train_feats,
        train_labels=train_labels,
        test_features=test_feats,
        test_labels=test_labels,
    )


def main():
    args = parse_args()

    task_dirs = sorted(
        [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir)
         if os.path.isdir(os.path.join(args.data_dir, d))]
    )
    if not task_dirs:
        print(f"No task directories found in {args.data_dir}")
        sys.exit(1)

    tasks = [load_task(d, args.device) for d in task_dirs]
    print(f"Loaded {len(tasks)} tasks: {[t.name for t in tasks]}")

    train_cfg = TrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    bank, matrix = run_experiment_a(
        tasks=tasks,
        d_model=args.d_model,
        n_tail=args.n_tail,
        probe_type=args.probe_type,
        train_config=train_cfg,
        threshold=args.threshold,
        device=args.device,
        verbose=args.verbose,
    )

    summary = matrix.summary()
    print(f"\n{'='*50}")
    print(f"ACC  = {summary['ACC']:.3f}")
    print(f"BWT  = {summary['BWT']:.3f}  (LAVA: always 0 by construction)")
    print(f"FWT  = {summary['FWT']:.3f}")
    print(f"{'='*50}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
