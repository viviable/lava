"""
Continual learning evaluation protocol for LAVA (Section 6).

Metrics follow Lopez-Paz & Ranzato (2017):
  ACC  = (1/K) Σ_i A_{i,K}                    — final average accuracy
  BWT  = (1/(K-1)) Σ_{i=1}^{K-1} (A_{i,K} - A_{i,i})  — backward transfer
  FWT  = (1/(K-1)) Σ_{i=2}^{K} (A_{i,i-1} - ā_i)      — forward transfer

For LAVA, BWT = 0 by construction (Prop. 6.1).

This module also implements:
  - The cumulative false-accept bound (Eq. 15)
  - Experiment A–E helpers as described in Section 6.5
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np

from .probe_bank import ProbeBank, ConceptConfig
from .training import TrainConfig, evaluate_probe


@dataclass
class TaskData:
    """Train + test data for a single concept/task."""
    name: str
    supervision: str              # 'classification' or 'preference'
    train_features: torch.Tensor
    train_labels: Optional[torch.Tensor]
    test_features: torch.Tensor
    test_labels: torch.Tensor


class AccuracyMatrix:
    """K×K matrix A where A[i, j] = accuracy on task i after training through task j."""

    def __init__(self, k: int):
        self.k = k
        self._data = np.full((k, k), np.nan)

    def set(self, task_i: int, after_j: int, acc: float):
        self._data[task_i, after_j] = acc

    def get(self, task_i: int, after_j: int) -> float:
        return float(self._data[task_i, after_j])

    def acc(self) -> float:
        """Final average accuracy (Eq. 12)."""
        k = self.k
        vals = [self._data[i, k - 1] for i in range(k) if not np.isnan(self._data[i, k - 1])]
        return float(np.mean(vals)) if vals else float("nan")

    def bwt(self) -> float:
        """Backward transfer (Eq. 13). For LAVA this is always 0."""
        k = self.k
        diffs = []
        for i in range(k - 1):
            if not (np.isnan(self._data[i, k - 1]) or np.isnan(self._data[i, i])):
                diffs.append(self._data[i, k - 1] - self._data[i, i])
        return float(np.mean(diffs)) if diffs else float("nan")

    def fwt(self, random_baselines: Optional[list[float]] = None) -> float:
        """Forward transfer (Eq. 14).

        random_baselines[i] = ā_i, accuracy of an untrained probe on task i.
        Defaults to 0.5 (random binary classification) if not provided.
        """
        k = self.k
        if random_baselines is None:
            random_baselines = [0.5] * k
        diffs = []
        for i in range(1, k):
            a_prev = self._data[i, i - 1]
            a_rand = random_baselines[i]
            if not np.isnan(a_prev):
                diffs.append(a_prev - a_rand)
        return float(np.mean(diffs)) if diffs else float("nan")

    def summary(self, random_baselines: Optional[list[float]] = None) -> dict:
        return {
            "ACC": self.acc(),
            "BWT": self.bwt(),
            "FWT": self.fwt(random_baselines),
            "matrix": self._data.tolist(),
        }

    def __repr__(self) -> str:
        return f"AccuracyMatrix(K={self.k}, ACC={self.acc():.3f}, BWT={self.bwt():.3f})"


# ---------------------------------------------------------------------------
# Experiment A: Sequential concept learning
# ---------------------------------------------------------------------------

def run_experiment_a(
    tasks: list[TaskData],
    d_model: int,
    n_tail: int = 5,
    probe_type: str = "mlp",
    train_config: Optional[TrainConfig] = None,
    threshold: float = 0.5,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[ProbeBank, AccuracyMatrix]:
    """Train probes sequentially on tasks T_1, ..., T_K.

    After each new task, evaluate ALL prior tasks to populate A.
    Returns the final ProbeBank and the full AccuracyMatrix.
    """
    K = len(tasks)
    matrix = AccuracyMatrix(K)
    bank = ProbeBank(d_model=d_model, n_tail=n_tail, device=device)

    if train_config is None:
        train_config = TrainConfig(device=device)

    for j, task in enumerate(tasks):
        if verbose:
            print(f"[Exp A] Training task {j+1}/{K}: {task.name}")

        cfg = ConceptConfig(
            name=task.name,
            probe_type=probe_type,
            supervision=task.supervision,
            threshold=threshold,
        )
        bank.add_concept(
            cfg,
            features=task.train_features,
            labels=task.train_labels,
            train_config=train_config,
            verbose=verbose,
        )

        # Evaluate all concepts seen so far
        for i in range(j + 1):
            probe = bank._probes[i]
            metrics = evaluate_probe(
                probe,
                tasks[i].test_features,
                tasks[i].test_labels,
                threshold=threshold,
                device=device,
            )
            matrix.set(task_i=i, after_j=j, acc=metrics["accuracy"])
            if verbose:
                print(f"  A[{i},{j}] ({tasks[i].name}) = {metrics['accuracy']:.3f}")

    return bank, matrix


# ---------------------------------------------------------------------------
# Cumulative false-accept bound (Eq. 15)
# ---------------------------------------------------------------------------

def cumulative_false_accept_bound(per_probe_fn_rates: list[float]) -> float:
    """Upper bound on joint false-accept probability (Eq. 15).

    Assumes conditional independence of probe errors given a truly invalid step.

    P(all K probes accept | step invalid) ≤ ∏_k FN_k
    """
    result = 1.0
    for fn in per_probe_fn_rates:
        result *= fn
    return result


def empirical_false_accept_rate(
    bank: ProbeBank,
    invalid_features: torch.Tensor,
) -> float:
    """Empirically compute joint false-accept rate on invalid steps.

    Args:
        invalid_features: (N, d_in) features for steps known to be invalid.

    Returns:
        Fraction of invalid steps that all probes incorrectly accept.
    """
    if bank.k == 0 or len(invalid_features) == 0:
        return 0.0
    accept_mask, _ = bank.verify_batch(invalid_features)
    return float(accept_mask.float().mean().item())


# ---------------------------------------------------------------------------
# Experiment E: K-scaling inference latency
# ---------------------------------------------------------------------------

def measure_inference_latency(
    bank: ProbeBank,
    features: torch.Tensor,
    n_warmup: int = 10,
    n_repeat: int = 100,
) -> float:
    """Measure average per-step verification latency in milliseconds.

    Returns:
        Mean latency in ms over n_repeat runs.
    """
    import time

    features = features.to(bank.device)
    # Warmup
    for _ in range(n_warmup):
        bank.verify(features[0])

    times = []
    for i in range(n_repeat):
        f = features[i % len(features)]
        t0 = time.perf_counter()
        bank.verify(f)
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times))
