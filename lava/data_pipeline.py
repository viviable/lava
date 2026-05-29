"""
Automated Probe Construction Pipeline (Section 4).

Covers:
  - Stratified dataset sampling
  - Reasoning trajectory decomposition (delimiter-based)
  - Confidence-aware annotation denoising (γ = 0.8)
  - Feature dataset construction from raw hidden states
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import torch

from .feature_extraction import extract_step_feature, AggMode


STEP_DELIMITER = "\n\n"
CONFIDENCE_THRESHOLD = 0.8  # γ in the paper


@dataclass
class AnnotatedStep:
    """A single labeled reasoning step."""
    context: str
    step_text: str
    label: int                      # 0 = incorrect/unsafe, 1 = correct/safe
    confidence: float               # annotator confidence in [0, 1]
    hidden_states: Optional[torch.Tensor] = None  # (L, T, d) or None


@dataclass
class PreferenceStep:
    """A pair of steps for preference supervision."""
    context: str
    step_pos: str                   # preferred step
    step_neg: str                   # dispreferred step
    hidden_pos: Optional[torch.Tensor] = None
    hidden_neg: Optional[torch.Tensor] = None


def decompose_trajectory(
    trajectory: str,
    delimiter: str = STEP_DELIMITER,
    unitary: bool = False,
) -> list[str]:
    """Split a reasoning trajectory into atomic steps.

    Args:
        trajectory: Full generated reasoning text.
        delimiter:  Token(s) that mark step boundaries (default: '\\n\\n').
        unitary:    If True, treat the entire trajectory as one step (for
                    safety-critical domains where responses are succinct).

    Returns:
        List of step strings (stripped of leading/trailing whitespace).
    """
    if unitary:
        return [trajectory.strip()]
    steps = [s.strip() for s in trajectory.split(delimiter) if s.strip()]
    return steps


def confidence_filter(
    items: list[AnnotatedStep],
    gamma: float = CONFIDENCE_THRESHOLD,
) -> list[AnnotatedStep]:
    """Retain only items with annotator confidence ≥ γ (Eq. 10)."""
    return [item for item in items if item.confidence >= gamma]


def stratified_split(
    items: list,
    labels: list[int],
    train_size: int,
    test_size: int,
    seed: int = 42,
) -> tuple[list, list]:
    """Balanced stratified split into train / test sets.

    Attempts to maintain the positive/negative ratio in both splits.
    """
    rng = random.Random(seed)
    pos = [x for x, y in zip(items, labels) if y == 1]
    neg = [x for x, y in zip(items, labels) if y == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_train_pos = int(train_size * len(pos) / len(items))
    n_train_neg = train_size - n_train_pos
    n_test_pos = int(test_size * len(pos) / len(items))
    n_test_neg = test_size - n_test_pos

    train = pos[:n_train_pos] + neg[:n_train_neg]
    test = pos[n_train_pos: n_train_pos + n_test_pos] + neg[n_train_neg: n_train_neg + n_test_neg]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def build_classification_dataset(
    steps: list[AnnotatedStep],
    mode: AggMode = "concat",
    n_tail: int = 5,
    layer_idx: int = -1,
    gamma: float = CONFIDENCE_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (features, labels) tensors for classification training.

    Applies confidence filtering (Eq. 10) before extracting features.

    Returns:
        features: (N, d_in)
        labels:   (N,) float tensor of 0/1 values
    """
    filtered = confidence_filter(steps, gamma)
    feat_list, label_list = [], []
    for step in filtered:
        if step.hidden_states is None:
            raise ValueError("hidden_states must be populated before calling build_dataset.")
        f = extract_step_feature(step.hidden_states, mode=mode, n_tail=n_tail, layer_idx=layer_idx)
        feat_list.append(f)
        label_list.append(float(step.label))

    features = torch.stack(feat_list)       # (N, d_in)
    labels = torch.tensor(label_list)       # (N,)
    return features, labels


def build_preference_dataset(
    pairs: list[PreferenceStep],
    mode: AggMode = "concat",
    n_tail: int = 5,
    layer_idx: int = -1,
) -> torch.Tensor:
    """Build (N, 2, d_in) tensor for preference training.

    Index 0 = positive (preferred), index 1 = negative (dispreferred).
    """
    pair_list = []
    for pair in pairs:
        if pair.hidden_pos is None or pair.hidden_neg is None:
            raise ValueError("hidden_pos/hidden_neg must be populated.")
        f_pos = extract_step_feature(pair.hidden_pos, mode=mode, n_tail=n_tail, layer_idx=layer_idx)
        f_neg = extract_step_feature(pair.hidden_neg, mode=mode, n_tail=n_tail, layer_idx=layer_idx)
        pair_list.append(torch.stack([f_pos, f_neg], dim=0))  # (2, d_in)

    return torch.stack(pair_list)  # (N, 2, d_in)
