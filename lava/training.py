"""
Probe training utilities (Section 3.2.4 and 4.3).

Supports two supervision modes:
  - 'classification': binary labels z ∈ {0, 1} → BCEWithLogitsLoss (Eq. 8)
  - 'preference':     paired features (f+, f-) → Bradley-Terry loss (Eq. 9)

Optimizer: Adam(lr=5e-3, betas=(0.9, 0.999), weight_decay=1e-4)
Gradient clipping: max_norm=1.0
Batch size: 64, Epochs: 50–200
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SupervisionMode = Literal["classification", "preference"]


@dataclass
class TrainConfig:
    lr: float = 5e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 100
    max_grad_norm: float = 1.0
    device: str = "cpu"
    concept_weights: List[float] = field(default_factory=list)


def bradley_terry_loss(logits_pos: torch.Tensor, logits_neg: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry pairwise preference loss (Eq. 9).

    L = -E[log σ(s(f+) - s(f-))]
    """
    return -torch.log(torch.sigmoid(logits_pos - logits_neg) + 1e-8).mean()


def train_probe(
    probe: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    mode: SupervisionMode = "classification",
    config: TrainConfig | None = None,
    verbose: bool = False,
) -> dict:
    """Train a single probe.

    Args:
        probe: LinearProbe or MLPProbe.
        features: For 'classification': (N, d_in). For 'preference': (N, 2, d_in)
                  where [:, 0, :] = positive, [:, 1, :] = negative.
        labels: For 'classification': (N,) binary float. Ignored for 'preference'.
        mode: Loss type.
        config: Training hyperparameters.
        verbose: Print loss every 10 epochs.

    Returns:
        dict with 'train_loss_history' list.
    """
    if config is None:
        config = TrainConfig()

    probe = probe.to(config.device)
    features = features.to(config.device)
    if labels is not None:
        labels = labels.to(config.device)

    optimizer = torch.optim.Adam(
        probe.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    if mode == "classification":
        loss_fn = nn.BCEWithLogitsLoss()
        if config.concept_weights:
            w = torch.tensor(config.concept_weights, device=config.device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=w)
        dataset = TensorDataset(features, labels)
    else:
        # preference: features shape (N, 2, d_in)
        pos_feats = features[:, 0, :]
        neg_feats = features[:, 1, :]
        dataset = TensorDataset(pos_feats, neg_feats)

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    history = []

    probe.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            if mode == "classification":
                f, z = batch
                logits = probe(f)
                loss = loss_fn(logits, z.float())
            else:
                f_pos, f_neg = batch
                logits_pos = probe(f_pos)
                logits_neg = probe(f_neg)
                loss = bradley_terry_loss(logits_pos, logits_neg)

            loss.backward()
            nn.utils.clip_grad_norm_(probe.parameters(), config.max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item() * len(batch[0])

        avg_loss = epoch_loss / len(dataset)
        history.append(avg_loss)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config.epochs}  loss={avg_loss:.4f}")

    probe.eval()
    return {"train_loss_history": history}


@torch.no_grad()
def evaluate_probe(
    probe: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    device: str = "cpu",
) -> dict:
    """Evaluate a trained probe on held-out features/labels.

    Returns accuracy, F1 (on the negative class, i.e., incorrect/unsafe),
    precision, and recall.
    """
    probe = probe.to(device).eval()
    features = features.to(device)
    labels = labels.to(device)

    scores = probe.score(features)  # (N,)
    preds = (scores >= threshold).long()
    gt = labels.long()

    acc = (preds == gt).float().mean().item()

    # F1 on the negative class (label=0: incorrect/unsafe)
    tn = ((preds == 0) & (gt == 0)).sum().item()
    fp = ((preds == 1) & (gt == 0)).sum().item()
    fn = ((preds == 0) & (gt == 1)).sum().item()

    neg_precision = tn / (tn + fn + 1e-8)
    neg_recall = tn / (tn + fp + 1e-8)
    f1_neg = 2 * neg_precision * neg_recall / (neg_precision + neg_recall + 1e-8)

    return {
        "accuracy": acc,
        "f1_negative": f1_neg,
        "neg_precision": neg_precision,
        "neg_recall": neg_recall,
        "threshold": threshold,
    }
