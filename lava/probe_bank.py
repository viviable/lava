"""
ProbeBank: manages a growing set of K independent per-concept probes (Section 3.2.3).

Each concept gets its own probe trained on ~1k samples. Once trained, a probe's
parameters are frozen and never modified — this is the architectural guarantee
behind zero backward interference (Prop. 6.1).

At inference, ALL K probes evaluate in a single forward pass (O(Kd) cost),
and the step is accepted only if every probe's score exceeds its threshold τ_k.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .probes import build_probe, LinearProbe, MLPProbe
from .training import SupervisionMode, TrainConfig, train_probe, evaluate_probe
from .feature_extraction import AggMode


@dataclass
class ConceptConfig:
    name: str
    probe_type: str = "mlp"               # 'linear' or 'mlp'
    supervision: SupervisionMode = "classification"
    threshold: float = 0.5
    agg_mode: AggMode = "concat"
    n_tail: int = 5


class ProbeBank:
    """A bank of independent per-concept probes.

    Probes are stored in insertion order (concept index 0, 1, ..., K-1).
    After training, each probe is frozen (requires_grad=False).

    Args:
        d_model: Hidden dimension of the backbone LLM.
        n_tail:  Number of tail tokens concatenated for feature extraction.
        device:  Torch device string.
    """

    def __init__(self, d_model: int, n_tail: int = 5, device: str = "cpu"):
        self.d_model = d_model
        self.n_tail = n_tail
        self.device = device

        self._probes: list[nn.Module] = []
        self._configs: list[ConceptConfig] = []
        self._thresholds: list[float] = []

    @property
    def k(self) -> int:
        return len(self._probes)

    def add_concept(
        self,
        config: ConceptConfig,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        train_config: Optional[TrainConfig] = None,
        verbose: bool = False,
    ) -> dict:
        """Train a new probe for a concept and add it to the bank.

        After training the probe's parameters are frozen. Prior probes are
        untouched — BWT = 0 by construction.

        Args:
            config:        Concept metadata + probe architecture choice.
            features:      Training features. Shape (N, d_in) for classification,
                           (N, 2, d_in) for preference.
            labels:        Binary labels (N,) for classification; None for preference.
            train_config:  Optimizer/training hyperparameters.
            verbose:       Print training progress.

        Returns:
            Training history dict.
        """
        d_in = self.n_tail * self.d_model  # concat aggregation
        probe = build_probe(config.probe_type, d_in)

        history = train_probe(
            probe,
            features,
            labels,
            mode=config.supervision,
            config=train_config,
            verbose=verbose,
        )

        # Freeze: prior probes are never touched again
        for p in probe.parameters():
            p.requires_grad_(False)
        probe.eval()

        self._probes.append(probe.to(self.device))
        self._configs.append(config)
        self._thresholds.append(config.threshold)

        return history

    @torch.no_grad()
    def verify(self, f: torch.Tensor) -> tuple[bool, torch.Tensor]:
        """Run all K probes on feature vector f and decide accept/reject.

        Implements Eq. (6): ACCEPT ⟺ ∀k Pk(f) ≥ τk

        Args:
            f: Feature vector of shape (d_in,) or (1, d_in).

        Returns:
            (accept: bool, scores: Tensor of shape (K,))
        """
        if f.dim() == 1:
            f = f.unsqueeze(0)  # (1, d_in)
        f = f.to(self.device)

        scores = torch.stack([p.score(f).squeeze(0) for p in self._probes])  # (K,)
        thresholds = torch.tensor(self._thresholds, device=self.device)
        accept = bool((scores >= thresholds).all().item())
        return accept, scores

    @torch.no_grad()
    def verify_batch(self, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch verification for N feature vectors.

        Args:
            F: (N, d_in)

        Returns:
            accept_mask: (N,) bool tensor
            scores:      (N, K) float tensor
        """
        F = F.to(self.device)
        scores = torch.stack([p.score(F) for p in self._probes], dim=1)  # (N, K)
        thresholds = torch.tensor(self._thresholds, device=self.device)  # (K,)
        accept_mask = (scores >= thresholds).all(dim=1)  # (N,)
        return accept_mask, scores

    def get_probe(self, concept_name: str) -> Optional[nn.Module]:
        for cfg, probe in zip(self._configs, self._probes):
            if cfg.name == concept_name:
                return probe
        return None

    def concept_names(self) -> list[str]:
        return [c.name for c in self._configs]

    def save(self, path: str):
        """Save all probe state dicts and metadata."""
        os.makedirs(path, exist_ok=True)
        for i, (probe, cfg) in enumerate(zip(self._probes, self._configs)):
            torch.save(probe.state_dict(), os.path.join(path, f"probe_{i}_{cfg.name}.pt"))
        meta = {
            "d_model": self.d_model,
            "n_tail": self.n_tail,
            "device": self.device,
            "concepts": [
                {
                    "name": c.name,
                    "probe_type": c.probe_type,
                    "supervision": c.supervision,
                    "threshold": c.threshold,
                    "agg_mode": c.agg_mode,
                    "n_tail": c.n_tail,
                }
                for c in self._configs
            ],
        }
        torch.save(meta, os.path.join(path, "metadata.pt"))

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ProbeBank":
        """Load a previously saved ProbeBank."""
        meta = torch.load(os.path.join(path, "metadata.pt"), map_location=device)
        bank = cls(d_model=meta["d_model"], n_tail=meta["n_tail"], device=device)
        d_in = meta["n_tail"] * meta["d_model"]

        for i, c in enumerate(meta["concepts"]):
            cfg = ConceptConfig(**c)
            probe = build_probe(cfg.probe_type, d_in)
            state = torch.load(
                os.path.join(path, f"probe_{i}_{cfg.name}.pt"), map_location=device
            )
            probe.load_state_dict(state)
            for p in probe.parameters():
                p.requires_grad_(False)
            probe.eval()
            bank._probes.append(probe.to(device))
            bank._configs.append(cfg)
            bank._thresholds.append(cfg.threshold)

        return bank
