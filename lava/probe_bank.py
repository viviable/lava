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
from .feature_extraction import AggMode, extract_step_feature


@dataclass
class ConceptConfig:
    name: str
    probe_type: str = "mlp"               # 'linear' or 'mlp'
    supervision: SupervisionMode = "classification"
    threshold: float = 0.5
    agg_mode: AggMode = "concat"
    n_tail: int = 5
    best_layer: Optional[int] = None      # L* chosen by layer search; None = use global layer_idx


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
            features:      Training features. For classification: (N, d_in) for a
                           single pre-selected layer, or (N, L, d_in) to trigger a
                           layer search that picks the best layer L*. For
                           preference: (N, 2, d_in).
            labels:        Binary labels (N,) for classification; None for preference.
            train_config:  Optimizer/training hyperparameters.
            verbose:       Print training progress.

        Returns:
            Training history dict (includes 'best_layer'/'per_layer_acc' when a
            layer search was performed).
        """
        # (N, L, d_in) classification features -> search layers for L*.
        if config.supervision == "classification" and features.dim() == 3:
            return self._add_concept_layer_search(
                config, features, labels, train_config, verbose
            )

        d_in = features.shape[-1]
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

    def _add_concept_layer_search(
        self,
        config: ConceptConfig,
        features: torch.Tensor,   # (N, L, d_in)
        labels: torch.Tensor,     # (N,)
        train_config: Optional[TrainConfig],
        verbose: bool,
        val_size: float = 0.2,
        seed: int = 42,
    ) -> dict:
        """Train one probe per layer, keep the one with best validation accuracy.

        Mirrors MoC's L* selection: carve a stratified internal val split out of
        the training data, score each layer's probe on it, and retain the winner
        (trained on train-minus-val). The chosen layer is recorded on the config
        so inference reads features from the same layer.
        """
        N, L, d_in = features.shape
        device = (train_config.device if train_config is not None else self.device)

        # Stratified train/val split (fixed seed) for early-stop-free layer picking.
        g = torch.Generator().manual_seed(seed)
        labels_long = labels.long()
        tr_idx_parts, val_idx_parts = [], []
        for cls in (0, 1):
            cls_idx = torch.nonzero(labels_long == cls, as_tuple=False).squeeze(1)
            if cls_idx.numel() == 0:
                continue
            cls_idx = cls_idx[torch.randperm(cls_idx.numel(), generator=g)]
            n_val = max(1, int(cls_idx.numel() * val_size)) if cls_idx.numel() > 1 else 0
            val_idx_parts.append(cls_idx[:n_val])
            tr_idx_parts.append(cls_idx[n_val:])
        tr_idx = torch.cat(tr_idx_parts)
        val_idx = torch.cat(val_idx_parts) if val_idx_parts else tr_idx

        best_acc, best_layer, best_probe, best_hist = -1.0, 0, None, None
        per_layer_acc: list[float] = []
        for layer in range(L):
            Xl = features[:, layer, :]
            probe = build_probe(config.probe_type, d_in)
            hist = train_probe(
                probe, Xl[tr_idx], labels[tr_idx],
                mode="classification", config=train_config, verbose=False,
            )
            metrics = evaluate_probe(
                probe, Xl[val_idx], labels[val_idx],
                threshold=config.threshold, device=device,
            )
            acc = metrics["accuracy"]
            per_layer_acc.append(acc)
            if verbose:
                print(f"  layer {layer}: val_acc={acc:.3f}")
            if acc > best_acc:
                best_acc, best_layer, best_probe, best_hist = acc, layer, probe, hist

        config.best_layer = best_layer
        for p in best_probe.parameters():
            p.requires_grad_(False)
        best_probe.eval()

        self._probes.append(best_probe.to(self.device))
        self._configs.append(config)
        self._thresholds.append(config.threshold)

        best_hist["best_layer"] = best_layer
        best_hist["per_layer_acc"] = per_layer_acc
        best_hist["val_acc"] = best_acc
        return best_hist

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
    def verify_hidden(
        self,
        hidden: torch.Tensor,
        agg_mode: AggMode = "concat",
        n_tail: Optional[int] = None,
        default_layer_idx: int = -1,
    ) -> tuple[bool, torch.Tensor]:
        """Verify a step from raw per-layer hidden states, honoring per-concept L*.

        A single backbone forward produces `hidden` of shape (L, T, d); each
        concept's feature is aggregated at its own `best_layer` (falling back to
        `default_layer_idx` when the concept has none), so probes trained on
        different layers all read the layer they were trained on.

        Args:
            hidden:            (L, T, d) stacked hidden states for the step tokens.
            agg_mode:          Aggregation mode (must match training).
            n_tail:            Tail tokens for 'concat' (defaults to self.n_tail).
            default_layer_idx: Layer used for concepts with best_layer=None.

        Returns:
            (accept: bool, scores: Tensor of shape (K,))
        """
        n_tail = self.n_tail if n_tail is None else n_tail
        score_list = []
        for cfg, probe in zip(self._configs, self._probes):
            layer = cfg.best_layer if cfg.best_layer is not None else default_layer_idx
            f = extract_step_feature(hidden, mode=agg_mode, n_tail=n_tail, layer_idx=layer)
            f = f.unsqueeze(0).to(self.device)  # (1, d_in)
            score_list.append(probe.score(f).squeeze(0))
        scores = torch.stack(score_list)  # (K,)
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
                    "best_layer": c.best_layer,
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
