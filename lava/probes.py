"""
Probe architectures for LAVA: LinearProbe and MLPProbe.

Both map a latent feature vector f ∈ R^d to a scalar score in [0, 1].
Feature vectors are typically the concatenation of the last 5 token
hidden states from the backbone's final layer (d_in = 5 * d_model).
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Tuple


class LinearProbe(nn.Module):
    """Single linear layer W ∈ R^{d_in × 1}.

    Xavier uniform init (gain=0.1) on weights; zero bias.
    Outputs a raw logit; apply sigmoid externally or use BCEWithLogitsLoss.
    """

    def __init__(self, d_in: int):
        super().__init__()
        self.linear = nn.Linear(d_in, 1)
        init.xavier_uniform_(self.linear.weight, gain=0.1)
        init.zeros_(self.linear.bias)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """f: (B, d_in) → logits: (B,)"""
        return self.linear(f).squeeze(-1)

    def score(self, f: torch.Tensor) -> torch.Tensor:
        """Return sigmoid-calibrated score in [0, 1]. Shape: (B,)"""
        return torch.sigmoid(self.forward(f))


class MLPProbe(nn.Module):
    """Two-hidden-layer MLP: d_in → 512 → 256 → 1 with ReLU + Dropout(0.2).

    Outputs a raw logit; apply sigmoid externally or use BCEWithLogitsLoss.
    """

    def __init__(self, d_in: int, hidden: Tuple[int, int] = (512, 256), dropout: float = 0.2):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(d_in, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=0.1)
                init.zeros_(m.bias)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """f: (B, d_in) → logits: (B,)"""
        return self.net(f).squeeze(-1)

    def score(self, f: torch.Tensor) -> torch.Tensor:
        """Return sigmoid-calibrated score in [0, 1]. Shape: (B,)"""
        return torch.sigmoid(self.forward(f))


def build_probe(probe_type: str, d_in: int, **kwargs) -> nn.Module:
    if probe_type == "linear":
        return LinearProbe(d_in)
    elif probe_type == "mlp":
        return MLPProbe(d_in, **kwargs)
    else:
        raise ValueError(f"Unknown probe_type '{probe_type}'. Choose 'linear' or 'mlp'.")
