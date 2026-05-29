"""
Latent feature extraction from LLM hidden states (Section 3.2.2).

The default aggregation ϕ(·) concatenates the last `n_tail` token
representations from the final layer:

    f = [h^(L*)_{T-n+1}, ..., h^(L*)_T]  ∈ R^{n*d_model}

Additional aggregation strategies (pooling, min) are provided for ablation.
"""

from __future__ import annotations
from typing import Literal

import torch


AggMode = Literal["concat", "pooling", "min", "last"]


def aggregate_hidden_states(
    hidden_states: torch.Tensor,
    mode: AggMode = "concat",
    n_tail: int = 5,
    layer_idx: int = -1,
) -> torch.Tensor:
    """Aggregate per-token hidden states into a single step feature vector.

    Args:
        hidden_states: Tensor of shape (L, T, d) where L = number of layers,
                       T = sequence length of the step, d = hidden dim.
                       Can also be (T, d) if a single layer is passed.
        mode: Aggregation strategy.
            'concat'  – concatenate last n_tail tokens from `layer_idx`.
            'pooling' – mean-pool all tokens across last 4 layers + `layer_idx`.
            'min'     – element-wise min across last 4 layers + `layer_idx`.
            'last'    – single token (position -1) from `layer_idx`.
        n_tail: Number of trailing tokens to concatenate in 'concat' mode.
        layer_idx: Which layer to use (-1 = last layer).

    Returns:
        Feature vector f of shape (d_in,) where d_in depends on mode/n_tail.
    """
    if hidden_states.dim() == 2:
        # (T, d) — single layer already selected
        h = hidden_states
        return _aggregate_single_layer(h, mode, n_tail)

    # hidden_states: (L, T, d)
    L, T, d = hidden_states.shape

    if mode == "concat":
        h = hidden_states[layer_idx]  # (T, d)
        return _aggregate_single_layer(h, "concat", n_tail)

    elif mode == "last":
        h = hidden_states[layer_idx]  # (T, d)
        return h[-1]  # (d,)

    elif mode in ("pooling", "min"):
        # Use last 4 layers + the designated layer_idx
        layer_indices = list(range(max(0, L - 4), L))
        selected = hidden_states[layer_indices]  # (k, T, d)
        # Mean-pool or min over tokens first, then aggregate layers
        pooled = selected.mean(dim=1)  # (k, d)
        if mode == "pooling":
            return pooled.mean(dim=0)  # (d,)
        else:
            return pooled.min(dim=0).values  # (d,)

    else:
        raise ValueError(f"Unknown aggregation mode '{mode}'.")


def _aggregate_single_layer(h: torch.Tensor, mode: AggMode, n_tail: int) -> torch.Tensor:
    """h: (T, d) → feature vector."""
    T, d = h.shape
    if mode == "concat":
        tail = h[max(0, T - n_tail):]  # (min(n_tail,T), d)
        # Pad to exactly n_tail tokens if sequence is shorter
        if tail.shape[0] < n_tail:
            pad = torch.zeros(n_tail - tail.shape[0], d, dtype=tail.dtype, device=tail.device)
            tail = torch.cat([pad, tail], dim=0)
        return tail.reshape(-1)  # (n_tail * d,)
    elif mode == "pooling":
        return h.mean(dim=0)  # (d,)
    elif mode == "min":
        return h.min(dim=0).values  # (d,)
    elif mode == "last":
        return h[-1]  # (d,)
    else:
        raise ValueError(f"Unknown mode '{mode}'.")


def extract_step_feature(
    step_hidden_states: list[torch.Tensor] | torch.Tensor,
    mode: AggMode = "concat",
    n_tail: int = 5,
    layer_idx: int = -1,
) -> torch.Tensor:
    """Extract a single feature vector for one reasoning step.

    Args:
        step_hidden_states: Either a list of per-layer tensors (each (T, d)),
                            or a stacked tensor of shape (L, T, d).
        mode: Aggregation strategy (default: 'concat').
        n_tail: Tail tokens to concatenate (default: 5, as in the paper).
        layer_idx: Which layer index to use for 'concat'/'last' modes.

    Returns:
        Flat feature vector, shape (d_in,).
    """
    if isinstance(step_hidden_states, list):
        step_hidden_states = torch.stack(step_hidden_states, dim=0)  # (L, T, d)
    return aggregate_hidden_states(step_hidden_states, mode=mode, n_tail=n_tail, layer_idx=layer_idx)


def compute_feature_dim(d_model: int, mode: AggMode = "concat", n_tail: int = 5) -> int:
    """Return the feature dimensionality produced by the given aggregation."""
    if mode == "concat":
        return n_tail * d_model
    else:
        return d_model
