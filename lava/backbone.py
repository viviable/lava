"""
Hidden-state backbone for probe feature extraction.

vLLM (via OpenAI-compatible API) handles step *generation* but does not expose
per-token hidden states. We therefore load a separate frozen HF transformers
model and run a no-grad forward pass over (context + step) to extract the
hidden states needed by the probes (Eq. 3, Sec 3.2.2).

The paper explicitly permits the feature backbone to differ from the draft
model: "M is the backbone whose activations are used for verification
(typically Mw, but other choices, including Ms or an assistance model are
possible)."
"""

from __future__ import annotations

from typing import Any, Optional

import torch


class HFHiddenStateBackbone:
    """Frozen HF causal-LM used only to expose hidden states.

    Args:
        model:        A loaded transformers AutoModelForCausalLM.
        tokenizer:    Corresponding tokenizer.
        device:       Torch device string.
        layer_idx:    Layer to expose (-1 = last). When the aggregation mode is
                      'concat' or 'last' a single layer is read; for 'pooling'
                      and 'min' the aggregator pulls the last 4 layers from the
                      stacked tensor we return.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cpu",
        layer_idx: int = -1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.layer_idx = layer_idx
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def extract_step_hidden(self, context: str, step_text: str) -> torch.Tensor:
        """Return hidden states for *only the step tokens*.

        Tokenize context and (context + step) separately to locate the step's
        token span, then forward the concatenated ids and slice out the trailing
        positions. Returns a tensor of shape (L, T_step, d).
        """
        ctx_ids = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(
            context + step_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"]

        ctx_len = ctx_ids.shape[1]
        full_len = full_ids.shape[1]
        step_len = max(1, full_len - ctx_len)

        full_ids = full_ids.to(self.device)
        outputs = self.model(full_ids, output_hidden_states=True, use_cache=False)

        # outputs.hidden_states: tuple of (L+1) tensors each (B=1, T, d)
        # (embedding output + per-layer outputs)
        per_layer = outputs.hidden_states  # tuple length L+1
        stacked = torch.stack([h[0] for h in per_layer], dim=0)  # (L+1, T, d)

        # Slice the trailing `step_len` token positions
        step_hidden = stacked[:, -step_len:, :]  # (L+1, T_step, d)
        return step_hidden.detach().cpu()
