"""
Speculative reasoning pipeline with LAVA verification (Section 3, Algorithm 1).

Generation uses vLLM via its OpenAI-compatible HTTP API (the pattern from the
SpecReason reference implementation). vLLM enforces real per-step stopping
through `stop=["\\n\\n"]`, so each step costs only the tokens it actually
emits — unlike a post-hoc truncated `transformers.generate` call.

Hidden states come from a separate `HFHiddenStateBackbone` (lava/backbone.py),
since vLLM's OpenAI server does not expose per-token activations.

At each step:
  1. Draft model Mw (vLLM) generates a candidate step, stopping at "\\n\\n".
  2. Hidden backbone runs a no-grad forward over (context + step).
  3. LAVA extracts the step feature and runs the probe bank.
  4. If ALL probes accept (score ≥ τ_k), keep the draft step.
  5. Otherwise, the strong model Ms (vLLM) regenerates the step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any

import time

import torch

from .probe_bank import ProbeBank
from .feature_extraction import AggMode
from .backbone import HFHiddenStateBackbone


STEP_DELIMITER = "\n\n"

# Default math system prompt — matches SpecReason's `get_first_user_msg`.
DEFAULT_MATH_SYSTEM_PROMPT = (
    "Solve the following math problem efficiently and clearly. Please reason "
    "step by step, separate logical reasoning steps with two newline characters "
    "(\\n\\n), and put your final answer within \\boxed{{}}.\nProblem: {problem}"
)

DEFAULT_MCQ_SYSTEM_PROMPT = (
    "What is the correct answer to the following problem? Please reason step "
    "by step. Separate logical reasoning steps with two newline characters "
    "(\\n\\n).\nPut the final answer **strictly** in the format \\boxed{{X}}, "
    "where X is a single letter (A, B, C, or D).\n\n"
    "**Example output:** \\boxed{{A}}\n\n"
    "Problem: {problem}.\nChoices:\n"
    "(A) {ans_a}\n(B) {ans_b}\n(C) {ans_c}\n(D) {ans_d}"
)

# Tokens that indicate the model has produced its final answer.
ANSWER_MARKERS = ("boxed", "Answer:", "ANSWER:")


@dataclass
class LAVAConfig:
    step_delimiter: str = STEP_DELIMITER
    agg_mode: AggMode = "concat"
    n_tail: int = 5
    layer_idx: int = -1
    max_steps: int = 64
    max_tokens_per_step: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    device: str = "cpu"


@dataclass
class StepResult:
    step_index: int
    text: str
    accepted: bool                                # True = draft accepted; False = strong used
    scores: Optional[torch.Tensor] = None         # (K,) probe scores
    draft_text: Optional[str] = None              # the draft proposal (kept even on reject)
    draft_tokens: int = 0
    strong_tokens: int = 0
    draft_latency: float = 0.0
    strong_latency: float = 0.0
    verify_latency: float = 0.0
    finished: bool = False                        # answer marker reached this step


@dataclass
class GenerationResult:
    query: str
    steps: list[StepResult] = field(default_factory=list)
    draft_accept_rate: float = 0.0
    total_steps: int = 0
    total_draft_tokens: int = 0
    total_strong_tokens: int = 0
    total_time: float = 0.0

    @property
    def full_text(self) -> str:
        return STEP_DELIMITER.join(s.text for s in self.steps)

    def compute_stats(self):
        if not self.steps:
            return
        self.total_steps = len(self.steps)
        self.draft_accept_rate = sum(s.accepted for s in self.steps) / self.total_steps
        self.total_draft_tokens = sum(s.draft_tokens for s in self.steps)
        self.total_strong_tokens = sum(s.strong_tokens for s in self.steps)
        self.total_time = sum(
            s.draft_latency + s.strong_latency + s.verify_latency for s in self.steps
        )


# ---------------------------------------------------------------------------
# Generation interface
# ---------------------------------------------------------------------------

class ModelInterface:
    """Abstract interface a concrete generation backend must implement."""

    def generate_step(
        self,
        problem: str,
        steps_so_far: list[str],
        max_tokens: int = 512,
        stop: str = STEP_DELIMITER,
    ) -> tuple[str, int, float, bool]:
        """Return (step_text, num_output_tokens, latency_seconds, finished)."""
        raise NotImplementedError


class VLLMModelInterface(ModelInterface):
    """OpenAI-API client targeting a local vLLM server.

    Mirrors SpecReason's `generate_new_step` pattern:
      - first step: pass the system prompt as a user message, generate fresh.
      - continuing: pass `<think>{steps_so_far}\\n\\n` as an assistant message
        and use `continue_final_message=True` so vLLM resumes that turn.
    """

    def __init__(
        self,
        base_url: str,
        model_name: Optional[str] = None,
        api_key: str = "EMPTY",
        system_prompt: str = DEFAULT_MATH_SYSTEM_PROMPT,
        temperature: float = 0.6,
        top_p: float = 0.95,
        prompt_kwargs: Optional[dict] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required for VLLMModelInterface. "
                "Install with: pip install openai"
            ) from e

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.prompt_kwargs = prompt_kwargs or {}

        if model_name is None:
            model_name = self.client.models.list().data[0].id
        self.model_name = model_name

    def _format_user_message(self, problem: str) -> str:
        return self.system_prompt.format(problem=problem, **self.prompt_kwargs)

    def generate_step(
        self,
        problem: str,
        steps_so_far: list[str],
        max_tokens: int = 512,
        stop: str = STEP_DELIMITER,
    ) -> tuple[str, int, float, bool]:
        user_msg = self._format_user_message(problem)

        if not steps_so_far:
            messages = [{"role": "user", "content": user_msg}]
            extra_body = {"add_generation_prompt": True}
        else:
            prior = STEP_DELIMITER.join(steps_so_far) + STEP_DELIMITER
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"<think>{prior}"},
            ]
            extra_body = {
                "add_generation_prompt": False,
                "continue_final_message": True,
            }

        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_tokens,
            stop=[stop],
            extra_body=extra_body,
        )
        elapsed = time.perf_counter() - t0

        step_text = response.choices[0].message.content
        n_out = response.usage.completion_tokens
        finished = any(marker in step_text for marker in ANSWER_MARKERS)
        return step_text, n_out, elapsed, finished


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class LAVAPipeline:
    """LAVA speculative reasoning pipeline.

    Args:
        draft_model:      ModelInterface for the weak/fast model Mw (vLLM).
        strong_model:     ModelInterface for the strong model Ms (vLLM).
        probe_bank:       Trained ProbeBank with K concepts.
        hidden_backbone:  Frozen HF model used to extract probe features.
                          If None, verification is skipped and every draft is
                          accepted (useful for sanity-checking the gen loop).
        config:           Runtime configuration.
    """

    def __init__(
        self,
        draft_model: ModelInterface,
        strong_model: ModelInterface,
        probe_bank: ProbeBank,
        hidden_backbone: Optional[HFHiddenStateBackbone] = None,
        config: Optional[LAVAConfig] = None,
    ):
        self.draft_model = draft_model
        self.strong_model = strong_model
        self.probe_bank = probe_bank
        self.hidden_backbone = hidden_backbone
        self.config = config or LAVAConfig()

    def _verify(self, context: str, step_text: str) -> tuple[bool, Optional[torch.Tensor], float]:
        """Extract hidden states + run probe bank. Returns (accept, scores, latency)."""
        if self.hidden_backbone is None or self.probe_bank.k == 0:
            return True, None, 0.0

        t0 = time.perf_counter()
        hidden = self.hidden_backbone.extract_step_hidden(context, step_text)
        # Per-concept L*: each probe reads features from the layer it was trained
        # on (ConceptConfig.best_layer), falling back to the global layer_idx.
        accept, scores = self.probe_bank.verify_hidden(
            hidden,
            agg_mode=self.config.agg_mode,
            n_tail=self.config.n_tail,
            default_layer_idx=self.config.layer_idx,
        )
        return accept, scores, time.perf_counter() - t0

    def run(self, problem: str, token_budget: Optional[int] = None) -> GenerationResult:
        """Execute LAVA inference for a single problem.

        Args:
            problem:       The question text.
            token_budget:  Optional cap on the sum of (draft + strong) tokens
                           across the whole trajectory. When the budget is hit
                           the loop exits and the last step is tagged
                           `finished=False`.
        """
        result = GenerationResult(query=problem)
        cfg = self.config

        steps_so_far: list[str] = []
        # Context string used for hidden-state extraction. We mirror the prompt
        # template the draft model sees so the backbone's tokenization matches.
        prompt_text = self.draft_model._format_user_message(problem) \
            if hasattr(self.draft_model, "_format_user_message") else problem

        for step_i in range(cfg.max_steps):
            # 1. Draft proposes a step.
            draft_text, draft_tokens, draft_latency, draft_finished = self.draft_model.generate_step(
                problem,
                steps_so_far,
                max_tokens=cfg.max_tokens_per_step,
                stop=cfg.step_delimiter,
            )

            # 2. + 3. Hidden states → probe bank.
            ctx_for_features = prompt_text + cfg.step_delimiter + cfg.step_delimiter.join(steps_so_far)
            if steps_so_far:
                ctx_for_features += cfg.step_delimiter
            accept, scores, verify_latency = self._verify(ctx_for_features, draft_text)

            strong_tokens = 0
            strong_latency = 0.0
            strong_finished = False
            if accept:
                chosen = draft_text
                finished = draft_finished
            else:
                # 4. Strong regenerates.
                strong_text, strong_tokens, strong_latency, strong_finished = self.strong_model.generate_step(
                    problem,
                    steps_so_far,
                    max_tokens=cfg.max_tokens_per_step,
                    stop=cfg.step_delimiter,
                )
                chosen = strong_text
                finished = strong_finished

            result.steps.append(
                StepResult(
                    step_index=step_i,
                    text=chosen,
                    accepted=accept,
                    scores=scores,
                    draft_text=draft_text,
                    draft_tokens=draft_tokens,
                    strong_tokens=strong_tokens,
                    draft_latency=draft_latency,
                    strong_latency=strong_latency,
                    verify_latency=verify_latency,
                    finished=finished,
                )
            )
            steps_so_far.append(chosen)

            # Edge case from SpecReason: model repeats the previous step.
            if len(steps_so_far) >= 2 and steps_so_far[-1] == steps_so_far[-2]:
                finished = True

            if finished:
                break

            if token_budget is not None:
                used = sum(s.draft_tokens + s.strong_tokens for s in result.steps)
                if used >= token_budget:
                    break

        result.compute_stats()
        return result
