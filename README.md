# LAVA: Latent Activation Verification Assistance

A continual-learning framework for step-level verification of LLM reasoning. LAVA pairs a fast draft model with a strong fallback model and uses a bank of frozen, per-concept probes over a backbone's hidden states to decide, step-by-step, whether to accept the draft or regenerate with the strong model.

Reference: *Accelerating Reasoning Through Step Speculation with Parallel Hidden States Verification* (paper PDF included in this repo).

## Key idea

At each reasoning step:

1. The draft model `Mw` proposes a candidate step (vLLM, stops at `\n\n`).
2. A frozen HF backbone runs a no-grad forward over `(context + step)` to expose hidden states.
3. LAVA extracts a feature from those hidden states (Eq. 3).
4. All `K` concept probes evaluate the feature in parallel (`O(Kd)`).
5. The step is accepted iff every probe's score clears its threshold (Eq. 6); otherwise the strong model `Ms` regenerates the step.

Each concept gets its own probe trained on ~1k samples, then frozen. Adding a new concept never touches prior probes — backward transfer is zero by construction (Prop. 6.1).

## Repository layout

```
lava/
  probes.py             Linear / MLP probe modules
  feature_extraction.py Hidden-state aggregation (concat / mean / min / last)
  training.py           Probe training: classification + Bradley-Terry preference
  probe_bank.py         ProbeBank: grow, freeze, verify, save/load
  data_pipeline.py      Trajectory decomposition + dataset builders
  backbone.py           HFHiddenStateBackbone — frozen feature backbone
  speculative.py        LAVAPipeline + VLLMModelInterface (OpenAI client)
  continual.py          Continual-learning eval: accuracy matrix, FA bounds, latency
scripts/
  extract_features.py   Run the backbone over annotated steps -> train/test .pt tensors
  extract_features.sh     ^ shell wrapper
  train_probe.py        Train one concept probe from extracted features
  train_probe.sh          ^ shell wrapper
  run_lava.py           Run the speculative pipeline over GSM8K / AIME / MATH-500 / HMMT
  inference.sh            ^ shell wrapper
  score_runs.py         Score runs the way G-OPD does (boxed extraction + math_verify)
  score.sh                ^ shell wrapper
  eval_continual.py     Run the Experiment-A continual-learning protocol
  eval_continual.sh       ^ shell wrapper
tests/
  test_lava.py          Unit tests (probes, training, probe bank, continual)
```

## Install

```bash
pip install -r requirements.txt
# vLLM is served as a separate process; install it in the same env:
pip install vllm
```

## End-to-end pipeline (.sh entry points)

```bash
# 1. Extract features from a JSONL of annotated steps (one concept at a time).
#    Input row: {"context": "...", "step_text": "...", "label": 0|1, "confidence": 0..1}
bash scripts/extract_features.sh data/raw/math_correctness.jsonl \
                                 data/task_0_math_correctness

# 2. Train a frozen probe for that concept.
bash scripts/train_probe.sh data/task_0_math_correctness \
                            math_correctness \
                            probes/

# 3. Run speculative reasoning (launch the two vLLM servers first — see below).
bash scripts/inference.sh aime 0 0      # dataset, problem_id, repeat_id

# 4. Score the run dir (G-OPD-style: last \boxed{} + math_verify).
bash scripts/score.sh results/lava

# 5. (Optional) Continual-learning eval across multiple concept tasks.
bash scripts/eval_continual.sh data/ results/continual.json
```

Every wrapper exposes its knobs as environment variables (see the comment block at the top of each `.sh`). For example: `D_MODEL=2048 PROBE_TYPE=linear bash scripts/train_probe.sh ...`.

## Train a probe

Given features `(N, n_tail * d_model)` and binary labels `(N,)`:

```bash
python scripts/train_probe.py \
    --features path/to/features.pt \
    --labels path/to/labels.pt \
    --concept math_correctness \
    --probe_type mlp \
    --d_model 2048 \
    --output probes/math_correctness
```

For preference supervision pass `--supervision preference` and features of shape `(N, 2, d_in)` (no labels file; the first element of each pair is the preferred one).

## Run inference

LAVA generation is split across three processes (the same shape as the SpecReason reference implementation): one vLLM server for the draft model, one for the strong model, and the runner.

```bash
# 1. Draft server (small, fast)
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --port 30001 --enable-prefix-caching

# 2. Strong server (large, slow)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/QwQ-32B --tensor-parallel-size 2 \
    --port 30000 --enable-prefix-caching

# 3. Runner
python scripts/run_lava.py \
    --dataset_name aime --problem_id 0 --repeat_id 0 \
    --probe_bank probes/ \
    --hidden_backbone deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --draft_url  http://localhost:30001/v1 \
    --strong_url http://localhost:30000/v1 \
    --output_dir results/lava
```

The runner writes per-problem pickle + txt files containing one metadata dict per step (draft text, probe scores, accept/reject, strong-regenerated text if any, per-stage token counts and latencies, stop reason). The schema matches the SpecReason logs with the LLM-as-judge score replaced by `probe_scores` + `concept_names`. A sibling `<repeat>.gold.json` records the dataset, problem text, and ground-truth answer so the scorer can grade later.

Supported benchmarks (all math reasoning):

| `--dataset_name` | HF id | # problems |
| --- | --- | --- |
| `gsm8k`   | `openai/gsm8k` (`main`, test) | 1319 |
| `aime`    | `HuggingFaceH4/aime_2024` (train) | 30 |
| `math500` | `HuggingFaceH4/MATH-500` (test) | 500 |
| `hmmt`    | `MathArena/hmmt_feb_2025` (train) | 30 |

## Score

After a sweep, aggregate accuracy / tokens / latency / draft-accept rate using G-OPD-style grading (last `\boxed{...}` + `math_verify` for math equivalence):

```bash
python scripts/score_runs.py --results_dir results/lava
# or restrict: --datasets gsm8k math500
```

## Programmatic use

```python
from lava import (
    ProbeBank, ConceptConfig, TrainConfig,
    LAVAPipeline, LAVAConfig, VLLMModelInterface, HFHiddenStateBackbone,
)

bank = ProbeBank(d_model=1536, n_tail=5, device="cuda")
bank.add_concept(
    ConceptConfig(name="math_correctness", probe_type="mlp", threshold=0.5),
    features, labels,
    train_config=TrainConfig(lr=5e-3, epochs=100),
)
# add more concepts the same way — prior probes stay frozen

draft  = VLLMModelInterface("http://localhost:30001/v1")
strong = VLLMModelInterface("http://localhost:30000/v1")

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
mdl = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
backbone = HFHiddenStateBackbone(mdl, tok, device="cuda")

pipeline = LAVAPipeline(draft, strong, bank, hidden_backbone=backbone,
                        config=LAVAConfig(max_steps=32))
result = pipeline.run("Prove that sqrt(2) is irrational.", token_budget=8192)
print(result.draft_accept_rate, result.full_text)
```

## Tests

```bash
python -m pytest tests/
```
