"""
LAVA: Latent Activation Verification Assistance

A continual-learning framework for step-level verification of LLM reasoning.
Probes are trained on frozen backbone hidden states; each new concept becomes
a new probe, prior probes are never touched.
"""

from .probes import LinearProbe, MLPProbe, build_probe
from .feature_extraction import extract_step_feature, compute_feature_dim, aggregate_hidden_states
from .training import train_probe, evaluate_probe, TrainConfig, bradley_terry_loss
from .probe_bank import ProbeBank, ConceptConfig
from .data_pipeline import (
    AnnotatedStep,
    PreferenceStep,
    decompose_trajectory,
    confidence_filter,
    build_classification_dataset,
    build_preference_dataset,
)
from .speculative import (
    LAVAPipeline,
    LAVAConfig,
    VLLMModelInterface,
    ModelInterface,
    GenerationResult,
    StepResult,
    DEFAULT_MATH_SYSTEM_PROMPT,
    DEFAULT_MCQ_SYSTEM_PROMPT,
)
from .backbone import HFHiddenStateBackbone
from .continual import (
    AccuracyMatrix,
    TaskData,
    run_experiment_a,
    cumulative_false_accept_bound,
    empirical_false_accept_rate,
    measure_inference_latency,
)

__all__ = [
    "LinearProbe",
    "MLPProbe",
    "build_probe",
    "extract_step_feature",
    "compute_feature_dim",
    "aggregate_hidden_states",
    "train_probe",
    "evaluate_probe",
    "TrainConfig",
    "bradley_terry_loss",
    "ProbeBank",
    "ConceptConfig",
    "AnnotatedStep",
    "PreferenceStep",
    "decompose_trajectory",
    "confidence_filter",
    "build_classification_dataset",
    "build_preference_dataset",
    "LAVAPipeline",
    "LAVAConfig",
    "VLLMModelInterface",
    "ModelInterface",
    "GenerationResult",
    "StepResult",
    "DEFAULT_MATH_SYSTEM_PROMPT",
    "DEFAULT_MCQ_SYSTEM_PROMPT",
    "HFHiddenStateBackbone",
    "AccuracyMatrix",
    "TaskData",
    "run_experiment_a",
    "cumulative_false_accept_bound",
    "empirical_false_accept_rate",
    "measure_inference_latency",
]
