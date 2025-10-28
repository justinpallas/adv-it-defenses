"""Core orchestration primitives for advdef."""

from __future__ import annotations

from .context import RunContext
from .pipeline import Pipeline, PipelineStep
from .registry import (
    ATTACKS,
    DATASETS,
    DEFENSES,
    EVALUATORS,
    INFERENCE_BACKENDS,
    MODELS,
    register_attack,
    register_dataset,
    register_defense,
    register_evaluator,
    register_inference,
    register_model,
)

__all__ = [
    "RunContext",
    "Pipeline",
    "PipelineStep",
    "ATTACKS",
    "DATASETS",
    "DEFENSES",
    "MODELS",
    "INFERENCE_BACKENDS",
    "EVALUATORS",
    "register_dataset",
    "register_attack",
    "register_defense",
    "register_model",
    "register_inference",
    "register_evaluator",
]
