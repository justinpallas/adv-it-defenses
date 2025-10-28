"""Evaluation configuration."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from .base import ModuleConfig


class EvaluationMetric(BaseModel):
    """Individual metric configuration."""

    name: str
    params: dict[str, object] = Field(default_factory=dict)


class EvaluationConfig(ModuleConfig):
    """Configuration for evaluation stage."""

    metrics: List[EvaluationMetric] = Field(default_factory=list)
