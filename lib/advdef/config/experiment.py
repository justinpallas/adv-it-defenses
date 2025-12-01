"""Top-level experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from .dataset import DatasetConfig
from .defense import DefenseConfig
from .evaluation import EvaluationConfig
from .inference import InferenceConfig
from .model import ModelConfig


class ExperimentConfig(BaseModel):
    """Aggregates the full experiment pipeline configuration."""

    name: str = Field(..., description="Human readable experiment name.")
    description: str | None = Field(default=None, description="Optional description.")
    seed: int = Field(default=123, description="Random seed for reproducibility.")
    work_dir: Path = Field(default=Path("runs"), description="Base directory for run artifacts.")
    dataset: DatasetConfig
    defenses: List[DefenseConfig] = Field(default_factory=list)
    model: ModelConfig = Field(default_factory=lambda: ModelConfig(type="timm"))
    additional_inference_models: List[ModelConfig] = Field(
        default_factory=list,
        description=(
            "Optional list of extra models to evaluate during inference. "
            "The primary model still drives attack generation."
        ),
    )
    inference: InferenceConfig = Field(default_factory=lambda: InferenceConfig(type="timm"))
    evaluation: EvaluationConfig = Field(default_factory=lambda: EvaluationConfig(type="imagenet"))
    tags: List[str] = Field(default_factory=list)

    @field_validator("work_dir", mode="before")
    def _coerce_path(cls, value: str | Path) -> Path:
        return Path(value)

    def run_directory(self, run_id: str) -> Path:
        return self.work_dir / run_id
