"""Inference configuration."""

from __future__ import annotations

from pydantic import Field

from .base import ModuleConfig


class InferenceConfig(ModuleConfig):
    """Configuration for inference backend."""

    results_dir: str | None = Field(
        default=None, description="Optional directory to store raw inference results."
    )
