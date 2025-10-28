"""Model configuration."""

from __future__ import annotations

from pydantic import Field

from .base import ModuleConfig


class ModelConfig(ModuleConfig):
    """Configuration for model selection or loading."""

    checkpoint: str | None = Field(
        default=None, description="Optional path to a model checkpoint."
    )
