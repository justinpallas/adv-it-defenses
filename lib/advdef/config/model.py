"""Model configuration."""

from __future__ import annotations

from pydantic import Field

from .base import ModuleConfig


class ModelConfig(ModuleConfig):
    """Configuration for model selection or loading."""

    checkpoint: str | None = Field(
        default=None, description="Optional path to a model checkpoint."
    )

    defense_checkpoints: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional map of defense names to fine-tuned checkpoints for extra inference runs."
        ),
    )
