"""Attack configuration."""

from __future__ import annotations

from pydantic import Field

from .base import ModuleConfig


class AttackConfig(ModuleConfig):
    """Configuration for a single adversarial attack."""

    targeted: bool = Field(
        default=False,
        description="Whether to run the attack in targeted mode when supported.",
    )
