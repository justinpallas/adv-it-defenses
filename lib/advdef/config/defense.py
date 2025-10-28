"""Defense configuration."""

from __future__ import annotations

from pydantic import Field

from .base import ModuleConfig


class DefenseConfig(ModuleConfig):
    """Configuration for a defense stage."""

    enabled: bool = Field(
        default=True,
        description="Allow disabling a defense without removing it from the pipeline.",
    )
