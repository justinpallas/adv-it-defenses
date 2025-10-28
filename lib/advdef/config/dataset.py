"""Dataset configuration."""

from __future__ import annotations

from typing import List

from pydantic import Field

from .attack import AttackConfig
from .base import ModuleConfig


class DatasetConfig(ModuleConfig):
    """Configuration for dataset preparation."""

    attacks: List[AttackConfig] = Field(
        default_factory=list,
        description="Attacks to generate adversarial variants for this dataset.",
    )


DatasetConfig.model_rebuild()
