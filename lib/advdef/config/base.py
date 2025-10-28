"""Base configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ModuleConfig(BaseModel):
    """Generic config with type-discriminated parameters."""

    type: str = Field(..., description="Registry name of the component.")
    name: Optional[str] = Field(
        default=None, description="Optional human-friendly label for this component."
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary keyword arguments for the component."
    )


class PathConfig(BaseModel):
    """Helper for filesystem related configuration."""

    path: Path
