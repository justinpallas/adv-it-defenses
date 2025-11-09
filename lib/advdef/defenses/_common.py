"""Shared helpers for defense implementations."""

from __future__ import annotations

from advdef.config import DefenseConfig
from advdef.utils.naming import build_identifier, slugify_label as _slugify_label


def slugify_label(label: str, *, default: str = "defense") -> str:
    return _slugify_label(label, default=default)


def build_config_identifier(config: DefenseConfig, *, default_prefix: str) -> str:
    return build_identifier(
        name=config.name,
        params=config.params,
        default_prefix=default_prefix,
    )


__all__ = ["slugify_label", "build_config_identifier"]
