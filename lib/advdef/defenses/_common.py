"""Shared helpers for defense implementations."""

from __future__ import annotations

import hashlib
import json

from advdef.config import DefenseConfig


def slugify_label(label: str, *, default: str = "defense") -> str:
    sanitized = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in label.strip().lower())
    sanitized = "-".join(filter(None, sanitized.split("-")))
    return sanitized or default


def build_config_identifier(config: DefenseConfig, *, default_prefix: str) -> str:
    name = config.name
    if name:
        return slugify_label(name, default=slugify_label(default_prefix))

    params = config.params or {}
    serialized = json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:8]
    base = slugify_label(default_prefix)
    return f"{base}-{digest}"


__all__ = ["slugify_label", "build_config_identifier"]
