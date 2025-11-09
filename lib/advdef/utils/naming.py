"""Naming utilities for reproducible identifiers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def slugify_label(label: str, *, default: str = "label") -> str:
    """Convert an arbitrary label into a filesystem-friendly slug."""

    sanitized = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in label.strip().lower())
    sanitized = "-".join(filter(None, sanitized.split("-")))
    return sanitized or default


def build_identifier(
    *,
    name: str | None,
    params: Mapping[str, Any] | None,
    default_prefix: str,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Create a stable identifier for a configuration."""

    base = slugify_label(default_prefix or "identifier")
    if name:
        return slugify_label(name, default=base)

    payload: dict[str, Any] = {"params": params or {}}
    if extra:
        payload["extra"] = extra

    serialized = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{digest}"


__all__ = ["slugify_label", "build_identifier"]
