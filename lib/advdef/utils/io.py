"""Serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml_file(path: Path) -> Any:
    """Load a YAML file and return the parsed object."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(obj: Any, path: Path) -> None:
    """Persist an object as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(obj, handle, sort_keys=False)


def save_json(obj: Any, path: Path, *, indent: int = 2) -> None:
    """Persist an object as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=indent, sort_keys=False)
        handle.write("\n")
