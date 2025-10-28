"""Filesystem utilities."""

from __future__ import annotations

import shutil
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def symlink_or_copy(src: Path, dst: Path) -> None:
    """Create a symlink pointing to src; fallback to copy if unsupported."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
