"""Environment inspection helpers."""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any, Dict


def _try_git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def capture_environment() -> Dict[str, Any]:
    """Collect lightweight environment metadata for experiment reproducibility."""
    env: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }

    git_commit = _try_git("rev-parse", "HEAD")
    if git_commit:
        env["git_commit"] = git_commit
    git_branch = _try_git("rev-parse", "--abbrev-ref", "HEAD")
    if git_branch:
        env["git_branch"] = git_branch

    try:
        import torch  # noqa: F401

        import torch.cuda

        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_device_count"] = torch.cuda.device_count()
            env["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ModuleNotFoundError:
        env["torch_version"] = None
        env["cuda_available"] = False

    return env
