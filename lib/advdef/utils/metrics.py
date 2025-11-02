"""Metric helpers for adversarial perturbations."""

from __future__ import annotations

import math
from typing import Dict

import torch


def normalized_l2(original: torch.Tensor, adversarial: torch.Tensor) -> torch.Tensor:
    """Return per-sample normalized L2 distances between original and adversarial tensors.

    The tensors are expected to be batches in CHW format with values in [0, 1].
    The normalization factor is sqrt(C * H * W) to match common reporting.
    """
    if original.shape != adversarial.shape:
        raise ValueError(
            f"Shape mismatch when computing normalized L2: {tuple(original.shape)} vs {tuple(adversarial.shape)}"
        )

    delta = adversarial - original
    flat = delta.view(delta.shape[0], -1)
    norms = flat.pow(2).sum(dim=1).sqrt()
    if flat.shape[1] == 0:
        raise ValueError("Cannot compute normalized L2 with zero-sized tensors.")
    return norms / math.sqrt(flat.shape[1])


def summarize_tensor(values: torch.Tensor) -> Dict[str, float]:
    """Return basic summary statistics for a 1-D tensor."""
    if values.ndim != 1:
        raise ValueError(f"Expected 1-D tensor, received shape {tuple(values.shape)}")
    if values.numel() == 0:
        raise ValueError("Cannot summarize an empty tensor.")

    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
    minimum = float(values.min().item())
    maximum = float(values.max().item())
    median = float(torch.median(values).item())
    return {
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "median": median,
    }

