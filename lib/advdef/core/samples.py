"""Shared sample metadata structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence


@dataclass
class SampleInfo:
    """Stores metadata about a sampled input image."""

    path: Path
    predicted_label: int
    confidence: float
    target_label: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "predicted_label": int(self.predicted_label),
            "confidence": float(self.confidence),
            "target_label": None if self.target_label is None else int(self.target_label),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SampleInfo":
        return cls(
            path=Path(payload["path"]),
            predicted_label=int(payload["predicted_label"]),
            confidence=float(payload["confidence"]),
            target_label=None if payload.get("target_label") in (None, "") else int(payload["target_label"]),
        )


def serialize_sample_infos(samples: Sequence[SampleInfo]) -> list[Dict[str, Any]]:
    """Convert SampleInfo objects into JSON-serialisable dictionaries."""
    return [sample.to_dict() for sample in samples]


def deserialize_sample_infos(entries: Sequence[Dict[str, Any]]) -> list[SampleInfo]:
    """Reconstruct SampleInfo objects from dictionaries."""
    return [SampleInfo.from_dict(entry) for entry in entries]
