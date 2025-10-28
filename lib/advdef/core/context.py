"""Execution context for a single experiment run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from advdef.utils import ensure_dir, save_json, save_yaml

if TYPE_CHECKING:
    from advdef.config import ExperimentConfig


@dataclass
class RunContext:
    """Runtime bookkeeping for a pipeline execution."""

    experiment: "ExperimentConfig"
    run_id: str
    run_dir: Path
    timestamp: str
    work_dir: Path
    options: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_dir = ensure_dir(self.run_dir)
        self.artifacts_dir = ensure_dir(self.run_dir / "artifacts")
        self.logs_dir = ensure_dir(self.run_dir / "logs")
        self.metrics_path = self.run_dir / "metrics.json"
        self.metadata_path = self.run_dir / "metadata.json"
        self.config_path = self.run_dir / "config.yaml"

    def artifact_path(self, *parts: str) -> Path:
        path = self.artifacts_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment": self.experiment.model_dump(),
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "timestamp": self.timestamp,
            "options": self.options,
            "state": self.state,
        }

    def save_config(self) -> None:
        save_yaml(self.experiment.model_dump(mode="json"), self.config_path)

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        save_json(metadata, self.metadata_path)

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        save_json(metrics, self.metrics_path)
