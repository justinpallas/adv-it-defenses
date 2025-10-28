"""Evaluation utilities for ImageNet-style experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from advdef.config import EvaluationConfig
from advdef.core.pipeline import DatasetArtifacts, DatasetVariant, Evaluator, InferenceResult
from advdef.core.registry import register_evaluator


def _build_label_map(samples: Sequence) -> Dict[str, int]:
    label_map: Dict[str, int] = {}
    for info in samples:
        label = info.target_label if getattr(info, "target_label", None) is not None else info.predicted_label
        stem = info.path.stem
        original = info.path.name
        candidates = {
            original,
            original.lower(),
            stem,
            f"{stem}.png",
            f"{stem}.jpg",
            f"{stem}.jpeg",
            f"{stem}.JPEG",
        }
        for candidate in candidates:
            label_map.setdefault(candidate, label)
    return label_map


def _load_predictions(result: InferenceResult, topk: int) -> tuple[np.ndarray, List[str]]:
    predictions_path = Path(result.predictions_path)
    if predictions_path.suffix == ".csv":
        df = pd.read_csv(predictions_path)
    elif predictions_path.suffix == ".parquet":
        df = pd.read_parquet(predictions_path)
    elif predictions_path.suffix == ".json":
        df = pd.read_json(predictions_path)
    else:
        raise ValueError(f"Unsupported predictions file format: {predictions_path}")

    if "filename" not in df.columns:
        raise ValueError(f"'filename' column missing from predictions file {predictions_path}")

    columns: List[str] = []
    for rank in range(1, topk + 1):
        column_name = f"top{rank}_index"
        if column_name in df.columns:
            columns.append(column_name)
    if not columns:
        raise ValueError(
            f"No prediction index columns found in {predictions_path}. Expected columns like 'top1_index'."
        )

    preds = df[columns].to_numpy(dtype=np.int64)
    filenames = df["filename"].astype(str).tolist()
    return preds, filenames


@register_evaluator("imagenet")
class ImageNetEvaluator(Evaluator):
    """Compute ImageNet top-k accuracies from inference outputs."""

    def __init__(self, config: EvaluationConfig) -> None:
        super().__init__(config)

    def run(
        self,
        context,
        dataset: DatasetArtifacts,
        variants: Iterable[DatasetVariant],
        inferences: Iterable[InferenceResult],
    ) -> Dict[str, object]:
        params = self.config.params
        topk_config = params.get("topk", [1, 5])
        if isinstance(topk_config, int):
            topk_values = [topk_config]
        else:
            topk_values = sorted({int(k) for k in topk_config if int(k) > 0})
        max_topk = max(topk_values) if topk_values else 1

        samples = dataset.metadata.get("samples", [])
        if not samples:
            raise RuntimeError("Dataset metadata does not include sampled image information required for evaluation.")

        label_map = _build_label_map(samples)

        metrics: Dict[str, object] = {"variants": {}, "topk": topk_values}

        for result in inferences:
            preds, filenames = _load_predictions(result, max_topk)
            ground_truth: List[int] = []
            missing: List[str] = []
            for name in filenames:
                label = label_map.get(name)
                if label is None:
                    label = label_map.get(Path(name).name)
                if label is None:
                    missing.append(name)
                else:
                    ground_truth.append(label)

            if missing:
                raise KeyError(
                    f"Ground-truth labels missing for {len(missing)} images when evaluating {result.variant.name}. "
                    f"Example filenames: {missing[:5]}"
                )

            ground_truth_arr = np.asarray(ground_truth, dtype=np.int64)
            if preds.shape[0] != ground_truth_arr.shape[0]:
                raise ValueError(
                    f"Row count mismatch for variant {result.variant.name}: "
                    f"{preds.shape[0]} predictions vs {ground_truth_arr.shape[0]} labels."
                )

            variant_metrics: Dict[str, float] = {}

            for k in topk_values:
                topk_slice = preds[:, :k]
                correct = (topk_slice == ground_truth_arr[:, None]).any(axis=1)
                accuracy = float(correct.mean())
                variant_metrics[f"top{k}_accuracy"] = accuracy
                variant_metrics[f"top{k}_correct"] = int(correct.sum())

            variant_metrics["num_samples"] = int(ground_truth_arr.size)
            metrics["variants"][result.variant.name] = variant_metrics

        return metrics
