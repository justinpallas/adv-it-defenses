"""Tests for serialising pipeline state objects."""

from __future__ import annotations

from pathlib import Path

from advdef.core.pipeline import DatasetArtifacts, DatasetVariant, InferenceResult
from advdef.core.samples import SampleInfo


def test_dataset_artifacts_state_roundtrip():
    samples = [
        SampleInfo(path=Path("/tmp/image1.png"), predicted_label=5, confidence=0.99, target_label=5),
        SampleInfo(path=Path("/tmp/image2.png"), predicted_label=12, confidence=0.87, target_label=None),
    ]
    artifacts = DatasetArtifacts(
        clean_dir="/tmp/baseline",
        labels_path="/tmp/manifest.csv",
        metadata={
            "samples": samples,
            "seed": 123,
        },
    )

    stored = artifacts.to_state()
    assert isinstance(stored["metadata"]["samples"][0], dict)

    restored = DatasetArtifacts.from_state(stored)
    restored_samples = restored.metadata["samples"]
    assert isinstance(restored_samples[0], SampleInfo)
    assert restored_samples[0].path == Path("/tmp/image1.png")
    assert restored.metadata["seed"] == 123


def test_dataset_variant_state_roundtrip():
    variant = DatasetVariant(
        name="attack",
        data_dir="/tmp/attack",
        parent="baseline",
        metadata={"manifest": "/tmp/attack/manifest.csv", "count": 2},
    )

    stored = variant.to_state()
    assert stored["name"] == "attack"

    restored = DatasetVariant.from_state(stored)
    assert restored.name == "attack"
    assert restored.metadata["count"] == 2


def test_inference_result_state_roundtrip():
    variant = DatasetVariant(name="baseline", data_dir="/tmp/baseline")
    result = InferenceResult(
        variant=variant,
        predictions_path="/tmp/preds.csv",
        metadata={"batch_size": 32},
    )

    stored = result.to_state()
    assert stored["predictions_path"].endswith("preds.csv")

    restored = InferenceResult.from_state(variant, stored)
    assert restored.predictions_path == "/tmp/preds.csv"
    assert restored.metadata["batch_size"] == 32
