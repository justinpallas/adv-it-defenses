"""Pipeline assembly and base component interfaces."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from advdef.config import (
    AttackConfig,
    DefenseConfig,
    EvaluationConfig,
    ExperimentConfig,
    InferenceConfig,
    ModelConfig,
)
from advdef.core.context import RunContext
from advdef.core.samples import SampleInfo, deserialize_sample_infos, serialize_sample_infos
from advdef.core.registry import (
    ATTACKS,
    DATASETS,
    DEFENSES,
    EVALUATORS,
    INFERENCE_BACKENDS,
)


@dataclass
class DatasetArtifacts:
    """Outputs produced by dataset builders."""

    clean_dir: str
    labels_path: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_state(self) -> Dict[str, Any]:
        return {
            "clean_dir": self.clean_dir,
            "labels_path": self.labels_path,
            "metadata": _prepare_metadata_for_state(self.metadata),
        }

    @classmethod
    def from_state(cls, payload: Dict[str, Any]) -> "DatasetArtifacts":
        metadata = _restore_metadata_from_state(payload.get("metadata", {}))
        return cls(
            clean_dir=payload["clean_dir"],
            labels_path=payload.get("labels_path"),
            metadata=metadata,
        )


@dataclass
class DatasetVariant:
    """Represents a specific dataset variant (baseline or attacked/defended)."""

    name: str
    data_dir: str
    parent: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_dir": self.data_dir,
            "parent": self.parent,
            "metadata": _prepare_metadata_for_state(self.metadata),
        }

    @classmethod
    def from_state(cls, payload: Dict[str, Any]) -> "DatasetVariant":
        metadata = _restore_metadata_from_state(payload.get("metadata", {}))
        return cls(
            name=payload["name"],
            data_dir=payload["data_dir"],
            parent=payload.get("parent"),
            metadata=metadata,
        )


@dataclass
class InferenceResult:
    """Structured inference result metadata."""

    variant: DatasetVariant
    predictions_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_state(self) -> Dict[str, Any]:
        return {
            "predictions_path": self.predictions_path,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_state(cls, variant: DatasetVariant, payload: Dict[str, Any]) -> "InferenceResult":
        return cls(
            variant=variant,
            predictions_path=payload["predictions_path"],
            metadata=copy.deepcopy(payload.get("metadata", {})),
        )


def _prepare_metadata_for_state(metadata: Dict[str, Any]) -> Dict[str, Any]:
    prepared = copy.deepcopy(metadata)
    samples = prepared.get("samples")
    if isinstance(samples, list) and samples and isinstance(samples[0], SampleInfo):
        prepared["samples"] = serialize_sample_infos(samples)
    return prepared


def _restore_metadata_from_state(metadata: Dict[str, Any]) -> Dict[str, Any]:
    restored = copy.deepcopy(metadata)
    samples = restored.get("samples")
    if (
        isinstance(samples, list)
        and samples
        and isinstance(samples[0], dict)
        and "path" in samples[0]
        and "predicted_label" in samples[0]
    ):
        try:
            restored["samples"] = deserialize_sample_infos(samples)
        except (TypeError, KeyError):
            pass
    return restored


class PipelineStep(ABC):
    """Generic base class for typed pipeline steps."""

    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def run(self, context: RunContext, *args, **kwargs):
        raise NotImplementedError


class DatasetBuilder(PipelineStep):
    """Base class for dataset preparation components."""

    config: ExperimentConfig

    @abstractmethod
    def run(self, context: RunContext) -> DatasetArtifacts:
        raise NotImplementedError


class Attack(PipelineStep):
    """Base class for adversarial attacks."""

    config: AttackConfig

    @abstractmethod
    def run(
        self, context: RunContext, dataset: DatasetArtifacts
    ) -> Iterable[DatasetVariant]:
        raise NotImplementedError


class Defense(PipelineStep):
    """Base class for defenses that operate on dataset variants."""

    config: DefenseConfig

    @abstractmethod
    def run(self, context: RunContext, variant: DatasetVariant) -> DatasetVariant:
        raise NotImplementedError


class InferenceBackend(PipelineStep):
    """Base class for inference execution."""

    config: InferenceConfig

    def __init__(self, config: InferenceConfig, model_config: ModelConfig) -> None:
        super().__init__(config)
        self.model_config = model_config

    @abstractmethod
    def run(
        self,
        context: RunContext,
        variant: DatasetVariant,
        *,
        namespace: str | None = None,
    ) -> InferenceResult:
        raise NotImplementedError


class Evaluator(PipelineStep):
    """Base class for metrics computation."""

    config: EvaluationConfig

    @abstractmethod
    def run(
        self,
        context: RunContext,
        dataset: DatasetArtifacts,
        variants: Iterable[DatasetVariant],
        inferences: Iterable[InferenceResult],
    ) -> Dict[str, Any]:
        raise NotImplementedError


class Pipeline:
    """Orchestrate the experiment described by an ExperimentConfig."""

    def __init__(self, context: RunContext) -> None:
        self.context = context
        self.config = context.experiment
        self.variant_records: List[Dict[str, Any]] = []
        self._steps_state = self.context.state.setdefault("steps", {})
        self._resume_enabled = bool(self.context.resume)

    @staticmethod
    def _label_for_model(model_config: ModelConfig) -> str:
        """Human-friendly label for identifying inference outputs."""
        return str(
            model_config.name
            or model_config.params.get("architecture")
            or model_config.params.get("model_name")
            or model_config.type
        )

    def build_dataset(self) -> DatasetArtifacts:
        cached = self._steps_state.get("dataset")
        if self._resume_enabled and cached:
            return DatasetArtifacts.from_state(cached)

        dataset_cls = DATASETS.get(self.config.dataset.type)
        builder: DatasetBuilder = dataset_cls(self.config.dataset)
        dataset = builder.run(self.context)
        self._steps_state["dataset"] = dataset.to_state()
        self.context.save_state()
        return dataset

    def run_attacks(
        self, dataset: DatasetArtifacts
    ) -> List[DatasetVariant]:
        cached = self._steps_state.get("attacks")
        if (
            self._resume_enabled
            and cached
            and isinstance(cached.get("variants"), list)
        ):
            return [DatasetVariant.from_state(entry) for entry in cached["variants"]]

        variants: List[DatasetVariant] = [
            DatasetVariant(
                name="baseline",
                data_dir=dataset.clean_dir,
                metadata={
                    "source": "baseline",
                    "image_hw": dataset.metadata.get("image_hw"),
                },
            )
        ]
        for attack_cfg in self.config.dataset.attacks:
            attack_cls = ATTACKS.get(attack_cfg.type)
            attack: Attack = attack_cls(attack_cfg)
            produced = list(attack.run(self.context, dataset))
            variants.extend(produced)
        self._steps_state["attacks"] = {"variants": [variant.to_state() for variant in variants]}
        self.context.save_state()
        return variants

    def run_defenses(self, variants: Iterable[DatasetVariant]) -> List[DatasetVariant]:
        cached = self._steps_state.get("defenses")
        if (
            self._resume_enabled
            and cached
            and isinstance(cached.get("variants"), list)
        ):
            return [DatasetVariant.from_state(entry) for entry in cached["variants"]]

        base_variants: List[DatasetVariant] = list(variants)
        defended: List[DatasetVariant] = []

        for defense_cfg in self.config.defenses:
            if not defense_cfg.enabled:
                continue
            defense_cls = DEFENSES.get(defense_cfg.type)
            defense: Defense = defense_cls(defense_cfg)

            if hasattr(defense, "initialize"):
                defense.initialize(self.context, base_variants)

            for variant in base_variants:
                defended.append(defense.run(self.context, variant))

            if hasattr(defense, "finalize"):
                defense.finalize()

        self._steps_state["defenses"] = {"variants": [variant.to_state() for variant in defended]}
        self.context.save_state()
        return defended

    def run_inference(
        self,
        variants: Iterable[DatasetVariant],
        namespace: str,
        model_config: ModelConfig | None = None,
        inference_label: str | None = None,
    ) -> List[InferenceResult]:
        model_cfg = model_config or self.config.model
        inference_cls = INFERENCE_BACKENDS.get(self.config.inference.type)
        backend: InferenceBackend = inference_cls(self.config.inference, model_cfg)
        inference_state = self._steps_state.setdefault("inference", {})
        namespace_state: Dict[str, Dict[str, Any]] = inference_state.setdefault(namespace, {})
        results: List[InferenceResult] = []
        for variant in variants:
            cached = namespace_state.get(variant.name)
            if self._resume_enabled and cached:
                result = InferenceResult.from_state(variant, cached)
                result.metadata = self._annotate_inference_metadata(result.metadata, namespace, model_cfg, inference_label)
                results.append(result)
                continue
            print(f"[debug] running inference for {variant.name} [{namespace}]")
            result = backend.run(self.context, variant, namespace=namespace)
            result.metadata = self._annotate_inference_metadata(result.metadata, namespace, model_cfg, inference_label)
            namespace_state[variant.name] = result.to_state()
            self.context.save_state()
            results.append(result)
        return results

    def _annotate_inference_metadata(
        self,
        metadata: Dict[str, Any],
        namespace: str,
        model_config: ModelConfig,
        inference_label: str | None,
    ) -> Dict[str, Any]:
        annotated = copy.deepcopy(metadata) if metadata is not None else {}
        annotated.setdefault("namespace", namespace)
        model_label = model_config.name or model_config.type
        annotated.setdefault("model_label", model_label)
        if inference_label:
            annotated.setdefault("inference_label", inference_label)
        if model_config.checkpoint:
            annotated.setdefault("checkpoint", str(model_config.checkpoint))
        return annotated

    def _run_defense_finetuned_inference(
        self, defended_variants: Iterable[DatasetVariant]
    ) -> tuple[list[DatasetVariant], list[InferenceResult]]:
        defense_checkpoints = getattr(self.config.model, "defense_checkpoints", {}) or {}
        if not defense_checkpoints:
            return [], []

        variants_by_defense: Dict[str, List[DatasetVariant]] = {}

        for variant in defended_variants:
            defense_name = None
            if isinstance(variant.metadata, dict):
                defense_name = variant.metadata.get("defense")
            checkpoint = defense_checkpoints.get(defense_name) if defense_name else None
            if not checkpoint:
                continue

            metadata = copy.deepcopy(variant.metadata)
            metadata["source_variant"] = variant.name
            metadata["inference_role"] = "fine_tuned"
            metadata["inference_checkpoint"] = str(checkpoint)
            metadata["inference_target_defense"] = defense_name
            tuned_variant = DatasetVariant(
                name=f"{variant.name}-finetuned",
                data_dir=variant.data_dir,
                parent=variant.name,
                metadata=metadata,
            )
            variants_by_defense.setdefault(defense_name, []).append(tuned_variant)

        finetuned_variants: List[DatasetVariant] = []
        finetuned_results: List[InferenceResult] = []

        for defense_name, variants in variants_by_defense.items():
            checkpoint = defense_checkpoints[defense_name]
            finetuned_variants.extend(variants)
            model_cfg = self.config.model.model_copy(update={"checkpoint": checkpoint})
            label = f"{defense_name}-finetuned"
            finetuned_results.extend(
                self.run_inference(
                    variants,
                    namespace=f"defended-{defense_name}-finetuned",
                    model_config=model_cfg,
                    inference_label=label,
                )
            )

        return finetuned_variants, finetuned_results

    def run_evaluation(
        self,
        dataset: DatasetArtifacts,
        variants: Iterable[DatasetVariant],
        inference_results: Iterable[InferenceResult],
    ) -> Dict[str, Any]:
        cached = self._steps_state.get("evaluation")
        if self._resume_enabled and cached and "metrics" in cached:
            return copy.deepcopy(cached["metrics"])

        evaluator_cls = EVALUATORS.get(self.config.evaluation.type)
        evaluator: Evaluator = evaluator_cls(self.config.evaluation)
        metrics = evaluator.run(self.context, dataset, list(variants), list(inference_results))
        self._steps_state["evaluation"] = {"metrics": copy.deepcopy(metrics)}
        self.context.save_state()
        return metrics

    def run(self) -> Dict[str, Any]:
        dataset = self.build_dataset()
        variants = self.run_attacks(dataset)

        baseline_inferences = self.run_inference(variants, "baseline")

        defended = self.run_defenses(variants)
        defended_inferences = self.run_inference(defended, "defended")

        extra_inferences: List[InferenceResult] = []
        for model_cfg in self.config.additional_inference_models:
            label = self._label_for_model(model_cfg)
            extra_inferences.extend(
                self.run_inference(
                    variants,
                    namespace=f"baseline-{label}",
                    model_config=model_cfg,
                    inference_label=label,
                )
            )
            extra_inferences.extend(
                self.run_inference(
                    defended,
                    namespace=f"defended-{label}",
                    model_config=model_cfg,
                    inference_label=label,
                )
            )

        finetuned_variants, finetuned_inferences = self._run_defense_finetuned_inference(defended)

        combined_variants = list(variants) + list(defended) + list(finetuned_variants)
        combined_inferences = (
            list(baseline_inferences)
            + list(defended_inferences)
            + list(extra_inferences)
            + list(finetuned_inferences)
        )

        self.variant_records = [
            {
                "name": variant.name,
                "parent": variant.parent,
                "data_dir": variant.data_dir,
                "metadata": variant.metadata,
            }
            for variant in combined_variants
        ]

        metrics = self.run_evaluation(dataset, combined_variants, combined_inferences)
        attack_normalized: Dict[str, Dict[str, Any]] = {}
        for record in self.variant_records:
            metadata = record.get("metadata", {})
            if metadata.get("inference_role") == "fine_tuned":
                continue
            attack_name = metadata.get("attack")
            stats = metadata.get("normalized_l2")
            if not attack_name or not isinstance(stats, dict):
                continue
            mean_nonzero = stats.get("mean_nonzero")
            count_nonzero = stats.get("count_nonzero")
            count_total = stats.get("count_total")
            if mean_nonzero is None:
                continue
            attack_normalized[record["name"]] = {
                "attack": attack_name,
                "mean_normalized_l2_nonzero": mean_nonzero,
                "nonzero_count": count_nonzero,
                "total_count": count_total,
            }
        if attack_normalized:
            metrics.setdefault("attack_metrics", {})
            metrics["attack_metrics"]["normalized_l2"] = attack_normalized

        self.context.save_metrics(metrics)
        return metrics
