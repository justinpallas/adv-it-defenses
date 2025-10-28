"""Pipeline assembly and base component interfaces."""

from __future__ import annotations

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


@dataclass
class DatasetVariant:
    """Represents a specific dataset variant (baseline or attacked/defended)."""

    name: str
    data_dir: str
    parent: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Structured inference result metadata."""

    variant: DatasetVariant
    predictions_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    def run(self, context: RunContext, variant: DatasetVariant) -> InferenceResult:
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

    def build_dataset(self) -> DatasetArtifacts:
        dataset_cls = DATASETS.get(self.config.dataset.type)
        builder: DatasetBuilder = dataset_cls(self.config.dataset)
        return builder.run(self.context)

    def run_attacks(
        self, dataset: DatasetArtifacts
    ) -> List[DatasetVariant]:
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
        return variants

    def run_defenses(self, variants: Iterable[DatasetVariant]) -> List[DatasetVariant]:
        results: List[DatasetVariant] = []
        for variant in variants:
            current = variant
            for defense_cfg in self.config.defenses:
                if not defense_cfg.enabled:
                    continue
                defense_cls = DEFENSES.get(defense_cfg.type)
                defense: Defense = defense_cls(defense_cfg)
                current = defense.run(self.context, current)
            results.append(current)
        return results

    def run_inference(self, variants: Iterable[DatasetVariant]) -> List[InferenceResult]:
        inference_cls = INFERENCE_BACKENDS.get(self.config.inference.type)
        backend: InferenceBackend = inference_cls(self.config.inference, self.config.model)
        return [backend.run(self.context, variant) for variant in variants]

    def run_evaluation(
        self,
        dataset: DatasetArtifacts,
        variants: Iterable[DatasetVariant],
        inference_results: Iterable[InferenceResult],
    ) -> Dict[str, Any]:
        evaluator_cls = EVALUATORS.get(self.config.evaluation.type)
        evaluator: Evaluator = evaluator_cls(self.config.evaluation)
        return evaluator.run(self.context, dataset, list(variants), list(inference_results))

    def run(self) -> Dict[str, Any]:
        dataset = self.build_dataset()
        variants = self.run_attacks(dataset)

        baseline_inferences = self.run_inference(variants)

        defended = self.run_defenses(variants)
        defended_inferences = self.run_inference(defended)

        combined_variants = list(variants) + list(defended)
        combined_inferences = list(baseline_inferences) + list(defended_inferences)

        metrics = self.run_evaluation(dataset, combined_variants, combined_inferences)
        self.context.save_metrics(metrics)
        return metrics
