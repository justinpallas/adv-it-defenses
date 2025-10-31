"""Adapters for Torchattacks adversarial methods."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image

from advdef.config import AttackConfig
from advdef.core.context import RunContext
from advdef.core.pipeline import Attack, DatasetArtifacts, DatasetVariant
from advdef.core.registry import register_attack
from advdef.datasets.imagenet_autoattack import SampleInfo, build_transform, load_image, write_manifest
from advdef.utils import ensure_dir

try:
    import torchattacks

    _HAS_TORCHATTACKS = True
except ImportError:  # pragma: no cover - dependency missing
    torchattacks = None
    _HAS_TORCHATTACKS = False


class _SampleDataset(Dataset):
    def __init__(self, samples: list[SampleInfo]) -> None:
        self.samples = samples
        self.transform = build_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        info = self.samples[idx]
        image = load_image(info.path, self.transform)
        label = info.target_label if info.target_label is not None else info.predicted_label
        return image, label


def _prepare_model(dataset: DatasetArtifacts, device: torch.device, model_name: str):
    from timm.models import create_model

    mean_values = torch.tensor(dataset.metadata.get("mean", [0.485, 0.456, 0.406]), dtype=torch.float32)
    std_values = torch.tensor(dataset.metadata.get("std", [0.229, 0.224, 0.225]), dtype=torch.float32)

    from advdef.datasets.imagenet_autoattack import NormalizedModel

    backbone = create_model(model_name, pretrained=True)
    backbone.eval()
    model = NormalizedModel(backbone, mean=mean_values, std=std_values).to(device)
    model.eval()
    return model


def _get_device(device_choice: str) -> torch.device:
    device = torch.device(
        "cuda"
        if (device_choice == "cuda" or (device_choice == "auto" and torch.cuda.is_available()))
        else "cpu"
    )
    if device.type == "cpu" and device_choice == "cuda":
        logging.warning("CUDA requested but not available. Falling back to CPU.")
    return device


def _generate_adversarial(
    attack,
    dataset: DatasetArtifacts,
    samples: list[SampleInfo],
    attack_root: Path,
    variant_prefix: str,
    batch_size: int,
    device: torch.device,
):
    dataset_loader = DataLoader(_SampleDataset(samples), batch_size=batch_size)
    outputs = []

    for batch_idx, (images, labels) in enumerate(dataset_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        adv_images = attack(images, labels)
        adv_images = adv_images.detach().cpu()

        start = (batch_idx - 1) * batch_size
        end = min(start + batch_size, len(samples))
        batch_infos = samples[start:end]
        for adv_tensor, info in zip(adv_images, batch_infos):
            adv_image = adv_tensor.clamp(0.0, 1.0)
            save_path = attack_root / f"{info.path.stem}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            to_pil_image(adv_image).save(save_path)
            outputs.append((info, save_path))

    return outputs


def _build_variant(
    name: str,
    attack_dir: Path,
    base_variant: str,
    outputs: list[tuple[SampleInfo, Path]],
    metadata: dict,
) -> DatasetVariant:
    manifest_path = attack_dir / "manifest.csv"
    write_manifest(manifest_path, [info for info, _ in outputs], [path for _, path in outputs])

    return DatasetVariant(
        name=name,
        data_dir=str(attack_dir),
        parent=base_variant,
        metadata=metadata | {"manifest": manifest_path.as_posix(), "count": len(outputs)},
    )


class TorchAttackBase(Attack):
    ATTACK_CLASS = None
    DEFAULT_NAME = "attack"

    def __init__(self, config: AttackConfig) -> None:
        if not _HAS_TORCHATTACKS:
            raise RuntimeError(
                "Missing dependency 'torchattacks'. Install it with `pip install torchattacks`."
            )
        super().__init__(config)

    def make_attack(self, model, params: dict[str, Any]):
        if self.ATTACK_CLASS is None:
            raise NotImplementedError("ATTACK_CLASS must be defined on subclasses.")
        return self.ATTACK_CLASS(model, **params)

    def build_attack_kwargs(
        self,
        *,
        params: dict,
        eps: float,
        alpha: float,
        steps: int,
        random_start: bool,
        norm: str,
        cw_steps: int,
        cw_confidence: float,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def run(self, context: RunContext, dataset: DatasetArtifacts) -> Iterable[DatasetVariant]:
        params = self.config.params
        eps = float(params.get("eps", 8 / 255))
        alpha = float(params.get("alpha", eps / 4))
        steps = int(params.get("steps", 10))
        random_start = bool(params.get("random_start", True))
        cw_steps = int(params.get("cw_steps", 100))
        cw_confidence = float(params.get("cw_confidence", 0.0))
        device_choice = str(params.get("device", "auto"))
        batch_size = int(params.get("batch_size", dataset.metadata.get("batch_size", 16)))
        norm = str(params.get("norm", "Linf"))
        variant_prefix = params.get("suffix", self.DEFAULT_NAME)
        base_variant_name = "baseline"

        samples = dataset.metadata.get("samples")
        if not samples:
            raise RuntimeError("Dataset metadata is missing sampled images required for attack.")

        device = _get_device(device_choice)
        model_name = dataset.metadata.get("model_name", "resnet50")
        model = _prepare_model(dataset, device, model_name)

        attack_dir = ensure_dir(context.artifacts_dir / "attacks" / variant_prefix)

        attack_kwargs = self.build_attack_kwargs(
            params=params,
            eps=eps,
            alpha=alpha,
            steps=steps,
            random_start=random_start,
            norm=norm,
            cw_steps=cw_steps,
            cw_confidence=cw_confidence,
        )
        attack = self.make_attack(model, attack_kwargs)

        outputs = _generate_adversarial(
            attack,
            dataset,
            list(samples),
            attack_dir,
            variant_prefix,
            batch_size,
            device,
        )

        metadata: dict[str, Any] = {
            "attack": variant_prefix,
            "eps": eps,
            "alpha": alpha,
            "steps": steps,
            "random_start": random_start,
            "norm": norm,
        }
        metadata.update(self.additional_metadata(params))

        return [
            _build_variant(
                name=f"{base_variant_name}-{variant_prefix}",
                attack_dir=attack_dir,
                base_variant=base_variant_name,
                outputs=outputs,
                metadata=metadata,
            )
        ]

    def additional_metadata(self, params: dict) -> dict[str, Any]:
        return {}


@register_attack("pgd")
class PGDAttack(TorchAttackBase):
    DEFAULT_NAME = "pgd"

    def make_attack(self, model, params: dict[str, Any]):  # type: ignore[override]
        norm = params.pop("norm", "Linf")
        if norm.lower() in {"linf", "inf"}:
            return torchattacks.PGD(model, **params)  # type: ignore[attr-defined]
        if norm.lower() == "l2":
            return torchattacks.PGDL2(model, **params)  # type: ignore[attr-defined]
        raise ValueError(f"Unsupported norm '{norm}' for PGD attack.")

    def build_attack_kwargs(  # type: ignore[override]
        self,
        *,
        params: dict,
        eps: float,
        alpha: float,
        steps: int,
        random_start: bool,
        norm: str,
        cw_steps: int,
        cw_confidence: float,
    ) -> dict[str, Any]:
        return {
            "eps": eps,
            "alpha": alpha,
            "steps": steps,
            "random_start": random_start,
            "norm": norm,
        }


@register_attack("fgsm")
class FGSMAttack(TorchAttackBase):
    DEFAULT_NAME = "fgsm"

    @property
    def ATTACK_CLASS(self):  # type: ignore[override]
        return torchattacks.FGSM  # type: ignore[attr-defined]

    def build_attack_kwargs(  # type: ignore[override]
        self,
        *,
        params: dict,
        eps: float,
        alpha: float,
        steps: int,
        random_start: bool,
        norm: str,
        cw_steps: int,
        cw_confidence: float,
    ) -> dict[str, Any]:
        return {"eps": eps}


@register_attack("cw-l2")
class CWL2Attack(TorchAttackBase):
    DEFAULT_NAME = "cw"

    @property
    def ATTACK_CLASS(self):  # type: ignore[override]
        return torchattacks.CW  # type: ignore[attr-defined]

    def build_attack_kwargs(  # type: ignore[override]
        self,
        *,
        params: dict,
        eps: float,
        alpha: float,
        steps: int,
        random_start: bool,
        norm: str,
        cw_steps: int,
        cw_confidence: float,
    ) -> dict[str, Any]:
        return {
            "c": float(params.get("cw_c", 1.0)),
            "kappa": cw_confidence,
            "steps": cw_steps,
            "lr": float(params.get("cw_lr", 0.01)),
        }

    def additional_metadata(self, params: dict) -> dict[str, Any]:  # type: ignore[override]
        return {
            "cw_c": float(params.get("cw_c", 1.0)),
            "cw_lr": float(params.get("cw_lr", 0.01)),
        }


@register_attack("deepfool")
class DeepFoolAttack(TorchAttackBase):
    DEFAULT_NAME = "deepfool"

    @property
    def ATTACK_CLASS(self):  # type: ignore[override]
        return torchattacks.DeepFool  # type: ignore[attr-defined]

    def build_attack_kwargs(  # type: ignore[override]
        self,
        *,
        params: dict,
        eps: float,
        alpha: float,
        steps: int,
        random_start: bool,
        norm: str,
        cw_steps: int,
        cw_confidence: float,
    ) -> dict[str, Any]:
        return {
            "steps": int(params.get("steps", 50)),
            "overshoot": float(params.get("overshoot", 0.02)),
        }

    def additional_metadata(self, params: dict) -> dict[str, Any]:  # type: ignore[override]
        return {
            "overshoot": float(params.get("overshoot", 0.02)),
        }
