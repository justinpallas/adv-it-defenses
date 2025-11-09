"""PyAutoAttack integration."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from torch import nn

from advdef.config import AttackConfig
from advdef.core.context import RunContext
from advdef.core.pipeline import Attack, DatasetArtifacts, DatasetVariant
from advdef.core.registry import register_attack
from advdef.core.samples import SampleInfo
from advdef.datasets.imagenet_autoattack import (
    NormalizedModel,
    build_transform,
    load_image,
    save_images,
    write_manifest,
)
from advdef.utils import Progress, build_identifier, ensure_dir, normalized_l2, summarize_tensor

try:
    from pyautoattack import AutoAttack
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "Missing dependency 'pyautoattack'. Install it with `pip install pyautoattack`."
    ) from exc

try:
    from timm.models import create_model
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'timm'. Install it with `pip install timm`."
    ) from exc


ATTACK_SPECS: dict[str, dict[str, object]] = {
    "apgd-ce": {
        "display": "APGD-CE",
        "key": "apgd-ce",
        "subdir": "apgd-ce",
        "seed_offset": 0,
    },
    "apgd-t": {
        "display": "APGD-T",
        "key": "apgd-t",
        "subdir": "apgd-t",
        "seed_offset": 1,
    },
    "fab": {
        "display": "FAB",
        "key": "fab",
        "subdir": "fab",
        "seed_offset": 2,
    },
    "square": {
        "display": "SquareAttack",
        "key": "square",
        "subdir": "squareattack",
        "seed_offset": 3,
    },
}


def create_autoattack(
    model: NormalizedModel,
    eps: float,
    norm: str,
    seed: int,
    device: torch.device,
    verbose: bool,
) -> AutoAttack:
    kwargs = dict(norm=norm, eps=eps, seed=seed)
    version = "standard"

    try:
        attack = AutoAttack(model, device=str(device), version=version, **kwargs)  # type: ignore[arg-type]
    except TypeError:
        try:
            attack = AutoAttack(model, version=version, **kwargs)
        except TypeError:
            attack = AutoAttack(model, **kwargs)
            if hasattr(attack, "version"):
                attack.version = version
        if hasattr(attack, "device"):
            attack.device = str(device)
    if hasattr(attack, "verbose"):
        attack.verbose = verbose
    elif verbose:
        print("[warn] AutoAttack instance does not expose a 'verbose' flag; progress output unavailable.")
    return attack


def configure_autoattack(attack: AutoAttack, params: dict) -> None:
    fab_attack = getattr(attack, "fab_attack", None)
    if fab_attack is not None:
        if "fab_steps" in params and hasattr(fab_attack, "n_iter"):
            fab_attack.n_iter = int(params["fab_steps"])
        if "fab_restarts" in params and hasattr(fab_attack, "n_restarts"):
            fab_attack.n_restarts = int(params["fab_restarts"])


def run_attack(
    attack_name: str,
    attack_key: str,
    model: NormalizedModel,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    eps: float,
    norm: str,
    seed: int,
    device: torch.device,
    verbose: bool,
    params: dict,
    *,
    chunk_id: int | None = None,
    chunk_count: int | None = None,
) -> torch.Tensor:
    attack = create_autoattack(model, eps=eps, norm=norm, seed=seed, device=device, verbose=verbose)
    configure_autoattack(attack, params)
    attack.attacks_to_run = [attack_key]
    attack.batch_size = batch_size
    chunk_detail = ""
    if chunk_id is not None and chunk_count is not None:
        chunk_detail = f" (chunk {chunk_id}/{chunk_count})"
    elif chunk_id is not None:
        chunk_detail = f" (chunk {chunk_id})"
    logging.info("[%s] processing %s samples%s", attack_name, len(inputs), chunk_detail)
    chunk_begin = time.monotonic()
    try:
        adversarial = attack.run_standard_evaluation(inputs.clone(), labels.clone(), bs=batch_size)
    except TypeError:
        try:
            adversarial = attack.run_standard_evaluation(inputs.clone(), labels.clone(), batch_size=batch_size)
        except TypeError:
            adversarial = attack.run_standard_evaluation(inputs.clone(), labels.clone())
    if isinstance(adversarial, tuple):  # older autoattack returns (adv, _)
        adversarial = adversarial[0]
    elapsed = time.monotonic() - chunk_begin
    logging.debug("%s%s finished in %.1fs", attack_name, chunk_detail, elapsed)
    return adversarial.detach().cpu()


@register_attack("autoattack")
class AutoAttackAttack(Attack):
    """Generate adversarial variants using PyAutoAttack."""

    def __init__(self, config: AttackConfig) -> None:
        super().__init__(config)
        prefix = config.type or "autoattack"
        self._config_identifier = build_identifier(
            name=config.name,
            params=config.params,
            default_prefix=prefix,
            extra={"targeted": config.targeted},
        )
        self._config_label = config.name or self._config_identifier

    def run(self, context: RunContext, dataset: DatasetArtifacts) -> Iterable[DatasetVariant]:
        params = self.config.params
        attack_names = params.get("attacks")
        if not attack_names:
            return []

        attacks = [name for name in attack_names if name in ATTACK_SPECS]
        if not attacks:
            return []

        eps = float(params.get("eps", 8 / 255))
        norm = str(params.get("norm", "Linf"))
        batch_size = int(params.get("batch_size", dataset.metadata.get("batch_size", 16)))
        device_choice = str(params.get("device", "auto"))
        verbose = bool(params.get("verbose", True))
        seed_base = int(params.get("seed", dataset.metadata.get("seed", 123)))

        model_name = dataset.metadata.get("model_name", "resnet50")
        mean_values = torch.tensor(dataset.metadata.get("mean", [0.485, 0.456, 0.406]), dtype=torch.float32)
        std_values = torch.tensor(dataset.metadata.get("std", [0.229, 0.224, 0.225]), dtype=torch.float32)

        samples: Sequence[SampleInfo] = dataset.metadata.get("samples", [])
        if not samples:
            raise RuntimeError("Dataset metadata is missing sampled images required for AutoAttack.")

        device = torch.device(
            "cuda"
            if (device_choice == "cuda" or (device_choice == "auto" and torch.cuda.is_available()))
            else "cpu"
        )
        if device.type == "cpu" and device_choice == "cuda":
            print("[warn] CUDA requested but not available. Falling back to CPU.")

        backbone = create_model(model_name, pretrained=True)
        backbone.eval()

        normalized_model = NormalizedModel(backbone, mean=mean_values, std=std_values).to(device)
        normalized_model.eval()

        transform = build_transform()

        config_identifier = self._config_identifier
        config_label = self._config_label
        attack_root = ensure_dir(context.artifacts_dir / "attacks" / config_identifier)
        image_hw = dataset.metadata.get("image_hw")
        variants: list[DatasetVariant] = []

        for attack_name in attacks:
            spec = ATTACK_SPECS[attack_name]
            display = str(spec["display"])
            attack_key = str(spec["key"])
            seed_offset = int(spec["seed_offset"])

            variant_name = f"{config_identifier}-{attack_name}"
            output_dir = ensure_dir(attack_root / str(spec["subdir"]))
            outputs = [output_dir / f"{info.path.stem}.png" for info in samples]
            manifest_path = output_dir / "manifest.csv"

            print(f"[info] Starting {display} attack [{variant_name}] on {len(samples)} samples.")

            tensors = [load_image(info.path, transform) for info in samples]
            labels = [info.target_label if info.target_label is not None else info.predicted_label for info in samples]

            chunk_size = int(params.get("chunk_size", batch_size))
            if chunk_size <= 0:
                raise ValueError("chunk_size must be positive.")
            chunk_size = min(chunk_size, len(samples))
            total_chunks = (len(samples) + chunk_size - 1) // chunk_size

            normalized_values: list[float] = []
            normalized_l2_column: list[str] = []

            with Progress(total=len(samples), description=f"{display} attack", unit="images") as progress:
                for chunk_idx in range(total_chunks):
                    start = chunk_idx * chunk_size
                    end = min(start + chunk_size, len(samples))
                    batch_infos = samples[start:end]
                    batch_outputs = outputs[start:end]
                    batch_tensors = tensors[start:end]

                    if not batch_infos:
                        continue

                    inputs_tensor = torch.stack(batch_tensors, dim=0).to(device)
                    labels_tensor = torch.tensor(labels[start:end], dtype=torch.long, device=device)

                    adv = run_attack(
                        display,
                        attack_key,
                        normalized_model,
                        inputs_tensor,
                        labels_tensor,
                        batch_size=min(batch_size, len(batch_infos)),
                        eps=eps,
                        norm=norm,
                        seed=seed_base + seed_offset + chunk_idx,
                        device=device,
                        verbose=verbose,
                        params=params,
                        chunk_id=chunk_idx + 1,
                        chunk_count=total_chunks,
                    )

                    originals_cpu = inputs_tensor.detach().cpu()
                    normalized_chunk = normalized_l2(originals_cpu, adv)
                    normalized_values.extend(normalized_chunk.tolist())
                    normalized_l2_column.extend(f"{value:.8f}" for value in normalized_chunk.tolist())

                    save_images(
                        adv,
                        batch_outputs,
                        description=f"{display} outputs chunk {chunk_idx + 1}",
                        progress=progress,
                    )
                    progress.refresh()

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            normalized_tensor = torch.tensor(normalized_values, dtype=torch.float32)
            normalized_l2_stats = summarize_tensor(normalized_tensor)
            total_count = int(normalized_tensor.numel())
            nonzero_mask = normalized_tensor > 1e-12
            nonzero_count = int(nonzero_mask.sum().item())
            mean_nonzero = float(normalized_tensor[nonzero_mask].mean().item()) if nonzero_count else 0.0
            normalized_l2_stats.update(
                {
                    "mean_nonzero": mean_nonzero,
                    "count_total": total_count,
                    "count_nonzero": nonzero_count,
                }
            )

            write_manifest(
                manifest_path,
                samples,
                outputs,
                extra_columns={"normalized_l2": normalized_l2_column},
            )

            variants.append(
                DatasetVariant(
                    name=variant_name,
                    data_dir=str(output_dir),
                    parent="baseline",
                    metadata={
                        "attack": attack_name,
                        "display": display,
                        "config_name": self.config.name,
                        "config_label": config_label,
                        "config_identifier": config_identifier,
                        "variant_name": variant_name,
                        "eps": eps,
                        "norm": norm,
                        "seed": seed_base + seed_offset,
                        "chunk_size": chunk_size,
                        "chunk_count": total_chunks,
                        "count": len(samples),
                        "manifest": manifest_path.as_posix(),
                        "image_hw": image_hw,
                        "normalized_l2": normalized_l2_stats,
                    },
                )
            )

        return variants
