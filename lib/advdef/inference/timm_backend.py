"""Inference backend using timm models."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from advdef.config import InferenceConfig, ModelConfig
from advdef.core.context import RunContext
from advdef.core.pipeline import DatasetVariant, InferenceBackend, InferenceResult
from advdef.core.registry import register_inference
from advdef.utils import ensure_dir

try:
    from timm.data import resolve_data_config, create_transform, ImageNetInfo
    from timm.models import create_model
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "Missing dependency 'timm'. Install it with `pip install timm`."
    ) from exc


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class MeanPredictionAggregator:
    """Aggregate multiple predictions per base filename by averaging probabilities."""

    def __init__(self, crop_suffix: str, expected_count: int | None = None) -> None:
        self.crop_suffix = crop_suffix
        self.expected_count = expected_count
        self._sums: Dict[str, np.ndarray] = {}
        self._counts: Dict[str, int] = {}
        self.total_members = 0

    def _base_name(self, filename: str) -> str:
        stem, ext = os.path.splitext(filename)
        if self.crop_suffix and self.crop_suffix in stem:
            stem = stem.split(self.crop_suffix)[0]
        return f"{stem}{ext}"

    def add(self, filename: str, probs: np.ndarray) -> None:
        base = self._base_name(filename)
        vector = probs.astype(np.float64, copy=False)
        if base in self._sums:
            self._sums[base] += vector
            self._counts[base] += 1
        else:
            self._sums[base] = np.array(vector, copy=True)
            self._counts[base] = 1
        self.total_members += 1

    def finalize(self) -> Dict[str, np.ndarray]:
        return {base: total / float(self._counts[base]) for base, total in self._sums.items()}

    @property
    def group_count(self) -> int:
        return len(self._sums)


class ExactEvalTransform:
    """Apply normalization without resizing for pre-cropped adversarial images."""

    def __init__(self, hw: Tuple[int, int], mean: Sequence[float], std: Sequence[float]) -> None:
        self.expected_height = int(hw[0])
        self.expected_width = int(hw[1])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, img: Image.Image) -> torch.Tensor:  # type: ignore[override]
        if not hasattr(img, "size"):
            raise ValueError("Expected a PIL image with size attribute.")
        width, height = img.size
        if (height, width) != (self.expected_height, self.expected_width):
            img = img.resize((self.expected_width, self.expected_height), Image.BICUBIC)
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensor = self.to_tensor(img)
        return self.normalize(tensor)


class ImageListDataset(Dataset):
    """Simple dataset over a list of image paths."""

    def __init__(self, paths: Sequence[Path], transform) -> None:
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.paths)

    def __getitem__(self, index: int):  # type: ignore[override]
        path = self.paths[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        return self.transform(image), path.name


@register_inference("timm")
class TimmInferenceBackend(InferenceBackend):
    """Run classification using a timm model."""

    def __init__(self, config: InferenceConfig, model_config: ModelConfig) -> None:
        super().__init__(config, model_config)

    def run(self, context: RunContext, variant: DatasetVariant) -> InferenceResult:
        params = self.config.params
        model_params = self.model_config.params

        architecture = model_params.get("architecture") or model_params.get("model_name") or "resnet50"
        pretrained = bool(model_params.get("pretrained", True))
        checkpoint = self.model_config.checkpoint
        batch_size = int(params.get("batch_size", 128))
        workers = int(params.get("workers", 2))
        topk = int(params.get("topk", 5))
        device_choice = str(params.get("device", "auto"))
        output_format = str(params.get("output_format", "csv"))

        device = torch.device(
            "cuda"
            if (device_choice == "cuda" or (device_choice == "auto" and torch.cuda.is_available()))
            else "cpu"
        )
        if device.type == "cpu" and device_choice == "cuda":
            print("[warn] CUDA requested for inference but not available. Falling back to CPU.")

        model = create_model(architecture, pretrained=pretrained)
        model.eval()
        model.to(device)

        if checkpoint:
            checkpoint_strict = bool(model_params.get("checkpoint_strict", False))
            try:
                raw_state = torch.load(checkpoint, map_location=device)
            except Exception as exc:  # pragma: no cover - runtime dependent
                raise RuntimeError(f"Failed to load checkpoint at {checkpoint}: {exc}") from exc

            state_dict = raw_state
            if isinstance(raw_state, dict):
                if "state_dict" in raw_state:
                    state_dict = raw_state["state_dict"]
                elif "model" in raw_state:
                    state_dict = raw_state["model"]

            if not isinstance(state_dict, dict):
                raise TypeError(
                    f"Checkpoint at {checkpoint} did not contain a state_dict; keys: "
                    f"{list(raw_state.keys()) if isinstance(raw_state, dict) else type(raw_state)}"
                )

            load_result = model.load_state_dict(state_dict, strict=checkpoint_strict)
            missing = getattr(load_result, "missing_keys", [])
            unexpected = getattr(load_result, "unexpected_keys", [])
            if missing or unexpected:
                print(
                    "[warn] Checkpoint loaded with mismatches: "
                    f"missing={missing or '<none>'}, unexpected={unexpected or '<none>'}, "
                    f"strict={checkpoint_strict}"
                )

        data_config = resolve_data_config({}, model=model)

        expected_hw = variant.metadata.get("image_hw") if isinstance(variant.metadata, dict) else None
        if expected_hw is not None:
            if isinstance(expected_hw, (list, tuple)) and len(expected_hw) == 2:
                mean = data_config.get("mean") or (0.485, 0.456, 0.406)
                std = data_config.get("std") or (0.229, 0.224, 0.225)
                transform = ExactEvalTransform((int(expected_hw[0]), int(expected_hw[1])), mean, std)
            else:
                transform = create_transform(**data_config)
        else:
            transform = create_transform(**data_config)

        data_dir = Path(variant.data_dir)
        image_paths = sorted(
            path for path in data_dir.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not image_paths:
            raise FileNotFoundError(f"No supported images found in {data_dir}.")

        dataset = ImageListDataset(image_paths, transform)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=(device.type == "cuda"))

        records: List[dict] = []

        aggregation_cfg = None
        if isinstance(variant.metadata, dict):
            aggregation_cfg = variant.metadata.get("aggregation")

        aggregator: MeanPredictionAggregator | None = None
        if isinstance(aggregation_cfg, dict) and aggregation_cfg.get("method") == "mean":
            crop_suffix = str(aggregation_cfg.get("crop_suffix", "__crop"))
            expected = aggregation_cfg.get("num_inputs_per_example")
            expected_count = int(expected) if isinstance(expected, int) and expected > 0 else None
            aggregator = MeanPredictionAggregator(crop_suffix, expected_count)

        imagenet_info = ImageNetInfo()
        class_map: Dict[int, str]
        label_to_name = None
        if hasattr(imagenet_info, "get_label_to_name"):
            try:
                label_to_name = imagenet_info.get_label_to_name()
            except TypeError:
                # Some versions expose it as a property instead of method
                label_to_name = imagenet_info.get_label_to_name
        elif hasattr(imagenet_info, "label_to_name"):
            label_to_name = getattr(imagenet_info, "label_to_name")

        if isinstance(label_to_name, dict):
            class_map = {int(k): v for k, v in label_to_name.items()}
        else:
            class_map = {}
            if hasattr(imagenet_info, "class_names"):
                class_names = getattr(imagenet_info, "class_names")
                if isinstance(class_names, (list, tuple)):
                    class_map = {idx: name for idx, name in enumerate(class_names)}
            elif hasattr(imagenet_info, "label_to_idx"):
                mapping = getattr(imagenet_info, "label_to_idx")
                if isinstance(mapping, dict):
                    class_map = {int(idx): label for label, idx in mapping.items()}

        if not class_map:
            class_map = {}

        with torch.no_grad():
            for images, filenames in loader:
                images = images.to(device)
                logits = model(images)
                probabilities = torch.softmax(logits, dim=1)

                if aggregator is not None:
                    for file_name, prob_row in zip(filenames, probabilities):
                        aggregator.add(file_name, prob_row.detach().cpu().numpy())
                else:
                    values, indices = probabilities.topk(topk, dim=1)
                    for file_name, idx_row, val_row in zip(filenames, indices, values):
                        record = {
                            "filename": file_name,
                        }
                        for rank, (index, value) in enumerate(zip(idx_row.tolist(), val_row.tolist()), start=1):
                            record[f"top{rank}_index"] = int(index)
                            record[f"top{rank}_prob"] = float(value)
                            record[f"top{rank}_label"] = class_map.get(int(index), str(int(index)))
                        records.append(record)

        if aggregator is not None:
            averaged = aggregator.finalize()
            for file_name in sorted(averaged):
                avg_probs = averaged[file_name]
                prob_tensor = torch.from_numpy(avg_probs).float()
                values, indices = prob_tensor.topk(topk)
                record = {"filename": file_name}
                for rank, (index, value) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
                    record[f"top{rank}_index"] = int(index)
                    record[f"top{rank}_prob"] = float(value)
                    record[f"top{rank}_label"] = class_map.get(int(index), str(int(index)))
                records.append(record)

        results_dir = ensure_dir(context.artifacts_dir / "inference" / variant.name)
        predictions_path = results_dir / f"predictions.{output_format}"

        df = pd.DataFrame.from_records(records)
        if output_format == "csv":
            df.to_csv(predictions_path, index=False)
        elif output_format == "parquet":
            df.to_parquet(predictions_path, index=False)
        elif output_format == "json":
            df.to_json(predictions_path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        metadata = {
            "model": architecture,
            "pretrained": pretrained,
            "device": str(device),
            "batch_size": batch_size,
            "topk": topk,
            "num_samples": len(records),
        }

        if aggregator is not None:
            metadata["aggregation"] = {
                "method": "mean",
                "groups": aggregator.group_count,
                "total_members": aggregator.total_members,
                "crop_suffix": aggregator.crop_suffix,
                "expected_per_group": aggregator.expected_count,
            }

        return InferenceResult(
            variant=variant,
            predictions_path=predictions_path.as_posix(),
            metadata=metadata,
        )
