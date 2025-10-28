"""Inference backend using timm models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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

        if checkpoint:
            model = create_model(architecture, pretrained=pretrained, checkpoint_path=str(checkpoint))
        else:
            model = create_model(architecture, pretrained=pretrained)
        model.eval()
        model.to(device)

        data_config = resolve_data_config({}, model=model)
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

        imagenet_info = ImageNetInfo()
        class_map = imagenet_info.get_label_to_name()

        with torch.no_grad():
            for images, filenames in loader:
                images = images.to(device)
                logits = model(images)
                probabilities = torch.softmax(logits, dim=1)
                values, indices = probabilities.topk(topk, dim=1)

                for file_name, idx_row, val_row in zip(filenames, indices, values):
                    record = {
                        "filename": file_name,
                    }
                    for rank, (index, value) in enumerate(zip(idx_row.tolist(), val_row.tolist()), start=1):
                        record[f"top{rank}_index"] = int(index)
                        record[f"top{rank}_prob"] = float(value)
                        record[f"top{rank}_label"] = class_map.get(int(index), "")
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

        return InferenceResult(
            variant=variant,
            predictions_path=predictions_path.as_posix(),
            metadata=metadata,
        )
