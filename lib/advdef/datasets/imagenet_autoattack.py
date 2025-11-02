"""ImageNet dataset sampling for adversarial evaluation."""

from __future__ import annotations

import csv
import random
import shutil
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from PIL import Image, ImageFile
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from advdef.config import DatasetConfig
from advdef.core.context import RunContext
from advdef.core.pipeline import DatasetArtifacts, DatasetBuilder
from advdef.core.registry import register_dataset
from advdef.utils import Progress, ensure_dir

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from timm.models import create_model
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "Missing dependency 'timm'. Install it with `pip install timm`."
    ) from exc

try:
    from advdef.datasets.imagenet_labels import load_ground_truth_labels
except ModuleNotFoundError:
    raise


@dataclass
class SampleInfo:
    path: Path
    predicted_label: int
    confidence: float
    target_label: Optional[int] = None


class NormalizedModel(nn.Module):
    """Wrap a model to apply normalization inside the forward pass."""

    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean[None, :, None, None])
        self.register_buffer("std", std[None, :, None, None])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model((inputs - self.mean) / self.std)


def discover_images(root: Path, patterns: Sequence[str]) -> List[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.rglob(pattern))
    return sorted(paths)


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


def load_image(path: Path, transform: transforms.Compose) -> torch.Tensor:
    with Image.open(path) as img:
        return transform(img.convert("RGB"))


def collect_samples(
    image_paths: Sequence[Path],
    count: int,
    transform: transforms.Compose,
    model: NormalizedModel,
    device: torch.device,
    min_confidence: float,
    seed: int,
    ground_truth: Optional[Dict[str, int]] = None,
    require_correct: bool = False,
) -> List[SampleInfo]:
    candidates = list(image_paths)
    rng = random.Random(seed)
    rng.shuffle(candidates)

    infos: List[SampleInfo] = []
    skipped_low_conf = 0
    skipped_missing_gt = 0
    skipped_mismatch = 0

    with Progress(total=count, description="Selecting baselines", unit="samples") as progress:
        for path in candidates:
            try:
                image_tensor = load_image(path, transform)
            except Exception as exc:
                print(f"[warn] Skipping {path}: {exc}")
                continue

            with torch.no_grad():
                logits = model(image_tensor.unsqueeze(0).to(device))
                probabilities = logits.softmax(dim=1)
                confidence, label = probabilities.max(dim=1)

            if confidence.item() < min_confidence:
                skipped_low_conf += 1
                continue

            target_label: Optional[int] = None
            if ground_truth is not None:
                filename = path.name
                for variant in {filename, filename.lower(), filename.upper()}:
                    target_label = ground_truth.get(variant)
                    if target_label is not None:
                        break

            predicted_label = int(label.item())

            if require_correct:
                if target_label is None:
                    skipped_missing_gt += 1
                    continue
                if predicted_label != target_label:
                    skipped_mismatch += 1
                    continue

            infos.append(
                SampleInfo(
                    path=path,
                    predicted_label=predicted_label,
                    confidence=confidence.item(),
                    target_label=target_label,
                )
            )

            progress.update()

            if len(infos) == count:
                break

    if len(infos) < count:
        raise RuntimeError(
            f"Only {len(infos)} usable images found; requested {count}. "
            "Consider lowering min_confidence, disabling require_correct, "
            "or ensuring the input directory is correct."
        )

    if require_correct and (skipped_missing_gt or skipped_mismatch):
        print(
            "[info] Correctness filtering skipped "
            f"{skipped_mismatch} misclassified and {skipped_missing_gt} without ground truth."
        )
    if min_confidence > 0.0 and skipped_low_conf:
        print(
            f"[info] Confidence filtering skipped {skipped_low_conf} candidates below {min_confidence:.3f}."
        )

    return infos


def save_images(
    batch: Iterable[torch.Tensor],
    destinations: Sequence[Path],
    *,
    description: str,
    progress: Progress | None = None,
) -> None:
    to_pil = transforms.ToPILImage()
    total = len(destinations)
    if isinstance(batch, torch.Tensor) and batch.shape[0] != total:
        raise ValueError("Number of images and destinations must match.")
    for tensor, out_path in zip(batch, destinations):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        to_pil(tensor.detach().cpu().clamp(0.0, 1.0)).save(out_path)
        if progress is not None:
            progress.update()
    if total and progress is None:
        target_dir = destinations[0].parent
        print(f"[info] Saved {total} {description} images to {target_dir}")


def resolve_mean_std(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract normalization constants, falling back to default ImageNet values."""
    default_mean = (0.485, 0.456, 0.406)
    default_std = (0.229, 0.224, 0.225)

    cfg = getattr(model, "pretrained_cfg", {}) or {}
    mean = cfg.get("mean")
    std = cfg.get("std")

    mean = mean if mean is not None else default_mean
    std = std if std is not None else default_std

    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


@register_dataset("imagenet-autoattack")
class ImageNetAutoAttackBuilder(DatasetBuilder):
    """Prepare ImageNet samples and baseline images for downstream attacks."""

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)

    def run(self, context: RunContext) -> DatasetArtifacts:
        params = self.config.params
        options = context.options or {}

        imagenet_root_override = options.get("imagenet_root")
        if imagenet_root_override is not None:
            imagenet_root = Path(imagenet_root_override)
        else:
            imagenet_root = context.work_dir / "datasets" / "imagenet"

        input_dir = self._resolve_path(
            params.get("input_dir"),
            default=imagenet_root / "val",
            base=imagenet_root,
        )
        count = int(params.get("count", 200))
        extensions = tuple(params.get("extensions", ("*.JPEG", "*.JPG", "*.jpeg", "*.jpg", "*.png")))
        min_confidence = float(params.get("min_confidence", 0.0))
        seed = int(params.get("seed", 123))
        device_choice = str(params.get("device", "auto"))
        devkit_dir = self._resolve_path(
            params.get("devkit_dir"),
            default=imagenet_root / "ILSVRC2012_devkit_t12",
            base=imagenet_root,
        )
        ground_truth_path = devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt"
        devkit_url = params.get(
            "devkit_url",
            "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz",
        )
        devkit_archive = self._resolve_path(
            params.get("devkit_archive"),
            default=context.work_dir / "downloads" / "ILSVRC2012_devkit_t12.tar.gz",
            base=context.work_dir,
        )
        require_correct = bool(params.get("require_correct", False))
        manifest_path = params.get("manifest")
        model_name = params.get("model_name", "resnet50")
        download_if_missing = bool(params.get("download_if_missing", True))
        download_url = params.get(
            "download_url",
            "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
        )
        archive_path = self._resolve_path(
            params.get("download_archive"),
            default=context.work_dir / "downloads" / "ILSVRC2012_img_val.tar",
            base=context.work_dir,
        )

        if not input_dir.exists():
            if download_if_missing:
                self._ensure_dataset_available(
                    input_dir=input_dir,
                    archive_path=archive_path,
                    download_url=download_url,
                )
            else:
                raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

        if not ground_truth_path.exists():
            self._ensure_devkit_available(
                devkit_dir=devkit_dir,
                archive_path=devkit_archive,
                download_url=devkit_url,
            )
            if not ground_truth_path.exists():
                raise FileNotFoundError(
                    f"Ground-truth file {ground_truth_path} could not be prepared automatically."
                )

        if ground_truth_path.exists():
            ground_truth_labels, _ = load_ground_truth_labels(ground_truth_path)
        else:
            raise FileNotFoundError(
                f"Ground-truth file {ground_truth_path} is required but was not found."
            )

        device = torch.device(
            "cuda"
            if (device_choice == "cuda" or (device_choice == "auto" and torch.cuda.is_available()))
            else "cpu"
        )
        if device.type == "cpu" and device_choice == "cuda":
            print("[warn] CUDA requested but not available. Falling back to CPU.")

        backbone = create_model(model_name, pretrained=True)
        backbone.eval()

        mean, std = resolve_mean_std(backbone)
        normalized_model = NormalizedModel(backbone, mean=mean, std=std).to(device)
        normalized_model.eval()

        transform = build_transform()

        all_images = discover_images(input_dir, extensions)
        if len(all_images) < count:
            raise RuntimeError(
                f"Found {len(all_images)} images under {input_dir}, fewer than requested count={count}."
            )

        print(f"[info] Discovered {len(all_images)} candidate images. Sampling {count} baselines.")

        sample_infos = collect_samples(
            image_paths=all_images,
            count=count,
            transform=transform,
            model=normalized_model,
            device=device,
            min_confidence=min_confidence,
            seed=seed,
            ground_truth=ground_truth_labels or None,
            require_correct=require_correct,
        )

        dataset_root = ensure_dir(context.artifacts_dir / "dataset")
        baseline_dir = dataset_root / "baseline"
        reset_directory(baseline_dir)

        batch_size = max(1, int(params.get("batch_size", 16)))

        def iter_batches(infos: Sequence[SampleInfo], size: int):
            for start in range(0, len(infos), size):
                end = min(start + size, len(infos))
                yield start, end, infos[start:end]

        baseline_paths = [baseline_dir / f"{info.path.stem}.png" for info in sample_infos]

        recorded_hw: Optional[tuple[int, int]] = None

        with Progress(total=len(sample_infos), description="Saving baselines", unit="images") as progress:
            for batch_idx, (start, end, batch_infos) in enumerate(
                iter_batches(sample_infos, batch_size), start=1
            ):
                tensors = [load_image(info.path, transform) for info in batch_infos]
                batch_tensor = torch.stack(tensors, dim=0)

                if recorded_hw is None and batch_tensor.numel() > 0:
                    _, _, height, width = batch_tensor.shape
                    recorded_hw = (height, width)

                save_images(
                    batch_tensor,
                    baseline_paths[start:end],
                    description=f"baseline batch {batch_idx}",
                    progress=progress,
                )

        if manifest_path:
            manifest_file = Path(manifest_path)
        else:
            manifest_file = dataset_root / "manifest.csv"

        write_manifest(manifest_file, sample_infos, baseline_paths)

        if recorded_hw is None:
            recorded_hw = (224, 224)

        metadata = {
            "samples": sample_infos,
            "model_name": model_name,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "batch_size": batch_size,
            "seed": seed,
            "manifest_path": str(manifest_file),
            "ground_truth_path": str(ground_truth_path) if ground_truth_path.exists() else None,
            "devkit_dir": str(devkit_dir),
            "imagenet_root": str(imagenet_root.resolve()),
            "input_dir": str(input_dir),
            "image_hw": list(recorded_hw),
            "timestamp": time.time(),
        }

        return DatasetArtifacts(
            clean_dir=str(baseline_dir),
            labels_path=str(manifest_file),
            metadata=metadata,
        )

    @staticmethod
    def _resolve_path(value: Optional[str | Path], *, default: Path, base: Path) -> Path:
        path = Path(value).expanduser() if value is not None else Path(default)
        if not path.is_absolute():
            path = (base / path).resolve()
        return path

    @staticmethod
    def _ensure_dataset_available(
        input_dir: Path,
        archive_path: Path,
        download_url: str,
    ) -> None:
        print(f"[info] Input directory {input_dir} not found. Attempting to download ImageNet validation set.")
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        if archive_path.exists():
            print(f"[info] Archive already present at {archive_path}. Skipping download.")
        else:
            ImageNetAutoAttackBuilder._download_file(download_url, archive_path)

        parent_dir = input_dir.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        print(f"[info] Extracting {archive_path} into {parent_dir} (this may take a while).")
        ImageNetAutoAttackBuilder._extract_tar_with_retry(
            archive_path=archive_path,
            target_root=parent_dir,
            download_url=download_url,
            mode="r",
            redownload_callback=lambda: ImageNetAutoAttackBuilder._download_file(download_url, archive_path),
        )

        extracted_dir = parent_dir / "ILSVRC2012_img_val"
        if extracted_dir.exists() and extracted_dir != input_dir:
            if input_dir.exists():
                print(f"[info] Desired input directory {input_dir} already exists after extraction.")
            else:
                extracted_dir.rename(input_dir)
        elif not input_dir.exists():
            jpeg_candidates = list(parent_dir.glob("ILSVRC2012_val_*.JPEG"))
            jpeg_candidates += list(parent_dir.glob("ILSVRC2012_val_*.jpg"))
            jpeg_candidates += list(parent_dir.glob("ILSVRC2012_val_*.jpeg"))
            if jpeg_candidates:
                input_dir.mkdir(parents=True, exist_ok=True)
                for candidate in jpeg_candidates:
                    target_path = input_dir / candidate.name
                    if target_path.exists():
                        candidate.unlink()
                    else:
                        candidate.rename(target_path)

        if not input_dir.exists():
            raise FileNotFoundError(
                f"Failed to prepare dataset. Expected directory {input_dir} after extraction."
            )
        print(f"[info] ImageNet validation images available at {input_dir}.")

    @staticmethod
    def _ensure_devkit_available(
        devkit_dir: Path,
        archive_path: Path,
        download_url: str,
    ) -> None:
        if (devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt").exists():
            return

        print(f"[info] Ground-truth devkit not found. Downloading from {download_url}.")
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        if not archive_path.exists():
            ImageNetAutoAttackBuilder._download_file(download_url, archive_path)
        else:
            print(f"[info] Devkit archive already present at {archive_path}.")

        target_root = devkit_dir.parent
        target_root.mkdir(parents=True, exist_ok=True)

        print(f"[info] Extracting devkit {archive_path} into {target_root}.")
        ImageNetAutoAttackBuilder._extract_tar_with_retry(
            archive_path=archive_path,
            target_root=target_root,
            download_url=download_url,
            mode="r:gz",
            redownload_callback=lambda: ImageNetAutoAttackBuilder._download_file(download_url, archive_path),
        )

        extracted_dir = target_root / "ILSVRC2012_devkit_t12"
        if extracted_dir.exists() and extracted_dir != devkit_dir:
            if devkit_dir.exists():
                print(f"[info] Devkit directory already exists at {devkit_dir}.")
            else:
                extracted_dir.rename(devkit_dir)

        if not (devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt").exists():
            raise FileNotFoundError(
                f"Failed to prepare ImageNet devkit under {devkit_dir}."
            )
        print(f"[info] ImageNet devkit available at {devkit_dir}.")

    @staticmethod
    def _download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
        print(f"[info] Downloading {url} -> {destination}")

        with urllib.request.urlopen(url) as response, destination.open("wb") as out_file:
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded / total * 100
                    print(
                        f"\r[info] Downloaded {downloaded / (1024 ** 2):.1f} MB / {total / (1024 ** 2):.1f} MB ({percent:.1f}%)",
                        end="",
                        flush=True,
                    )
        print()

        print(f"[info] Download complete: {destination}")

    @staticmethod
    def _extract_tar_with_retry(
        archive_path: Path,
        target_root: Path,
        download_url: str,
        mode: str,
        redownload_callback,
        max_attempts: int = 2,
    ) -> None:
        def is_within_directory(directory: Path, target: Path) -> bool:
            try:
                target.relative_to(directory)
            except ValueError:
                return False
            return True

        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            try:
                with tarfile.open(archive_path, mode) as tar:
                    for member in tar.getmembers():
                        member_path = target_root / member.name
                        if not is_within_directory(target_root, member_path.parent):
                            raise RuntimeError(f"Unsafe member path detected in archive: {member.name}")
                    tar.extractall(path=target_root)
                return
            except (tarfile.ReadError, EOFError) as exc:
                print(
                    f"[warn] Failed to extract {archive_path} (attempt {attempt}): {exc}. "
                    f"{'Re-downloading archive.' if attempt < max_attempts else 'Giving up.'}"
                )
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"Failed to extract archive {archive_path} after {max_attempts} attempts."
                    ) from exc
                if archive_path.exists():
                    archive_path.unlink()
                redownload_callback()


def write_manifest(
    manifest_path: Path,
    samples: Sequence[SampleInfo],
    outputs: Sequence[Path],
    extra_columns: dict[str, Sequence[object]] | None = None,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        include_target = any(info.target_label is not None for info in samples)
        header = ["output_path", "source_filename", "predicted_label", "confidence"]
        if include_target:
            header.append("ground_truth_label")
        column_keys: list[str] = []
        if extra_columns:
            column_keys = list(extra_columns.keys())
            for key in column_keys:
                values = extra_columns[key]
                if len(values) != len(samples):
                    raise ValueError(
                        f"Extra column '{key}' has {len(values)} entries but {len(samples)} samples were provided."
                    )
                header.append(key)
        writer.writerow(header)
        for index, (info, output_path) in enumerate(zip(samples, outputs)):
            row = [
                output_path.as_posix(),
                info.path.name,
                info.predicted_label,
                f"{info.confidence:.6f}",
            ]
            if include_target:
                row.append("" if info.target_label is None else str(info.target_label))
            if column_keys:
                for key in column_keys:
                    row.append(extra_columns[key][index])
            writer.writerow(row)
