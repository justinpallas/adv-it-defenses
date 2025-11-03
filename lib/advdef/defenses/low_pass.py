"""Low-pass filtering defense using Gaussian or box blur."""

from __future__ import annotations

import concurrent.futures
import functools
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageFile

try:
    from scipy.ndimage import gaussian_filter, uniform_filter
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "Missing dependency 'scipy'. Install it with `pip install scipy`."
    ) from exc

from advdef.config import DefenseConfig
from advdef.core.context import RunContext
from advdef.core.pipeline import DatasetVariant, Defense
from advdef.core.registry import register_defense
from advdef.utils import Progress, ensure_dir
from ._common import build_config_identifier

ImageFile.LOAD_TRUNCATED_IMAGES = True


def discover_images(root: Path, patterns: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.rglob(pattern))
    return sorted(paths)


def apply_low_pass(
    src: Path,
    *,
    input_root: Path,
    output_root: Path,
    filter_type: str,
    sigma: float,
    kernel_size: int,
    overwrite: bool,
    dry_run: bool,
    format_hint: str | None,
) -> tuple[str, str]:
    relative = src.relative_to(input_root)
    destination = output_root / relative

    if format_hint:
        destination = destination.with_suffix(f".{format_hint.lower()}")

    if not overwrite and destination.exists():
        return "skipped", destination.as_posix()

    if dry_run:
        return "dry-run", destination.as_posix()

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(src) as img:
            mode = img.mode
            if mode not in ("L", "RGB", "RGBA"):
                img = img.convert("RGB")
                mode = img.mode

            arr = np.asarray(img, dtype=np.float32) / 255.0

            if mode == "RGBA":
                rgb = arr[..., :3]
                alpha = arr[..., 3:]
                filtered = _filter_array(rgb, filter_type, sigma, kernel_size)
                filtered = np.concatenate([filtered, alpha], axis=-1)
            else:
                filtered = _filter_array(arr, filter_type, sigma, kernel_size)

            filtered_uint8 = np.clip(filtered * 255.0, 0.0, 255.0).round().astype(np.uint8)
            result = Image.fromarray(filtered_uint8, mode=mode)

            save_kwargs = {}
            if format_hint:
                save_kwargs["format"] = format_hint.upper()
            elif img.format:
                save_kwargs["format"] = img.format

            result.save(destination, **save_kwargs)
    except Exception as exc:  # pragma: no cover - file system dependent
        return f"failed: {exc}", destination.as_posix()

    return "written", destination.as_posix()


def _filter_array(
    array: np.ndarray,
    filter_type: str,
    sigma: float,
    kernel_size: int,
) -> np.ndarray:
    if filter_type == "gaussian":
        if array.ndim == 3:
            sigma_tuple = (sigma, sigma, 0.0)
        else:
            sigma_tuple = sigma
        return gaussian_filter(array, sigma=sigma_tuple, mode="reflect")
    if filter_type == "box":
        if array.ndim == 3:
            size = (kernel_size, kernel_size, 1)
        else:
            size = kernel_size
        return uniform_filter(array, size=size, mode="reflect")
    raise ValueError(f"Unsupported filter_type '{filter_type}'. Expected 'gaussian' or 'box'.")


@register_defense("low-pass")
class LowPassDefense(Defense):
    """Blur images with a low-pass filter to suppress high-frequency perturbations."""

    def __init__(self, config: DefenseConfig) -> None:
        super().__init__(config)
        self._settings_reported = False
        self._progress: Progress | None = None
        self._variant_images: dict[str, list[Path]] = {}
        self._params_cache: dict[str, object] | None = None
        self._config_identifier = build_config_identifier(config, default_prefix="low-pass")

    def _get_params(self) -> dict[str, object]:
        if self._params_cache is None:
            params = self.config.params
            patterns = tuple(params.get("extensions", ("*.png", "*.jpg", "*.jpeg", "*.bmp")))
            filter_type = str(params.get("filter", "gaussian")).lower()
            if filter_type not in {"gaussian", "box"}:
                raise ValueError("filter must be 'gaussian' or 'box'.")
            sigma = float(params.get("sigma", 1.0))
            if sigma <= 0.0 and filter_type == "gaussian":
                raise ValueError("sigma must be positive for gaussian filter.")
            kernel_size = int(params.get("kernel_size", 3))
            if kernel_size <= 1:
                kernel_size = 3
            if kernel_size % 2 == 0:
                kernel_size += 1
            overwrite = bool(params.get("overwrite", True))
            dry_run = bool(params.get("dry_run", False))
            format_hint = params.get("format")
            workers = int(params.get("workers", max(1, (os.cpu_count() or 2) - 1)))

            self._params_cache = {
                "patterns": patterns,
                "filter_type": filter_type,
                "sigma": sigma,
                "kernel_size": kernel_size,
                "overwrite": overwrite,
                "dry_run": dry_run,
                "format_hint": format_hint,
                "workers": workers,
            }
        return self._params_cache

    def initialize(self, context: RunContext, variants: list[DatasetVariant]) -> None:
        details = self._get_params()
        patterns = details["patterns"]  # type: ignore[index]

        self._variant_images = {}
        total_images = 0
        for variant in variants:
            images = discover_images(Path(variant.data_dir), patterns)
            if not images:
                raise FileNotFoundError(f"No images matched the provided extensions in {variant.data_dir}.")
            self._variant_images[variant.name] = images
            total_images += len(images)

        if not self._settings_reported:
            print(
                "[info] Low-pass defense settings: "
                f"config={self.config.name or self._config_identifier} "
                f"filter={details['filter_type']}, sigma={details['sigma']}, kernel_size={details['kernel_size']}, "
                f"overwrite={details['overwrite']}, dry_run={details['dry_run']}, "
                f"workers={details['workers']}, format={details['format_hint']}"
            )
            self._settings_reported = True

        if self._progress is not None:
            self._progress.close()
        self._progress = Progress(total=total_images, description="Low-pass filtering", unit="images")

    def run(self, context: RunContext, variant: DatasetVariant) -> DatasetVariant:
        details = self._get_params()
        patterns = details["patterns"]  # type: ignore[index]
        filter_type = details["filter_type"]  # type: ignore[index]
        sigma = details["sigma"]  # type: ignore[index]
        kernel_size = details["kernel_size"]  # type: ignore[index]
        overwrite = details["overwrite"]  # type: ignore[index]
        dry_run = details["dry_run"]  # type: ignore[index]
        format_hint = details["format_hint"]
        workers = details["workers"]  # type: ignore[index]

        input_dir = Path(variant.data_dir)
        output_root = ensure_dir(
            context.artifacts_dir / "defenses" / "low-pass" / variant.name / self._config_identifier
        )

        images = self._variant_images.get(variant.name)
        if images is None:
            images = discover_images(input_dir, patterns)
            if self._progress is not None:
                self._progress.add_total(len(images))

        if not images:
            raise FileNotFoundError(f"No images matched the provided extensions in {input_dir}.")

        progress = self._progress
        if progress is None:
            progress = Progress(total=len(images), description="Low-pass filtering", unit="images")
            self._progress = progress

        task = functools.partial(
            apply_low_pass,
            input_root=input_dir,
            output_root=output_root,
            filter_type=filter_type,
            sigma=sigma,
            kernel_size=kernel_size,
            overwrite=overwrite,
            dry_run=dry_run,
            format_hint=format_hint,
        )

        written = skipped = failed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for status, path in executor.map(task, images):
                if status == "written":
                    written += 1
                elif status == "skipped":
                    skipped += 1
                elif status == "dry-run":
                    print(f"[dry-run] would write {path}")
                else:
                    failed += 1
                    print(f"[warn] Failed to low-pass {path}: {status}")
                progress.update()

        metadata = {
            "defense": "low-pass",
            "filter": filter_type,
            "sigma": sigma,
            "kernel_size": kernel_size,
            "format": format_hint,
            "written": written,
            "skipped": skipped,
            "failed": failed,
            "source_variant": variant.name,
            "config_name": self.config.name,
            "config_identifier": self._config_identifier,
        }

        return DatasetVariant(
            name=f"{variant.name}-low-pass-{self._config_identifier}",
            data_dir=str(output_root),
            parent=variant.name,
            metadata=metadata,
        )

    def finalize(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        self._variant_images.clear()


__all__ = ["LowPassDefense"]
