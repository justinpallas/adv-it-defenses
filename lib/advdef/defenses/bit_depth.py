"""Bit-depth reduction defense."""

from __future__ import annotations

import concurrent.futures
import functools
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageFile

from advdef.config import DefenseConfig
from advdef.core.context import RunContext
from advdef.core.pipeline import DatasetVariant, Defense
from advdef.core.registry import register_defense
from advdef.utils import Progress, ensure_dir

ImageFile.LOAD_TRUNCATED_IMAGES = True


def discover_images(root: Path, patterns: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.rglob(pattern))
    return sorted(paths)


def quantize_array(
    array: np.ndarray,
    *,
    bits: int,
    normalize: bool,
) -> np.ndarray:
    if array.ndim not in {2, 3}:
        raise ValueError("Expected a 2D (grayscale) or 3D (color) array.")

    if not normalize:
        info = np.iinfo(array.dtype)
        scale_in = info.max
        array_float = array.astype(np.float32) / scale_in
    else:
        array_float = array

    levels = 2**bits
    quantized = np.floor(array_float * (levels - 1) + 0.5) / (levels - 1)
    quantized = np.clip(quantized, 0.0, 1.0)

    return quantized


def reduce_bit_depth(
    src: Path,
    *,
    input_root: Path,
    output_root: Path,
    bits: int,
    normalize: bool,
    dither: bool,
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

            arr = np.asarray(img, dtype=np.float32)
            if mode == "RGBA":
                rgb = arr[..., :3] / 255.0
                alpha = arr[..., 3:]
                reduced = quantize_array(rgb, bits=bits, normalize=True)
                reduced = np.concatenate([reduced, alpha / 255.0], axis=-1)
            else:
                reduced = quantize_array(arr / 255.0, bits=bits, normalize=True)

            if dither:
                noise = (np.random.rand(*reduced.shape) - 0.5) / (2**bits)
                reduced = np.clip(reduced + noise, 0.0, 1.0)

            reduced_uint8 = (reduced * 255.0).round().astype(np.uint8)
            result = Image.fromarray(reduced_uint8, mode=mode)

            save_kwargs = {}
            if format_hint:
                save_kwargs["format"] = format_hint.upper()
            elif img.format:
                save_kwargs["format"] = img.format

            result.save(destination, **save_kwargs)
    except Exception as exc:  # pragma: no cover - file system dependent
        return f"failed: {exc}", destination.as_posix()

    return "written", destination.as_posix()


@register_defense("bit-depth")
class BitDepthDefense(Defense):
    """Reduce image bit-depth to limit information content."""

    def __init__(self, config: DefenseConfig) -> None:
        super().__init__(config)
        self._settings_reported = False
        self._progress: Progress | None = None
        self._variant_images: dict[str, list[Path]] = {}
        self._params_cache: dict[str, object] | None = None

    def _get_params(self) -> dict[str, object]:
        if self._params_cache is None:
            params = self.config.params
            patterns = tuple(params.get("extensions", ("*.png", "*.jpg", "*.jpeg", "*.bmp")))
            bits = int(params.get("bits", 4))
            if bits <= 0 or bits > 8:
                raise ValueError("bits must be between 1 and 8.")
            normalize = bool(params.get("normalize_input", True))
            dither = bool(params.get("dither", False))
            overwrite = bool(params.get("overwrite", True))
            dry_run = bool(params.get("dry_run", False))
            format_hint = params.get("format")
            workers = int(params.get("workers", max(1, (os.cpu_count() or 2) - 1)))

            self._params_cache = {
                "patterns": patterns,
                "bits": bits,
                "normalize": normalize,
                "dither": dither,
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
                "[info] Bit-depth defense settings: "
                f"bits={details['bits']}, normalize_input={details['normalize']}, "
                f"dither={details['dither']}, overwrite={details['overwrite']}, "
                f"dry_run={details['dry_run']}, workers={details['workers']}, "
                f"format={details['format_hint']}"
            )
            self._settings_reported = True

        if self._progress is not None:
            self._progress.close()
        self._progress = Progress(total=total_images, description="Bit-depth reduction", unit="images")

    def run(self, context: RunContext, variant: DatasetVariant) -> DatasetVariant:
        details = self._get_params()
        patterns = details["patterns"]  # type: ignore[index]
        bits = details["bits"]  # type: ignore[index]
        normalize = details["normalize"]  # type: ignore[index]
        dither = details["dither"]  # type: ignore[index]
        overwrite = details["overwrite"]  # type: ignore[index]
        dry_run = details["dry_run"]  # type: ignore[index]
        format_hint = details["format_hint"]
        workers = details["workers"]  # type: ignore[index]

        input_dir = Path(variant.data_dir)
        output_root = ensure_dir(context.artifacts_dir / "defenses" / "bit-depth" / variant.name)

        images = self._variant_images.get(variant.name)
        if images is None:
            images = discover_images(input_dir, patterns)
            if self._progress is not None:
                self._progress.add_total(len(images))

        if not images:
            raise FileNotFoundError(f"No images matched the provided extensions in {input_dir}.")

        progress = self._progress
        if progress is None:
            progress = Progress(total=len(images), description="Bit-depth reduction", unit="images")
            self._progress = progress

        task = functools.partial(
            reduce_bit_depth,
            input_root=input_dir,
            output_root=output_root,
            bits=bits,
            normalize=normalize,
            dither=dither,
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
                    print(f"[warn] Failed to quantize {path}: {status}")
                progress.update()

        metadata = {
            "defense": "bit-depth",
            "bits": bits,
            "normalize_input": normalize,
            "dither": dither,
            "format": format_hint,
            "written": written,
            "skipped": skipped,
            "failed": failed,
            "source_variant": variant.name,
        }

        return DatasetVariant(
            name=f"{variant.name}-bit-depth",
            data_dir=str(output_root),
            parent=variant.name,
            metadata=metadata,
        )

    def finalize(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        self._variant_images.clear()


__all__ = ["BitDepthDefense"]
