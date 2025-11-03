"""Image cropping and resizing defense."""

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageFile
from numpy.random import SeedSequence

from advdef.config import DefenseConfig
from advdef.core.context import RunContext
from advdef.core.pipeline import DatasetVariant, Defense
from advdef.core.registry import register_defense
from advdef.utils import Progress, ensure_dir
from ._common import build_config_identifier

ImageFile.LOAD_TRUNCATED_IMAGES = True

INTERPOLATION_MODES = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def discover_images(root: Path, patterns: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.rglob(pattern))
    return sorted(paths)


def clamp_crop_box(
    width: int,
    height: int,
    crop_w: int,
    crop_h: int,
    mode: str,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[int, int, int, int]:
    crop_w = min(crop_w, width)
    crop_h = min(crop_h, height)

    if crop_w == width and crop_h == height:
        return 0, 0, width, height

    if mode == "center":
        left = max(0, (width - crop_w) // 2)
        top = max(0, (height - crop_h) // 2)
    elif mode == "top-left":
        left = 0
        top = 0
    elif mode == "bottom-right":
        left = max(0, width - crop_w)
        top = max(0, height - crop_h)
    elif mode == "random":
        if rng is None:
            raise ValueError("Random crop mode requires an RNG.")
        max_left = max(width - crop_w, 0)
        max_top = max(height - crop_h, 0)
        left = int(rng.integers(0, max_left + 1)) if max_left > 0 else 0
        top = int(rng.integers(0, max_top + 1)) if max_top > 0 else 0
    else:
        raise ValueError(
            f"Unsupported crop_mode '{mode}'. Expected one of: center, top-left, bottom-right, random."
        )

    right = left + crop_w
    bottom = top + crop_h
    return left, top, right, bottom


def parse_size(value: object, *, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        width, height = value
    elif isinstance(value, dict) and {"width", "height"} <= set(value):
        width, height = value["width"], value["height"]
    else:
        raise ValueError(f"{name} must be a sequence [width, height] or mapping with width/height.")
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} values must be positive integers.")
    return width, height


def validate_crop_resize(
    crop: tuple[int, int] | None,
    resize: tuple[int, int] | None,
    *,
    name: str,
    num_crops: int,
    crop_mode: str,
) -> None:
    if crop is not None and resize is not None:
        crop_w, crop_h = crop
        resize_w, resize_h = resize
        if crop_w < resize_w or crop_h < resize_h:
            if not (num_crops > 1 or crop_mode == "random"):
                print(
                    f"[warn] {name}: crop_size {crop} smaller than resize_size {resize}; image will be upsampled."
                )


def transform_image(
    src: Path,
    *,
    crop_size: tuple[int, int] | None,
    crop_mode: str,
    resize_size: tuple[int, int] | None,
    interpolation: Image.Resampling,
    output_root: Path,
    input_root: Path,
    overwrite: bool,
    dry_run: bool,
    format_hint: str | None,
    num_crops: int,
    crop_suffix: str,
    rng_seed: SeedSequence | int | None,
) -> tuple[str, list[str]]:
    relative = src.relative_to(input_root)
    destination = output_root / relative

    if format_hint:
        destination = destination.with_suffix(f".{format_hint.lower()}")

    if num_crops <= 0:
        raise ValueError("num_crops must be positive.")

    expected_paths: list[Path]
    if num_crops == 1:
        expected_paths = [destination]
    else:
        stem = destination.stem
        suffix = destination.suffix
        parent = destination.parent
        expected_paths = [
            parent / f"{stem}{crop_suffix}{idx:03d}{suffix}"
            for idx in range(num_crops)
        ]

    if not overwrite and all(path.exists() for path in expected_paths):
        return "skipped", [path.as_posix() for path in expected_paths]

    if dry_run:
        return "dry-run", [path.as_posix() for path in expected_paths]

    destination.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(rng_seed) if rng_seed is not None else np.random.default_rng()

    try:
        with Image.open(src) as img:
            img.load()

            outputs: list[str] = []

            if num_crops == 1:
                working = img
                if crop_size is not None:
                    left, top, right, bottom = clamp_crop_box(
                        img.width, img.height, crop_size[0], crop_size[1], crop_mode, rng=rng if crop_mode == "random" else None
                    )
                    if (right - left, bottom - top) != (img.width, img.height):
                        working = img.crop((left, top, right, bottom))

                if resize_size is not None:
                    working = working.resize(resize_size, interpolation)

                save_kwargs = {}
                if format_hint:
                    save_kwargs["format"] = format_hint.upper()
                elif img.format:
                    save_kwargs["format"] = img.format

                working.save(expected_paths[0], **save_kwargs)
                outputs.append(expected_paths[0].as_posix())
                return "written", outputs

            if crop_size is None:
                raise ValueError("crop_size must be provided when num_crops > 1.")

            save_kwargs = {}
            if format_hint:
                save_kwargs["format"] = format_hint.upper()
            elif img.format:
                save_kwargs["format"] = img.format

            for idx, path in enumerate(expected_paths):
                left, top, right, bottom = clamp_crop_box(
                    img.width,
                    img.height,
                    crop_size[0],
                    crop_size[1],
                    crop_mode,
                    rng=rng,
                )
                crop = img.crop((left, top, right, bottom))

                if resize_size is not None:
                    crop = crop.resize(resize_size, interpolation)

                crop.save(path, **save_kwargs)
                outputs.append(path.as_posix())
            return "written", outputs
    except Exception as exc:  # pragma: no cover - file system dependent
        return f"failed: {exc}", [path.as_posix() for path in expected_paths]


@register_defense("crop-resize")
class CropResizeDefense(Defense):
    """Crop and/or resize images as a defense step."""

    def __init__(self, config: DefenseConfig) -> None:
        super().__init__(config)
        self._settings_reported = False
        self._progress: Progress | None = None
        self._variant_images: dict[str, list[Path]] = {}
        self._params_cache: dict[str, object] | None = None
        self._variant_image_seeds: dict[str, list[SeedSequence | None]] = {}
        self._config_identifier = build_config_identifier(config, default_prefix="crop-resize")

    def _get_params(self) -> dict[str, object]:
        if self._params_cache is None:
            params = self.config.params
            patterns = tuple(params.get("extensions", ("*.png", "*.jpg", "*.jpeg", "*.bmp")))
            crop_size = parse_size(params.get("crop_size"), name="crop_size")
            resize_size = parse_size(params.get("resize_size"), name="resize_size")
            num_crops = int(params.get("num_crops", 1))
            if num_crops <= 0:
                raise ValueError("num_crops must be positive.")
            crop_mode = str(params.get("crop_mode", "center")).lower()
            validate_crop_resize(crop_size, resize_size, name="crop-resize", num_crops=num_crops, crop_mode=crop_mode)
            interpolation_name = str(params.get("interpolation", "bilinear")).lower()
            interpolation = INTERPOLATION_MODES.get(interpolation_name)
            if interpolation is None:
                raise ValueError(
                    f"Unsupported interpolation '{interpolation_name}'. "
                    f"Expected one of: {', '.join(INTERPOLATION_MODES)}."
                )

            overwrite = bool(params.get("overwrite", True))
            dry_run = bool(params.get("dry_run", False))
            format_hint = params.get("format")
            workers = int(params.get("workers", max(1, (os.cpu_count() or 2) - 1)))
            crop_suffix = str(params.get("crop_suffix", "__crop"))
            seed_param = params.get("seed")
            seed = int(seed_param) if seed_param is not None else None

            self._params_cache = {
                "patterns": patterns,
                "crop_size": crop_size,
                "crop_mode": crop_mode,
                "resize_size": resize_size,
                "interpolation": interpolation,
                "interpolation_name": interpolation_name,
                "overwrite": overwrite,
                "dry_run": dry_run,
                "format_hint": format_hint,
                "workers": workers,
                "num_crops": num_crops,
                "crop_suffix": crop_suffix,
                "seed": seed,
            }
        return self._params_cache

    def initialize(self, context: RunContext, variants: list[DatasetVariant]) -> None:
        details = self._get_params()
        patterns = details["patterns"]  # type: ignore[index]

        self._variant_images = {}
        self._variant_image_seeds = {}
        total_images = 0
        min_width: int | None = None
        min_height: int | None = None

        keep_seed = details["seed"]  # type: ignore[index]
        num_crops = details["num_crops"]  # type: ignore[index]

        if keep_seed is not None and (num_crops > 1 or details["crop_mode"] == "random"):
            master_seed = SeedSequence(keep_seed)
            variant_seed_sequences = master_seed.spawn(len(variants))
        else:
            variant_seed_sequences = [None] * len(variants)

        for idx, variant in enumerate(variants):
            images = discover_images(Path(variant.data_dir), patterns)
            if not images:
                raise FileNotFoundError(f"No images matched the provided extensions in {variant.data_dir}.")
            self._variant_images[variant.name] = images
            variant_seed = variant_seed_sequences[idx] if idx < len(variant_seed_sequences) else None
            if isinstance(variant_seed, SeedSequence) and images:
                self._variant_image_seeds[variant.name] = variant_seed.spawn(len(images))
            else:
                self._variant_image_seeds[variant.name] = [None] * len(images)
            total_images += len(images)

            try:
                with Image.open(images[0]) as sample:
                    width, height = sample.size
            except Exception:
                continue

            if min_width is None or width < min_width:
                min_width = width
            if min_height is None or height < min_height:
                min_height = height

        if not self._settings_reported:
            crop_size = details["crop_size"]
            resize_size = details["resize_size"]
            print(
                "[info] Crop-Resize defense settings: "
                f"config={self.config.name or self._config_identifier} "
                f"crop_size={crop_size}, crop_mode={details['crop_mode']}, "
                f"resize_size={resize_size}, interpolation={details['interpolation_name']}, "
                f"num_crops={details['num_crops']}, crop_suffix={details['crop_suffix']}, "
                f"seed={details['seed']}, overwrite={details['overwrite']}, dry_run={details['dry_run']}, "
                f"workers={details['workers']}, format={details['format_hint']}"
            )
            self._settings_reported = True

        crop_size = details["crop_size"]
        if crop_size is not None and min_width is not None and min_height is not None:
            crop_w, crop_h = crop_size
            if crop_w >= min_width and crop_h >= min_height:
                print(
                    "[warn] Crop-Resize defense crop_size >= minimum image size; crop step will have no effect."
                )

        if self._progress is not None:
            self._progress.close()
        self._progress = Progress(total=total_images, description="Crop/Resize", unit="images")

    def run(self, context: RunContext, variant: DatasetVariant) -> DatasetVariant:
        details = self._get_params()
        patterns = details["patterns"]  # type: ignore[index]
        crop_size = details["crop_size"]  # type: ignore[index]
        crop_mode = details["crop_mode"]  # type: ignore[index]
        resize_size = details["resize_size"]  # type: ignore[index]
        interpolation = details["interpolation"]  # type: ignore[index]
        overwrite = details["overwrite"]  # type: ignore[index]
        dry_run = details["dry_run"]  # type: ignore[index]
        format_hint = details["format_hint"]
        workers = details["workers"]  # type: ignore[index]
        num_crops = details["num_crops"]  # type: ignore[index]
        crop_suffix = details["crop_suffix"]  # type: ignore[index]
        seed_value = details["seed"]  # type: ignore[index]

        input_dir = Path(variant.data_dir)
        output_root = ensure_dir(
            context.artifacts_dir / "defenses" / "crop-resize" / variant.name / self._config_identifier
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
            progress = Progress(total=len(images), description="Crop/Resize", unit="images")
            self._progress = progress

        image_seeds = self._variant_image_seeds.get(variant.name)
        if image_seeds is None or len(image_seeds) != len(images):
            image_seeds = [None] * len(images)
            self._variant_image_seeds[variant.name] = image_seeds

        def task(src: Path, rng_seed: SeedSequence | None) -> tuple[str, list[str]]:
            return transform_image(
                src,
                crop_size=crop_size,
                crop_mode=crop_mode,
                resize_size=resize_size,
                interpolation=interpolation,
                output_root=output_root,
                input_root=input_dir,
                overwrite=overwrite,
                dry_run=dry_run,
                format_hint=format_hint,
                num_crops=num_crops,
                crop_suffix=crop_suffix,
                rng_seed=rng_seed,
            )

        written = skipped = failed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for status, paths in executor.map(task, images, image_seeds):
                if status == "written":
                    written += len(paths)
                elif status == "skipped":
                    skipped += len(paths)
                elif status == "dry-run":
                    for path in paths:
                        print(f"[dry-run] would write {path}")
                else:
                    failed += len(paths)
                    print(f"[warn] Failed to transform {paths}: {status}")
                progress.update()

        metadata = {
            "defense": "crop-resize",
            "crop_size": crop_size,
            "crop_mode": crop_mode,
            "resize_size": resize_size,
            "interpolation": details["interpolation_name"],
            "format": format_hint,
            "num_crops": num_crops,
            "crop_suffix": crop_suffix,
            "seed": seed_value,
            "written": written,
            "skipped": skipped,
            "failed": failed,
            "source_variant": variant.name,
            "config_name": self.config.name,
            "config_identifier": self._config_identifier,
        }

        if num_crops > 1:
            metadata["aggregation"] = {
                "method": "mean",
                "crop_suffix": crop_suffix,
                "num_inputs_per_example": num_crops,
            }

        return DatasetVariant(
            name=f"{variant.name}-crop-resize-{self._config_identifier}",
            data_dir=str(output_root),
            parent=variant.name,
            metadata=metadata,
        )

    def finalize(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        self._variant_images.clear()
        self._variant_image_seeds.clear()


__all__ = ["CropResizeDefense"]
