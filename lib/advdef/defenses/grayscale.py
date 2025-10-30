"""Grayscale conversion defense."""

from __future__ import annotations

import concurrent.futures
import functools
import os
from pathlib import Path
from typing import Sequence

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


def convert_to_grayscale(
    src: Path,
    *,
    input_root: Path,
    output_root: Path,
    mode: str,
    replicate_rgb: bool,
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
            transformed = img.convert(mode)
            if replicate_rgb and mode != "RGB":
                transformed = transformed.convert("RGB")

            save_kwargs = {}
            if format_hint:
                save_kwargs["format"] = format_hint.upper()
            elif img.format:
                save_kwargs["format"] = img.format

            transformed.save(destination, **save_kwargs)
    except Exception as exc:  # pragma: no cover - file system dependent
        return f"failed: {exc}", destination.as_posix()

    return "written", destination.as_posix()


@register_defense("grayscale")
class GrayscaleDefense(Defense):
    """Convert images to grayscale."""

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
            mode = str(params.get("mode", "L")).upper()
            replicate_rgb = bool(params.get("replicate_rgb", False))
            overwrite = bool(params.get("overwrite", True))
            dry_run = bool(params.get("dry_run", False))
            format_hint = params.get("format")
            workers = int(params.get("workers", max(1, (os.cpu_count() or 2) - 1)))

            if replicate_rgb and mode == "RGB":
                replicate_rgb = False

            self._params_cache = {
                "patterns": patterns,
                "mode": mode,
                "replicate_rgb": replicate_rgb,
                "overwrite": overwrite,
                "dry_run": dry_run,
                "format_hint": format_hint,
                "workers": workers,
            }
        return self._params_cache

    def initialize(self, context: RunContext, variants: list[DatasetVariant]) -> None:
        params = self._get_params()
        patterns = params["patterns"]  # type: ignore[index]

        self._variant_images = {}
        total_images = 0
        for variant in variants:
            input_dir = Path(variant.data_dir)
            images = discover_images(input_dir, patterns)
            if not images:
                raise FileNotFoundError(f"No images matched the provided extensions in {input_dir}.")
            self._variant_images[variant.name] = images
            total_images += len(images)

        if not self._settings_reported:
            print(
                "[info] Grayscale defense settings: "
                f"mode={params['mode']}, replicate_rgb={params['replicate_rgb']}, format={params['format_hint']}, "
                f"overwrite={params['overwrite']}, dry_run={params['dry_run']}, workers={params['workers']}"
            )
            self._settings_reported = True

        if self._progress is not None:
            self._progress.close()
        self._progress = Progress(total=total_images, description="Grayscale conversion", unit="images")

    def run(self, context: RunContext, variant: DatasetVariant) -> DatasetVariant:
        params = self._get_params()
        patterns = params["patterns"]  # type: ignore[index]
        mode = params["mode"]  # type: ignore[index]
        replicate_rgb = params["replicate_rgb"]  # type: ignore[index]
        overwrite = params["overwrite"]  # type: ignore[index]
        dry_run = params["dry_run"]  # type: ignore[index]
        format_hint = params["format_hint"]
        workers = params["workers"]  # type: ignore[index]

        input_dir = Path(variant.data_dir)
        output_root = ensure_dir(context.artifacts_dir / "defenses" / "grayscale" / variant.name)

        images = self._variant_images.get(variant.name)
        if images is None:
            images = discover_images(input_dir, patterns)
        if not images:
            raise FileNotFoundError(f"No images matched the provided extensions in {input_dir}.")

        progress = self._progress
        if progress is None:
            progress = Progress(total=len(images), description="Grayscale conversion", unit="images")
            self._progress = progress

        task = functools.partial(
            convert_to_grayscale,
            input_root=input_dir,
            output_root=output_root,
            mode=mode,
            replicate_rgb=replicate_rgb,
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
                    print(f"[warn] Failed to convert {path}: {status}")
                progress.update()

        metadata = {
            "defense": "grayscale",
            "mode": mode,
            "replicate_rgb": replicate_rgb,
            "written": written,
            "skipped": skipped,
            "failed": failed,
            "source_variant": variant.name,
        }
        if format_hint:
            metadata["format"] = format_hint

        return DatasetVariant(
            name=f"{variant.name}-grayscale",
            data_dir=str(output_root),
            parent=variant.name,
            metadata=metadata,
        )

    def finalize(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        self._variant_images.clear()


__all__ = ["GrayscaleDefense"]
