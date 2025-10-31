"""Horizontal flip defense."""

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


def flip_image(
    src: Path,
    *,
    input_root: Path,
    output_root: Path,
    overwrite: bool,
    dry_run: bool,
    format_hint: str | None,
    direction: str,
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
            if direction == "horizontal":
                flipped = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            elif direction == "vertical":
                flipped = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            else:
                raise ValueError(f"Unsupported flip direction '{direction}'.")

            save_kwargs = {}
            if format_hint:
                save_kwargs["format"] = format_hint.upper()
            elif img.format:
                save_kwargs["format"] = img.format

            flipped.save(destination, **save_kwargs)
    except Exception as exc:  # pragma: no cover - file system dependent
        return f"failed: {exc}", destination.as_posix()

    return "written", destination.as_posix()


@register_defense("flip")
class FlipDefense(Defense):
    """Flip images horizontally or vertically."""

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
            direction = str(params.get("direction", "horizontal")).lower()
            if direction not in {"horizontal", "vertical"}:
                raise ValueError("direction must be 'horizontal' or 'vertical'.")
            overwrite = bool(params.get("overwrite", True))
            dry_run = bool(params.get("dry_run", False))
            format_hint = params.get("format")
            workers = int(params.get("workers", max(1, (os.cpu_count() or 2) - 1)))

            self._params_cache = {
                "patterns": patterns,
                "direction": direction,
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
                "[info] Flip defense settings: "
                f"direction={details['direction']}, overwrite={details['overwrite']}, "
                f"dry_run={details['dry_run']}, workers={details['workers']}, format={details['format_hint']}"
            )
            self._settings_reported = True

        if self._progress is not None:
            self._progress.close()
        self._progress = Progress(total=total_images, description="Image flipping", unit="images")

    def run(self, context: RunContext, variant: DatasetVariant) -> DatasetVariant:
        details = self._get_params()
        patterns = details["patterns"]  # type: ignore[index]
        direction = details["direction"]  # type: ignore[index]
        overwrite = details["overwrite"]  # type: ignore[index]
        dry_run = details["dry_run"]  # type: ignore[index]
        format_hint = details["format_hint"]
        workers = details["workers"]  # type: ignore[index]

        input_dir = Path(variant.data_dir)
        output_root = ensure_dir(context.artifacts_dir / "defenses" / "flip" / variant.name)

        images = self._variant_images.get(variant.name)
        if images is None:
            images = discover_images(input_dir, patterns)
            if self._progress is not None:
                self._progress.add_total(len(images))

        if not images:
            raise FileNotFoundError(f"No images matched the provided extensions in {input_dir}.")

        progress = self._progress
        if progress is None:
            progress = Progress(total=len(images), description="Image flipping", unit="images")
            self._progress = progress

        task = functools.partial(
            flip_image,
            input_root=input_dir,
            output_root=output_root,
            overwrite=overwrite,
            dry_run=dry_run,
            format_hint=format_hint,
            direction=direction,
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
                    print(f"[warn] Failed to flip {path}: {status}")
                progress.update()

        metadata = {
            "defense": "flip",
            "direction": direction,
            "format": format_hint,
            "written": written,
            "skipped": skipped,
            "failed": failed,
            "source_variant": variant.name,
        }

        return DatasetVariant(
            name=f"{variant.name}-flip",
            data_dir=str(output_root),
            parent=variant.name,
            metadata=metadata,
        )

    def finalize(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        self._variant_images.clear()


__all__ = ["FlipDefense"]
