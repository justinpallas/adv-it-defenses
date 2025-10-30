"""JPEG recompression defense."""

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


def compress_image(
    src: Path,
    *,
    input_root: Path,
    output_root: Path,
    quality: int,
    progressive: bool,
    optimize: bool,
    overwrite: bool,
    dry_run: bool,
) -> tuple[str, str]:
    relative = src.relative_to(input_root)
    destination = (output_root / relative).with_suffix(".jpg")

    if not overwrite and destination.exists():
        return "skipped", destination.as_posix()

    if dry_run:
        return "dry-run", destination.as_posix()

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(src) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            save_kwargs = {
                "format": "JPEG",
                "quality": max(1, min(95, quality)),
                "optimize": optimize,
                "progressive": progressive,
            }
            img.save(destination, **save_kwargs)
    except Exception as exc:  # pragma: no cover - file system dependent
        return f"failed: {exc}", destination.as_posix()

    return "written", destination.as_posix()


@register_defense("jpeg")
class JPEGDefense(Defense):
    """Re-encode images with JPEG compression."""

    def __init__(self, config: DefenseConfig) -> None:
        super().__init__(config)
        self._settings_reported = False
        self._progress: Progress | None = None

    def run(self, context: RunContext, variant: DatasetVariant) -> DatasetVariant:
        params = self.config.params
        patterns = params.get("extensions", ("*.png", "*.jpg", "*.jpeg", "*.bmp"))
        quality = int(params.get("quality", 75))
        progressive = bool(params.get("progressive", False))
        optimize = bool(params.get("optimize", False))
        overwrite = bool(params.get("overwrite", True))
        dry_run = bool(params.get("dry_run", False))
        workers = int(params.get("workers", max(1, (os.cpu_count() or 2) - 1)))

        if not self._settings_reported:
            print(
                "[info] JPEG defense settings: "
                f"quality={quality}, progressive={progressive}, optimize={optimize}, "
                f"overwrite={overwrite}, dry_run={dry_run}, workers={workers}"
            )
            self._settings_reported = True

        input_dir = Path(variant.data_dir)
        output_root = ensure_dir(context.artifacts_dir / "defenses" / "jpeg" / variant.name)

        images = discover_images(input_dir, patterns)
        if not images:
            raise FileNotFoundError(f"No images matched the provided extensions in {input_dir}.")

        if self._progress is None:
            self._progress = Progress(total=len(images), description="JPEG compression", unit="images")
        else:
            self._progress.add_total(len(images))
        progress = self._progress

        task = functools.partial(
            compress_image,
            input_root=input_dir,
            output_root=output_root,
            quality=quality,
            progressive=progressive,
            optimize=optimize,
            overwrite=overwrite,
            dry_run=dry_run,
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
                    print(f"[warn] Failed to compress {path}: {status}")
                progress.update()

        metadata = {
            "defense": "jpeg",
            "quality": quality,
            "progressive": progressive,
            "optimize": optimize,
            "written": written,
            "skipped": skipped,
            "failed": failed,
            "source_variant": variant.name,
        }

        return DatasetVariant(
            name=f"{variant.name}-jpeg",
            data_dir=str(output_root),
            parent=variant.name,
            metadata=metadata,
        )

    def finalize(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None


__all__ = ["JPEGDefense"]
