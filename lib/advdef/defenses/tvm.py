"""Total variation minimization defense (Rudin et al., 1992)."""

from __future__ import annotations

import concurrent.futures
import inspect
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageFile
from numpy.random import SeedSequence

try:
    from skimage.restoration import denoise_tv_chambolle
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "Missing dependency 'scikit-image'. Install it with `pip install scikit-image`."
    ) from exc

from advdef.config import DefenseConfig
from advdef.core.context import RunContext
from advdef.core.pipeline import DatasetVariant, Defense
from advdef.core.registry import register_defense
from advdef.utils import Progress, ensure_dir

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _resolve_tvm_parameters():
    try:
        return inspect.signature(denoise_tv_chambolle).parameters
    except (TypeError, ValueError):  # pragma: no cover - fallback when signature unavailable
        return {}


_TVM_PARAMS = _resolve_tvm_parameters()


def _run_tvm(
    array: np.ndarray,
    *,
    weight: float,
    eps: float,
    n_iter_max: int,
    channel_axis: int | None,
) -> np.ndarray:
    kwargs: dict[str, object] = {"weight": weight, "eps": eps}

    if "channel_axis" in _TVM_PARAMS:
        kwargs["channel_axis"] = channel_axis
    elif "multichannel" in _TVM_PARAMS:
        kwargs["multichannel"] = channel_axis is not None

    if "n_iter_max" in _TVM_PARAMS:
        kwargs["n_iter_max"] = n_iter_max
    elif "max_num_iter" in _TVM_PARAMS:
        kwargs["max_num_iter"] = n_iter_max

    return denoise_tv_chambolle(array, **kwargs)


def discover_images(root: Path, patterns: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.rglob(pattern))
    return sorted(paths)


def apply_tvm(
    src: Path,
    *,
    input_root: Path,
    output_root: Path,
    weight: float,
    eps: float,
    n_iter_max: int,
    multichannel: bool,
    overwrite: bool,
    dry_run: bool,
    format_hint: str | None,
    keep_probability: float,
    projection_steps: int,
    rng_seed: SeedSequence | int | None,
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

            mask: np.ndarray | None = None
            if 0.0 < keep_probability < 1.0:
                rng = np.random.default_rng(rng_seed)
                height, width = arr.shape[:2]
                mask = rng.random((height, width)) < keep_probability
                if not mask.any():
                    y = rng.integers(height)
                    x = rng.integers(width)
                    mask[y, x] = True

            def _denoise(array: np.ndarray) -> np.ndarray:
                if mask is None:
                    channel_axis = -1 if (multichannel and array.ndim == 3) else None
                    return _run_tvm(
                        array,
                        weight=weight,
                        eps=eps,
                        n_iter_max=n_iter_max,
                        channel_axis=channel_axis,
                    )

                working = array.copy()
                for _ in range(projection_steps):
                    channel_axis = -1 if (multichannel and working.ndim == 3) else None
                    working = _run_tvm(
                        working,
                        weight=weight,
                        eps=eps,
                        n_iter_max=n_iter_max,
                        channel_axis=channel_axis,
                    )
                    working[mask] = array[mask]
                return working

            if mode == "RGBA":
                rgb = arr[..., :3]
                alpha = arr[..., 3:]
                denoised_rgb = _denoise(rgb)
                denoised = np.concatenate([denoised_rgb, alpha], axis=-1)
            else:
                denoised = _denoise(arr)

            denoised_uint8 = np.clip(denoised * 255.0, 0.0, 255.0).round().astype(np.uint8)
            result = Image.fromarray(denoised_uint8, mode=mode)

            save_kwargs = {}
            if format_hint:
                save_kwargs["format"] = format_hint.upper()
            elif img.format:
                save_kwargs["format"] = img.format

            result.save(destination, **save_kwargs)
    except Exception as exc:  # pragma: no cover - file system dependent
        return f"failed: {exc}", destination.as_posix()

    return "written", destination.as_posix()


@register_defense("tvm")
class TVMDefense(Defense):
    """Reduce adversarial noise via total variation minimization."""

    def __init__(self, config: DefenseConfig) -> None:
        super().__init__(config)
        self._settings_reported = False
        self._progress: Progress | None = None
        self._variant_images: dict[str, list[Path]] = {}
        self._params_cache: dict[str, object] | None = None
        self._variant_image_seeds: dict[str, list[SeedSequence | None]] = {}

    def _get_params(self) -> dict[str, object]:
        if self._params_cache is None:
            params = self.config.params
            patterns = tuple(params.get("extensions", ("*.png", "*.jpg", "*.jpeg", "*.bmp")))
            weight = float(params.get("weight", 0.1))
            if weight < 0.0:
                raise ValueError("weight must be non-negative.")
            eps = float(params.get("eps", 1e-4))
            if eps <= 0.0:
                raise ValueError("eps must be positive.")
            n_iter_max = int(params.get("n_iter_max", 200))
            if n_iter_max <= 0:
                raise ValueError("n_iter_max must be positive.")
            multichannel = bool(params.get("multichannel", True))
            overwrite = bool(params.get("overwrite", True))
            dry_run = bool(params.get("dry_run", False))
            format_hint = params.get("format")
            workers = int(params.get("workers", max(1, (os.cpu_count() or 2) - 1)))
            keep_probability = float(params.get("keep_probability", 0.5))
            if not 0.0 <= keep_probability <= 1.0:
                raise ValueError("keep_probability must be within [0.0, 1.0].")
            projection_steps = int(params.get("projection_steps", 5))
            if projection_steps <= 0:
                raise ValueError("projection_steps must be positive.")
            seed_param = params.get("seed")
            seed = int(seed_param) if seed_param is not None else None

            self._params_cache = {
                "patterns": patterns,
                "weight": weight,
                "eps": eps,
                "n_iter_max": n_iter_max,
                "multichannel": multichannel,
                "overwrite": overwrite,
                "dry_run": dry_run,
                "format_hint": format_hint,
                "workers": workers,
                "keep_probability": keep_probability,
                "projection_steps": projection_steps,
                "seed": seed,
            }
        return self._params_cache

    def initialize(self, context: RunContext, variants: list[DatasetVariant]) -> None:
        details = self._get_params()
        patterns = details["patterns"]  # type: ignore[index]
        keep_probability = details["keep_probability"]  # type: ignore[index]
        seed_value = details["seed"]  # type: ignore[index]

        self._variant_images = {}
        self._variant_image_seeds = {}

        if seed_value is not None and 0.0 < keep_probability < 1.0:
            master_seed = SeedSequence(seed_value)
            variant_seed_sequences = master_seed.spawn(len(variants))
        else:
            variant_seed_sequences = [None] * len(variants)

        total_images = 0
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

        if not self._settings_reported:
            print(
                "[info] TVM defense settings: "
                f"weight={details['weight']}, eps={details['eps']}, n_iter_max={details['n_iter_max']}, "
                f"multichannel={details['multichannel']}, overwrite={details['overwrite']}, "
                f"dry_run={details['dry_run']}, workers={details['workers']}, format={details['format_hint']}, "
                f"keep_probability={details['keep_probability']}, projection_steps={details['projection_steps']}, "
                f"seed={details['seed']}"
            )
            self._settings_reported = True

        if self._progress is not None:
            self._progress.close()
        self._progress = Progress(total=total_images, description="TVM denoising", unit="images")

    def run(self, context: RunContext, variant: DatasetVariant) -> DatasetVariant:
        details = self._get_params()
        patterns = details["patterns"]  # type: ignore[index]
        weight = details["weight"]  # type: ignore[index]
        eps = details["eps"]  # type: ignore[index]
        n_iter_max = details["n_iter_max"]  # type: ignore[index]
        multichannel = details["multichannel"]  # type: ignore[index]
        overwrite = details["overwrite"]  # type: ignore[index]
        dry_run = details["dry_run"]  # type: ignore[index]
        format_hint = details["format_hint"]
        workers = details["workers"]  # type: ignore[index]
        keep_probability = details["keep_probability"]  # type: ignore[index]
        projection_steps = details["projection_steps"]  # type: ignore[index]
        seed_value = details["seed"]  # type: ignore[index]

        input_dir = Path(variant.data_dir)
        output_root = ensure_dir(context.artifacts_dir / "defenses" / "tvm" / variant.name)

        images = self._variant_images.get(variant.name)
        if images is None:
            images = discover_images(input_dir, patterns)
            if self._progress is not None:
                self._progress.add_total(len(images))

        if not images:
            raise FileNotFoundError(f"No images matched the provided extensions in {input_dir}.")

        progress = self._progress
        if progress is None:
            progress = Progress(total=len(images), description="TVM denoising", unit="images")
            self._progress = progress

        image_seeds = self._variant_image_seeds.get(variant.name)
        if image_seeds is None or len(image_seeds) != len(images):
            image_seeds = [None] * len(images)
            self._variant_image_seeds[variant.name] = image_seeds

        def task(src: Path, rng_seed: SeedSequence | None) -> tuple[str, str]:
            return apply_tvm(
                src,
                input_root=input_dir,
                output_root=output_root,
                weight=weight,
                eps=eps,
                n_iter_max=n_iter_max,
                multichannel=multichannel,
                overwrite=overwrite,
                dry_run=dry_run,
                format_hint=format_hint,
                keep_probability=keep_probability,
                projection_steps=projection_steps,
                rng_seed=rng_seed,
            )

        written = skipped = failed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for status, path in executor.map(task, images, image_seeds):
                if status == "written":
                    written += 1
                elif status == "skipped":
                    skipped += 1
                elif status == "dry-run":
                    print(f"[dry-run] would write {path}")
                else:
                    failed += 1
                    print(f"[warn] Failed to denoise {path}: {status}")
                progress.update()

        if failed:
            raise RuntimeError(
                f"TVM defense failed for {failed} image(s) in variant '{variant.name}'."
            )

        metadata = {
            "defense": "tvm",
            "weight": weight,
            "eps": eps,
            "n_iter_max": n_iter_max,
            "multichannel": multichannel,
            "format": format_hint,
            "keep_probability": keep_probability,
            "projection_steps": projection_steps,
            "seed": seed_value,
            "written": written,
            "skipped": skipped,
            "failed": failed,
            "source_variant": variant.name,
        }

        return DatasetVariant(
            name=f"{variant.name}-tvm",
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


__all__ = ["TVMDefense"]
