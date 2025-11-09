"""Block-Matching 3D (BM3D) denoising defense."""

from __future__ import annotations

import concurrent.futures
import functools
import importlib
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageFile

try:  # Torch is an optional dependency for CUDA-focused backends.
    import torch
except ImportError:  # pragma: no cover - torch is already required in the project deps
    torch = None  # type: ignore[assignment]

from advdef.config import DefenseConfig
from advdef.core.context import RunContext
from advdef.core.pipeline import DatasetVariant, Defense
from advdef.core.registry import register_defense
from advdef.utils import Progress, ensure_dir
from ._common import build_config_identifier

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(slots=True)
class _BM3DBackend:
    """Holds callables for grayscale and RGB BM3D entry points."""

    gray: Callable[[np.ndarray, float], np.ndarray]
    color: Callable[[np.ndarray, float], np.ndarray]
    module_name: str
    gray_name: str
    color_name: str


def discover_images(root: Path, patterns: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.rglob(pattern))
    return sorted(paths)


def _import_backend(module_name: str, gray_name: str, color_name: str) -> _BM3DBackend:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise SystemExit(
            f"Missing dependency '{module_name}'. Install it with `pip install {module_name}` "
            "or point `backend_module` at a compatible implementation."
        ) from exc

    try:
        gray = getattr(module, gray_name)
    except AttributeError as exc:  # pragma: no cover - configuration hint
        raise SystemExit(
            f"Module '{module_name}' does not expose '{gray_name}'. "
            "Adjust the `function_gray` parameter to match the backend API."
        ) from exc

    try:
        color = getattr(module, color_name)
    except AttributeError as exc:  # pragma: no cover - configuration hint
        raise SystemExit(
            f"Module '{module_name}' does not expose '{color_name}'. "
            "Adjust the `function_rgb` parameter to match the backend API."
        ) from exc

    return _BM3DBackend(
        gray=gray,
        color=color,
        module_name=module_name,
        gray_name=gray_name,
        color_name=color_name,
    )


def _filter_kwargs(func: Callable[..., Any], candidates: Mapping[str, Any]) -> dict[str, Any]:
    if not candidates:
        return {}

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - native/opaque callables
        return {}

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return {key: value for key, value in candidates.items() if value is not None}

    filtered: dict[str, Any] = {}
    for key, value in candidates.items():
        if value is None:
            continue
        if key in signature.parameters:
            filtered[key] = value
    return filtered


def _normalize_result(array: Any, clip_output: bool) -> np.ndarray:
    if torch is not None and isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()

    result = np.asarray(array, dtype=np.float32)
    if result.size == 0:
        return result

    # Some backends operate in [0, 255] whereas others use [0, 1].
    max_value = float(np.max(result))
    if max_value > 1.5:
        result /= 255.0

    min_value = float(np.min(result))
    if min_value < -0.01:
        result -= min_value

    if clip_output:
        result = np.clip(result, 0.0, 1.0)
    return result


def apply_bm3d(
    src: Path,
    *,
    input_root: Path,
    output_root: Path,
    backend: _BM3DBackend,
    sigma_psd: float,
    bm3d_kwargs: Mapping[str, Any],
    convert_mode: str,
    preserve_alpha: bool,
    format_hint: str | None,
    overwrite: bool,
    dry_run: bool,
    clip_output: bool,
) -> tuple[str, str]:
    bm3d_kwargs = dict(bm3d_kwargs)
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
            alpha_channel = None
            working = img
            if img.mode in ("RGBA", "LA") and preserve_alpha:
                alpha_channel = np.asarray(img.getchannel("A"), dtype=np.uint8)

            if convert_mode == "rgb":
                working = img.convert("RGB")
            elif convert_mode == "luma":
                working = img.convert("L")
            else:
                if img.mode == "RGBA":
                    working = img.convert("RGB")
                elif img.mode == "LA":
                    working = img.convert("L")
                elif img.mode not in ("RGB", "L"):
                    working = img.convert("RGB")

            array = np.asarray(working, dtype=np.float32) / 255.0
            array = np.ascontiguousarray(array)

            if array.ndim == 2:
                fn = backend.gray
            else:
                fn = backend.color

            kwargs = _filter_kwargs(fn, bm3d_kwargs)
            result = fn(array, sigma_psd, **kwargs)
            normalized = _normalize_result(result, clip_output=clip_output)
            result_uint8 = np.clip(normalized * 255.0, 0.0, 255.0).round().astype(np.uint8)

            if result_uint8.ndim == 2:
                result_img = Image.fromarray(result_uint8, mode="L")
            else:
                result_img = Image.fromarray(result_uint8, mode="RGB")

            if alpha_channel is not None and preserve_alpha:
                alpha_img = Image.fromarray(alpha_channel, mode="L")
                if result_img.mode == "RGB":
                    result_img.putalpha(alpha_img)
                else:
                    result_img = result_img.convert("LA")
                    result_img.putalpha(alpha_img)

            save_kwargs = {}
            if format_hint:
                save_kwargs["format"] = format_hint.upper()
            elif img.format:
                save_kwargs["format"] = img.format

            result_img.save(destination, **save_kwargs)
    except Exception as exc:  # pragma: no cover - file system / backend dependent
        return f"failed: {exc}", destination.as_posix()

    return "written", destination.as_posix()


@register_defense("bm3d")
class BM3DDefense(Defense):
    """Apply BM3D denoising to mitigate adversarial perturbations."""

    def __init__(self, config: DefenseConfig) -> None:
        super().__init__(config)
        self._settings_reported = False
        self._progress: Progress | None = None
        self._variant_images: dict[str, list[Path]] = {}
        self._params_cache: dict[str, Any] | None = None
        self._backend: _BM3DBackend | None = None
        self._config_identifier = build_config_identifier(config, default_prefix="bm3d")

    def _get_params(self) -> dict[str, Any]:
        if self._params_cache is None:
            params = self.config.params
            patterns = tuple(params.get("extensions", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")))
            sigma = float(params.get("sigma", 25.0))
            if sigma < 0.0:
                raise ValueError("sigma must be non-negative.")
            sigma_scale = str(params.get("sigma_scale", "pixel")).lower()
            if sigma_scale not in {"pixel", "relative"}:
                raise ValueError("sigma_scale must be 'pixel' or 'relative'.")
            sigma_psd = sigma / 255.0 if sigma_scale == "pixel" else sigma
            profile = params.get("profile", "np")
            profile = str(profile) if profile is not None else None
            profile_parameter = str(params.get("profile_parameter", "profile"))
            stage = params.get("stage", "all")
            stage = str(stage) if stage is not None else None
            stage_parameter = str(params.get("stage_parameter", "stage_arg"))
            backend_label = str(params.get("backend", "cpu")).lower()
            backend_module = str(
                params.get("backend_module")
                or ("bm3d_cuda" if backend_label == "cuda" else "bm3d")
            )
            default_gray = "bm3d" if backend_label == "cpu" else "bm3d_gray_cuda"
            default_rgb = "bm3d_rgb" if backend_label == "cpu" else "bm3d_rgb_cuda"
            function_gray = str(params.get("function_gray") or default_gray)
            function_rgb = str(params.get("function_rgb") or default_rgb)
            bm3d_kwargs = dict(params.get("bm3d_kwargs", {}))
            bm3d_kwargs = {str(key): value for key, value in bm3d_kwargs.items()}
            if profile and profile_parameter and profile_parameter not in bm3d_kwargs:
                bm3d_kwargs[profile_parameter] = profile
            if stage and stage_parameter and stage_parameter not in bm3d_kwargs:
                bm3d_kwargs[stage_parameter] = stage

            device = params.get("device")
            if backend_label == "cuda" and device is None:
                device = "cuda"
            if device is not None and "device" not in bm3d_kwargs:
                bm3d_kwargs["device"] = device

            convert_mode = str(params.get("convert_mode", "auto")).lower()
            if convert_mode not in {"auto", "rgb", "luma"}:
                raise ValueError("convert_mode must be 'auto', 'rgb', or 'luma'.")
            preserve_alpha = bool(params.get("preserve_alpha", True))
            clip_output = bool(params.get("clip_output", True))
            format_hint = params.get("format")
            overwrite = bool(params.get("overwrite", True))
            dry_run = bool(params.get("dry_run", False))
            workers_default = 1 if backend_label == "cuda" else max(1, (os.cpu_count() or 2) - 1)
            workers = int(params.get("workers", workers_default))
            if workers <= 0:
                raise ValueError("workers must be positive.")

            self._params_cache = {
                "patterns": patterns,
                "sigma": sigma,
                "sigma_scale": sigma_scale,
                "sigma_psd": sigma_psd,
                "profile": profile,
                "profile_parameter": profile_parameter,
                "stage": stage,
                "stage_parameter": stage_parameter,
                "backend_label": backend_label,
                "backend_module": backend_module,
                "function_gray": function_gray,
                "function_rgb": function_rgb,
                "bm3d_kwargs": bm3d_kwargs,
                "convert_mode": convert_mode,
                "preserve_alpha": preserve_alpha,
                "clip_output": clip_output,
                "format_hint": format_hint,
                "overwrite": overwrite,
                "dry_run": dry_run,
                "workers": workers,
            }
        return self._params_cache

    def _get_backend(self, params: Mapping[str, Any]) -> _BM3DBackend:
        if self._backend is None:
            self._backend = _import_backend(
                module_name=params["backend_module"],
                gray_name=params["function_gray"],
                color_name=params["function_rgb"],
            )
        return self._backend

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
            bm3d_kwargs = params["bm3d_kwargs"]  # type: ignore[index]
            described_kwargs = ", ".join(sorted(bm3d_kwargs)) or "<none>"
            print(
                "[info] BM3D defense settings: "
                f"config={self.config.name or self._config_identifier} "
                f"sigma={params['sigma']} ({params['sigma_scale']}), "
                f"backend={params['backend_label']}::{params['backend_module']} "
                f"functions=({params['function_gray']}, {params['function_rgb']}), "
                f"convert_mode={params['convert_mode']}, preserve_alpha={params['preserve_alpha']}, "
                f"clip_output={params['clip_output']}, format={params['format_hint']}, "
                f"workers={params['workers']}, bm3d_kwargs={described_kwargs}"
            )
            self._settings_reported = True

        if self._progress is not None:
            self._progress.close()
        self._progress = Progress(total=total_images, description="BM3D denoising", unit="images")

    def run(self, context: RunContext, variant: DatasetVariant) -> DatasetVariant:
        params = self._get_params()
        backend = self._get_backend(params)
        sigma_psd = params["sigma_psd"]  # type: ignore[index]
        bm3d_kwargs = params["bm3d_kwargs"]  # type: ignore[index]
        convert_mode = params["convert_mode"]  # type: ignore[index]
        preserve_alpha = params["preserve_alpha"]  # type: ignore[index]
        clip_output = params["clip_output"]  # type: ignore[index]
        format_hint = params["format_hint"]
        overwrite = params["overwrite"]  # type: ignore[index]
        dry_run = params["dry_run"]  # type: ignore[index]
        workers = params["workers"]  # type: ignore[index]
        patterns = params["patterns"]  # type: ignore[index]

        input_dir = Path(variant.data_dir)
        output_root = ensure_dir(
            context.artifacts_dir / "defenses" / "bm3d" / variant.name / self._config_identifier
        )

        images = self._variant_images.get(variant.name)
        if images is None:
            images = discover_images(input_dir, patterns)
        if not images:
            raise FileNotFoundError(f"No images matched the provided extensions in {input_dir}.")

        progress = self._progress
        if progress is None:
            progress = Progress(total=len(images), description="BM3D denoising", unit="images")
            self._progress = progress

        task = functools.partial(
            apply_bm3d,
            input_root=input_dir,
            output_root=output_root,
            backend=backend,
            sigma_psd=sigma_psd,
            bm3d_kwargs=bm3d_kwargs,
            convert_mode=convert_mode,
            preserve_alpha=preserve_alpha,
            format_hint=format_hint,
            overwrite=overwrite,
            dry_run=dry_run,
            clip_output=clip_output,
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
                    print(f"[warn] Failed to denoise {path}: {status}")
                progress.update()

        metadata = {
            "defense": "bm3d",
            "sigma": params["sigma"],
            "sigma_scale": params["sigma_scale"],
            "profile": params["profile"],
            "stage": params["stage"],
            "backend": params["backend_label"],
            "backend_module": params["backend_module"],
            "function_gray": params["function_gray"],
            "function_rgb": params["function_rgb"],
            "convert_mode": convert_mode,
            "preserve_alpha": preserve_alpha,
            "clip_output": clip_output,
            "written": written,
            "skipped": skipped,
            "failed": failed,
            "source_variant": variant.name,
            "config_name": self.config.name,
            "config_identifier": self._config_identifier,
        }
        if format_hint:
            metadata["format"] = format_hint

        return DatasetVariant(
            name=f"{variant.name}-bm3d-{self._config_identifier}",
            data_dir=str(output_root),
            parent=variant.name,
            metadata=metadata,
        )

    def finalize(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        self._variant_images.clear()
        self._backend = None


__all__ = ["BM3DDefense"]
