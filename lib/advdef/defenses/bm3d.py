"""Block-Matching 3D (BM3D) denoising defense."""

from __future__ import annotations

import concurrent.futures
import functools
import importlib
import inspect
import os
import subprocess
import tempfile
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


def apply_bm3d_python(
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
            original_format = img.format
            alpha_channel = None
            working = img
            original_format = img.format
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


def apply_bm3d_cli(
    src: Path,
    *,
    input_root: Path,
    output_root: Path,
    binary_path: Path,
    sigma_cli: float,
    convert_mode: str,
    preserve_alpha: bool,
    format_hint: str | None,
    overwrite: bool,
    dry_run: bool,
    cli_color_mode: str,
    cli_twostep: bool,
    cli_quiet: bool,
    cli_extra_args: Sequence[str],
    cli_log_output: bool,
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
        original_format: str | None = None
        with Image.open(src) as img:
            original_format = img.format
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

            color_flag: bool
            if cli_color_mode == "auto":
                color_flag = working.mode != "L"
            else:
                color_flag = cli_color_mode == "color"

            with tempfile.TemporaryDirectory(prefix="bm3d_cli_") as tmpdir:
                tmp_dir = Path(tmpdir)
                tmp_input = tmp_dir / "input.bmp"
                tmp_output = tmp_dir / "output.bmp"
                working.save(tmp_input, format="BMP")

                sigma_arg = f"{sigma_cli:.6f}".rstrip("0").rstrip(".")
                cmd: list[str] = [
                    str(binary_path),
                    tmp_input.as_posix(),
                    tmp_output.as_posix(),
                    sigma_arg or "0",
                    "color" if color_flag else "nocolor",
                ]
                if cli_twostep:
                    cmd.append("twostep")
                if cli_quiet:
                    cmd.append("quiet")
                if cli_extra_args:
                    cmd.extend(cli_extra_args)

                run_kwargs: dict[str, Any] = {}
                if not cli_log_output:
                    run_kwargs.update({"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True})

                result = subprocess.run(cmd, check=True, **run_kwargs)

                if tmp_output.exists():
                    result_img = Image.open(tmp_output)
                    result_img.load()
                else:
                    captured = ""
                    if not cli_log_output and result is not None:
                        captured = (result.stdout or "") + (result.stderr or "")
                    raise RuntimeError(
                        "bm3d-gpu did not produce an output image."
                        + (f" Output:\n{captured}" if captured else "")
                    )

            # preserve original format information outside context

            if alpha_channel is not None and preserve_alpha:
                if result_img.mode != "RGB":
                    result_img = result_img.convert("RGB")
                alpha_img = Image.fromarray(alpha_channel, mode="L")
                result_img.putalpha(alpha_img)

            save_kwargs = {}
            if format_hint:
                save_kwargs["format"] = format_hint.upper()
            elif original_format:
                save_kwargs["format"] = original_format

            result_img.save(destination, **save_kwargs)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - external binary failure
        return f"failed: bm3d-gpu exited with code {exc.returncode}", destination.as_posix()
    except Exception as exc:  # pragma: no cover - file system / binary dependent
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
            sigma_cli = sigma if sigma_scale == "pixel" else sigma * 255.0
            profile = params.get("profile", "np")
            profile = str(profile) if profile is not None else None
            profile_parameter = str(params.get("profile_parameter", "profile"))
            stage = params.get("stage", "all")
            stage = str(stage) if stage is not None else None
            stage_parameter = str(params.get("stage_parameter", "stage_arg"))
            backend_label = str(params.get("backend", "cpu")).lower()
            if backend_label not in {"cpu", "cuda", "cli"}:
                raise ValueError("backend must be one of 'cpu', 'cuda', or 'cli'.")
            backend_kind = "cli" if backend_label == "cli" else "python"

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

            bm3d_kwargs: dict[str, Any] = {}
            backend_module: str | None = None
            function_gray: str | None = None
            function_rgb: str | None = None
            cli_binary: Path | None = None
            cli_color_mode = str(params.get("cli_color_mode", "auto")).lower()
            if cli_color_mode not in {"auto", "color", "grayscale"}:
                raise ValueError("cli_color_mode must be 'auto', 'color', or 'grayscale'.")
            cli_twostep = bool(params.get("cli_twostep", True))
            cli_quiet = bool(params.get("cli_quiet", True))
            cli_extra_args = tuple(str(arg) for arg in params.get("cli_extra_args", ()))
            cli_log_output = bool(params.get("cli_log_output", False))

            if backend_kind == "python":
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
            else:
                default_binary = Path("external/bm3d-gpu/build/bm3d")
                cli_binary_param = params.get("cli_binary") or params.get("binary_path") or default_binary
                cli_binary = Path(cli_binary_param)
                if not cli_binary.exists():
                    raise FileNotFoundError(
                        f"bm3d-gpu binary not found at {cli_binary}. Build it via `advdef setup bm3d-gpu` "
                        "or point `cli_binary` to an existing executable."
                    )

            self._params_cache = {
                "patterns": patterns,
                "sigma": sigma,
                "sigma_scale": sigma_scale,
                "sigma_psd": sigma_psd,
                "sigma_cli": sigma_cli,
                "profile": profile,
                "profile_parameter": profile_parameter,
                "stage": stage,
                "stage_parameter": stage_parameter,
                "backend_label": backend_label,
                "backend_kind": backend_kind,
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
                "cli_binary": cli_binary,
                "cli_color_mode": cli_color_mode,
                "cli_twostep": cli_twostep,
                "cli_quiet": cli_quiet,
                "cli_extra_args": cli_extra_args,
                "cli_log_output": cli_log_output,
            }
        return self._params_cache

    def _get_backend(self, params: Mapping[str, Any]) -> _BM3DBackend:
        if params["backend_kind"] != "python":
            raise RuntimeError("CLI backend does not provide callable functions.")
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
            if params["backend_kind"] == "python":
                bm3d_kwargs = params["bm3d_kwargs"]  # type: ignore[index]
                described_kwargs = ", ".join(sorted(bm3d_kwargs)) or "<none>"
                backend_summary = (
                    f"backend={params['backend_label']}::{params['backend_module']} "
                    f"functions=({params['function_gray']}, {params['function_rgb']}), "
                    f"bm3d_kwargs={described_kwargs}"
                )
            else:
                extra_args = list(params["cli_extra_args"])  # type: ignore[index]
                backend_summary = (
                    f"backend=cli binary={params['cli_binary']} "
                    f"color_mode={params['cli_color_mode']} twostep={params['cli_twostep']} "
                    f"quiet={params['cli_quiet']} log_output={params['cli_log_output']} "
                    f"extra_args={extra_args or '<none>'}"
                )

            print(
                "[info] BM3D defense settings: "
                f"config={self.config.name or self._config_identifier} "
                f"sigma={params['sigma']} ({params['sigma_scale']}), "
                f"{backend_summary} "
                f"convert_mode={params['convert_mode']}, preserve_alpha={params['preserve_alpha']}, "
                f"clip_output={params['clip_output']}, format={params['format_hint']}, "
                f"workers={params['workers']}"
            )
            self._settings_reported = True

        if self._progress is not None:
            self._progress.close()
        self._progress = Progress(total=total_images, description="BM3D denoising", unit="images")

    def run(self, context: RunContext, variant: DatasetVariant) -> DatasetVariant:
        params = self._get_params()
        backend_kind = params["backend_kind"]  # type: ignore[index]
        backend = self._get_backend(params) if backend_kind == "python" else None
        sigma_psd = params["sigma_psd"]  # type: ignore[index]
        sigma_cli = params["sigma_cli"]  # type: ignore[index]
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

        if backend_kind == "python":
            task = functools.partial(
                apply_bm3d_python,
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
        else:
            task = functools.partial(
                apply_bm3d_cli,
                input_root=input_dir,
                output_root=output_root,
                binary_path=params["cli_binary"],
                sigma_cli=sigma_cli,
                convert_mode=convert_mode,
                preserve_alpha=preserve_alpha,
                format_hint=format_hint,
                overwrite=overwrite,
                dry_run=dry_run,
                cli_color_mode=params["cli_color_mode"],
                cli_twostep=params["cli_twostep"],
                cli_quiet=params["cli_quiet"],
                cli_extra_args=params["cli_extra_args"],
                cli_log_output=params["cli_log_output"],
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
            "backend_kind": backend_kind,
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
        if backend_kind == "python":
            metadata["backend_module"] = params["backend_module"]
            metadata["function_gray"] = params["function_gray"]
            metadata["function_rgb"] = params["function_rgb"]
            metadata["bm3d_kwargs"] = dict(params["bm3d_kwargs"] or {})
        else:
            metadata["cli_binary"] = str(params["cli_binary"])
            metadata["cli_color_mode"] = params["cli_color_mode"]
            metadata["cli_twostep"] = params["cli_twostep"]
            metadata["cli_quiet"] = params["cli_quiet"]
            metadata["cli_extra_args"] = list(params["cli_extra_args"])
            metadata["cli_log_output"] = params["cli_log_output"]
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
