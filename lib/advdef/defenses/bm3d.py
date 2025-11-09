"""Block-Matching 3D (BM3D) denoising defense (CLI backend only)."""

from __future__ import annotations

import concurrent.futures
import functools
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageFile

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


def run_cli_denoiser(
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

            color_flag = working.mode != "L" if cli_color_mode == "auto" else cli_color_mode == "color"

            with tempfile.TemporaryDirectory(prefix="bm3d_cli_") as tmpdir:
                tmp_dir = Path(tmpdir)
                tmp_input = tmp_dir / "input.bmp"
                tmp_output = tmp_dir / "output.bmp"
                working.save(tmp_input, format="BMP")

                sigma_arg = f"{sigma_cli:.6f}".rstrip("0").rstrip(".") or "0"
                cmd: list[str] = [
                    str(binary_path),
                    tmp_input.as_posix(),
                    tmp_output.as_posix(),
                    sigma_arg,
                    "color" if color_flag else "nocolor",
                ]
                if cli_twostep:
                    cmd.append("twostep")
                if cli_quiet:
                    cmd.append("quiet")
                if cli_extra_args:
                    cmd.extend(cli_extra_args)

                run_kwargs = {}
                if not cli_log_output:
                    run_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}

                result = subprocess.run(cmd, check=True, **run_kwargs)

                if tmp_output.exists():
                    result_img = Image.open(tmp_output)
                    result_img.load()
                else:
                    captured = ""
                    if not cli_log_output:
                        captured = (result.stdout or "") + (result.stderr or "")
                    raise RuntimeError(
                        "bm3d-gpu did not produce an output image."
                        + (f" Output:\n{captured}" if captured else "")
                    )

        if preserve_alpha and alpha_channel is not None:
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
    """Apply BM3D denoising via the external CUDA/CLI backend."""

    def __init__(self, config: DefenseConfig) -> None:
        super().__init__(config)
        self._settings_reported = False
        self._progress: Progress | None = None
        self._variant_images: dict[str, list[Path]] = {}
        self._params_cache: dict[str, object] | None = None
        self._config_identifier = build_config_identifier(config, default_prefix="bm3d")

    def _get_params(self) -> dict[str, object]:
        if self._params_cache is None:
            params = self.config.params
            backend = str(params.get("backend", "cli")).lower()
            if backend != "cli":
                raise ValueError("BM3D defense only supports backend='cli'.")

            patterns = tuple(params.get("extensions", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")))
            sigma = float(params.get("sigma", 25.0))
            if sigma < 0.0:
                raise ValueError("sigma must be non-negative.")
            sigma_scale = str(params.get("sigma_scale", "pixel")).lower()
            if sigma_scale not in {"pixel", "relative"}:
                raise ValueError("sigma_scale must be 'pixel' or 'relative'.")
            sigma_cli = sigma if sigma_scale == "pixel" else sigma * 255.0

            convert_mode = str(params.get("convert_mode", "auto")).lower()
            if convert_mode not in {"auto", "rgb", "luma"}:
                raise ValueError("convert_mode must be 'auto', 'rgb', or 'luma'.")
            preserve_alpha = bool(params.get("preserve_alpha", True))
            clip_output = bool(params.get("clip_output", True))
            format_hint = params.get("format")
            overwrite = bool(params.get("overwrite", True))
            dry_run = bool(params.get("dry_run", False))
            workers = int(params.get("workers", max(1, (os.cpu_count() or 2) - 1)))
            if workers <= 0:
                raise ValueError("workers must be positive.")

            default_binary = Path("external/bm3d-gpu/build/bm3d")
            cli_binary = Path(params.get("cli_binary") or params.get("binary_path") or default_binary)
            if not cli_binary.exists():
                raise FileNotFoundError(
                    f"bm3d-gpu binary not found at {cli_binary}. Build it via `advdef setup bm3d-gpu` "
                    "or point `cli_binary` to an existing executable."
                )

            cli_color_mode = str(params.get("cli_color_mode", "auto")).lower()
            if cli_color_mode not in {"auto", "color", "grayscale"}:
                raise ValueError("cli_color_mode must be 'auto', 'color', or 'grayscale'.")
            cli_twostep = bool(params.get("cli_twostep", True))
            cli_quiet = bool(params.get("cli_quiet", True))
            cli_log_output = bool(params.get("cli_log_output", False))
            cli_extra_args = tuple(str(arg) for arg in params.get("cli_extra_args", ()))

            self._params_cache = {
                "patterns": patterns,
                "sigma": sigma,
                "sigma_scale": sigma_scale,
                "sigma_cli": sigma_cli,
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
                "cli_log_output": cli_log_output,
                "cli_extra_args": cli_extra_args,
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
            extra_args = list(params["cli_extra_args"])  # type: ignore[index]
            print(
                "[info] BM3D defense settings: "
                f"config={self.config.name or self._config_identifier} "
                f"sigma={params['sigma']} ({params['sigma_scale']}), "
                f"binary={params['cli_binary']} "
                f"color_mode={params['cli_color_mode']} twostep={params['cli_twostep']} "
                f"quiet={params['cli_quiet']} log_output={params['cli_log_output']} "
                f"extra_args={extra_args or '<none>'} "
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
        sigma_cli = params["sigma_cli"]  # type: ignore[index]
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
            run_cli_denoiser,
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
            "convert_mode": convert_mode,
            "preserve_alpha": preserve_alpha,
            "clip_output": clip_output,
            "backend": "cli",
            "cli_binary": str(params["cli_binary"]),
            "cli_color_mode": params["cli_color_mode"],
            "cli_twostep": params["cli_twostep"],
            "cli_quiet": params["cli_quiet"],
            "cli_log_output": params["cli_log_output"],
            "cli_extra_args": list(params["cli_extra_args"]),
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


__all__ = ["BM3DDefense"]
