"""R-SMOE defense integration."""

from __future__ import annotations

import importlib
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import SimpleQueue
from threading import Lock
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from advdef.config import DefenseConfig
from advdef.core.context import RunContext
from advdef.core.pipeline import DatasetVariant, Defense
from advdef.core.registry import register_defense
from advdef.utils import Progress, ensure_dir
from ._common import build_config_identifier

DEFAULT_IMAGE_FILENAME = "0000001.png"
DEFAULT_CAMERA_ID = 1
DEFAULT_IMAGE_ID = 1
SECONDARY_IMAGE_FILENAMES = ("000001.png",)
SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}

FUSION_EXPECTED_SHIFTS = 64


def find_images(root: Path) -> List[Path]:
    candidates = sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"No supported images found in {root}.")
    return candidates


def ensure_empty_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target directory already exists: {path}. "
                "Set overwrite=True to replace it."
            )
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def convert_to_png(src: Path, dst: Path) -> None:
    with Image.open(src) as image:
        image.save(dst, format="PNG")


def copy_image(src: Path, dst: Path, convert: bool) -> None:
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dst)
        return

    if not convert:
        raise ValueError(
            f"Image {src} is not a PNG. Enable convert_to_png to convert automatically "
            "or provide PNG files."
        )
    convert_to_png(src, dst)


def get_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.width, image.height


def write_sparse_stub(
    sparse_dir: Path,
    image_filename: str,
    width: int,
    height: int,
) -> None:
    colmap_dir = sparse_dir / "0"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    fx = fy = max(width, height)
    cx = width / 2.0
    cy = height / 2.0

    cameras_txt = "\n".join(
        [
            "# Camera list with one line of data per camera:",
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
            f"{DEFAULT_CAMERA_ID} PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}",
            "",
        ]
    )
    (colmap_dir / "cameras.txt").write_text(cameras_txt)

    images_txt = "\n".join(
        [
            "# Image list with two lines of data per image:",
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
            f"{DEFAULT_IMAGE_ID} 1.0 0.0 0.0 0.0 0.0 0.0 0.0 {DEFAULT_CAMERA_ID} {image_filename}",
            "",
        ]
    )
    (colmap_dir / "images.txt").write_text(images_txt)

    points_txt = "\n".join(
        [
            "# 3D point list with one line of data per point:",
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
            "1 0.0 0.0 0.0 255 255 255 1.0",
            "",
        ]
    )
    (colmap_dir / "points3D.txt").write_text(points_txt)

    xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    rgb = np.array([[255, 255, 255]], dtype=np.uint8)
    normals = np.zeros_like(xyz, dtype=np.float32)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    vertices = np.empty(xyz.shape[0], dtype=dtype)
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    vertices["nx"] = normals[:, 0]
    vertices["ny"] = normals[:, 1]
    vertices["nz"] = normals[:, 2]
    vertices["red"] = rgb[:, 0]
    vertices["green"] = rgb[:, 1]
    vertices["blue"] = rgb[:, 2]
    vertex_element = PlyElement.describe(vertices, "vertex")
    PlyData([vertex_element]).write(colmap_dir / "points3D.ply")


def prepare_dataset(
    images: Sequence[Path],
    output_root: Path,
    image_filename: str,
    overwrite: bool,
    convert_to_png_flag: bool,
    progress: Progress | None = None,
) -> List[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    prepared_dirs: List[Path] = []

    manage_progress = progress is None
    if manage_progress:
        progress = Progress(total=len(images), description="Preparing R-SMOE inputs", unit="images")

    assert progress is not None

    for image_path in images:
        scene_dir = output_root / image_path.stem
        images_dir = scene_dir / "images"
        sparse_dir = scene_dir / "sparse"
        ensure_empty_directory(scene_dir, overwrite=overwrite)
        images_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)

        destination = images_dir / image_filename
        logging.debug("Copying %s -> %s", image_path, destination)
        copy_image(image_path, destination, convert=convert_to_png_flag)
        for alt_name in SECONDARY_IMAGE_FILENAMES:
            alt_path = images_dir / alt_name
            if alt_path.name != destination.name:
                shutil.copy2(destination, alt_path)

        width, height = get_image_size(destination)
        write_sparse_stub(sparse_dir, image_filename, width, height)
        prepared_dirs.append(scene_dir)
        progress.update()

    if manage_progress:
        progress.close()

    return prepared_dirs


def ensure_denoise_layout(
    base_model_dir: Path,
    iterations: Optional[int],
    expected_shifts: int = FUSION_EXPECTED_SHIFTS,
) -> None:
    iteration_folder = iterations if iterations is not None else 30_000
    for shift in range(expected_shifts):
        shift_dir = base_model_dir.parent / f"{base_model_dir.name}_shift{shift:02d}"
        renders_dir = shift_dir / "train" / f"ours_{iteration_folder}" / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)


def find_render(model_dir: Path) -> Path:
    search_patterns = [
        "train/ours_*/renders/00000.png",
        "train/ours_*/renders/00001.png",
    ]
    render_candidates: List[Path] = []
    for pattern in search_patterns:
        render_candidates.extend(sorted(model_dir.glob(pattern)))
    if not render_candidates:
        render_candidates.extend(sorted(model_dir.glob("train/ours_*/renders/*.png")))
    if not render_candidates:
        raise FileNotFoundError(
            f"Could not locate a render at {model_dir}/train/ours_*/renders/*.png. "
            "Verify the training finished successfully."
        )
    return render_candidates[-1]


def run_training(
    train_script: Path,
    dataset_dir: Path,
    model_dir: Path,
    iterations: Optional[int],
    npcs: Optional[int],
    file_name: str,
    extra_args: Optional[Sequence[str]],
    working_dir: Path,
    log_path: Path,
    gui_port: int | None,
    env: dict[str, str] | None = None,
) -> None:
    script_path = train_script.resolve()
    dataset_dir = dataset_dir.resolve()
    model_dir = model_dir.resolve()
    log_path = log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        sys.executable,
        str(script_path),
        "-s",
        str(dataset_dir),
        "-m",
        str(model_dir),
        "--file_name",
        file_name,
    ]

    if iterations is not None:
        cmd.extend(["--iterations", str(iterations)])
    if npcs is not None:
        cmd.extend(["--npcs", str(npcs)])
    if extra_args:
        cmd.extend(extra_args)
    if gui_port is not None:
        cmd.extend(["--port", str(gui_port)])

    logging.debug("Running: %s", " ".join(cmd))
    ensure_dir(working_dir / "all_result")
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            cmd,
            cwd=str(working_dir),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if result.returncode != 0:
        tail = []
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = lines[-20:]
        except Exception:  # pragma: no cover - best effort
            tail = ["<unable to read log file>"]
        raise RuntimeError(
            "R-SMOE training failed with exit code "
            f"{result.returncode}. See {log_path} for details.\n"
            + "\n".join(tail)
        )


@register_defense("r-smoe")
class RSMoEDefense(Defense):
    """Apply the R-SMOE reconstruction pipeline as a defense stage."""

    def __init__(self, config: DefenseConfig) -> None:
        super().__init__(config)
        self._settings_reported = False
        self._prep_progress: Progress | None = None
        self._recon_progress: Progress | None = None
        self._variant_images: dict[str, list[Path]] = {}
        self._params_cache: dict[str, object] | None = None
        self._config_identifier = self._build_config_identifier()

    def _build_config_identifier(self) -> str:
        return build_config_identifier(self.config, default_prefix=self.config.type or "r-smoe")

    def _get_params(self) -> dict[str, object]:
        if self._params_cache is None:
            params = self.config.params

            default_root = Path(__file__).resolve().parents[3] / "external" / "r-smoe"
            r_smoe_root = Path(params.get("root", default_root))
            if not r_smoe_root.exists():
                raise FileNotFoundError(
                    f"R-SMOE root directory not found at {r_smoe_root}. "
                    "Provide 'root' in the defense configuration."
                )

            mode = str(params.get("mode", "standard"))
            if mode not in {"standard", "denoise"}:
                raise ValueError("R-SMOE mode must be 'standard' or 'denoise'.")

            iterations = params.get("iterations")
            npcs = params.get("npcs")
            file_name_prefix = params.get("file_name_prefix")
            skip_existing = bool(params.get("skip_existing", True))
            image_filename = params.get("image_filename", DEFAULT_IMAGE_FILENAME)

            extra_args = params.get("extra_args")
            extra_args_list: List[str] = list(extra_args) if extra_args else []
            user_override_port = False
            for arg in extra_args_list:
                if arg == "--port" or arg.startswith("--port="):
                    user_override_port = True
                    break

            n_multi_model: Optional[int]
            if mode == "denoise":
                n_multi_model = int(params.get("n_multi_model", 8))
                extra_args_list.extend(["--n_multi_model", str(n_multi_model)])
            else:
                n_multi_model = None

            extra_args_seq: Optional[Sequence[str]] = extra_args_list if extra_args_list else None

            gui_port_param = params.get("gui_port")
            gui_port: int | None = None
            if gui_port_param is not None:
                gui_port = int(gui_port_param)
                if gui_port < 0 or gui_port > 65535:
                    raise ValueError("gui_port must be between 0 and 65535.")

            raw_device_ids = params.get("device_ids")
            device_ids: list[str] | None = None
            if raw_device_ids is not None:
                if not isinstance(raw_device_ids, Sequence) or isinstance(raw_device_ids, (str, bytes)):
                    raise ValueError("device_ids must be a sequence of GPU identifiers.")
                device_ids = [str(device_id) for device_id in raw_device_ids]
                if not device_ids:
                    raise ValueError("device_ids cannot be empty.")

            max_jobs_param = params.get("max_concurrent_jobs")
            if max_jobs_param is None:
                max_concurrent_jobs = len(device_ids) if device_ids else 1
            else:
                max_concurrent_jobs = int(max_jobs_param)
                if max_concurrent_jobs < 1:
                    raise ValueError("max_concurrent_jobs must be at least 1.")
            if device_ids:
                max_concurrent_jobs = min(max_concurrent_jobs, len(device_ids)) or 1
            if gui_port is None and not user_override_port:
                gui_port = 0 if max_concurrent_jobs > 1 else None

            if params.get("train_script"):
                train_script = Path(params["train_script"])
            else:
                default_name = "train_noisy_final.py" if mode == "denoise" else "train_render_metrics_final.py"
                train_script = r_smoe_root / default_name
            if not train_script.exists():
                raise FileNotFoundError(f"Training script not found: {train_script}")

            self._params_cache = {
                "root": r_smoe_root,
                "mode": mode,
                "iterations": iterations,
                "npcs": npcs,
                "file_name_prefix": file_name_prefix,
                "skip_existing": skip_existing,
                "extra_args_list": extra_args_list,
                "extra_args_seq": extra_args_seq,
                "train_script": train_script,
                "n_multi_model": n_multi_model,
                "image_filename": image_filename,
                "max_concurrent_jobs": max_concurrent_jobs,
                "device_ids": device_ids,
                "gui_port": gui_port,
            }
        return self._params_cache

    def initialize(self, context: RunContext, variants: list[DatasetVariant]) -> None:
        params = self._get_params()
        r_smoe_root = params["root"]  # type: ignore[index]
        mode = params["mode"]  # type: ignore[index]
        iterations = params["iterations"]
        npcs = params["npcs"]
        skip_existing = params["skip_existing"]  # type: ignore[index]
        extra_args_list = params["extra_args_list"]  # type: ignore[index]
        train_script = params["train_script"]  # type: ignore[index]
        n_multi_model = params["n_multi_model"]
        file_name_prefix_param = params["file_name_prefix"]
        max_concurrent_jobs = params["max_concurrent_jobs"]  # type: ignore[index]
        device_ids = params["device_ids"]
        gui_port = params["gui_port"]

        self._variant_images = {}
        total_images = 0
        for variant in variants:
            images = find_images(Path(variant.data_dir))
            if not images:
                raise FileNotFoundError(f"No supported images found in {variant.data_dir}.")
            self._variant_images[variant.name] = images
            total_images += len(images)

        if not self._settings_reported:
            extra_args_display = " ".join(extra_args_list) if extra_args_list else "<none>"
            n_multi_model_display = n_multi_model if n_multi_model is not None else "<unused>"
            file_prefix_display = file_name_prefix_param if file_name_prefix_param is not None else "<variant>"
            device_ids_display = ", ".join(device_ids) if device_ids else "<all-visible>"
            gui_port_display = gui_port if gui_port is not None else "<default>"
            print(
                "[info] R-SMOE settings: "
                f"config={self.config.name or self._config_identifier} "
                f"root={r_smoe_root} mode={mode} iterations={iterations} npcs={npcs} "
                f"n_multi_model={n_multi_model_display} skip_existing={skip_existing} "
                f"file_prefix={file_prefix_display} extra_args={extra_args_display} "
                f"train_script={train_script} parallel_jobs={max_concurrent_jobs} "
                f"device_ids={device_ids_display} gui_port={gui_port_display}"
            )
            self._settings_reported = True

        if self._prep_progress is not None:
            self._prep_progress.close()
        if self._recon_progress is not None:
            self._recon_progress.close()

        self._prep_progress = Progress(total=total_images, description="Preparing R-SMOE inputs", unit="images")
        self._recon_progress = Progress(total=total_images, description="R-SMOE reconstruction", unit="images")

    def run(self, context: RunContext, variant: DatasetVariant) -> DatasetVariant:
        _ensure_dependencies()
        params = self._get_params()

        r_smoe_root = params["root"]  # type: ignore[index]
        mode = params["mode"]  # type: ignore[index]
        iterations = params["iterations"]
        npcs = params["npcs"]
        skip_existing = params["skip_existing"]  # type: ignore[index]
        extra_args_seq = params["extra_args_seq"]  # type: ignore[index]
        train_script = params["train_script"]  # type: ignore[index]
        n_multi_model = params["n_multi_model"]
        image_filename = params["image_filename"]  # type: ignore[index]
        file_name_prefix_param = params["file_name_prefix"]
        max_concurrent_jobs = params["max_concurrent_jobs"]  # type: ignore[index]
        device_ids = params["device_ids"]
        gui_port = params["gui_port"]

        file_name_prefix = file_name_prefix_param if file_name_prefix_param is not None else variant.name

        variant_root = ensure_dir(
            context.artifacts_dir / "defenses" / "r-smoe" / variant.name / self._config_identifier
        )
        prepared_root = ensure_dir(variant_root / "prepared")
        models_root = ensure_dir(variant_root / "models")
        recon_root = ensure_dir(variant_root / "reconstructed")

        images = self._variant_images.get(variant.name)
        if images is None:
            images = find_images(Path(variant.data_dir))

        if self._prep_progress is None:
            self._prep_progress = Progress(total=len(images), description="Preparing R-SMOE inputs", unit="images")

        prepared_datasets = prepare_dataset(
            images=images,
            output_root=prepared_root,
            image_filename=image_filename,
            overwrite=True,
            convert_to_png_flag=True,
            progress=self._prep_progress,
        )

        logging.info("Found %d dataset(s) for R-SMOE defense in %s", len(prepared_datasets), prepared_root)

        progress = self._recon_progress
        if progress is None:
            progress = Progress(total=len(prepared_datasets), description="R-SMOE reconstruction", unit="images")
            self._recon_progress = progress
        results: List[Path] = []

        rsmoe_data_root = ensure_dir(r_smoe_root / "data")
        worker_count = max(1, max_concurrent_jobs)
        device_queue: SimpleQueue[str] | None = None
        if device_ids:
            worker_count = min(worker_count, len(device_ids)) or 1
            device_queue = SimpleQueue()
            for device_id in device_ids:
                device_queue.put(device_id)
        worker_count = min(worker_count, len(prepared_datasets)) or 1

        duration_lock = Lock()
        total_duration = 0.0
        completed_images = 0

        def record_completion(duration: float) -> float | None:
            nonlocal total_duration, completed_images
            with duration_lock:
                completed_images += 1
                total_duration += max(0.0, duration)
                count = completed_images
                total = total_duration
            remaining = max(0, len(prepared_datasets) - count)
            if count == 0:
                return None
            if remaining == 0:
                return 0.0
            if total <= 0.0:
                return None
            avg_duration = total / count
            batches = math.ceil(remaining / worker_count)
            return avg_duration * batches

        def process_dataset(dataset_dir: Path) -> Path:
            image_name = dataset_dir.name
            dest_image = recon_root / f"{image_name}.png"
            model_dir = models_root / image_name
            log_path = model_dir / "rsmoe.log"

            if skip_existing and dest_image.exists():
                logging.debug("Skipping %s (reconstruction already exists).", image_name)
                eta_override = record_completion(0.0)
                if progress is not None:
                    if worker_count > 1 and eta_override is not None:
                        progress.set_eta_override(eta_override)
                    progress.update()
                return dest_image

            ensure_dir(model_dir)
            file_name = f"{file_name_prefix}_{image_name}"

            if mode == "denoise":
                ensure_denoise_layout(model_dir, iterations)

            symlink_dir = rsmoe_data_root / image_name
            dataset_dir_abs = dataset_dir.resolve()
            if symlink_dir.exists() or symlink_dir.is_symlink():
                if symlink_dir.is_symlink():
                    symlink_dir.unlink()
                else:
                    shutil.rmtree(symlink_dir)
            try:
                symlink_dir.symlink_to(dataset_dir_abs, target_is_directory=True)
            except OSError:
                shutil.copytree(dataset_dir_abs, symlink_dir)

            assigned_device: str | None = None
            env: dict[str, str] | None = None
            if device_queue is not None:
                assigned_device = device_queue.get()
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = assigned_device

            job_start = time.monotonic()
            try:
                logging.debug("R-SMOE processing %s", image_name)
                logging.info("R-SMOE processing %s (log: %s)", image_name, log_path)
                run_training(
                    train_script=train_script,
                    dataset_dir=dataset_dir_abs,
                    model_dir=model_dir,
                    iterations=iterations,
                    npcs=npcs,
                    file_name=file_name,
                    extra_args=extra_args_seq,
                    working_dir=r_smoe_root,
                    log_path=log_path,
                    gui_port=gui_port,
                    env=env,
                )
            finally:
                if assigned_device is not None and device_queue is not None:
                    device_queue.put(assigned_device)

            render_path = find_render(model_dir)
            logging.debug("Copying %s -> %s", render_path, dest_image)
            shutil.copy2(render_path, dest_image)
            eta_override = record_completion(time.monotonic() - job_start)
            if progress is not None:
                if worker_count > 1 and eta_override is not None:
                    progress.set_eta_override(eta_override)
                progress.update()
            return dest_image

        if worker_count == 1:
            for dataset_dir in prepared_datasets:
                dest_image = process_dataset(dataset_dir)
                results.append(dest_image)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {executor.submit(process_dataset, dataset_dir): dataset_dir for dataset_dir in prepared_datasets}
                for future in as_completed(futures):
                    dest_image = future.result()
                    results.append(dest_image)

        metadata = {
            "defense": "r-smoe",
            "mode": mode,
            "iterations": iterations,
            "npcs": npcs,
            "train_script": str(train_script),
            "models_root": models_root.as_posix(),
            "recon_root": recon_root.as_posix(),
            "source_variant": variant.name,
            "image_hw": variant.metadata.get("image_hw"),
            "config_name": self.config.name,
            "config_identifier": self._config_identifier,
            "file_name_prefix": file_name_prefix,
            "max_concurrent_jobs": max_concurrent_jobs,
        }
        if gui_port is not None:
            metadata["gui_port"] = gui_port
        if device_ids:
            metadata["device_ids"] = list(device_ids)
        if n_multi_model is not None:
            metadata["n_multi_model"] = n_multi_model

        return DatasetVariant(
            name=f"{variant.name}-r-smoe-{self._config_identifier}",
            data_dir=str(recon_root),
            parent=variant.name,
            metadata=metadata,
        )

    def finalize(self) -> None:
        if self._prep_progress is not None:
            self._prep_progress.close()
            self._prep_progress = None
        if self._recon_progress is not None:
            self._recon_progress.close()
            self._recon_progress = None


__all__ = ["RSMoEDefense"]

def _ensure_dependencies() -> None:
    missing: List[str] = []
    dependency_hints = {
        "skimage": "Install scikit-image: `pip install scikit-image imageio`.",
        "simple_knn": "Build simple-knn extension: `pip install -e ./external/r-smoe/submodules-2d-smoe/simple-knn`.",
        "diff_gaussian_rasterization": "Build diff-gaussian-rasterization extension: `pip install -e ./external/r-smoe/submodules-2d-smoe/diff-gaussian-rasterization`.",
    }

    for module_name, hint in dependency_hints.items():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(hint)

    if missing:
        message = "\n - ".join(
            [
                "Missing dependencies for R-SMOE defense:",
                *missing,
                "Run `advdef setup r-smoe` to install them automatically.",
            ]
        )
        raise RuntimeError(message)
