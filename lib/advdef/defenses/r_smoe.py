"""R-SMOE defense integration."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from advdef.config import DefenseConfig
from advdef.core.context import RunContext
from advdef.core.pipeline import DatasetVariant, Defense
from advdef.core.registry import register_defense
from advdef.utils import ensure_dir

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
) -> List[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    prepared_dirs: List[Path] = []

    for image_path in images:
        scene_dir = output_root / image_path.stem
        images_dir = scene_dir / "images"
        sparse_dir = scene_dir / "sparse"
        ensure_empty_directory(scene_dir, overwrite=overwrite)
        images_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)

        destination = images_dir / image_filename
        logging.info("Copying %s -> %s", image_path, destination)
        copy_image(image_path, destination, convert=convert_to_png_flag)
        for alt_name in SECONDARY_IMAGE_FILENAMES:
            alt_path = images_dir / alt_name
            if alt_path.name != destination.name:
                shutil.copy2(destination, alt_path)

        width, height = get_image_size(destination)
        write_sparse_stub(sparse_dir, image_filename, width, height)
        prepared_dirs.append(scene_dir)

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
) -> None:
    cmd: List[str] = [
        sys.executable,
        str(train_script),
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

    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


@register_defense("r-smoe")
class RSMoEDefense(Defense):
    """Apply the R-SMOE reconstruction pipeline as a defense stage."""

    def __init__(self, config: DefenseConfig) -> None:
        super().__init__(config)

    def run(self, context: RunContext, variant: DatasetVariant) -> DatasetVariant:
        params = self.config.params

        default_root = Path(__file__).resolve().parents[3] / "external" / "r-smoe"
        r_smoe_root = Path(params.get("root", default_root))
        if not r_smoe_root.exists():
            raise FileNotFoundError(
                f"R-SMOE root directory not found at {r_smoe_root}. "
                "Provide 'root' in the defense configuration."
            )

        mode = str(params.get("mode", "standard"))
        iterations = params.get("iterations")
        npcs = params.get("npcs")
        file_name_prefix = params.get("file_name_prefix", variant.name)
        n_multi_model = int(params.get("n_multi_model", 8))
        skip_existing = bool(params.get("skip_existing", True))
        extra_args = params.get("extra_args")
        extra_args_list: List[str] = list(extra_args) if extra_args else []

        if mode not in {"standard", "denoise"}:
            raise ValueError("R-SMOE mode must be 'standard' or 'denoise'.")

        if mode == "denoise":
            extra_args_list.extend(["--n_multi_model", str(n_multi_model)])

        extra_args_seq: Optional[Sequence[str]] = extra_args_list if extra_args_list else None

        if params.get("train_script"):
            train_script = Path(params["train_script"])
        else:
            default_name = "train_noisy_final.py" if mode == "denoise" else "train_render_metrics_final.py"
            train_script = r_smoe_root / default_name
        if not train_script.exists():
            raise FileNotFoundError(f"Training script not found: {train_script}")

        variant_root = ensure_dir(context.artifacts_dir / "defenses" / "r-smoe" / variant.name)
        prepared_root = ensure_dir(variant_root / "prepared")
        models_root = ensure_dir(variant_root / "models")
        recon_root = ensure_dir(variant_root / "reconstructed")

        images = find_images(Path(variant.data_dir))
        prepared_datasets = prepare_dataset(
            images=images,
            output_root=prepared_root,
            image_filename=params.get("image_filename", DEFAULT_IMAGE_FILENAME),
            overwrite=True,
            convert_to_png_flag=True,
        )

        logging.info("Found %d dataset(s) for R-SMOE defense in %s", len(prepared_datasets), prepared_root)

        results: List[Path] = []

        for dataset_dir in prepared_datasets:
            image_name = dataset_dir.name
            dest_image = recon_root / f"{image_name}.png"
            if skip_existing and dest_image.exists():
                logging.info("Skipping %s (reconstruction already exists).", image_name)
                results.append(dest_image)
                continue

            model_dir = models_root / image_name
            ensure_dir(model_dir)
            file_name = f"{file_name_prefix}_{image_name}"

            if mode == "denoise":
                ensure_denoise_layout(model_dir, iterations)

            logging.info("R-SMOE processing %s", image_name)
            run_training(
                train_script=train_script,
                dataset_dir=dataset_dir,
                model_dir=model_dir,
                iterations=iterations,
                npcs=npcs,
                file_name=file_name,
                extra_args=extra_args_seq,
            )

            render_path = find_render(model_dir)
            logging.info("Copying %s -> %s", render_path, dest_image)
            shutil.copy2(render_path, dest_image)
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
        }

        return DatasetVariant(
            name=f"{variant.name}-r-smoe",
            data_dir=str(recon_root),
            parent=variant.name,
            metadata=metadata,
        )


__all__ = ["RSMoEDefense"]
