"""Unit tests verifying individual defense image transformations."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from advdef.config import DefenseConfig
from advdef.core.pipeline import DatasetVariant
from advdef.defenses.bit_depth import BitDepthDefense
from advdef.defenses.crop_resize import CropResizeDefense
from advdef.defenses.flip import FlipDefense
from advdef.defenses.grayscale import GrayscaleDefense
from advdef.defenses.jpeg import JPEGDefense
from advdef.defenses.low_pass import LowPassDefense
from advdef.defenses.tvm import TVMDefense
from advdef.utils import ensure_dir


class DummyContext(SimpleNamespace):
    """Lightweight context providing just artifacts_dir/options."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.artifacts_dir = ensure_dir(root / "artifacts")
        self.options = {}


def _execute_defense(
    tmp_path: Path,
    defense_cls,
    type_name: str,
    params: dict,
    image_array: np.ndarray,
    *,
    mode: str = "RGB",
    image_name: str = "sample.png",
    pre_existing: bool = False,
) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    Image.fromarray(image_array.astype(np.uint8), mode=mode).save(input_dir / image_name)

    context = DummyContext(tmp_path)
    variant = DatasetVariant(name="variant", data_dir=str(input_dir))
    config = DefenseConfig(type=type_name, name=f"{type_name}_test", params=dict(params))

    defense = defense_cls(config)
    defense.initialize(context, [variant])
    result_variant = defense.run(context, variant)
    defense.finalize()

    output_path = Path(result_variant.data_dir) / image_name
    if not output_path.exists():
        stem = Path(image_name).stem
        matches = list(Path(result_variant.data_dir).glob(f"{stem}.*"))
        if matches:
            output_path = matches[0]
    if not pre_existing:
        assert output_path.exists(), f"Expected output file {output_path}."
    return output_path


def test_crop_resize_crops_center(tmp_path):
    # Construct a 4x4 RGB image with distinct quadrant colours.
    arr = np.array(
        [
            [[255, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 0]],
            [[255, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 0]],
            [[0, 0, 255], [0, 0, 255], [255, 255, 0], [255, 255, 0]],
            [[0, 0, 255], [0, 0, 255], [255, 255, 0], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )

    output_path = _execute_defense(
        tmp_path,
        CropResizeDefense,
        "crop-resize",
        {"crop_size": [2, 2]},
        arr,
    )

    result = np.array(Image.open(output_path))
    expected = arr[1:3, 1:3, :]

    assert result.shape == (2, 2, 3)
    np.testing.assert_array_equal(result, expected)


def test_crop_resize_random_multicrop(tmp_path):
    base = np.arange(25, dtype=np.uint8).reshape(5, 5)
    arr = np.stack([base, base, base], axis=-1)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    image_name = "sample.png"
    Image.fromarray(arr).save(input_dir / image_name)

    context = DummyContext(tmp_path)
    variant = DatasetVariant(name="variant", data_dir=str(input_dir))
    params = {
        "crop_size": [2, 2],
        "resize_size": [2, 2],
        "crop_mode": "random",
        "num_crops": 3,
        "seed": 123,
        "crop_suffix": "__crop",
        "interpolation": "nearest",
    }
    config = DefenseConfig(type="crop-resize", name="cropping_random", params=params)
    defense = CropResizeDefense(config)
    defense.initialize(context, [variant])
    result_variant = defense.run(context, variant)
    defense.finalize()

    output_dir = Path(result_variant.data_dir)
    files = sorted(path for path in output_dir.iterdir() if path.suffix == ".png")
    assert [path.name for path in files] == ["sample__crop000.png", "sample__crop001.png", "sample__crop002.png"]

    master_seed = np.random.SeedSequence(123)
    variant_seed = master_seed.spawn(1)[0]
    image_seed = variant_seed.spawn(1)[0]
    rng = np.random.default_rng(image_seed)

    expected_crops = []
    for _ in range(3):
        left = int(rng.integers(0, 5 - 2 + 1))
        top = int(rng.integers(0, 5 - 2 + 1))
        expected_crops.append(arr[top : top + 2, left : left + 2])

    for path, expected in zip(files, expected_crops):
        observed = np.array(Image.open(path))
        np.testing.assert_array_equal(observed, expected)

    aggregation_meta = result_variant.metadata.get("aggregation")
    assert aggregation_meta is not None
    assert aggregation_meta["method"] == "mean"
    assert aggregation_meta["num_inputs_per_example"] == 3


def test_flip_horizontal(tmp_path):
    arr = np.array(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255]]],
        dtype=np.uint8,
    )

    output_path = _execute_defense(
        tmp_path,
        FlipDefense,
        "flip",
        {"direction": "horizontal"},
        arr,
    )

    result = np.array(Image.open(output_path))
    expected = arr[:, ::-1, :]

    np.testing.assert_array_equal(result, expected)


def test_flip_vertical(tmp_path):
    arr = np.array(
        [
            [[255, 0, 0]],
            [[0, 255, 0]],
            [[0, 0, 255]],
        ],
        dtype=np.uint8,
    )

    output_path = _execute_defense(
        tmp_path,
        FlipDefense,
        "flip",
        {"direction": "vertical"},
        arr,
    )

    result = np.array(Image.open(output_path))
    expected = arr[::-1, :, :]

    np.testing.assert_array_equal(result, expected)


def test_bit_depth_quantization(tmp_path):
    arr = np.array(
        [[[0, 0, 0], [60, 120, 180], [128, 200, 255], [240, 250, 255]]],
        dtype=np.uint8,
    )

    output_path = _execute_defense(
        tmp_path,
        BitDepthDefense,
        "bit-depth",
        {"bits": 2, "dither": False},
        arr,
    )

    result = np.array(Image.open(output_path))
    unique_values = np.unique(result)
    assert set(unique_values.tolist()) <= {0, 85, 170, 255}


def test_bit_depth_dither(tmp_path):
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)

    output_path = _execute_defense(
        tmp_path,
        BitDepthDefense,
        "bit-depth",
        {"bits": 3, "dither": True},
        arr,
    )

    result = np.array(Image.open(output_path))
    assert result.shape == arr.shape


def test_low_pass_preserves_colour_channels(tmp_path):
    # Half red, half green image to test channel separation.
    arr = np.zeros((5, 5, 3), dtype=np.uint8)
    arr[:, :3, 0] = 255  # Red half
    arr[:, 3:, 1] = 255  # Green half

    output_path = _execute_defense(
        tmp_path,
        LowPassDefense,
        "low-pass",
        {"filter": "gaussian", "sigma": 0.8},
        arr,
    )

    result = np.array(Image.open(output_path))

    # Ensure we still have distinct channel information after filtering.
    channel_diff = np.abs(result[..., 0].astype(int) - result[..., 1].astype(int))
    assert channel_diff.max() > 10, "Low-pass filtering collapsed colour channels unexpectedly."


def test_low_pass_box_filter(tmp_path):
    arr = np.zeros((5, 5), dtype=np.uint8)
    arr[:, :3] = 255

    output_path = _execute_defense(
        tmp_path,
        LowPassDefense,
        "low-pass",
        {"filter": "box", "kernel_size": 3},
        arr,
        mode="L",
    )

    result = np.array(Image.open(output_path))
    assert result.std() < arr.std()


def test_tvm_reduces_noise(tmp_path):
    rng = np.random.default_rng(0)
    base = np.full((16, 16, 3), 128, dtype=np.uint8)
    noise = rng.integers(-40, 40, size=base.shape, dtype=np.int16)
    arr = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    output_path = _execute_defense(
        tmp_path,
        TVMDefense,
        "tvm",
        {"weight": 0.1, "eps": 1e-4, "n_iter_max": 50, "keep_probability": 0.0},
        arr,
    )

    result = np.array(Image.open(output_path))

    original_std = arr.astype(np.float32).std()
    denoised_std = result.astype(np.float32).std()
    assert denoised_std < original_std, "TVM defense did not reduce overall variation."


def test_tvm_respects_pixel_mask(tmp_path):
    rng = np.random.default_rng(1)
    base = np.full((12, 12, 3), 128, dtype=np.uint8)
    noise = rng.integers(-60, 60, size=base.shape, dtype=np.int16)
    arr = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    params = {
        "weight": 0.15,
        "n_iter_max": 40,
        "keep_probability": 0.3,
        "projection_steps": 4,
        "seed": 1234,
    }

    output_path = _execute_defense(
        tmp_path,
        TVMDefense,
        "tvm",
        params,
        arr,
    )

    result = np.array(Image.open(output_path))

    master_seed = np.random.SeedSequence(params["seed"])
    variant_seed = master_seed.spawn(1)[0]
    image_seed = variant_seed.spawn(1)[0]
    mask_rng = np.random.default_rng(image_seed)
    mask = mask_rng.random(arr.shape[:2]) < params["keep_probability"]
    if not mask.any():
        y = mask_rng.integers(arr.shape[0])
        x = mask_rng.integers(arr.shape[1])
        mask[y, x] = True

    np.testing.assert_array_equal(result[mask], arr[mask])
    if (~mask).any():
        assert np.any(result[~mask] != arr[~mask])
    else:
        assert np.any(result != arr)


def test_grayscale_replication(tmp_path):
    arr = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )

    output_path = _execute_defense(
        tmp_path,
        GrayscaleDefense,
        "grayscale",
        {"replicate_rgb": True},
        arr,
    )

    result = np.array(Image.open(output_path))
    assert result.shape[-1] == 3
    assert np.all(result[..., 0] == result[..., 1])
    assert np.all(result[..., 1] == result[..., 2])


def test_grayscale_single_channel(tmp_path):
    arr = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
        ],
        dtype=np.uint8,
    )

    output_path = _execute_defense(
        tmp_path,
        GrayscaleDefense,
        "grayscale",
        {"replicate_rgb": False},
        arr,
    )

    result = Image.open(output_path)
    assert result.mode == "L"


def test_jpeg_recompression(tmp_path):
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[..., 0] = 255

    output_path = _execute_defense(
        tmp_path,
        JPEGDefense,
        "jpeg",
        {"quality": 50},
        arr,
    )

    result = Image.open(output_path)
    assert result.format == "JPEG"
    assert result.mode == "RGB"


def test_crop_resize_warns_and_upsamples(tmp_path):
    arr = np.tile(np.arange(16, dtype=np.uint8), (16, 1))

    output_path = _execute_defense(
        tmp_path,
        CropResizeDefense,
        "crop-resize",
        {"crop_size": [8, 8], "resize_size": [16, 16]},
        arr,
        mode="L",
    )

    result = np.array(Image.open(output_path))
    assert result.shape == (16, 16)
