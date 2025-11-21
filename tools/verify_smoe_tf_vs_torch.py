#!/usr/bin/env python3
"""Compare TensorFlow SMoE-AE outputs against the PyTorch port on blocked data.

Example (8x8 steered):
    python tools/verify_smoe_tf_vs_torch.py \
        --checkpoint "../SMoE-AE/weights/8x8 - steered/cp.ckpt/cp.ckpt" \
        --torch-weights external/smoe-ae/smoe_ae_8x8.pt \
        --blocks "../SMoE-AE/images/blocked/8x8/lena.pckl" \
        --block-size 8
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:  # pragma: no cover - requires TF
    raise SystemExit("TensorFlow is required to run this checker.") from exc

from advdef.defenses.smoe_ae import SmoeAE, SmoeDecoder


def _build_tf_model(block_size: int, kernel_num: int, predict_covariance: bool) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(block_size, block_size, 1))

    if block_size == 8:
        conv_channels = [16, 32, 64, 128, 256, 512, 1024]
        dense_layers = [1024, 512, 256, 128, 64]
    else:
        conv_channels = [16, 32, 64, 128, 256, 512]
        dense_layers = [512, 256, 128, 64]

    x = inputs
    for filters in conv_channels:
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    for units in dense_layers:
        x = tf.keras.layers.Dense(units, activation="relu")(x)

    out_features = kernel_num * 3
    if predict_covariance:
        out_features += kernel_num * 4
    outputs = tf.keras.layers.Dense(out_features, activation="linear")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def _psnr(mse: float) -> float:
    if mse <= 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def _load_blocks(path: Path, limit: int | None) -> np.ndarray:
    data = pickle.load(open(path, "rb"))
    blocks = np.array(data["block"], dtype=np.float32)
    if blocks.max() > 1.5:
        blocks = blocks / 255.0
    if limit is not None:
        blocks = blocks[:limit]
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify TF vs. torch SMoE-AE outputs on blocked inputs.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to TF checkpoint (cp.ckpt).")
    parser.add_argument("--torch-weights", type=Path, required=True, help="Path to torch .pt weights.")
    parser.add_argument("--blocks", type=Path, required=True, help="Pickled blocked data (with 'block' entry).")
    parser.add_argument("--block-size", type=int, default=8, choices=[8, 16], help="Block size.")
    parser.add_argument("--kernel-num", type=int, default=4, help="Number of SMoE kernels.")
    parser.add_argument(
        "--predict-covariance",
        action="store_true",
        help="Enable covariance prediction (use for 8x8 steered variants).",
    )
    parser.add_argument("--limit", type=int, default=64, help="Limit number of blocks for the check.")
    args = parser.parse_args()

    predict_covariance = args.predict_covariance or args.block_size == 8
    blocks = _load_blocks(args.blocks, args.limit)
    if blocks.shape[1:] != (args.block_size, args.block_size):
        raise SystemExit(f"Block data shape {blocks.shape} does not match block_size={args.block_size}.")

    tf_model = _build_tf_model(args.block_size, args.kernel_num, predict_covariance)
    tf_ckpt = tf.train.Checkpoint(model=tf_model)
    tf_ckpt.restore(str(args.checkpoint)).expect_partial()

    torch_state = torch.load(args.torch_weights, map_location="cpu")
    if isinstance(torch_state, dict) and "state_dict" in torch_state:
        torch_state = torch_state["state_dict"]
    torch_model = SmoeAE(
        block_size=args.block_size,
        kernel_num=args.kernel_num,
        predict_covariance=predict_covariance,
    )
    torch_model.load_state_dict(torch_state)
    torch_model.eval()

    tf_inp = tf.convert_to_tensor(blocks[..., None])  # NHWC
    tf_params = tf_model(tf_inp).numpy()

    torch_blocks = torch.from_numpy(blocks).unsqueeze(1)  # NCHW
    with torch.no_grad():
        torch_params = torch_model.encoder(torch_blocks).cpu().numpy()

    decoder = SmoeDecoder(
        block_size=args.block_size,
        kernel_num=args.kernel_num,
        predict_covariance=predict_covariance,
    )
    with torch.no_grad():
        tf_recon = decoder(torch.from_numpy(tf_params).float()).cpu().numpy()
        torch_recon = decoder(torch.from_numpy(torch_params).float()).cpu().numpy()

    param_mse = float(np.mean((torch_params - tf_params) ** 2))
    recon_mse = float(np.mean((torch_recon - tf_recon) ** 2))
    input_mse = float(np.mean((torch_recon - blocks) ** 2))

    print(f"Params: mse={param_mse:.6e}, psnr={_psnr(param_mse):.2f} dB")
    print(f"Recons: mse={recon_mse:.6e}, psnr={_psnr(recon_mse):.2f} dB")
    print(f"Recons vs input: mse={input_mse:.6e}, psnr={_psnr(input_mse):.2f} dB")
    print(f"TF recon min/max/mean: {tf_recon.min():.4f}/{tf_recon.max():.4f}/{tf_recon.mean():.4f}")
    print(f"Torch recon min/max/mean: {torch_recon.min():.4f}/{torch_recon.max():.4f}/{torch_recon.mean():.4f}")


if __name__ == "__main__":
    main()
