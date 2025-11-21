#!/usr/bin/env python3
"""Convert SMoE-AE TensorFlow checkpoints to a PyTorch state_dict.

Usage:
    python tools/convert_smoe_ae_tf_to_torch.py \
        --checkpoint ../SMoE-AE/weights/8x8\ -\ steered/cp.ckpt/cp.ckpt \
        --output smoe_ae_8x8.pt \
        --block-size 8

This script mirrors the demo architectures:
- block_size=8 -> predicts covariance (28 outputs, conv stack up to 1024 filters)
- block_size=16 -> radial variant without covariance prediction (12 outputs)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:  # pragma: no cover - depends on user environment
    raise SystemExit("TensorFlow is required to run this conversion script.") from exc

from advdef.defenses.smoe_ae import SmoeAE


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


def _map_weights(tf_model: tf.keras.Model, torch_model: nn.Module) -> dict:
    torch_state = torch_model.state_dict()
    torch_keys: List[str] = [k for k in torch_state.keys() if k.startswith("encoder")]
    tf_weights = tf_model.trainable_variables

    if len(torch_keys) != len(tf_weights):
        raise RuntimeError(
            f"Mismatch in variable count: torch={len(torch_keys)}, tensorflow={len(tf_weights)}"
        )

    for key, tf_var in zip(torch_keys, tf_weights):
        array = tf_var.numpy()
        if "weight" in key and array.ndim == 4:
            array = array.transpose(3, 2, 0, 1)
        elif "weight" in key and array.ndim == 2:
            array = array.T
        torch_state[key] = torch.tensor(array)

    return torch_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SMoE-AE TensorFlow checkpoint to PyTorch.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to TensorFlow checkpoint (cp.ckpt).")
    parser.add_argument("--output", type=Path, required=True, help="Destination .pt path.")
    parser.add_argument("--block-size", type=int, default=8, choices=[8, 16], help="Block size of the model.")
    parser.add_argument("--kernel-num", type=int, default=4, help="Number of SMoE kernels.")
    parser.add_argument(
        "--predict-covariance",
        action="store_true",
        help="Enable covariance prediction (use for the 8x8 steered variant).",
    )
    args = parser.parse_args()

    predict_covariance = args.predict_covariance or args.block_size == 8

    tf_model = _build_tf_model(args.block_size, args.kernel_num, predict_covariance)
    # Older checkpoints are TF Checkpoints (.index/.data). Keras 3 dropped direct support
    # for the TF format in load_weights, so restore via tf.train.Checkpoint.
    ckpt = tf.train.Checkpoint(model=tf_model)
    ckpt.restore(str(args.checkpoint)).expect_partial()

    torch_model = SmoeAE(
        block_size=args.block_size,
        kernel_num=args.kernel_num,
        predict_covariance=predict_covariance,
    )
    mapped_state = _map_weights(tf_model, torch_model)
    torch.save(mapped_state, args.output)

    print(f"[ok] Saved PyTorch weights to {args.output}")


if __name__ == "__main__":
    main()
