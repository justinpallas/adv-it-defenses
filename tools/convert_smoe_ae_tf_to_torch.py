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
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:  # pragma: no cover - depends on user environment
    raise SystemExit("TensorFlow is required to run this conversion script.") from exc

from advdef.defenses.smoe_ae import SmoeAE


def _load_raw_checkpoint(path: Path) -> Dict[str, tf.Tensor]:
    """Load raw tensors from a TF checkpoint (no Keras restore)."""
    reader = tf.train.load_checkpoint(str(path))
    tensors: Dict[str, tf.Tensor] = {}
    for name in sorted(reader.get_variable_to_shape_map().keys()):
        tensors[name] = reader.get_tensor(name)
    return tensors


def _assign_weights_from_raw(tensors: Dict[str, tf.Tensor], torch_model: nn.Module) -> None:
    """Map raw checkpoint conv/dense tensors into the torch encoder."""
    from advdef.defenses.smoe_ae import _ConvBlock  # local import to avoid circular issues

    conv_modules: List[nn.Conv2d] = []
    for module in torch_model.encoder.features:  # type: ignore[attr-defined]
        if isinstance(module, nn.Conv2d):
            conv_modules.append(module)
        elif isinstance(module, _ConvBlock):
            conv_modules.append(module.conv)

    linear_modules: List[nn.Linear] = [m for m in torch_model.encoder.head if isinstance(m, nn.Linear)]  # type: ignore[attr-defined]

    conv_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    dense_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    # Collect layer_with_weights-{idx}/{kernel|bias}/.ATTRIBUTES/VARIABLE_VALUE
    entries: List[Tuple[int, np.ndarray | None, np.ndarray | None]] = []
    for name, tensor in tensors.items():
        if "layer_with_weights-" not in name or "OPTIMIZER_SLOT" in name:
            continue
        if "/kernel/" in name:
            idx = int(name.split("layer_with_weights-")[1].split("/")[0])
            entries.append((idx, tensor, None))
        elif "/bias/" in name:
            idx = int(name.split("layer_with_weights-")[1].split("/")[0])
            entries.append((idx, None, tensor))

    # Aggregate kernels/biases by index.
    by_idx: Dict[int, Dict[str, np.ndarray]] = {}
    for idx, kernel, bias in entries:
        slot = by_idx.setdefault(idx, {})
        if kernel is not None:
            slot["kernel"] = kernel
        if bias is not None:
            slot["bias"] = bias

    for idx in sorted(by_idx.keys()):
        entry = by_idx[idx]
        if "kernel" in entry and "bias" in entry:
            k = entry["kernel"]
            b = entry["bias"]
            if k.ndim == 4:
                conv_pairs.append((k, b))
            elif k.ndim == 2:
                dense_pairs.append((k, b))

    if len(conv_pairs) != len(conv_modules) or len(dense_pairs) != len(linear_modules):
        available = ", ".join(list(tensors.keys())[:10])
        raise RuntimeError(
            f"Mismatch conv_pairs={len(conv_pairs)} (torch={len(conv_modules)}), "
            f"dense_pairs={len(dense_pairs)} (torch={len(linear_modules)}). "
            f"Sample tensors: {available}"
        )

    for module, (k_np, b_np) in zip(conv_modules, conv_pairs, strict=False):
        module.weight.data = torch.tensor(k_np.transpose(3, 2, 0, 1), dtype=module.weight.dtype)
        module.bias.data = torch.tensor(b_np, dtype=module.bias.dtype)

    for module, (k_np, b_np) in zip(linear_modules, dense_pairs, strict=False):
        module.weight.data = torch.tensor(k_np.T, dtype=module.weight.dtype)  # TF: [H, W], torch: [W, H]
        module.bias.data = torch.tensor(b_np, dtype=module.bias.dtype)


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

    torch_model = SmoeAE(
        block_size=args.block_size,
        kernel_num=args.kernel_num,
        predict_covariance=predict_covariance,
    )

    tensors = _load_raw_checkpoint(args.checkpoint)
    _assign_weights_from_raw(tensors, torch_model)
    torch.save(torch_model.state_dict(), args.output)

    print(f"[ok] Saved PyTorch weights to {args.output}")


if __name__ == "__main__":
    main()
