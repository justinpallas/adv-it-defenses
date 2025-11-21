"""SMoE autoencoder defense (PyTorch port of the demo implementation)."""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from advdef.config import DefenseConfig
from advdef.core.context import RunContext
from advdef.core.pipeline import DatasetVariant, Defense
from advdef.core.registry import register_defense
from advdef.utils import Progress, ensure_dir
from ._common import build_config_identifier


def _discover_images(root: Path, patterns: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.rglob(pattern))
    return sorted(paths)


def _default_device(device_param: str | None) -> torch.device:
    if device_param:
        return torch.device(device_param)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _chunks(items: Sequence[Path], size: int) -> Iterable[Sequence[Path]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - standard forward
        return F.relu(self.conv(x))


class SmoeEncoder(nn.Module):
    """Convolutional encoder that predicts SMoE parameters for one grayscale block."""

    def __init__(
        self,
        block_size: int,
        kernel_num: int = 4,
        predict_covariance: bool = True,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.kernel_num = kernel_num
        self.predict_covariance = predict_covariance

        if block_size == 8:
            conv_channels = [16, 32, 64, 128, 256, 512, 1024]
            dense_layers = [1024, 512, 256, 128, 64]
        elif block_size == 16:
            conv_channels = [16, 32, 64, 128, 256, 512]
            dense_layers = [512, 256, 128, 64]
        else:
            raise ValueError("block_size must be 8 or 16.")

        features = [nn.Conv2d(1, conv_channels[0], kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for in_ch, out_ch in zip(conv_channels, conv_channels[1:], strict=False):
            features.append(_ConvBlock(in_ch, out_ch))
        self.features = nn.Sequential(*features)

        flattened = conv_channels[-1] * block_size * block_size
        layers: list[nn.Module] = []
        in_dim = flattened
        for out_dim in dense_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = out_dim

        out_features = kernel_num * 3
        if predict_covariance:
            out_features += kernel_num * 4
        layers.append(nn.Linear(in_dim, out_features))
        self.head = nn.Sequential(*layers)

        self._precompute = (block_size, kernel_num, predict_covariance, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw SMoE parameters for each block."""
        h = self.features(x)
        h = torch.flatten(h, 1)
        params = self.head(h)
        if self.predict_covariance:
            center_nus = torch.clamp(params[:, : self.kernel_num * 3], 0.0, 1.0)
            params = torch.cat([center_nus, params[:, self.kernel_num * 3 :]], dim=1)
        else:
            params = torch.clamp(params, 0.0, 1.0)
        return params


class SmoeDecoder(nn.Module):
    """Differentiable SMoE reconstruction."""

    def __init__(
        self,
        block_size: int,
        kernel_num: int = 4,
        predict_covariance: bool = True,
        sigma_x: float = 0.035,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.kernel_num = kernel_num
        self.predict_covariance = predict_covariance
        self.sigma_x = sigma_x

        grid = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, block_size),
                torch.linspace(0, 1, block_size),
                indexing="xy",
            ),
            dim=-1,
        )
        self.register_buffer("domain_grid", grid.reshape(-1, 2), persistent=False)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        k = self.kernel_num
        p = self.block_size * self.block_size

        center_x = params[:, 0:k]
        center_y = params[:, k : 2 * k]
        nue_e = params[:, 2 * k : 3 * k]

        centers = torch.stack([center_x, center_y], dim=-1)  # [B, k, 2]
        mus = centers.view(-1, k, 1, 2)
        domain = self.domain_grid.view(1, 1, p, 2)
        x_sub_mu = (domain - mus).unsqueeze(-1)  # [B, k, p, 2, 1]

        if self.predict_covariance:
            a_flat = params[:, 3 * k :].view(-1, k, 2, 2)
            a_tril = torch.tril(a_flat)
            exponent = -0.5 * torch.einsum(
                "abcli,ablm,abnm,abcnj->abc",
                x_sub_mu,
                a_tril,
                a_tril,
                x_sub_mu,
            )
        else:
            dist_sq = torch.sum(x_sub_mu.squeeze(-1) ** 2, dim=-1)
            exponent = -dist_sq / max(self.sigma_x, 1e-6)

        n_exp = torch.exp(exponent)  # [B, k, p]
        n_w_norm = torch.clamp(n_exp.sum(dim=1, keepdim=True), min=1e-8)
        weights = n_exp / n_w_norm

        res = torch.sum(weights * nue_e.unsqueeze(-1), dim=1)
        res = torch.clamp(res, 0.0, 1.0)
        return res.view(-1, self.block_size, self.block_size)


class SmoeAE(nn.Module):
    """Full SMoE autoencoder (encoder + decoder)."""

    def __init__(
        self,
        block_size: int,
        kernel_num: int = 4,
        predict_covariance: bool = True,
        sigma_x: float = 0.035,
    ) -> None:
        super().__init__()
        self.encoder = SmoeEncoder(
            block_size=block_size,
            kernel_num=kernel_num,
            predict_covariance=predict_covariance,
        )
        self.decoder = SmoeDecoder(
            block_size=block_size,
            kernel_num=kernel_num,
            predict_covariance=predict_covariance,
            sigma_x=sigma_x,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = self.encoder(x)
        return self.decoder(params)


@dataclass
class _SmoeParams:
    patterns: Sequence[str]
    block_size: int
    kernel_num: int
    predict_covariance: bool
    sigma_x: float
    weights_path: Path
    device: torch.device
    padding_mode: str
    output_format: str | None
    overwrite: bool
    workers: int
    batch_blocks: int
    config_identifier: str


class _SmoeRunner:
    """Performs inference over images with a shared model instance."""

    def __init__(self, params: _SmoeParams) -> None:
        self.params = params
        self.model = SmoeAE(
            block_size=params.block_size,
            kernel_num=params.kernel_num,
            predict_covariance=params.predict_covariance,
            sigma_x=params.sigma_x,
        )
        state = torch.load(params.weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)
        self.model.to(params.device)
        self.model.eval()

    def _process_channel(self, channel: torch.Tensor) -> torch.Tensor:
        """Run the AE per-channel; channel shape [1, 1, H, W]."""
        block = self.params.block_size

        pad_h = (block - channel.shape[2] % block) % block
        pad_w = (block - channel.shape[3] % block) % block
        if pad_h or pad_w:
            channel = F.pad(
                channel,
                (0, pad_w, 0, pad_h),
                mode=self.params.padding_mode,
            )

        h_blocks = channel.shape[2] // block
        w_blocks = channel.shape[3] // block

        blocks = (
            channel.unfold(2, block, block)
            .unfold(3, block, block)
            .contiguous()
            .view(-1, 1, block, block)
        )

        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for chunk in torch.split(blocks, self.params.batch_blocks):
                chunk = chunk.to(self.params.device)
                decoded = self.model(chunk)
                outputs.append(decoded.cpu())
        recon_blocks = torch.cat(outputs, dim=0)

        recon_blocks = (
            recon_blocks.view(1, h_blocks, w_blocks, block, block)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(1, 1, h_blocks * block, w_blocks * block)
        )

        if pad_h or pad_w:
            recon_blocks = recon_blocks[:, :, : channel.shape[2] - pad_h, : channel.shape[3] - pad_w]

        return recon_blocks

    def process_image(self, image_path: Path, output_path: Path) -> None:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        recon_channels: list[torch.Tensor] = []
        for idx in range(3):
            channel = tensor[:, idx : idx + 1, :, :]
            recon = self._process_channel(channel)
            recon_channels.append(recon)

        recon_tensor = torch.cat(recon_channels, dim=1).squeeze(0)
        recon_np = (recon_tensor.permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(recon_np, mode="RGB").save(
            output_path,
            format=self.params.output_format.upper() if self.params.output_format else None,
        )


@register_defense("smoe-ae")
class SmoeAEDefense(Defense):
    """Apply the SMoE autoencoder per RGB channel."""

    def __init__(self, config: DefenseConfig) -> None:
        super().__init__(config)
        self._settings_reported = False
        self._progress: Progress | None = None
        self._variant_images: dict[str, list[Path]] = {}
        self._params: _SmoeParams | None = None
        self._runner: _SmoeRunner | None = None
        self._config_identifier = build_config_identifier(config, default_prefix="smoe-ae")

    def _get_params(self) -> _SmoeParams:
        if self._params is not None:
            return self._params

        params = self.config.params
        patterns = tuple(params.get("extensions", ("*.png", "*.jpg", "*.jpeg", "*.bmp")))
        block_size = int(params.get("block_size", 8))
        kernel_num = int(params.get("kernel_num", 4))
        predict_covariance = bool(params.get("predict_covariance", block_size == 8))
        sigma_x = float(params.get("sigma_x", 0.035))
        weights_path = Path(params.get("weights_path", ""))
        if not weights_path:
            raise ValueError("weights_path is required for smoe-ae defense.")
        if not weights_path.exists():
            raise FileNotFoundError(f"weights_path not found: {weights_path}")
        device = _default_device(params.get("device"))
        padding_mode = params.get("padding_mode", "constant")
        if padding_mode not in {"constant", "reflect"}:
            raise ValueError("padding_mode must be 'constant' or 'reflect'.")
        output_format = params.get("format")
        overwrite = bool(params.get("overwrite", True))
        workers = int(params.get("workers", max(1, (os.cpu_count() or 2) - 1)))
        batch_blocks = int(params.get("batch_blocks", 256))

        self._params = _SmoeParams(
            patterns=patterns,
            block_size=block_size,
            kernel_num=kernel_num,
            predict_covariance=predict_covariance,
            sigma_x=sigma_x,
            weights_path=weights_path,
            device=device,
            padding_mode=padding_mode,
            output_format=output_format,
            overwrite=overwrite,
            workers=workers,
            batch_blocks=batch_blocks,
            config_identifier=self._config_identifier,
        )
        return self._params

    def initialize(self, context: RunContext, variants: list[DatasetVariant]) -> None:
        details = self._get_params()

        self._variant_images = {}
        total_images = 0
        for variant in variants:
            images = _discover_images(Path(variant.data_dir), details.patterns)
            if not images:
                raise FileNotFoundError(f"No images matched the provided extensions in {variant.data_dir}.")
            self._variant_images[variant.name] = images
            total_images += len(images)

        if not self._settings_reported:
            print(
                "[info] SMoE-AE settings: "
                f"config={self.config.name or self._config_identifier} "
                f"block_size={details.block_size} kernel_num={details.kernel_num} "
                f"predict_covariance={details.predict_covariance} sigma_x={details.sigma_x} "
                f"weights={details.weights_path} device={details.device} "
                f"padding_mode={details.padding_mode} batch_blocks={details.batch_blocks} "
                f"workers={details.workers}"
            )
            self._settings_reported = True

        if self._progress is not None:
            self._progress.close()
        self._progress = Progress(total=total_images, description="SMoE-AE reconstruction", unit="images")

    def _ensure_runner(self) -> _SmoeRunner:
        if self._runner is None:
            params = self._get_params()
            self._runner = _SmoeRunner(params)
        return self._runner

    def _process_batch(
        self,
        images: Sequence[Path],
        input_root: Path,
        output_root: Path,
        runner: _SmoeRunner,
        overwrite: bool,
    ) -> None:
        progress = self._progress
        for src in images:
            rel = src.relative_to(input_root)
            dst = output_root / rel
            if not overwrite and dst.exists():
                if progress:
                    progress.update()
                continue
            runner.process_image(src, dst)
            if progress:
                progress.update()

    def run(self, context: RunContext, variant: DatasetVariant) -> DatasetVariant:
        details = self._get_params()
        runner = self._ensure_runner()

        input_dir = Path(variant.data_dir)
        output_root = ensure_dir(
            context.artifacts_dir / "defenses" / "smoe-ae" / variant.name / self._config_identifier
        )

        images = self._variant_images.get(variant.name)
        if images is None:
            images = _discover_images(input_dir, details.patterns)

        if not images:
            raise FileNotFoundError(f"No images matched the provided extensions in {variant.data_dir}.")

        if details.workers <= 1:
            self._process_batch(images, input_dir, output_root, runner, details.overwrite)
        else:
            chunk_size = math.ceil(len(images) / details.workers)
            with ThreadPoolExecutor(max_workers=details.workers) as executor:
                futures = []
                for chunk in _chunks(images, chunk_size):
                    futures.append(
                        executor.submit(
                            self._process_batch, chunk, input_dir, output_root, runner, details.overwrite
                        )
                    )
                for future in futures:
                    future.result()

        return DatasetVariant(
            name=f"{variant.name}-smoe-ae-{self._config_identifier}",
            data_dir=str(output_root),
            parent=variant.name,
            metadata={
                "defense": "smoe-ae",
                "block_size": details.block_size,
                "kernel_num": details.kernel_num,
                "predict_covariance": details.predict_covariance,
                "weights_path": str(details.weights_path),
            },
        )

    def finalize(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        self._runner = None


__all__ = ["SmoeAEDefense", "SmoeAE"]
