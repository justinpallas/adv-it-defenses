"""Defense implementations made available for the pipeline."""

from __future__ import annotations

from .bit_depth import BitDepthDefense
from .crop_resize import CropResizeDefense
from .flip import FlipDefense
from .grayscale import GrayscaleDefense
from .jpeg import JPEGDefense
from .low_pass import LowPassDefense
from .r_smoe import RSMoEDefense
from .tvm import TVMDefense

__all__ = [
    "BitDepthDefense",
    "CropResizeDefense",
    "FlipDefense",
    "GrayscaleDefense",
    "JPEGDefense",
    "LowPassDefense",
    "RSMoEDefense",
    "TVMDefense",
]
