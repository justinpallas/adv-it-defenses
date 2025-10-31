"""Import side effects to populate registries."""

from __future__ import annotations

# Dataset builders
from .datasets import imagenet_autoattack  # noqa: F401

# Attacks
from .attacks import autoattack  # noqa: F401
from .attacks import torchattacks  # noqa: F401

# Inference backends
from .inference import timm_backend  # noqa: F401

# Evaluation helpers
from .evaluation import imagenet  # noqa: F401

# Defenses (wrappers point to external implementations)
from .defenses import jpeg  # noqa: F401
from .defenses import r_smoe  # noqa: F401
