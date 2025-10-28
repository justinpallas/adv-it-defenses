"""Import side effects to populate registries."""

from __future__ import annotations

# Dataset builders
from advdef.datasets import imagenet_autoattack  # noqa: F401

# Attacks
from advdef.attacks import autoattack  # noqa: F401

# Inference backends
from advdef.inference import timm_backend  # noqa: F401

# Evaluation helpers
from advdef.evaluation import imagenet  # noqa: F401

# Defenses (wrappers point to external implementations)
from advdef.defenses import jpeg  # noqa: F401
from advdef.defenses import r_smoe  # noqa: F401
