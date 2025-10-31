"""Attack implementations."""

from __future__ import annotations

from .autoattack import AutoAttackAttack
from .torchattacks import CWL2Attack, DeepFoolAttack, FGSMAttack, PGDAttack

__all__ = [
    "AutoAttackAttack",
    "PGDAttack",
    "FGSMAttack",
    "CWL2Attack",
    "DeepFoolAttack",
]
