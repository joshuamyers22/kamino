"""Kamino: Gaussian linear mixed-model numerical foundations."""

from kamino.model import ModelSpec, ObjectiveKind
from kamino.pls import FixedThetaResult, evaluate_fixed_theta

__all__ = [
    "FixedThetaResult",
    "ModelSpec",
    "ObjectiveKind",
    "evaluate_fixed_theta",
]

__version__ = "0.0.1"
