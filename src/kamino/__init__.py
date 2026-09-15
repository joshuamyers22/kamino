"""Kamino: versioned Gaussian linear mixed-model compatibility."""

from kamino.fit import FitControl, lmer
from kamino.model import ModelSpec, ObjectiveKind
from kamino.pls import FixedThetaResult, evaluate_fixed_theta
from kamino.results import (
    LinearMixedModelResult,
    OptimizerDiagnostics,
    PredictionResult,
)

__all__ = [
    "FitControl",
    "FixedThetaResult",
    "LinearMixedModelResult",
    "ModelSpec",
    "ObjectiveKind",
    "OptimizerDiagnostics",
    "PredictionResult",
    "evaluate_fixed_theta",
    "lmer",
]

__version__ = "0.0.1"
