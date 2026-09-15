"""Kamino: versioned Gaussian linear mixed-model compatibility."""

from kamino.bundle import BundleLimits, load_model_bundle, save_model_bundle
from kamino.errors import BundleError, ResourceLimitError
from kamino.fit import FitControl, lmer
from kamino.model import ModelSpec, ObjectiveKind
from kamino.pls import FixedThetaResult, evaluate_fixed_theta
from kamino.results import (
    LinearFunctionResult,
    LinearMixedModelResult,
    OptimizerDiagnostics,
    PredictionOnlyModel,
    PredictionResult,
)
from kamino.sparse import SparseBackendLimits

__all__ = [
    "BundleError",
    "BundleLimits",
    "FitControl",
    "FixedThetaResult",
    "LinearMixedModelResult",
    "LinearFunctionResult",
    "ModelSpec",
    "ObjectiveKind",
    "OptimizerDiagnostics",
    "PredictionOnlyModel",
    "PredictionResult",
    "ResourceLimitError",
    "SparseBackendLimits",
    "evaluate_fixed_theta",
    "lmer",
    "load_model_bundle",
    "save_model_bundle",
]

__version__ = "0.0.1"
