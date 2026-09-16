"""Kamino: versioned Gaussian linear mixed-model compatibility."""

from kamino.bundle import BundleLimits, load_model_bundle, save_model_bundle
from kamino.errors import BootstrapError, BundleError, ResourceLimitError
from kamino.fit import FitControl, lmer, refit
from kamino.inference import (
    BootstrapInterval,
    BootstrapReplicate,
    BootstrapResult,
    FailureAccounting,
    InferenceLimits,
    SimulationBatch,
    SimulationDraw,
    parametric_bootstrap,
    simulate,
)
from kamino.model import ModelSpec, ObjectiveKind
from kamino.pls import FixedThetaResult, evaluate_fixed_theta
from kamino.results import (
    LinearFunctionResult,
    LinearMixedModelResult,
    OptimizerDiagnostics,
    PredictionOnlyModel,
    PredictionResult,
)
from kamino.satterthwaite import (
    SatterthwaiteAnalysis,
    SatterthwaiteControl,
    SatterthwaiteJointTest,
    SatterthwaiteTest,
    satterthwaite,
)
from kamino.sparse import SparseBackendLimits

__all__ = [
    "BundleError",
    "BundleLimits",
    "BootstrapError",
    "BootstrapInterval",
    "BootstrapReplicate",
    "BootstrapResult",
    "FitControl",
    "FailureAccounting",
    "FixedThetaResult",
    "LinearMixedModelResult",
    "LinearFunctionResult",
    "InferenceLimits",
    "ModelSpec",
    "ObjectiveKind",
    "OptimizerDiagnostics",
    "PredictionOnlyModel",
    "PredictionResult",
    "ResourceLimitError",
    "SimulationBatch",
    "SimulationDraw",
    "SparseBackendLimits",
    "SatterthwaiteAnalysis",
    "SatterthwaiteControl",
    "SatterthwaiteJointTest",
    "SatterthwaiteTest",
    "evaluate_fixed_theta",
    "lmer",
    "load_model_bundle",
    "save_model_bundle",
    "parametric_bootstrap",
    "refit",
    "simulate",
    "satterthwaite",
]

__version__ = "0.0.1"
