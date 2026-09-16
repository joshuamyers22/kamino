"""Kamino: versioned Gaussian linear mixed-model compatibility."""

from kamino.bundle import BundleLimits, load_model_bundle, save_model_bundle
from kamino.cluster_robust import (
    ClusterCovarianceType,
    ClusterRobustAnalysis,
    ClusterRobustControl,
    ClusterRobustTest,
    cluster_robust,
)
from kamino.errors import BootstrapError, BundleError, PostfitError, ResourceLimitError
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
from kamino.kenward_roger import (
    KenwardRogerAnalysis,
    KenwardRogerControl,
    KenwardRogerTest,
    kenward_roger,
)
from kamino.model import ModelSpec, ObjectiveKind
from kamino.pls import FixedThetaResult, evaluate_fixed_theta
from kamino.postfit import (
    POSTFIT_CONTRACT_VERSION,
    ContrastResult,
    GridEstimate,
    LinearFunctionBasis,
    PerformanceSummary,
    PostfitAnalysis,
    ReferenceGridResult,
    adapt_kamino,
    adapt_statsmodels_mixedlm,
    adapt_statsmodels_ols,
)
from kamino.profile import (
    LikelihoodProfile,
    ProfileControl,
    ProfileInterval,
    ProfilePoint,
    ProfileTrace,
    likelihood_profile,
)
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
    "ContrastResult",
    "ClusterCovarianceType",
    "ClusterRobustAnalysis",
    "ClusterRobustControl",
    "ClusterRobustTest",
    "FitControl",
    "FailureAccounting",
    "FixedThetaResult",
    "GridEstimate",
    "LinearMixedModelResult",
    "LinearFunctionResult",
    "LinearFunctionBasis",
    "InferenceLimits",
    "KenwardRogerAnalysis",
    "KenwardRogerControl",
    "KenwardRogerTest",
    "LikelihoodProfile",
    "ModelSpec",
    "ObjectiveKind",
    "OptimizerDiagnostics",
    "POSTFIT_CONTRACT_VERSION",
    "PredictionOnlyModel",
    "PredictionResult",
    "PerformanceSummary",
    "PostfitAnalysis",
    "PostfitError",
    "ProfileControl",
    "ProfileInterval",
    "ProfilePoint",
    "ProfileTrace",
    "ReferenceGridResult",
    "ResourceLimitError",
    "SimulationBatch",
    "SimulationDraw",
    "SparseBackendLimits",
    "SatterthwaiteAnalysis",
    "SatterthwaiteControl",
    "SatterthwaiteJointTest",
    "SatterthwaiteTest",
    "evaluate_fixed_theta",
    "adapt_kamino",
    "adapt_statsmodels_mixedlm",
    "adapt_statsmodels_ols",
    "cluster_robust",
    "lmer",
    "kenward_roger",
    "likelihood_profile",
    "load_model_bundle",
    "save_model_bundle",
    "parametric_bootstrap",
    "refit",
    "simulate",
    "satterthwaite",
]

__version__ = "0.0.1"
