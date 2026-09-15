"""Typed errors for model validation and numerical evaluation."""


class KaminoError(Exception):
    """Base class for expected Kamino failures."""


class ModelSpecificationError(KaminoError, ValueError):
    """The supplied model arrays do not define a supported model."""


class NumericalError(KaminoError, ArithmeticError):
    """A factorization or objective evaluation failed."""


class UnsupportedFormulaError(ModelSpecificationError):
    """The formula is outside Kamino's advertised compatibility profile."""


class ConvergenceError(NumericalError):
    """The optimizer did not produce an accepted final state."""


class ResourceLimitError(ModelSpecificationError):
    """A validated model exceeds a configured allocation or fill limit."""


class PredictionError(KaminoError, ValueError):
    """Prediction inputs or conditioning are unsupported or invalid."""


class BundleError(KaminoError):
    """A model bundle could not be safely written, validated, or loaded."""
