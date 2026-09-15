"""Typed errors for model validation and numerical evaluation."""


class KaminoError(Exception):
    """Base class for expected Kamino failures."""


class ModelSpecificationError(KaminoError, ValueError):
    """The supplied model arrays do not define a supported model."""


class NumericalError(KaminoError, ArithmeticError):
    """A factorization or objective evaluation failed."""
