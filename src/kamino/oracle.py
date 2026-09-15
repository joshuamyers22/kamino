"""Independent small-n marginal-covariance oracle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kamino.errors import ModelSpecificationError, NumericalError
from kamino.model import (
    FloatArray,
    MatrixInput,
    ModelSpec,
    ObjectiveKind,
    RandomInterceptSpec,
)


@dataclass(frozen=True, slots=True)
class DenseOracleResult:
    """Independent GLS outputs at one relative covariance factor."""

    kind: ObjectiveKind
    objective: float
    beta: FloatArray
    sigma2: float
    beta_covariance: FloatArray
    penalized_residual_sum_squares: float
    logdet_h: float
    logdet_information: float


def evaluate_dense_oracle(
    spec: ModelSpec | RandomInterceptSpec,
    lambda_: MatrixInput,
    *,
    kind: ObjectiveKind = ObjectiveKind.REML,
    maximum_n: int = 2_000,
) -> DenseOracleResult:
    """Evaluate by direct H assembly without PLS implementation helpers."""
    if spec.n > maximum_n:
        raise ModelSpecificationError(
            f"dense oracle refuses n={spec.n}; maximum_n={maximum_n}"
        )
    covariance_factor: FloatArray = np.asarray(lambda_, dtype=np.float64)
    if covariance_factor.shape != (spec.q, spec.q):
        raise ModelSpecificationError("lambda_ must be a q by q matrix")
    if not np.isfinite(covariance_factor).all():
        raise ModelSpecificationError("lambda_ contains non-finite values")
    relative_random_covariance = covariance_factor @ covariance_factor.T
    if isinstance(spec, RandomInterceptSpec):
        indices = spec.group_indices
        random_component = relative_random_covariance[
            indices[:, None], indices[None, :]
        ]
    else:
        random_component = spec.z @ relative_random_covariance @ spec.z.T
    h = np.eye(spec.n, dtype=np.float64) * (1.0 / spec.weights) + random_component
    try:
        chol_h: FloatArray = np.linalg.cholesky(h)
        h_inv_x = np.linalg.solve(chol_h.T, np.linalg.solve(chol_h, spec.x))
        centered = spec.y - spec.offset
        h_inv_y = np.linalg.solve(chol_h.T, np.linalg.solve(chol_h, centered))
        information = spec.x.T @ h_inv_x
        chol_information: FloatArray = np.linalg.cholesky(information)
        beta = np.linalg.solve(
            chol_information.T,
            np.linalg.solve(chol_information, spec.x.T @ h_inv_y),
        )
        residual = centered - spec.x @ beta
        h_inv_residual = np.linalg.solve(chol_h.T, np.linalg.solve(chol_h, residual))
    except np.linalg.LinAlgError as error:
        raise NumericalError("dense oracle factorization failed") from error
    quadratic = float(residual @ h_inv_residual)
    degrees = spec.n if kind is ObjectiveKind.ML else spec.n - spec.p
    if degrees <= 0:
        raise ModelSpecificationError("REML requires n greater than fixed-design rank")
    if quadratic <= 0.0 or not np.isfinite(quadratic):
        raise NumericalError("oracle residual quadratic must be positive")
    logdet_h = float(2.0 * np.log(chol_h.diagonal()).sum())
    logdet_information = float(2.0 * np.log(chol_information.diagonal()).sum())
    objective = logdet_h
    if kind is ObjectiveKind.REML:
        objective += logdet_information
    objective += degrees * (1.0 + np.log(2.0 * np.pi * quadratic / degrees))
    sigma2 = quadratic / degrees
    beta_covariance = sigma2 * np.linalg.solve(
        chol_information.T,
        np.linalg.solve(chol_information, np.eye(spec.p, dtype=np.float64)),
    )
    beta.setflags(write=False)
    beta_covariance.setflags(write=False)
    return DenseOracleResult(
        kind=kind,
        objective=float(objective),
        beta=beta,
        sigma2=float(sigma2),
        beta_covariance=beta_covariance,
        penalized_residual_sum_squares=quadratic,
        logdet_h=logdet_h,
        logdet_information=logdet_information,
    )
