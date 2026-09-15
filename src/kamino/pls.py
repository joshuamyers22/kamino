"""Fixed-covariance weighted penalized least-squares evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kamino.errors import ModelSpecificationError, NumericalError
from kamino.model import FloatArray, MatrixInput, ModelSpec, ObjectiveKind


def _readonly(array: FloatArray) -> FloatArray:
    value = np.array(array, dtype=np.float64, copy=True)
    value.setflags(write=False)
    return value


def _chol_solve(cholesky: FloatArray, right: FloatArray) -> FloatArray:
    return np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, right))


@dataclass(frozen=True, slots=True)
class FixedThetaResult:
    """Immutable outputs for one covariance-factor evaluation."""

    kind: ObjectiveKind
    objective: float
    log_likelihood: float
    beta: FloatArray
    u: FloatArray
    b: FloatArray
    sigma2: float
    beta_covariance: FloatArray
    random_covariance: FloatArray
    penalized_residual_sum_squares: float
    weighted_residual_sum_squares: float
    random_effect_penalty: float
    logdet_c: float
    logdet_s: float
    logdet_weights: float


def evaluate_fixed_theta(
    spec: ModelSpec,
    lambda_: MatrixInput,
    *,
    kind: ObjectiveKind = ObjectiveKind.REML,
) -> FixedThetaResult:
    """Evaluate the profiled weighted ML or REML criterion at lambda_."""
    covariance_factor: FloatArray = np.asarray(lambda_, dtype=np.float64)
    if covariance_factor.shape != (spec.q, spec.q):
        raise ModelSpecificationError("lambda_ must be a q by q matrix")
    if not np.isfinite(covariance_factor).all():
        raise ModelSpecificationError("lambda_ contains non-finite values")

    sqrt_weights = np.sqrt(spec.weights)
    centered_response = spec.y - spec.offset
    a = sqrt_weights[:, None] * (spec.z @ covariance_factor)
    design = sqrt_weights[:, None] * spec.x
    response = sqrt_weights * centered_response

    c = np.eye(spec.q, dtype=np.float64) + a.T @ a
    try:
        chol_c: FloatArray = np.linalg.cholesky(c)
        d = a.T @ design
        f = a.T @ response
        c_inv_d = _chol_solve(chol_c, d)
        c_inv_f = _chol_solve(chol_c, f)
        schur = design.T @ design - d.T @ c_inv_d
        chol_s: FloatArray = np.linalg.cholesky(schur)
        target = design.T @ response - d.T @ c_inv_f
        beta = _chol_solve(chol_s, target)
        u = _chol_solve(chol_c, f - d @ beta)
    except np.linalg.LinAlgError as error:
        raise NumericalError("fixed-theta factorization failed") from error

    residual = response - design @ beta - a @ u
    weighted_rss = float(residual @ residual)
    penalty = float(u @ u)
    pwrss = weighted_rss + penalty
    degrees = spec.n if kind is ObjectiveKind.ML else spec.n - spec.p
    if degrees <= 0:
        raise ModelSpecificationError("REML requires n greater than fixed-design rank")
    if not np.isfinite(pwrss) or pwrss <= 0.0:
        raise NumericalError("penalized residual sum of squares must be positive")

    logdet_c = float(2.0 * np.log(chol_c.diagonal()).sum())
    logdet_s = float(2.0 * np.log(chol_s.diagonal()).sum())
    logdet_weights = float(np.log(spec.weights).sum())
    objective = logdet_c - logdet_weights
    if kind is ObjectiveKind.REML:
        objective += logdet_s
    objective += degrees * (1.0 + np.log(2.0 * np.pi * pwrss / degrees))
    sigma2 = pwrss / degrees
    identity_p = np.eye(spec.p, dtype=np.float64)
    beta_covariance = sigma2 * _chol_solve(chol_s, identity_p)
    random_covariance = sigma2 * covariance_factor @ covariance_factor.T

    return FixedThetaResult(
        kind=kind,
        objective=float(objective),
        log_likelihood=float(-0.5 * objective),
        beta=_readonly(beta),
        u=_readonly(u),
        b=_readonly(covariance_factor @ u),
        sigma2=float(sigma2),
        beta_covariance=_readonly(beta_covariance),
        random_covariance=_readonly(random_covariance),
        penalized_residual_sum_squares=pwrss,
        weighted_residual_sum_squares=weighted_rss,
        random_effect_penalty=penalty,
        logdet_c=logdet_c,
        logdet_s=logdet_s,
        logdet_weights=logdet_weights,
    )
