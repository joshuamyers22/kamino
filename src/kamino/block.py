"""Batched block PLS for one independent random-intercept grouping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from kamino.errors import ModelSpecificationError, NumericalError
from kamino.model import FloatArray, ModelSpec, ObjectiveKind

IntArray = npt.NDArray[np.int64]
BACKEND_NAME = "random-intercept-block-cholesky"


def _readonly(value: FloatArray) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _chol_solve(cholesky: FloatArray, right: FloatArray) -> FloatArray:
    return np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, right))


def _validated_group_indices(spec: ModelSpec, group_indices: IntArray) -> IntArray:
    indices = np.asarray(group_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.shape != (spec.n,):
        raise ModelSpecificationError("group_indices must have one value per row")
    if spec.q == 0 or (indices < 0).any() or (indices >= spec.q).any():
        raise ModelSpecificationError("group_indices contains an invalid group index")
    if len(set(indices.tolist())) != spec.q:
        raise ModelSpecificationError("every random-intercept group must be observed")
    return indices


@dataclass(frozen=True, slots=True)
class RandomInterceptBlockResult:
    """Outputs from one scalar-theta block evaluation."""

    kind: ObjectiveKind
    objective: float
    log_likelihood: float
    beta: FloatArray
    u: FloatArray
    b: FloatArray
    sigma2: float
    beta_covariance: FloatArray
    random_variance: float
    penalized_residual_sum_squares: float
    weighted_residual_sum_squares: float
    random_effect_penalty: float
    logdet_c: float
    logdet_s: float
    logdet_weights: float


def evaluate_random_intercept_block(
    spec: ModelSpec,
    group_indices: IntArray,
    theta: float,
    *,
    kind: ObjectiveKind = ObjectiveKind.REML,
) -> RandomInterceptBlockResult:
    """Evaluate a one-term random-intercept model without a q-by-q factorization."""
    if not np.isfinite(theta) or theta < 0.0:
        raise ModelSpecificationError("theta must be finite and nonnegative")
    indices = _validated_group_indices(spec, group_indices)
    groups = spec.q
    centered_response = spec.y - spec.offset

    group_weight = np.bincount(indices, weights=spec.weights, minlength=groups).astype(
        np.float64, copy=False
    )
    group_response = np.bincount(
        indices,
        weights=spec.weights * centered_response,
        minlength=groups,
    ).astype(np.float64, copy=False)
    group_design = np.zeros((groups, spec.p), dtype=np.float64)
    for column in range(spec.p):
        group_design[:, column] = np.bincount(
            indices,
            weights=spec.weights * spec.x[:, column],
            minlength=groups,
        )

    c = 1.0 + theta * theta * group_weight
    d = theta * group_design
    f = theta * group_response
    inverse_c = 1.0 / c
    weighted_design = spec.weights[:, None] * spec.x
    schur = spec.x.T @ weighted_design - d.T @ (inverse_c[:, None] * d)
    target = spec.x.T @ (spec.weights * centered_response) - d.T @ (inverse_c * f)

    try:
        chol_s: FloatArray = np.linalg.cholesky(schur)
        beta = _chol_solve(chol_s, target)
    except np.linalg.LinAlgError as error:
        raise NumericalError("block fixed-effect factorization failed") from error

    u = inverse_c * (f - d @ beta)
    b = theta * u
    residual = np.sqrt(spec.weights) * (centered_response - spec.x @ beta - b[indices])
    weighted_rss = float(residual @ residual)
    penalty = float(u @ u)
    pwrss = weighted_rss + penalty
    degrees = spec.n if kind is ObjectiveKind.ML else spec.n - spec.p
    if degrees <= 0:
        raise ModelSpecificationError("REML requires n greater than fixed-design rank")
    if not np.isfinite(pwrss) or pwrss <= 0.0:
        raise NumericalError("penalized residual sum of squares must be positive")

    logdet_c = float(np.log(c).sum())
    logdet_s = float(2.0 * np.log(chol_s.diagonal()).sum())
    logdet_weights = float(np.log(spec.weights).sum())
    objective = logdet_c - logdet_weights
    if kind is ObjectiveKind.REML:
        objective += logdet_s
    objective += degrees * (1.0 + np.log(2.0 * np.pi * pwrss / degrees))
    sigma2 = pwrss / degrees
    beta_covariance = sigma2 * _chol_solve(chol_s, np.eye(spec.p, dtype=np.float64))

    return RandomInterceptBlockResult(
        kind=kind,
        objective=float(objective),
        log_likelihood=float(-0.5 * objective),
        beta=_readonly(beta),
        u=_readonly(u),
        b=_readonly(b),
        sigma2=float(sigma2),
        beta_covariance=_readonly(beta_covariance),
        random_variance=float(sigma2 * theta * theta),
        penalized_residual_sum_squares=pwrss,
        weighted_residual_sum_squares=weighted_rss,
        random_effect_penalty=penalty,
        logdet_c=logdet_c,
        logdet_s=logdet_s,
        logdet_weights=logdet_weights,
    )


__all__ = [
    "BACKEND_NAME",
    "RandomInterceptBlockResult",
    "evaluate_random_intercept_block",
]
