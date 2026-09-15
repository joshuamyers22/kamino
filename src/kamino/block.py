"""Batched block PLS for one independent grouping structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from kamino.errors import ModelSpecificationError, NumericalError
from kamino.model import FloatArray, ObjectiveKind, SingleGroupSpec, VectorInput

IntArray = npt.NDArray[np.int64]
BACKEND_NAME = "single-group-block-cholesky"


def _readonly(value: FloatArray) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _chol_solve(cholesky: FloatArray, right: FloatArray) -> FloatArray:
    return np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, right))


def _validated_group_indices(spec: SingleGroupSpec) -> IntArray:
    indices = np.asarray(spec.group_indices)
    if indices.ndim != 1 or indices.shape != (spec.n,):
        raise ModelSpecificationError("group_indices must have one value per row")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ModelSpecificationError("group_indices must contain integers")
    if (
        spec.group_count == 0
        or (indices < 0).any()
        or (indices >= spec.group_count).any()
    ):
        raise ModelSpecificationError("group_indices contains an invalid group index")
    if spec.group_count > spec.n:
        raise ModelSpecificationError("every random-intercept group must be observed")
    return indices


@dataclass(frozen=True, slots=True)
class SingleGroupBlockResult:
    """Outputs from one covariance-block evaluation."""

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


@dataclass(frozen=True, slots=True)
class SingleGroupBlockWorkspace:
    """Theta-independent sufficient statistics for one grouped fit.

    The workspace is O(g*k*(k+p)) rather than O(n*q) and is intentionally
    reusable across optimizer evaluations.  It retains the validated compact
    specification so accepted effects can still be mapped back to rows.
    """

    spec: SingleGroupSpec
    centered_response: FloatArray
    random_cross: FloatArray
    random_fixed_cross: FloatArray
    random_response_cross: FloatArray
    fixed_cross: FloatArray
    fixed_response_cross: FloatArray
    logdet_weights: float


def _lower_triangular(theta: VectorInput, k: int) -> FloatArray:
    values = np.asarray(theta, dtype=np.float64)
    expected = k * (k + 1) // 2
    if values.ndim != 1 or values.shape != (expected,):
        raise ModelSpecificationError(
            f"theta must contain {expected} lower-triangular parameters"
        )
    if not np.isfinite(values).all():
        raise ModelSpecificationError("theta contains non-finite values")
    factor = np.zeros((k, k), dtype=np.float64)
    cursor = 0
    for column in range(k):
        width = k - column
        factor[column:, column] = values[cursor : cursor + width]
        cursor += width
    if (factor.diagonal() < 0.0).any():
        raise ModelSpecificationError("theta diagonal parameters must be nonnegative")
    return factor


def _batched_chol_solve(cholesky: FloatArray, right: FloatArray) -> FloatArray:
    return np.linalg.solve(cholesky.swapaxes(-1, -2), np.linalg.solve(cholesky, right))


def prepare_single_group_block(spec: SingleGroupSpec) -> SingleGroupBlockWorkspace:
    """Assemble reusable block sufficient statistics without a dense ``Z``."""
    indices = _validated_group_indices(spec)
    groups = spec.group_count
    k = spec.k
    centered_response = spec.y - spec.offset

    group_weight = np.bincount(indices, weights=spec.weights, minlength=groups).astype(
        np.float64, copy=False
    )
    if (group_weight <= 0.0).any():
        raise ModelSpecificationError("every random-intercept group must be observed")
    random_cross = np.empty((groups, k, k), dtype=np.float64)
    random_fixed_cross = np.empty((groups, k, spec.p), dtype=np.float64)
    random_response_cross = np.empty((groups, k), dtype=np.float64)
    for left in range(k):
        random_response_cross[:, left] = np.bincount(
            indices,
            weights=(spec.weights * spec.random_design[:, left] * centered_response),
            minlength=groups,
        )
        for right in range(k):
            random_cross[:, left, right] = np.bincount(
                indices,
                weights=(
                    spec.weights
                    * spec.random_design[:, left]
                    * spec.random_design[:, right]
                ),
                minlength=groups,
            )
        for column in range(spec.p):
            random_fixed_cross[:, left, column] = np.bincount(
                indices,
                weights=(
                    spec.weights * spec.random_design[:, left] * spec.x[:, column]
                ),
                minlength=groups,
            )

    weighted_design = spec.weights[:, None] * spec.x
    weighted_response = spec.weights * centered_response
    return SingleGroupBlockWorkspace(
        spec=spec,
        centered_response=_readonly(centered_response),
        random_cross=_readonly(random_cross),
        random_fixed_cross=_readonly(random_fixed_cross),
        random_response_cross=_readonly(random_response_cross),
        fixed_cross=_readonly(spec.x.T @ weighted_design),
        fixed_response_cross=_readonly(spec.x.T @ weighted_response),
        logdet_weights=float(np.log(spec.weights).sum()),
    )


def evaluate_prepared_single_group_block(
    workspace: SingleGroupBlockWorkspace,
    theta: VectorInput,
    *,
    kind: ObjectiveKind = ObjectiveKind.REML,
) -> SingleGroupBlockResult:
    """Evaluate theta from preassembled single-group sufficient statistics."""
    spec = workspace.spec
    k = spec.k
    factor = _lower_triangular(theta, k)

    factor_transpose = factor.T
    c = (
        np.eye(k, dtype=np.float64)[None, :, :]
        + factor_transpose[None, :, :] @ workspace.random_cross @ factor[None, :, :]
    )
    d = factor_transpose[None, :, :] @ workspace.random_fixed_cross
    f = (factor_transpose[None, :, :] @ workspace.random_response_cross[:, :, None])[
        ..., 0
    ]
    try:
        chol_c: FloatArray = np.linalg.cholesky(c)
        c_inv_d = _batched_chol_solve(chol_c, d)
        c_inv_f = _batched_chol_solve(chol_c, f[:, :, None])[..., 0]
    except np.linalg.LinAlgError as error:
        raise NumericalError("group-block factorization failed") from error
    schur = workspace.fixed_cross - np.einsum("gkp,gkq->pq", d, c_inv_d)
    target = workspace.fixed_response_cross - np.einsum("gkp,gk->p", d, c_inv_f)

    try:
        chol_s: FloatArray = np.linalg.cholesky(schur)
        beta = _chol_solve(chol_s, target)
    except np.linalg.LinAlgError as error:
        raise NumericalError("block fixed-effect factorization failed") from error

    u_by_group = _batched_chol_solve(chol_c, (f - d @ beta)[:, :, None])[..., 0]
    b_by_group = (factor[None, :, :] @ u_by_group[:, :, None])[..., 0]
    contribution = np.einsum(
        "nk,nk->n", spec.random_design, b_by_group[spec.group_indices]
    )
    residual = np.sqrt(spec.weights) * (
        workspace.centered_response - spec.x @ beta - contribution
    )
    weighted_rss = float(residual @ residual)
    penalty = float(np.square(u_by_group).sum())
    pwrss = weighted_rss + penalty
    degrees = spec.n if kind is ObjectiveKind.ML else spec.n - spec.p
    if degrees <= 0:
        raise ModelSpecificationError("REML requires n greater than fixed-design rank")
    if not np.isfinite(pwrss) or pwrss <= 0.0:
        raise NumericalError("penalized residual sum of squares must be positive")

    logdet_c = float(2.0 * np.log(chol_c.diagonal(axis1=1, axis2=2)).sum())
    logdet_s = float(2.0 * np.log(chol_s.diagonal()).sum())
    objective = logdet_c - workspace.logdet_weights
    if kind is ObjectiveKind.REML:
        objective += logdet_s
    objective += degrees * (1.0 + np.log(2.0 * np.pi * pwrss / degrees))
    sigma2 = pwrss / degrees
    beta_covariance = sigma2 * _chol_solve(chol_s, np.eye(spec.p, dtype=np.float64))

    return SingleGroupBlockResult(
        kind=kind,
        objective=float(objective),
        log_likelihood=float(-0.5 * objective),
        beta=_readonly(beta),
        u=_readonly(u_by_group.reshape(-1)),
        b=_readonly(b_by_group.reshape(-1)),
        sigma2=float(sigma2),
        beta_covariance=_readonly(beta_covariance),
        random_covariance=_readonly(sigma2 * factor @ factor.T),
        penalized_residual_sum_squares=pwrss,
        weighted_residual_sum_squares=weighted_rss,
        random_effect_penalty=penalty,
        logdet_c=logdet_c,
        logdet_s=logdet_s,
        logdet_weights=workspace.logdet_weights,
    )


def evaluate_single_group_block(
    spec: SingleGroupSpec,
    theta: VectorInput,
    *,
    kind: ObjectiveKind = ObjectiveKind.REML,
) -> SingleGroupBlockResult:
    """Evaluate one grouped covariance term without a q-by-q factorization."""
    return evaluate_prepared_single_group_block(
        prepare_single_group_block(spec), theta, kind=kind
    )


__all__ = [
    "BACKEND_NAME",
    "SingleGroupBlockResult",
    "SingleGroupBlockWorkspace",
    "evaluate_prepared_single_group_block",
    "evaluate_single_group_block",
    "prepare_single_group_block",
]
