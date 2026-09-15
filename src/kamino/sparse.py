"""General sparse PLS backend for coupled random-effects terms."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportArgumentType=false, reportReturnType=false
# pyright: reportUnknownParameterType=false, reportUnnecessaryIsInstance=false

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import SuperLU, splu

from kamino.errors import ModelSpecificationError, NumericalError, ResourceLimitError
from kamino.model import FloatArray, GeneralSparseSpec, ObjectiveKind, VectorInput

BACKEND_NAME = "scipy-superlu-symmetric-sparse"


def _readonly(value: FloatArray) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _chol_solve(cholesky: FloatArray, right: FloatArray) -> FloatArray:
    return np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, right))


@dataclass(frozen=True, slots=True)
class SparseBackendLimits:
    """Preflight and post-factorization limits for sparse workspaces."""

    maximum_random_coefficients: int = 2_000_000
    maximum_design_nonzeros: int = 20_000_000
    maximum_structural_c_nonzeros: int = 20_000_000
    maximum_factor_nonzeros: int = 40_000_000

    def __post_init__(self) -> None:
        if not all(
            type(value) is int and value > 0
            for value in (
                self.maximum_random_coefficients,
                self.maximum_design_nonzeros,
                self.maximum_structural_c_nonzeros,
                self.maximum_factor_nonzeros,
            )
        ):
            raise ModelSpecificationError(
                "sparse backend limits must be positive integers"
            )


@dataclass(frozen=True, slots=True)
class GeneralSparseResult:
    """Outputs from one coupled sparse covariance evaluation."""

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
    factor_nonzeros: int


@dataclass(frozen=True, slots=True)
class GeneralSparseWorkspace:
    """Theta-independent sparse structure and weighted cross-products."""

    spec: GeneralSparseSpec
    random_design: sparse.csr_matrix
    random_cross: sparse.csc_matrix
    random_fixed_cross: FloatArray
    random_response_cross: FloatArray
    fixed_cross: FloatArray
    fixed_response_cross: FloatArray
    weighted_design: FloatArray
    weighted_response: FloatArray
    centered_response: FloatArray
    logdet_weights: float
    structural_c_pattern: sparse.csc_matrix
    structural_c_nonzeros: int
    limits: SparseBackendLimits


def _random_design(spec: GeneralSparseSpec, *, structural: bool) -> sparse.csr_matrix:
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    column_offset = 0
    observation_rows = np.arange(spec.n, dtype=np.int64)
    for term in spec.terms:
        rows.append(np.repeat(observation_rows, term.k))
        columns.append(
            (
                column_offset
                + term.group_indices[:, None] * term.k
                + np.arange(term.k, dtype=np.int64)[None, :]
            ).reshape(-1)
        )
        values.append(
            np.ones(spec.n * term.k, dtype=np.float64)
            if structural
            else term.random_design.reshape(-1)
        )
        column_offset += term.q
    matrix = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(spec.n, spec.q),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def prepare_general_sparse(
    spec: GeneralSparseSpec,
    *,
    limits: SparseBackendLimits | None = None,
) -> GeneralSparseWorkspace:
    """Assemble reusable sparse cross-products without a dense random design."""
    active_limits = limits or SparseBackendLimits()
    if not isinstance(active_limits, SparseBackendLimits):
        raise ModelSpecificationError("limits must be a SparseBackendLimits instance")
    design_nonzeros = sum(spec.n * term.k for term in spec.terms)
    if spec.q > active_limits.maximum_random_coefficients:
        raise ResourceLimitError("random coefficient count exceeds sparse limit")
    if design_nonzeros > active_limits.maximum_design_nonzeros:
        raise ResourceLimitError("random-design nonzeros exceed sparse limit")
    try:
        structural_design = _random_design(spec, structural=True)
        structural_cross = (structural_design.T @ structural_design).tocsc()
        structural_pattern = (
            structural_cross + sparse.eye(spec.q, format="csc")
        ).astype(np.bool_)
    except (MemoryError, ValueError) as error:
        raise NumericalError("sparse symbolic assembly failed") from error
    structural_c_nonzeros = int(structural_cross.nnz)
    if structural_c_nonzeros > active_limits.maximum_structural_c_nonzeros:
        raise ResourceLimitError("structural sparse cross-product exceeds limit")

    try:
        random_design = _random_design(spec, structural=False)
        sqrt_weights = np.sqrt(spec.weights)
        weighted_random = random_design.multiply(sqrt_weights[:, None]).tocsr()
        weighted_design = sqrt_weights[:, None] * spec.x
        centered_response = spec.y - spec.offset
        weighted_response = sqrt_weights * centered_response
        random_cross = (weighted_random.T @ weighted_random).tocsc()
        random_cross.sum_duplicates()
        random_cross.sort_indices()
        random_fixed_cross = np.asarray(weighted_random.T @ weighted_design)
        random_response_cross = np.asarray(weighted_random.T @ weighted_response)
    except (MemoryError, ValueError) as error:
        raise NumericalError("sparse numeric assembly failed") from error
    return GeneralSparseWorkspace(
        spec=spec,
        random_design=random_design,
        random_cross=random_cross,
        random_fixed_cross=_readonly(random_fixed_cross),
        random_response_cross=_readonly(random_response_cross),
        fixed_cross=_readonly(weighted_design.T @ weighted_design),
        fixed_response_cross=_readonly(weighted_design.T @ weighted_response),
        weighted_design=_readonly(weighted_design),
        weighted_response=_readonly(weighted_response),
        centered_response=_readonly(centered_response),
        logdet_weights=float(np.log(spec.weights).sum()),
        structural_c_pattern=structural_pattern,
        structural_c_nonzeros=structural_c_nonzeros,
        limits=active_limits,
    )


def _covariance_factor(
    theta: VectorInput, spec: GeneralSparseSpec
) -> sparse.csc_matrix:
    values = np.asarray(theta, dtype=np.float64)
    if values.ndim != 1 or values.shape != (spec.d,):
        raise ModelSpecificationError(
            f"theta must contain {spec.d} lower-triangular parameters"
        )
    if not np.isfinite(values).all():
        raise ModelSpecificationError("theta contains non-finite values")
    blocks: list[sparse.csc_matrix] = []
    cursor = 0
    for term in spec.terms:
        factor = np.zeros((term.k, term.k), dtype=np.float64)
        for column in range(term.k):
            width = term.k - column
            factor[column:, column] = values[cursor : cursor + width]
            cursor += width
        if (factor.diagonal() < 0.0).any():
            raise ModelSpecificationError(
                "theta diagonal parameters must be nonnegative"
            )
        blocks.append(
            sparse.kron(
                sparse.eye(term.group_count, format="csc"),
                sparse.csc_matrix(factor),
                format="csc",
            )
        )
    return sparse.block_diag(blocks, format="csc")


def _factorize(c: sparse.csc_matrix) -> SuperLU:
    try:
        return splu(
            c,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.0,
            options={"Equil": False, "SymmetricMode": True},
        )
    except (RuntimeError, MemoryError, ValueError) as error:
        raise NumericalError("general sparse factorization failed") from error


def evaluate_prepared_general_sparse(
    workspace: GeneralSparseWorkspace,
    theta: VectorInput,
    *,
    kind: ObjectiveKind = ObjectiveKind.REML,
) -> GeneralSparseResult:
    """Evaluate a covariance vector using the cached coupled sparse structure."""
    spec = workspace.spec
    covariance_factor = _covariance_factor(theta, spec)
    try:
        c = (
            sparse.eye(spec.q, format="csc")
            + covariance_factor.T @ workspace.random_cross @ covariance_factor
        ).tocsc()
        c.sum_duplicates()
        c.sort_indices()
        factor = _factorize(c)
        factor_nonzeros = int(factor.L.nnz + factor.U.nnz)
        if factor_nonzeros > workspace.limits.maximum_factor_nonzeros:
            raise ResourceLimitError("sparse factor fill exceeds configured limit")
        d = np.asarray(covariance_factor.T @ workspace.random_fixed_cross)
        f = np.asarray(covariance_factor.T @ workspace.random_response_cross)
        c_inv_d = np.asarray(factor.solve(d))
        c_inv_f = np.asarray(factor.solve(f))
        schur = workspace.fixed_cross - d.T @ c_inv_d
        target = workspace.fixed_response_cross - d.T @ c_inv_f
        chol_s: FloatArray = np.linalg.cholesky(schur)
        beta = _chol_solve(chol_s, target)
        u = np.asarray(factor.solve(f - d @ beta))
    except ModelSpecificationError:
        raise
    except (np.linalg.LinAlgError, RuntimeError, MemoryError, ValueError) as error:
        raise NumericalError("general sparse solve failed") from error

    b = np.asarray(covariance_factor @ u)
    contribution = np.asarray(workspace.random_design @ b).reshape(-1)
    residual = np.sqrt(spec.weights) * (
        workspace.centered_response - spec.x @ beta - contribution
    )
    weighted_rss = float(residual @ residual)
    penalty = float(u @ u)
    pwrss = weighted_rss + penalty
    degrees = spec.n if kind is ObjectiveKind.ML else spec.n - spec.p
    if degrees <= 0:
        raise ModelSpecificationError("REML requires n greater than fixed-design rank")
    if not np.isfinite(pwrss) or pwrss <= 0.0:
        raise NumericalError("penalized residual sum of squares must be positive")

    diagonal = np.asarray(factor.U.diagonal(), dtype=np.float64)
    if (diagonal == 0.0).any() or not np.isfinite(diagonal).all():
        raise NumericalError("general sparse factor has an invalid determinant")
    logdet_c = float(np.log(np.abs(diagonal)).sum())
    logdet_s = float(2.0 * np.log(chol_s.diagonal()).sum())
    objective = logdet_c - workspace.logdet_weights
    if kind is ObjectiveKind.REML:
        objective += logdet_s
    objective += degrees * (1.0 + np.log(2.0 * np.pi * pwrss / degrees))
    sigma2 = pwrss / degrees
    beta_covariance = sigma2 * _chol_solve(chol_s, np.eye(spec.p, dtype=np.float64))

    covariance_blocks: list[FloatArray] = []
    cursor = 0
    theta_values = np.asarray(theta, dtype=np.float64)
    for term in spec.terms:
        local = np.zeros((term.k, term.k), dtype=np.float64)
        local_cursor = cursor
        for column in range(term.k):
            width = term.k - column
            local[column:, column] = theta_values[local_cursor : local_cursor + width]
            local_cursor += width
        covariance_blocks.append(sigma2 * local @ local.T)
        cursor += term.d
    covariance_size = sum(block.shape[0] for block in covariance_blocks)
    random_covariance = np.zeros((covariance_size, covariance_size), dtype=np.float64)
    block_start = 0
    for block in covariance_blocks:
        block_stop = block_start + block.shape[0]
        random_covariance[block_start:block_stop, block_start:block_stop] = block
        block_start = block_stop

    return GeneralSparseResult(
        kind=kind,
        objective=float(objective),
        log_likelihood=float(-0.5 * objective),
        beta=_readonly(beta),
        u=_readonly(u),
        b=_readonly(b),
        sigma2=float(sigma2),
        beta_covariance=_readonly(beta_covariance),
        random_covariance=_readonly(random_covariance),
        penalized_residual_sum_squares=pwrss,
        weighted_residual_sum_squares=weighted_rss,
        random_effect_penalty=penalty,
        logdet_c=logdet_c,
        logdet_s=logdet_s,
        logdet_weights=workspace.logdet_weights,
        factor_nonzeros=factor_nonzeros,
    )


def evaluate_general_sparse(
    spec: GeneralSparseSpec,
    theta: VectorInput,
    *,
    kind: ObjectiveKind = ObjectiveKind.REML,
    limits: SparseBackendLimits | None = None,
) -> GeneralSparseResult:
    """Prepare and evaluate a coupled sparse model."""
    return evaluate_prepared_general_sparse(
        prepare_general_sparse(spec, limits=limits), theta, kind=kind
    )


__all__ = [
    "BACKEND_NAME",
    "GeneralSparseResult",
    "GeneralSparseWorkspace",
    "SparseBackendLimits",
    "evaluate_general_sparse",
    "evaluate_prepared_general_sparse",
    "prepare_general_sparse",
]
