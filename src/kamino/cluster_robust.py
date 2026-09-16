"""Cluster-robust covariance with clubSandwich-compatible CR2 inference.

The working target is the fitted marginal covariance, scaled by the residual
variance.  Scores use marginal residuals and clusters must contain every level
of every random-effects grouping factor used by the fitted model.
"""

# pyright: reportArgumentType=false, reportMissingTypeStubs=false
# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnnecessaryIsInstance=false

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve
from scipy.stats import chi2, norm
from scipy.stats import f as f_distribution
from scipy.stats import t as t_distribution

from kamino.errors import ModelSpecificationError, PostfitError, ResourceLimitError
from kamino.formula import GeneralDesign, SingleGroupDesign
from kamino.model import FloatArray, IntArray
from kamino.results import LinearMixedModelResult

ClusterCovarianceType: TypeAlias = Literal["CR0", "CR1", "CR2"]
ClusterDistribution: TypeAlias = Literal["normal", "t", "chi_square", "f"]


def _readonly_float(value: object, *, ndim: int | None = None) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ModelSpecificationError(f"value must have {ndim} dimensions")
    result.setflags(write=False)
    return result


def _readonly_int(value: object) -> IntArray:
    result = np.array(value, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ClusterRobustControl:
    """Explicit observation-space and numerical ceilings for robust inference."""

    maximum_observations: int = 100_000
    maximum_fixed_effects: int = 128
    maximum_cluster_size: int = 2_000
    maximum_cluster_matrix_elements: int = 5_000_000
    maximum_bytes: int = 512 * 1024 * 1024
    maximum_condition_number: float = 1e12
    eigenvalue_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "maximum_observations",
            "maximum_fixed_effects",
            "maximum_cluster_size",
            "maximum_cluster_matrix_elements",
            "maximum_bytes",
        ):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
                raise ModelSpecificationError(f"{name} must be a positive integer")
            if int(value) <= 0:
                raise ModelSpecificationError(f"{name} must be a positive integer")
        for name in ("maximum_condition_number", "eigenvalue_tolerance"):
            value = getattr(self, name)
            if (
                not isinstance(value, Real)
                or isinstance(value, (bool, np.bool_))
                or not np.isfinite(value)
                or float(value) <= 0.0
            ):
                raise ModelSpecificationError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class ClusterRobustTest:
    """One covariance-consistent cluster-robust linear hypothesis test."""

    covariance_type: ClusterCovarianceType
    method: str
    coefficient_names: tuple[str, ...]
    contrast: FloatArray
    rhs: FloatArray
    estimable: bool
    available: bool
    status: str
    message: str
    estimate: float | None
    standard_error: float | None
    statistic: float | None
    numerator_df: int | None
    denominator_df: float | None
    distribution: ClusterDistribution | None
    p_value: float | None
    scale: float | None
    confidence_level: float
    confidence_lower: float | None
    confidence_upper: float | None


@dataclass(frozen=True, slots=True)
class _RandomTargetTerm:
    group_indices: IntArray
    random_design: FloatArray
    relative_covariance: FloatArray


class ClusterRobustAnalysis:
    """Immutable robust covariance and cluster-specific adjustment state."""

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("ClusterRobustAnalysis is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        model: LinearMixedModelResult,
        covariance_type: ClusterCovarianceType,
        cluster_name: str | None,
        cluster_levels: tuple[str, ...],
        cluster_codes: IntArray,
        covariance: FloatArray,
        bread: FloatArray,
        marginal_residuals: FloatArray,
        working_targets: tuple[FloatArray, ...],
        weight_matrices: tuple[FloatArray, ...],
        adjustments: tuple[FloatArray, ...],
        estimating_matrices: tuple[FloatArray, ...],
        score_contributions: FloatArray,
        g_matrices: tuple[FloatArray, ...],
        h_matrices: tuple[FloatArray, ...],
    ) -> None:
        self._model = model
        self.covariance_type: ClusterCovarianceType = covariance_type
        self.cluster_name = cluster_name
        self.cluster_levels = cluster_levels
        self.cluster_codes = _readonly_int(cluster_codes)
        self.covariance = _readonly_float(covariance, ndim=2)
        self.bread = _readonly_float(bread, ndim=2)
        self.marginal_residuals = _readonly_float(marginal_residuals, ndim=1)
        self.working_targets = tuple(
            _readonly_float(value, ndim=2) for value in working_targets
        )
        self.weight_matrices = tuple(
            _readonly_float(value, ndim=2) for value in weight_matrices
        )
        self.adjustments = tuple(
            _readonly_float(value, ndim=2) for value in adjustments
        )
        self.estimating_matrices = tuple(
            _readonly_float(value, ndim=2) for value in estimating_matrices
        )
        self.score_contributions = _readonly_float(score_contributions, ndim=2)
        self._g_matrices = tuple(_readonly_float(value, ndim=2) for value in g_matrices)
        self._h_matrices = tuple(_readonly_float(value, ndim=2) for value in h_matrices)
        self._sealed = True

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        """Full, unreduced fixed-effect coefficient identity."""
        return self._model.full_fixed_names

    @property
    def retained_indices(self) -> tuple[int, ...]:
        """Full-coordinate positions corresponding to covariance rows."""
        return self._model.fixed_rank.retained_indices

    @property
    def cluster_sizes(self) -> tuple[int, ...]:
        """Observation count in first-appearance cluster order."""
        counts = np.bincount(self.cluster_codes, minlength=len(self.cluster_levels))
        return tuple(int(value) for value in counts)

    def test(
        self,
        contrast: Sequence[float] | Sequence[Sequence[float]] | FloatArray,
        *,
        rhs: float | Sequence[float] | FloatArray = 0.0,
        level: float = 0.95,
    ) -> ClusterRobustTest:
        """Test one contrast with Satterthwaite or a joint contrast with HTZ."""
        matrix = _contrast_matrix(contrast, len(self.coefficient_names))
        right = _rhs_vector(rhs, matrix.shape[0])
        _validate_level(level)
        estimable = all(self._model.fixed_rank.is_estimable(row) for row in matrix)
        if not estimable:
            return self._unavailable(
                matrix,
                right,
                level,
                "non_estimable",
                "a contrast row is not estimable under the coefficient rank",
                estimable=False,
            )
        reduced = matrix[:, list(self.retained_indices)]
        effect = reduced @ self._model.beta - right
        hypothesis_covariance = reduced @ self.covariance @ reduced.T
        try:
            chol = np.linalg.cholesky(hypothesis_covariance)
            solved = cho_solve((chol, True), effect)
        except np.linalg.LinAlgError:
            return self._unavailable(
                matrix,
                right,
                level,
                "invalid_variance",
                "contrast covariance is not positive definite",
                estimable=True,
            )
        q = matrix.shape[0]
        quadratic = float(effect @ solved)
        if q == 1:
            standard_error = float(np.sqrt(hypothesis_covariance[0, 0]))
            statistic = float(effect[0] / standard_error)
            estimate = float((reduced @ self._model.beta)[0])
            if self.covariance_type == "CR2":
                degrees = self._satterthwaite_df(reduced[0])
                if not np.isfinite(degrees) or degrees <= 0.0:
                    return self._unavailable(
                        matrix,
                        right,
                        level,
                        "invalid_degrees_of_freedom",
                        "CR2 Satterthwaite degrees of freedom are unavailable",
                        estimable=True,
                    )
                probability = float(2.0 * t_distribution.sf(abs(statistic), degrees))
                critical = float(t_distribution.ppf(0.5 + level / 2.0, degrees))
                distribution: ClusterDistribution = "t"
                method = "CR2 Satterthwaite"
            else:
                degrees = None
                probability = float(2.0 * norm.sf(abs(statistic)))
                critical = float(norm.ppf(0.5 + level / 2.0))
                distribution = "normal"
                method = f"{self.covariance_type} asymptotic"
            return ClusterRobustTest(
                covariance_type=self.covariance_type,
                method=method,
                coefficient_names=self.coefficient_names,
                contrast=_readonly_float(matrix),
                rhs=_readonly_float(right),
                estimable=True,
                available=True,
                status="ok",
                message=f"{method} contrast available",
                estimate=estimate,
                standard_error=standard_error,
                statistic=statistic,
                numerator_df=1,
                denominator_df=degrees,
                distribution=distribution,
                p_value=probability,
                scale=1.0,
                confidence_level=level,
                confidence_lower=estimate - critical * standard_error,
                confidence_upper=estimate + critical * standard_error,
            )
        if self.covariance_type == "CR2":
            scale, degrees = self._htz_state(reduced)
            if not np.isfinite(degrees) or degrees <= 0.0 or scale <= 0.0:
                return self._unavailable(
                    matrix,
                    right,
                    level,
                    "invalid_degrees_of_freedom",
                    "CR2 HTZ degrees of freedom are unavailable",
                    estimable=True,
                )
            statistic = float(scale * quadratic / q)
            probability = float(f_distribution.sf(statistic, q, degrees))
            distribution = "f"
            method = "CR2 HTZ"
        else:
            scale = 1.0
            degrees = None
            statistic = quadratic
            probability = float(chi2.sf(quadratic, q))
            distribution = "chi_square"
            method = f"{self.covariance_type} asymptotic"
        return ClusterRobustTest(
            covariance_type=self.covariance_type,
            method=method,
            coefficient_names=self.coefficient_names,
            contrast=_readonly_float(matrix),
            rhs=_readonly_float(right),
            estimable=True,
            available=True,
            status="ok",
            message=f"{method} joint contrast available",
            estimate=None,
            standard_error=None,
            statistic=statistic,
            numerator_df=q,
            denominator_df=degrees,
            distribution=distribution,
            p_value=probability,
            scale=scale,
            confidence_level=level,
            confidence_lower=None,
            confidence_upper=None,
        )

    def _p_array(self, contrast: FloatArray) -> FloatArray:
        q = contrast.shape[0]
        clusters = len(self.cluster_levels)
        result = np.empty((q, q, clusters, clusters), dtype=np.float64)
        g_values = tuple(contrast @ value for value in self._g_matrices)
        h_values = tuple(contrast @ value for value in self._h_matrices)
        for left in range(q):
            for right in range(q):
                block = np.empty((clusters, clusters), dtype=np.float64)
                for row in range(clusters):
                    for column in range(clusters):
                        block[row, column] = -float(
                            h_values[row][left] @ h_values[column][right]
                        )
                    block[row, row] += float(g_values[row][left] @ g_values[row][right])
                result[left, right] = block
        return result

    def _satterthwaite_df(self, contrast: FloatArray) -> float:
        p_array = self._p_array(contrast.reshape(1, -1))[0, 0]
        expected = float(np.trace(p_array))
        variance = float(np.sum(np.square(p_array)))
        return expected * expected / variance if variance > 0.0 else np.nan

    def _htz_state(self, contrast: FloatArray) -> tuple[float, float]:
        q = contrast.shape[0]
        p_array = self._p_array(contrast)
        omega = np.empty((q, q), dtype=np.float64)
        for left in range(q):
            for right in range(q):
                omega[left, right] = float(np.trace(p_array[left, right]))
        omega_inverse_half = _symmetric_power(omega, -0.5, 1e-12)
        transformed = np.empty_like(p_array)
        for row in range(p_array.shape[2]):
            for column in range(p_array.shape[3]):
                transformed[:, :, row, column] = (
                    omega_inverse_half @ p_array[:, :, row, column] @ omega_inverse_half
                )
        variance = np.empty((q, q), dtype=np.float64)
        for left in range(q):
            for right in range(q):
                variance[left, right] = float(
                    np.sum(transformed[left, right] * transformed[right, left])
                    + np.sum(transformed[left, left] * transformed[right, right])
                )
        denominator = float(np.sum(variance))
        if denominator <= 0.0:
            return np.nan, np.nan
        nu = q * (q + 1.0) / denominator
        scale = max((nu - q + 1.0) / nu, 0.0)
        return float(scale), float(nu - q + 1.0)

    def _unavailable(
        self,
        contrast: FloatArray,
        rhs: FloatArray,
        level: float,
        status: str,
        message: str,
        *,
        estimable: bool,
    ) -> ClusterRobustTest:
        return ClusterRobustTest(
            covariance_type=self.covariance_type,
            method=(
                "CR2 Satterthwaite/HTZ"
                if self.covariance_type == "CR2"
                else f"{self.covariance_type} asymptotic"
            ),
            coefficient_names=self.coefficient_names,
            contrast=_readonly_float(contrast),
            rhs=_readonly_float(rhs),
            estimable=estimable,
            available=False,
            status=status,
            message=message,
            estimate=None,
            standard_error=None,
            statistic=None,
            numerator_df=None,
            denominator_df=None,
            distribution=None,
            p_value=None,
            scale=None,
            confidence_level=level,
            confidence_lower=None,
            confidence_upper=None,
        )


def _contrast_matrix(value: object, columns: int) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelSpecificationError("contrast must be numeric") from error
    if result.ndim == 1:
        result = result.reshape(1, -1)
    if (
        result.ndim != 2
        or result.shape[0] == 0
        or result.shape[1] != columns
        or not np.isfinite(result).all()
    ):
        raise ModelSpecificationError(
            "contrast must be a finite vector or matrix in the full coefficient space"
        )
    if np.linalg.matrix_rank(result) != result.shape[0]:
        raise ModelSpecificationError(
            "joint contrast rows must be linearly independent"
        )
    return np.asarray(result, dtype=np.float64)


def _rhs_vector(value: object, rows: int) -> FloatArray:
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        result = np.repeat(float(value), rows)
    else:
        try:
            result = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ModelSpecificationError(
                "rhs must be finite with one value per row"
            ) from error
    if result.shape != (rows,) or not np.isfinite(result).all():
        raise ModelSpecificationError("rhs must be finite with one value per row")
    return result


def _validate_level(value: float) -> None:
    if (
        not isinstance(value, Real)
        or isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
        or not 0.0 < float(value) < 1.0
    ):
        raise ModelSpecificationError("confidence level must be between zero and one")


def _symmetric_power(matrix: FloatArray, power: float, tolerance: float) -> FloatArray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    scale = max(1.0, float(np.max(np.abs(values))))
    if float(values[0]) < -tolerance * scale:
        raise PostfitError("cluster adjustment matrix is not positive semidefinite")
    powered = np.zeros_like(values)
    keep = values > tolerance
    powered[keep] = np.power(values[keep], power)
    return np.asarray((vectors * powered) @ vectors.T, dtype=np.float64)


def _model_terms(model: LinearMixedModelResult) -> tuple[_RandomTargetTerm, ...]:
    design = model._training_design
    relative = np.asarray(model.random_covariance, dtype=np.float64) / model.sigma2
    if isinstance(design, SingleGroupDesign):
        return (
            _RandomTargetTerm(
                design.spec.group_indices,
                design.spec.random_design,
                relative,
            ),
        )
    if not isinstance(design, GeneralDesign):  # pragma: no cover - internal invariant
        raise PostfitError("unsupported retained model design")
    terms: list[_RandomTargetTerm] = []
    cursor = 0
    for term in design.spec.terms:
        stop = cursor + term.k
        terms.append(
            _RandomTargetTerm(
                term.group_indices,
                term.random_design,
                relative[cursor:stop, cursor:stop],
            )
        )
        cursor = stop
    return tuple(terms)


def _default_cluster(model: LinearMixedModelResult) -> tuple[str, tuple[str, ...]]:
    if model.random_terms:
        term = min(model.random_terms, key=lambda value: len(value.group_levels))
        return term.group_name, term.training_groups
    return model.group_name, model._training_groups


def _named_cluster(
    model: LinearMixedModelResult, name: str
) -> tuple[str, tuple[str, ...]]:
    if model.random_terms:
        matches = tuple(term for term in model.random_terms if term.group_name == name)
        if len(matches) != 1:
            raise ModelSpecificationError(
                f"cluster name {name!r} is not one fitted grouping factor"
            )
        return name, matches[0].training_groups
    if name != model.group_name:
        raise ModelSpecificationError(
            f"cluster name {name!r} is not the fitted grouping factor "
            f"{model.group_name!r}"
        )
    return name, model._training_groups


def _cluster_values(
    model: LinearMixedModelResult, cluster: object | None
) -> tuple[str | None, tuple[str, ...]]:
    if cluster is None:
        return _default_cluster(model)
    if isinstance(cluster, str):
        return _named_cluster(model, cluster)
    if isinstance(cluster, pd.Series):
        index = tuple(str(value) for value in cluster.index)
        if len(set(index)) != len(index):
            raise ModelSpecificationError(
                "cluster Series row identifiers must be unique"
            )
        positions = {value: position for position, value in enumerate(index)}
        if any(row not in positions for row in model.row_ids):
            raise ModelSpecificationError(
                "cluster Series must contain every retained model row identifier"
            )
        raw = tuple(cluster.iloc[positions[row]] for row in model.row_ids)
    else:
        if not isinstance(cluster, Sequence) and not isinstance(cluster, np.ndarray):
            raise ModelSpecificationError(
                "cluster must be a fitted grouping name or a row-aligned sequence"
            )
        raw = tuple(cast(Sequence[object], cluster))
        if len(raw) != len(model.row_ids):
            raise ModelSpecificationError(
                "cluster must have one value per retained model row"
            )
    if any(not isinstance(value, str) or not value for value in raw):
        raise ModelSpecificationError(
            "cluster labels must be nonempty strings without missing values"
        )
    return None, cast(tuple[str, ...], raw)


def _factorize(values: tuple[str, ...]) -> tuple[tuple[str, ...], IntArray]:
    levels: list[str] = []
    mapping: dict[str, int] = {}
    codes = np.empty(len(values), dtype=np.int64)
    for index, value in enumerate(values):
        if value not in mapping:
            mapping[value] = len(levels)
            levels.append(value)
        codes[index] = mapping[value]
    if len(levels) < 2:
        raise ModelSpecificationError(
            "cluster-robust inference requires at least two independent clusters"
        )
    return tuple(levels), codes


def _validate_nested(terms: tuple[_RandomTargetTerm, ...], codes: IntArray) -> None:
    for term_index, term in enumerate(terms):
        seen: dict[int, int] = {}
        for group, cluster in zip(term.group_indices, codes, strict=True):
            group_value = int(group)
            cluster_value = int(cluster)
            previous = seen.setdefault(group_value, cluster_value)
            if previous != cluster_value:
                raise ModelSpecificationError(
                    "every random-effect grouping factor must be nested in the "
                    f"independent cluster variable; term {term_index + 1} is not nested"
                )


def _target_block(
    indices: IntArray, terms: tuple[_RandomTargetTerm, ...]
) -> FloatArray:
    size = len(indices)
    target = np.eye(size, dtype=np.float64)
    for term in terms:
        local_groups = term.group_indices[indices]
        local_design = term.random_design[indices]
        for group in np.unique(local_groups):
            positions = np.flatnonzero(local_groups == group)
            z = local_design[positions]
            target[np.ix_(positions, positions)] += z @ term.relative_covariance @ z.T
    return target


def cluster_robust(
    model: LinearMixedModelResult,
    cluster: object | None = None,
    *,
    covariance_type: ClusterCovarianceType = "CR2",
    control: ClusterRobustControl | None = None,
) -> ClusterRobustAnalysis:
    """Build CR0, CR1, or mixed-model CR2 inference for independent clusters."""
    if not isinstance(model, LinearMixedModelResult):
        raise ModelSpecificationError("cluster_robust requires a live fitted result")
    if covariance_type not in ("CR0", "CR1", "CR2"):
        raise ModelSpecificationError("covariance_type must be CR0, CR1, or CR2")
    if control is not None and not isinstance(control, ClusterRobustControl):
        raise ModelSpecificationError("control must be a ClusterRobustControl")
    limits = control or ClusterRobustControl()
    if model.has_prior_weights:
        raise PostfitError(
            "cluster-robust inference does not support fits with prior weights"
        )
    x = np.asarray(model._training_fixed_design, dtype=np.float64)
    design = model._training_design
    y = np.asarray(design.spec.y, dtype=np.float64)
    offset = np.asarray(design.spec.offset, dtype=np.float64)
    n, p = x.shape
    if n > limits.maximum_observations or p > limits.maximum_fixed_effects:
        raise ResourceLimitError("cluster-robust design exceeds its resource limit")
    cluster_name, values = _cluster_values(model, cluster)
    levels, codes = _factorize(values)
    terms = _model_terms(model)
    _validate_nested(terms, codes)
    row_blocks = tuple(
        np.flatnonzero(codes == index).astype(np.int64) for index in range(len(levels))
    )
    sizes = tuple(len(value) for value in row_blocks)
    elements = sum(value * value for value in sizes)
    estimated_bytes = 8 * (8 * elements + 6 * n * p + 4 * p * p)
    if (
        max(sizes) > limits.maximum_cluster_size
        or elements > limits.maximum_cluster_matrix_elements
        or estimated_bytes > limits.maximum_bytes
    ):
        raise ResourceLimitError(
            "cluster-robust observation-space matrices exceed the configured limit"
        )

    targets: list[FloatArray] = []
    weights: list[FloatArray] = []
    x_blocks: list[FloatArray] = []
    xw_blocks: list[FloatArray] = []
    normal = np.zeros((p, p), dtype=np.float64)
    for indices in row_blocks:
        target = _target_block(indices, terms)
        try:
            chol = np.linalg.cholesky(target)
            weight = cho_solve((chol, True), np.eye(len(indices), dtype=np.float64))
        except np.linalg.LinAlgError as error:  # pragma: no cover - I + PSD invariant
            raise PostfitError(
                "working covariance target is not positive definite"
            ) from error
        block = x[indices]
        xw = block.T @ weight
        targets.append(target)
        weights.append(weight)
        x_blocks.append(block)
        xw_blocks.append(xw)
        normal += xw @ block
    condition = float(np.linalg.cond(normal))
    if not np.isfinite(condition) or condition > limits.maximum_condition_number:
        raise PostfitError("cluster-robust bread is ill-conditioned")
    try:
        normal_chol = np.linalg.cholesky(normal)
        bread = cho_solve((normal_chol, True), np.eye(p, dtype=np.float64))
    except np.linalg.LinAlgError as error:
        raise PostfitError("cluster-robust bread is not positive definite") from error

    adjustments: list[FloatArray] = []
    if covariance_type == "CR0":
        adjustments = [np.eye(size, dtype=np.float64) for size in sizes]
    elif covariance_type == "CR1":
        multiplier = np.sqrt(len(levels) / (len(levels) - 1.0))
        adjustments = [multiplier * np.eye(size, dtype=np.float64) for size in sizes]
    else:
        for target, _weight, block, xw in zip(
            targets, weights, x_blocks, xw_blocks, strict=True
        ):
            target_chol_upper = np.linalg.cholesky(target).T
            ih = np.eye(len(block), dtype=np.float64) - block @ bread @ xw
            adjustment_target = target_chol_upper @ ih @ target @ target_chol_upper.T
            inverse_half = _symmetric_power(
                adjustment_target, -0.5, limits.eigenvalue_tolerance
            )
            adjustments.append(target_chol_upper.T @ inverse_half @ target_chol_upper)

    residuals = y - offset - x @ np.asarray(model.beta)
    estimating: list[FloatArray] = []
    scores: list[FloatArray] = []
    g_matrices: list[FloatArray] = []
    h_matrices: list[FloatArray] = []
    bread_chol = np.linalg.cholesky(bread)
    for indices, target, block, xw, adjustment in zip(
        row_blocks, targets, x_blocks, xw_blocks, adjustments, strict=True
    ):
        estimate_matrix = xw @ adjustment
        estimating.append(estimate_matrix)
        scores.append(estimate_matrix @ residuals[indices])
        if covariance_type == "CR2":
            me = bread @ estimate_matrix
            g_matrices.append(me @ np.linalg.cholesky(target))
            h_matrices.append(me @ block @ bread_chol)
    score_matrix = np.column_stack(scores)
    covariance = bread @ (score_matrix @ score_matrix.T) @ bread
    covariance = (covariance + covariance.T) / 2.0
    return ClusterRobustAnalysis(
        model=model,
        covariance_type=covariance_type,
        cluster_name=cluster_name,
        cluster_levels=levels,
        cluster_codes=codes,
        covariance=covariance,
        bread=bread,
        marginal_residuals=residuals,
        working_targets=tuple(targets),
        weight_matrices=tuple(weights),
        adjustments=tuple(adjustments),
        estimating_matrices=tuple(estimating),
        score_contributions=score_matrix,
        g_matrices=tuple(g_matrices),
        h_matrices=tuple(h_matrices),
    )


__all__ = [
    "ClusterCovarianceType",
    "ClusterRobustAnalysis",
    "ClusterRobustControl",
    "ClusterRobustTest",
    "cluster_robust",
]
