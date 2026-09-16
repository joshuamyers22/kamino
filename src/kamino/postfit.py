"""Validated post-estimation capabilities for Kamino and explicit adapters.

The public factories in this module validate concrete result classes.  They do
not use attribute-based duck typing: coefficient identity, covariance identity,
estimability, design reconstruction, and degrees-of-freedom semantics are part
of the adapter contract.
"""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportMissingTypeStubs=false, reportOptionalMemberAccess=false
# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations, product
from numbers import Real
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from scipy.stats import f as f_distribution
from scipy.stats import t as t_distribution

from kamino.errors import ModelSpecificationError, PostfitError
from kamino.model import FloatArray
from kamino.results import LinearMixedModelResult

AdapterSource: TypeAlias = Literal[
    "kamino",
    "statsmodels_ols",
    "statsmodels_wls",
    "statsmodels_mixedlm",
]
InferenceMethod: TypeAlias = Literal[
    "asymptotic",
    "residual_t",
    "satterthwaite",
    "kenward_roger",
]
Distribution: TypeAlias = Literal["normal", "t", "chi_square", "f"]
WeightMethod: TypeAlias = Literal[
    "equal", "proportional", "outer", "cells", "flat", "user"
]
AdjustmentMethod: TypeAlias = Literal["none", "holm", "bonferroni", "sidak"]

POSTFIT_CONTRACT_VERSION = "1.0.0"


def _readonly(value: object, *, ndim: int | None = None) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ModelSpecificationError(f"value must have {ndim} dimensions")
    result.setflags(write=False)
    return result


def _string_tuple(values: Sequence[object], name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or len(set(result)) != len(result):
        raise PostfitError(f"{name} must be nonempty and unique")
    return result


@dataclass(frozen=True, slots=True)
class LinearFunctionBasis:
    """Immutable coefficient, covariance, rank, and inference identity."""

    contract_version: str
    source: AdapterSource
    full_coefficient_names: tuple[str, ...]
    retained_indices: tuple[int, ...]
    coefficients: FloatArray
    covariance: FloatArray
    null_basis: FloatArray
    covariance_kind: str
    inference_method: InferenceMethod
    denominator_df: float | None

    def __post_init__(self) -> None:
        if self.contract_version != POSTFIT_CONTRACT_VERSION:
            raise PostfitError("unsupported postfit contract version")
        if self.source not in (
            "kamino",
            "statsmodels_ols",
            "statsmodels_wls",
            "statsmodels_mixedlm",
        ):
            raise PostfitError("unsupported postfit adapter source")
        if self.inference_method not in (
            "asymptotic",
            "residual_t",
            "satterthwaite",
            "kenward_roger",
        ):
            raise PostfitError("unsupported postfit inference method")
        names = _string_tuple(self.full_coefficient_names, "coefficient names")
        retained = tuple(self.retained_indices)
        if (
            len(set(retained)) != len(retained)
            or any(type(index) is not int for index in retained)
            or any(index < 0 or index >= len(names) for index in retained)
        ):
            raise PostfitError("retained coefficient indices are invalid")
        coefficients = _readonly(self.coefficients, ndim=1)
        covariance = _readonly(self.covariance, ndim=2)
        null_basis = _readonly(self.null_basis, ndim=2)
        p = len(retained)
        if coefficients.shape != (p,) or covariance.shape != (p, p):
            raise PostfitError("coefficient and covariance shapes do not align")
        if null_basis.shape[0] != len(names):
            raise PostfitError("null basis does not align with full coefficients")
        if not np.isfinite(null_basis).all():
            raise PostfitError("null basis must be finite")
        if null_basis.shape[1] and not np.allclose(
            null_basis.T @ null_basis,
            np.eye(null_basis.shape[1]),
            atol=1e-10,
            rtol=1e-10,
        ):
            raise PostfitError("null basis must be orthonormal")
        if not np.isfinite(coefficients).all() or not np.isfinite(covariance).all():
            raise PostfitError("coefficient state must be finite")
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10):
            raise PostfitError("coefficient covariance must be symmetric")
        minimum = float(np.linalg.eigvalsh(covariance)[0]) if p else 0.0
        scale = max(1.0, float(np.max(np.abs(covariance))) if p else 1.0)
        if minimum < -1e-10 * scale:
            raise PostfitError("coefficient covariance is not positive semidefinite")
        if not self.covariance_kind:
            raise PostfitError("covariance kind must be explicit")
        if self.inference_method == "residual_t":
            if self.covariance_kind != "model_based":
                raise PostfitError(
                    "residual degrees of freedom require model-based covariance"
                )
            if (
                self.denominator_df is None
                or not np.isfinite(self.denominator_df)
                or self.denominator_df <= 0.0
            ):
                raise PostfitError(
                    "residual inference requires positive degrees of freedom"
                )
        elif self.denominator_df is not None:
            raise PostfitError(
                "only residual-t inference stores a basis-wide denominator DF"
            )
        object.__setattr__(self, "full_coefficient_names", names)
        object.__setattr__(self, "retained_indices", retained)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "null_basis", null_basis)

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        """Names corresponding exactly to the retained coefficient vector."""
        return tuple(
            self.full_coefficient_names[index] for index in self.retained_indices
        )

    @property
    def full_coefficients(self) -> FloatArray:
        """Full-coordinate coefficients, with dropped entries represented by NaN."""
        result = np.full(len(self.full_coefficient_names), np.nan, dtype=np.float64)
        result[list(self.retained_indices)] = self.coefficients
        result.setflags(write=False)
        return result

    def is_estimable(self, contrast: Sequence[float] | FloatArray) -> bool:
        """Return whether a full-coordinate linear function is identified."""
        values = _contrast_vector(contrast, len(self.full_coefficient_names))
        if self.null_basis.shape[1] == 0:
            return True
        discrepancy = self.null_basis.T @ values
        scale = max(1.0, float(np.linalg.norm(values)))
        return bool(np.max(np.abs(discrepancy)) <= np.sqrt(np.finfo(float).eps) * scale)


@dataclass(frozen=True, slots=True)
class ContrastResult:
    """One validated one- or multi-DF linear hypothesis result."""

    label: str
    coefficient_names: tuple[str, ...]
    contrast: FloatArray
    rhs: FloatArray
    constant: FloatArray
    estimable: bool
    available: bool
    status: str
    message: str
    estimate: float | None
    standard_error: float | None
    statistic: float | None
    numerator_df: int | None
    denominator_df: float | None
    distribution: Distribution | None
    p_value: float | None
    adjusted_p_value: float | None
    adjustment: AdjustmentMethod
    confidence_level: float
    confidence_lower: float | None
    confidence_upper: float | None
    covariance_kind: str
    inference_method: InferenceMethod


@dataclass(frozen=True, slots=True)
class GridEstimate:
    """One coefficient-aligned estimated marginal mean."""

    levels: tuple[tuple[str, object], ...]
    linear_function: FloatArray
    offset: float
    grid_weights: FloatArray
    result: ContrastResult


@dataclass(frozen=True, slots=True)
class ReferenceGridResult:
    """Estimated marginal means over one explicit reference grid."""

    specs: tuple[str, ...]
    by: tuple[str, ...]
    weights: WeightMethod
    grid_columns: tuple[str, ...]
    grid_rows: tuple[tuple[object, ...], ...]
    estimates: tuple[GridEstimate, ...]
    _analysis: PostfitAnalysis = field(repr=False, compare=False)

    def pairwise(
        self, *, adjustment: AdjustmentMethod = "holm"
    ) -> tuple[ContrastResult, ...]:
        """Compare every pair of specifications within each by-group."""
        _validate_adjustment(adjustment)
        grouped: dict[tuple[object, ...], list[GridEstimate]] = {}
        for estimate in self.estimates:
            values = dict(estimate.levels)
            key = tuple(values[name] for name in self.by)
            grouped.setdefault(key, []).append(estimate)
        results: list[ContrastResult] = []
        for estimates in grouped.values():
            for left, right in combinations(estimates, 2):
                left_levels = dict(left.levels)
                right_levels = dict(right.levels)
                left_label = ", ".join(
                    f"{name}={left_levels[name]}" for name in self.specs
                )
                right_label = ", ".join(
                    f"{name}={right_levels[name]}" for name in self.specs
                )
                results.append(
                    self._analysis.contrast(
                        left.linear_function - right.linear_function,
                        constant=left.offset - right.offset,
                        label=f"{left_label} - {right_label}",
                    )
                )
        return _adjust_results(tuple(results), adjustment)


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    """Named, explicitly scoped model-performance quantities."""

    source: AdapterSource
    averaging_measure: str
    metrics: tuple[tuple[str, float], ...]

    def get(self, name: str) -> float:
        for key, value in self.metrics:
            if key == name:
                return value
        raise ModelSpecificationError(f"performance metric {name!r} is unavailable")


@dataclass(frozen=True, slots=True)
class _Variable:
    name: str
    categorical: bool
    levels: tuple[object, ...]


class PostfitAnalysis:
    """A validated result bound to one covariance and DF contract."""

    def __init__(
        self,
        basis: LinearFunctionBasis,
        *,
        model: object,
        variables: tuple[_Variable, ...],
        training_data: pd.DataFrame | None,
        inference_state: object | None = None,
    ) -> None:
        self.basis = basis
        self._model = model
        self._variables = variables
        self._training_data = training_data
        self._inference_state = inference_state

    def _design(self, data: pd.DataFrame) -> FloatArray:
        if self.basis.source == "kamino":
            model = cast(LinearMixedModelResult, self._model)
            result = model.fixed_encoder.evaluate(data, len(data))
        else:
            from patsy import build_design_matrices

            raw = cast(Any, self._model)
            design_info = raw.model.data.design_info
            try:
                matrix = build_design_matrices(
                    [design_info], data, return_type="dataframe"
                )[0]
            except Exception as error:
                raise ModelSpecificationError(
                    f"new data cannot be encoded by the statsmodels formula: {error}"
                ) from error
            names = tuple(str(value) for value in matrix.columns)
            if names != self.basis.full_coefficient_names:
                raise PostfitError(
                    "statsmodels rebuilt design changed coefficient order"
                )
            result = np.asarray(matrix, dtype=np.float64)
        if result.shape != (len(data), len(self.basis.full_coefficient_names)):
            raise PostfitError(
                "rebuilt design does not align with coefficient identity"
            )
        if not np.isfinite(result).all():
            raise ModelSpecificationError("rebuilt design contains non-finite values")
        return result

    def contrast(
        self,
        contrast: Sequence[float] | Sequence[Sequence[float]] | FloatArray,
        *,
        rhs: float | Sequence[float] | FloatArray = 0.0,
        constant: float | Sequence[float] | FloatArray = 0.0,
        label: str = "contrast",
        level: float = 0.95,
    ) -> ContrastResult:
        """Evaluate a hypothesis using the analysis-bound covariance/DF method."""
        matrix = _contrast_matrix(contrast, len(self.basis.full_coefficient_names))
        right = _row_values(rhs, matrix.shape[0], "rhs")
        constants = _row_values(constant, matrix.shape[0], "constant")
        _validate_level(level)
        estimable = all(self.basis.is_estimable(row) for row in matrix)
        if not estimable:
            return _unavailable_contrast(
                self.basis,
                matrix,
                right,
                constants,
                label,
                level,
                "non_estimable",
                "a contrast row is not estimable under the coefficient rank",
                estimable=False,
            )
        if self.basis.source == "kamino" and self.basis.inference_method in (
            "satterthwaite",
            "kenward_roger",
        ):
            return self._kamino_contrast(
                matrix, right, constants, label=label, level=level
            )
        return _wald_contrast(
            self.basis,
            matrix,
            right,
            constants,
            label=label,
            level=level,
        )

    def _kamino_contrast(
        self,
        matrix: FloatArray,
        rhs: FloatArray,
        constants: FloatArray,
        *,
        label: str,
        level: float,
    ) -> ContrastResult:
        adjusted_rhs = rhs - constants
        method = self.basis.inference_method
        state = cast(Any, self._inference_state)
        if method == "satterthwaite":
            if matrix.shape[0] == 1:
                result = state.test(matrix[0], rhs=float(adjusted_rhs[0]))
                if not result.available:
                    return _unavailable_contrast(
                        self.basis,
                        matrix,
                        rhs,
                        constants,
                        label,
                        level,
                        result.status,
                        result.message,
                        estimable=result.estimable,
                    )
                assert result.estimate is not None
                assert result.standard_error is not None
                assert result.statistic is not None
                assert result.denominator_df is not None
                assert result.p_value is not None
                estimate = float(result.estimate + constants[0])
                critical = float(
                    t_distribution.ppf(0.5 + level / 2.0, result.denominator_df)
                )
                return _available_contrast(
                    self.basis,
                    matrix,
                    rhs,
                    constants,
                    label,
                    level,
                    estimate=estimate,
                    standard_error=float(result.standard_error),
                    statistic=float(result.statistic),
                    numerator_df=1,
                    denominator_df=float(result.denominator_df),
                    distribution="t",
                    p_value=float(result.p_value),
                    confidence_lower=estimate - critical * result.standard_error,
                    confidence_upper=estimate + critical * result.standard_error,
                )
            joint = state.joint_test(matrix, rhs=adjusted_rhs)
            if not joint.available:
                return _unavailable_contrast(
                    self.basis,
                    matrix,
                    rhs,
                    constants,
                    label,
                    level,
                    joint.status,
                    joint.message,
                    estimable=joint.estimable,
                )
            return _available_contrast(
                self.basis,
                matrix,
                rhs,
                constants,
                label,
                level,
                estimate=None,
                standard_error=None,
                statistic=cast(float, joint.statistic),
                numerator_df=cast(int, joint.numerator_df),
                denominator_df=cast(float, joint.denominator_df),
                distribution="f",
                p_value=cast(float, joint.p_value),
            )

        test = state.test(matrix, rhs=adjusted_rhs)
        if not test.available:
            return _unavailable_contrast(
                self.basis,
                matrix,
                rhs,
                constants,
                label,
                level,
                test.status,
                test.message,
                estimable=test.estimable,
            )
        retained = matrix[:, list(self.basis.retained_indices)]
        raw = retained @ self.basis.coefficients + constants
        estimate = float(raw[0]) if matrix.shape[0] == 1 else None
        standard_error = None
        if matrix.shape[0] == 1:
            variance = float(retained[0] @ self.basis.covariance @ retained[0])
            standard_error = float(np.sqrt(max(0.0, variance)))
        return _available_contrast(
            self.basis,
            matrix,
            rhs,
            constants,
            label,
            level,
            estimate=estimate,
            standard_error=standard_error,
            statistic=cast(float, test.statistic),
            numerator_df=cast(int, test.numerator_df),
            denominator_df=cast(float, test.denominator_df),
            distribution="f",
            p_value=cast(float, test.p_value),
        )

    def tidy(self, *, level: float = 0.95) -> tuple[ContrastResult, ...]:
        """Return coefficient-aligned inference without fabricating unavailable rows."""
        _validate_level(level)
        width = len(self.basis.full_coefficient_names)
        results = []
        for index, name in enumerate(self.basis.full_coefficient_names):
            contrast = np.zeros(width, dtype=np.float64)
            contrast[index] = 1.0
            results.append(self.contrast(contrast, label=name, level=level))
        return tuple(results)

    def reference_grid(
        self,
        *,
        specs: Sequence[str],
        by: Sequence[str] = (),
        data: pd.DataFrame | None = None,
        at: Mapping[str, Sequence[object]] | None = None,
        weights: WeightMethod | Sequence[float] = "equal",
        offset: float | Sequence[float] | FloatArray | None = None,
        level: float = 0.95,
    ) -> ReferenceGridResult:
        """Build and evaluate a bounded emmeans-style reference grid."""
        _validate_level(level)
        source_data = self._training_data if data is None else data
        if source_data is None:
            raise ModelSpecificationError(
                "reference grids require the original model-frame variables"
            )
        if not isinstance(source_data, pd.DataFrame) or len(source_data) == 0:
            raise ModelSpecificationError(
                "reference-grid data must be a nonempty DataFrame"
            )
        spec_names = _requested_names(specs, "specs")
        by_names = _requested_names(by, "by", allow_empty=True)
        if set(spec_names) & set(by_names):
            raise ModelSpecificationError("specs and by variables must be disjoint")
        variable_names = tuple(variable.name for variable in self._variables)
        requested = spec_names + by_names
        unknown = [name for name in requested if name not in variable_names]
        if unknown:
            raise ModelSpecificationError(
                f"reference-grid variable {unknown[0]!r} is not a fixed predictor"
            )
        at_values = (
            {} if at is None else {str(key): tuple(value) for key, value in at.items()}
        )
        formula_offsets: tuple[str, ...] = ()
        requires_explicit = False
        if self.basis.source == "kamino":
            model = cast(LinearMixedModelResult, self._model)
            formula_offsets = model.formula_offset_names
            requires_explicit = model.requires_explicit_offset
        allowed_at = set(variable_names + formula_offsets)
        invalid_at = [name for name in at_values if name not in allowed_at]
        if invalid_at:
            raise ModelSpecificationError(
                f"at variable {invalid_at[0]!r} is not a fixed predictor or offset"
            )
        columns: list[str] = []
        values: list[tuple[object, ...]] = []
        for variable in self._variables:
            columns.append(variable.name)
            if variable.name in at_values:
                selected = at_values[variable.name]
                if not selected:
                    raise ModelSpecificationError("at values must be nonempty")
            elif variable.categorical:
                selected = variable.levels
            else:
                selected = (float(pd.to_numeric(source_data[variable.name]).mean()),)
            values.append(tuple(selected))
        if self.basis.source == "kamino":
            for name in formula_offsets:
                if name not in source_data:
                    raise ModelSpecificationError(
                        f"reference-grid data is missing offset column {name!r}"
                    )
                columns.append(name)
                chosen = at_values.get(
                    name, (float(pd.to_numeric(source_data[name]).mean()),)
                )
                values.append(tuple(chosen))
        if requires_explicit and offset is None:
            raise ModelSpecificationError(
                "this fit requires an explicit reference-grid offset"
            )
        rows = tuple(product(*values))
        if not rows:
            raise ModelSpecificationError("reference grid is empty")
        if len(rows) > 100_000:
            raise ModelSpecificationError("reference grid exceeds 100000 rows")
        grid = pd.DataFrame(rows, columns=columns)
        design = self._design(grid)
        offsets = np.zeros(len(grid), dtype=np.float64)
        for name in formula_offsets:
            offsets += np.asarray(grid[name], dtype=np.float64)
        if offset is not None:
            offsets += _grid_offset(offset, len(grid))
        method: WeightMethod
        user_weights: FloatArray | None
        if isinstance(weights, str):
            if weights not in ("equal", "proportional", "outer", "cells", "flat"):
                raise ModelSpecificationError(
                    "unsupported reference-grid weight method"
                )
            method = cast(WeightMethod, weights)
            user_weights = None
        else:
            method = "user"
            user_weights = _readonly(weights, ndim=1)
            if user_weights.shape != (len(grid),):
                raise ModelSpecificationError(
                    "user weights must provide one value per reference-grid row"
                )
            if not np.isfinite(user_weights).all() or (user_weights < 0.0).any():
                raise ModelSpecificationError(
                    "user weights must be finite and nonnegative"
                )
        estimates: list[GridEstimate] = []
        groups = _grid_groups(grid, spec_names + by_names)
        for key, indices in groups:
            raw_weights = _reference_weights(
                method,
                grid,
                indices,
                source_data,
                self._variables,
                spec_names,
                by_names,
                user_weights,
            )
            total = float(np.sum(raw_weights))
            if not np.isfinite(total) or total <= 0.0:
                raise ModelSpecificationError(
                    "reference-grid weights have no positive denominator"
                )
            normalized = raw_weights / total
            linear = normalized @ design[indices]
            mean_offset = float(normalized @ offsets[indices])
            level_pairs = tuple(
                (name, key[position])
                for position, name in enumerate(spec_names + by_names)
            )
            label = ", ".join(f"{name}={value}" for name, value in level_pairs)
            result = self.contrast(
                linear,
                constant=mean_offset,
                label=label,
                level=level,
            )
            frozen_weights = _readonly(normalized)
            estimates.append(
                GridEstimate(
                    levels=level_pairs,
                    linear_function=_readonly(linear),
                    offset=mean_offset,
                    grid_weights=frozen_weights,
                    result=result,
                )
            )
        return ReferenceGridResult(
            specs=spec_names,
            by=by_names,
            weights=method,
            grid_columns=tuple(columns),
            grid_rows=rows,
            estimates=tuple(estimates),
            _analysis=self,
        )

    def performance(self) -> PerformanceSummary:
        """Return only metrics defined for the concrete adapter."""
        if self.basis.source == "kamino":
            return _kamino_performance(cast(LinearMixedModelResult, self._model))
        raw = cast(Any, self._model)
        metrics: list[tuple[str, float]] = [
            ("nobs", float(raw.nobs)),
            ("log_likelihood", float(raw.llf)),
        ]
        for name in ("aic", "bic", "rsquared", "rsquared_adj", "scale"):
            if hasattr(raw, name):
                value = float(getattr(raw, name))
                if np.isfinite(value):
                    metrics.append((name, value))
        return PerformanceSummary(
            source=self.basis.source,
            averaging_measure="model-defined",
            metrics=tuple(metrics),
        )


def _requested_names(
    values: Sequence[str], name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ModelSpecificationError(f"{name} must be a sequence of names")
    result = tuple(str(value) for value in values)
    if (not allow_empty and not result) or len(set(result)) != len(result):
        raise ModelSpecificationError(
            f"{name} must be {'unique' if allow_empty else 'nonempty and unique'}"
        )
    return result


def _contrast_vector(value: object, width: int) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelSpecificationError("contrast must be numeric") from error
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ModelSpecificationError(
            "contrast must be a finite vector in the full coefficient space"
        )
    return result


def _contrast_matrix(value: object, width: int) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelSpecificationError("contrast must be numeric") from error
    if result.ndim == 1:
        result = result[None, :]
    if (
        result.ndim != 2
        or result.shape[0] == 0
        or result.shape[1] != width
        or not np.isfinite(result).all()
    ):
        raise ModelSpecificationError(
            "contrast must be a finite vector or matrix in the full coefficient space"
        )
    return result


def _row_values(value: object, rows: int, name: str) -> FloatArray:
    try:
        result = (
            np.repeat(np.asarray(value, dtype=np.float64), rows)
            if np.isscalar(value)
            else np.asarray(value, dtype=np.float64)
        )
    except (TypeError, ValueError) as error:
        raise ModelSpecificationError(f"{name} must be numeric") from error
    if result.shape != (rows,) or not np.isfinite(result).all():
        raise ModelSpecificationError(f"{name} must be finite with one value per row")
    return result


def _validate_level(level: float) -> None:
    if (
        not isinstance(level, Real)
        or isinstance(level, (bool, np.bool_))
        or not np.isfinite(level)
        or not 0.0 < level < 1.0
    ):
        raise ModelSpecificationError(
            "confidence level must be strictly between zero and one"
        )


def _validate_adjustment(value: str) -> None:
    if value not in ("none", "holm", "bonferroni", "sidak"):
        raise ModelSpecificationError(
            "adjustment must be none, holm, bonferroni, or sidak"
        )


def _available_contrast(
    basis: LinearFunctionBasis,
    matrix: FloatArray,
    rhs: FloatArray,
    constants: FloatArray,
    label: str,
    level: float,
    *,
    estimate: float | None,
    standard_error: float | None,
    statistic: float,
    numerator_df: int,
    denominator_df: float | None,
    distribution: Distribution,
    p_value: float,
    confidence_lower: float | None = None,
    confidence_upper: float | None = None,
) -> ContrastResult:
    return ContrastResult(
        label=label,
        coefficient_names=basis.full_coefficient_names,
        contrast=_readonly(matrix),
        rhs=_readonly(rhs),
        constant=_readonly(constants),
        estimable=True,
        available=True,
        status="ok",
        message=f"{basis.inference_method} contrast available",
        estimate=estimate,
        standard_error=standard_error,
        statistic=float(statistic),
        numerator_df=numerator_df,
        denominator_df=denominator_df,
        distribution=distribution,
        p_value=float(p_value),
        adjusted_p_value=float(p_value),
        adjustment="none",
        confidence_level=level,
        confidence_lower=confidence_lower,
        confidence_upper=confidence_upper,
        covariance_kind=basis.covariance_kind,
        inference_method=basis.inference_method,
    )


def _unavailable_contrast(
    basis: LinearFunctionBasis,
    matrix: FloatArray,
    rhs: FloatArray,
    constants: FloatArray,
    label: str,
    level: float,
    status: str,
    message: str,
    *,
    estimable: bool,
) -> ContrastResult:
    return ContrastResult(
        label=label,
        coefficient_names=basis.full_coefficient_names,
        contrast=_readonly(matrix),
        rhs=_readonly(rhs),
        constant=_readonly(constants),
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
        adjusted_p_value=None,
        adjustment="none",
        confidence_level=level,
        confidence_lower=None,
        confidence_upper=None,
        covariance_kind=basis.covariance_kind,
        inference_method=basis.inference_method,
    )


def _wald_contrast(
    basis: LinearFunctionBasis,
    matrix: FloatArray,
    rhs: FloatArray,
    constants: FloatArray,
    *,
    label: str,
    level: float,
) -> ContrastResult:
    retained = matrix[:, list(basis.retained_indices)]
    effect = retained @ basis.coefficients + constants - rhs
    covariance = retained @ basis.covariance @ retained.T
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if eigenvalues.size == 0 or eigenvalues[0] <= 0.0:
        return _unavailable_contrast(
            basis,
            matrix,
            rhs,
            constants,
            label,
            level,
            "invalid_variance",
            "hypothesis covariance has no positive direction",
            estimable=True,
        )
    tolerance = np.sqrt(np.finfo(float).eps) * eigenvalues[0]
    keep = eigenvalues > tolerance
    rank = int(np.count_nonzero(keep))
    null_projection = eigenvectors[:, ~keep].T @ effect
    if null_projection.size and np.max(np.abs(null_projection)) > tolerance * max(
        1.0, float(np.linalg.norm(effect))
    ):
        return _unavailable_contrast(
            basis,
            matrix,
            rhs,
            constants,
            label,
            level,
            "inconsistent_hypothesis",
            "redundant hypothesis rows have an inconsistent right-hand side",
            estimable=True,
        )
    transformed = eigenvectors[:, keep].T @ effect
    wald = float(np.sum(np.square(transformed) / eigenvalues[keep]))
    if matrix.shape[0] == 1 and rank == 1:
        estimate = float((retained @ basis.coefficients + constants)[0])
        standard_error = float(np.sqrt(covariance[0, 0]))
        statistic = float(effect[0] / standard_error)
        if basis.inference_method == "residual_t":
            degrees = cast(float, basis.denominator_df)
            probability = float(2.0 * t_distribution.sf(abs(statistic), degrees))
            critical = float(t_distribution.ppf(0.5 + level / 2.0, degrees))
            distribution: Distribution = "t"
        else:
            degrees = None
            probability = float(2.0 * norm.sf(abs(statistic)))
            critical = float(norm.ppf(0.5 + level / 2.0))
            distribution = "normal"
        return _available_contrast(
            basis,
            matrix,
            rhs,
            constants,
            label,
            level,
            estimate=estimate,
            standard_error=standard_error,
            statistic=statistic,
            numerator_df=1,
            denominator_df=degrees,
            distribution=distribution,
            p_value=probability,
            confidence_lower=estimate - critical * standard_error,
            confidence_upper=estimate + critical * standard_error,
        )
    if basis.inference_method == "residual_t":
        degrees = cast(float, basis.denominator_df)
        statistic = wald / rank
        probability = float(f_distribution.sf(statistic, rank, degrees))
        distribution = "f"
    else:
        degrees = None
        statistic = wald
        probability = float(chi2.sf(statistic, rank))
        distribution = "chi_square"
    return _available_contrast(
        basis,
        matrix,
        rhs,
        constants,
        label,
        level,
        estimate=None,
        standard_error=None,
        statistic=statistic,
        numerator_df=rank,
        denominator_df=degrees,
        distribution=distribution,
        p_value=probability,
    )


def _adjust_results(
    results: tuple[ContrastResult, ...], adjustment: AdjustmentMethod
) -> tuple[ContrastResult, ...]:
    _validate_adjustment(adjustment)
    if adjustment == "none" or not results:
        return tuple(replace(result, adjustment=adjustment) for result in results)
    available = [
        (index, cast(float, result.p_value))
        for index, result in enumerate(results)
        if result.available and result.p_value is not None
    ]
    m = len(results)
    adjusted: dict[int, float] = {}
    if adjustment == "bonferroni":
        adjusted = {
            index: min(1.0, probability * m) for index, probability in available
        }
    elif adjustment == "sidak":
        adjusted = {
            index: min(1.0, -np.expm1(m * np.log1p(-probability)))
            for index, probability in available
        }
    else:
        ordered = sorted(available, key=lambda item: item[1])
        previous = 0.0
        for rank_index, (index, probability) in enumerate(ordered):
            value = max(previous, min(1.0, (m - rank_index) * probability))
            adjusted[index] = value
            previous = value
    return tuple(
        replace(
            result,
            adjusted_p_value=adjusted.get(index),
            adjustment=adjustment,
        )
        for index, result in enumerate(results)
    )


def _grid_groups(
    grid: pd.DataFrame, names: tuple[str, ...]
) -> tuple[tuple[tuple[object, ...], np.ndarray[Any, np.dtype[np.int64]]], ...]:
    keys: dict[tuple[object, ...], list[int]] = {}
    for index, row in grid.iterrows():
        key = tuple(row[name] for name in names)
        keys.setdefault(key, []).append(int(index))
    return tuple(
        (key, np.asarray(indices, dtype=np.int64)) for key, indices in keys.items()
    )


def _categorical_key(
    frame: pd.DataFrame, variables: Sequence[_Variable]
) -> list[tuple[object, ...]]:
    names = [variable.name for variable in variables if variable.categorical]
    if not names:
        return [()] * len(frame)
    return [tuple(row) for row in frame[names].itertuples(index=False, name=None)]


def _counts(values: Sequence[tuple[object, ...]]) -> dict[tuple[object, ...], int]:
    result: dict[tuple[object, ...], int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _reference_weights(
    method: WeightMethod,
    grid: pd.DataFrame,
    indices: np.ndarray[Any, np.dtype[np.int64]],
    data: pd.DataFrame,
    variables: tuple[_Variable, ...],
    specs: tuple[str, ...],
    by: tuple[str, ...],
    user: FloatArray | None,
) -> FloatArray:
    if method == "user":
        assert user is not None
        return np.asarray(user[indices], dtype=np.float64)
    if method == "equal":
        return np.ones(indices.size, dtype=np.float64)
    categorical = tuple(variable for variable in variables if variable.categorical)
    if not categorical:
        return np.ones(indices.size, dtype=np.float64)
    rows = grid.iloc[indices]
    primary = set(specs + by)
    nuisance = tuple(
        variable for variable in categorical if variable.name not in primary
    )
    if method == "proportional":
        nuisance_counts = _counts(_categorical_key(data, nuisance))
        return np.asarray(
            [
                nuisance_counts.get(
                    tuple(row[variable.name] for variable in nuisance), 0
                )
                for _, row in rows.iterrows()
            ],
            dtype=np.float64,
        )
    if method == "outer":
        margins = {
            variable.name: data[variable.name].value_counts(dropna=False).to_dict()
            for variable in nuisance
        }
        return np.asarray(
            [
                float(
                    np.prod(
                        [
                            margins[variable.name].get(row[variable.name], 0)
                            for variable in nuisance
                        ]
                    )
                )
                for _, row in rows.iterrows()
            ],
            dtype=np.float64,
        )
    cell_counts = _counts(_categorical_key(data, categorical))
    values = np.asarray(
        [
            cell_counts.get(tuple(row[variable.name] for variable in categorical), 0)
            for _, row in rows.iterrows()
        ],
        dtype=np.float64,
    )
    return (values > 0.0).astype(np.float64) if method == "flat" else values


def _grid_offset(value: object, rows: int) -> FloatArray:
    try:
        result = (
            np.repeat(np.asarray(value, dtype=np.float64), rows)
            if np.isscalar(value)
            else np.asarray(value, dtype=np.float64)
        )
    except (TypeError, ValueError) as error:
        raise ModelSpecificationError(
            "reference-grid offset must be numeric"
        ) from error
    if result.shape != (rows,) or not np.isfinite(result).all():
        raise ModelSpecificationError(
            "reference-grid offset must be finite with one value per grid row"
        )
    return result


def _null_basis(matrix: FloatArray) -> FloatArray:
    _, singular, vectors = np.linalg.svd(matrix, full_matrices=True)
    maximum = float(singular[0]) if singular.size else 0.0
    rank = int(np.count_nonzero(singular > np.sqrt(np.finfo(float).eps) * maximum))
    return _readonly(vectors[rank:].T)


def _kamino_variables(model: LinearMixedModelResult) -> tuple[_Variable, ...]:
    return tuple(
        _Variable(
            variable.name,
            variable.kind == "categorical",
            tuple(variable.levels),
        )
        for variable in model.fixed_encoder.variables
    )


def adapt_kamino(
    model: LinearMixedModelResult,
    *,
    inference: Literal["asymptotic", "satterthwaite", "kenward_roger"] = "asymptotic",
    data: pd.DataFrame | None = None,
) -> PostfitAnalysis:
    """Create a postfit analysis for one live Kamino fit."""
    if not isinstance(model, LinearMixedModelResult):
        raise ModelSpecificationError("Kamino postfit requires a live fitted result")
    if inference not in ("asymptotic", "satterthwaite", "kenward_roger"):
        raise ModelSpecificationError("unsupported Kamino postfit inference method")
    state: object | None = None
    covariance = model.beta_covariance
    covariance_kind = "model_based"
    if inference == "satterthwaite":
        state = model.satterthwaite()
        if not cast(Any, state).available:
            raise PostfitError(
                f"Satterthwaite postfit is unavailable: {cast(Any, state).status}"
            )
    elif inference == "kenward_roger":
        state = model.kenward_roger()
        if not cast(Any, state).available:
            raise PostfitError(
                f"Kenward-Roger postfit is unavailable: {cast(Any, state).status}"
            )
        covariance = cast(Any, state).adjusted_covariance
        covariance_kind = "kenward_roger_adjusted"
    basis = LinearFunctionBasis(
        contract_version=POSTFIT_CONTRACT_VERSION,
        source="kamino",
        full_coefficient_names=model.full_fixed_names,
        retained_indices=model.fixed_rank.retained_indices,
        coefficients=model.beta,
        covariance=covariance,
        null_basis=model.fixed_rank.null_basis,
        covariance_kind=covariance_kind,
        inference_method=inference,
        denominator_df=None,
    )
    return PostfitAnalysis(
        basis,
        model=model,
        variables=_kamino_variables(model),
        training_data=data,
        inference_state=state,
    )


def _unwrap_statsmodels_ols(result: object) -> Any:
    try:
        from statsmodels.regression.linear_model import (
            OLS,
            WLS,
            RegressionResults,
            RegressionResultsWrapper,
        )
    except ModuleNotFoundError as error:  # pragma: no cover - clean-wheel path
        raise PostfitError(
            "statsmodels adapters require kamino[postfit-statsmodels]"
        ) from error

    raw = result._results if isinstance(result, RegressionResultsWrapper) else result
    if not isinstance(raw, RegressionResults) or not isinstance(raw.model, (OLS, WLS)):
        raise ModelSpecificationError(
            "OLS/WLS postfit requires a statsmodels OLS or WLS result"
        )
    return raw


def _unwrap_statsmodels_mixedlm(result: object) -> Any:
    try:
        from statsmodels.regression.mixed_linear_model import (
            MixedLMResults,
            MixedLMResultsWrapper,
        )
    except ModuleNotFoundError as error:  # pragma: no cover - clean-wheel path
        raise PostfitError(
            "statsmodels adapters require kamino[postfit-statsmodels]"
        ) from error

    raw = result._results if isinstance(result, MixedLMResultsWrapper) else result
    if not isinstance(raw, MixedLMResults):
        raise ModelSpecificationError(
            "MixedLM postfit requires a statsmodels MixedLM result"
        )
    return raw


def _statsmodels_formula_state(raw: Any) -> tuple[tuple[_Variable, ...], pd.DataFrame]:
    from patsy import DesignInfo

    design_info = getattr(raw.model.data, "design_info", None)
    frame = getattr(raw.model.data, "frame", None)
    if not isinstance(design_info, DesignInfo) or not isinstance(frame, pd.DataFrame):
        raise ModelSpecificationError(
            "statsmodels postfit requires a formula fit with retained model data"
        )
    variables: list[_Variable] = []
    for factor, information in design_info.factor_infos.items():
        name = str(factor.name())
        if name not in frame.columns:
            raise ModelSpecificationError(
                "statsmodels postfit currently supports only direct predictor names"
            )
        categorical = str(information.type) == "categorical"
        levels = tuple(information.categories) if categorical else ()
        variables.append(_Variable(name, categorical, levels))
    return tuple(variables), frame.copy(deep=False)


def adapt_statsmodels_ols(result: object) -> PostfitAnalysis:
    """Adapt a formula-fitted statsmodels OLS/WLS result with explicit DF policy."""
    raw = _unwrap_statsmodels_ols(result)
    from statsmodels.regression.linear_model import OLS

    names = tuple(str(value) for value in raw.model.exog_names)
    variables, frame = _statsmodels_formula_state(raw)
    covariance = np.asarray(raw.cov_params(), dtype=np.float64)
    covariance_kind = (
        "model_based"
        if str(raw.cov_type).lower() == "nonrobust"
        else f"statsmodels_{str(raw.cov_type).lower()}"
    )
    inference: InferenceMethod = (
        "residual_t"
        if covariance_kind == "model_based" and bool(raw.use_t)
        else "asymptotic"
    )
    source: AdapterSource = (
        "statsmodels_ols" if isinstance(raw.model, OLS) else "statsmodels_wls"
    )
    exog = np.asarray(raw.model.exog, dtype=np.float64)
    basis = LinearFunctionBasis(
        contract_version=POSTFIT_CONTRACT_VERSION,
        source=source,
        full_coefficient_names=names,
        retained_indices=tuple(range(len(names))),
        coefficients=np.asarray(raw.params, dtype=np.float64),
        covariance=covariance,
        null_basis=_null_basis(exog),
        covariance_kind=covariance_kind,
        inference_method=inference,
        denominator_df=float(raw.df_resid) if inference == "residual_t" else None,
    )
    return PostfitAnalysis(
        basis,
        model=raw,
        variables=variables,
        training_data=frame,
    )


def adapt_statsmodels_mixedlm(result: object) -> PostfitAnalysis:
    """Adapt fixed effects from a formula-fitted statsmodels MixedLM result."""
    raw = _unwrap_statsmodels_mixedlm(result)
    names = tuple(str(value) for value in raw.model.exog_names)
    variables, frame = _statsmodels_formula_state(raw)
    k_fe = int(raw.model.k_fe)
    covariance = np.asarray(raw.cov_params(), dtype=np.float64)[:k_fe, :k_fe]
    exog = np.asarray(raw.model.exog, dtype=np.float64)
    basis = LinearFunctionBasis(
        contract_version=POSTFIT_CONTRACT_VERSION,
        source="statsmodels_mixedlm",
        full_coefficient_names=names,
        retained_indices=tuple(range(len(names))),
        coefficients=np.asarray(raw.fe_params, dtype=np.float64),
        covariance=covariance,
        null_basis=_null_basis(exog),
        covariance_kind="model_based",
        inference_method="asymptotic",
        denominator_df=None,
    )
    return PostfitAnalysis(
        basis,
        model=raw,
        variables=variables,
        training_data=frame,
    )


def _kamino_performance(model: LinearMixedModelResult) -> PerformanceSummary:
    fixed = np.asarray(model._training_fixed_design) @ np.asarray(model.beta)
    fixed_variance = float(np.var(fixed, ddof=0))
    random_variance = 0.0
    block_start = 0
    if model._training_random_design_terms:
        terms = model._training_random_design_terms
    else:
        terms_list: list[FloatArray] = []
        cursor = 0
        for size in model.covariance_term_sizes:
            terms_list.append(model._training_random_design[:, cursor : cursor + size])
            cursor += size
        terms = tuple(terms_list)
    for term, size in zip(terms, model.covariance_term_sizes, strict=True):
        covariance = model.random_covariance[
            block_start : block_start + size, block_start : block_start + size
        ]
        random_variance += float(
            np.mean(np.einsum("ij,jk,ik->i", term, covariance, term))
        )
        block_start += size
    weights = np.asarray(model._training_design.spec.weights)
    residual_variance = float(np.mean(model.sigma2 / weights))
    total = fixed_variance + random_variance + residual_variance
    metrics = [
        ("nobs", float(model._training_design.spec.n)),
        ("log_likelihood", float(model.log_likelihood)),
        ("sigma", float(model.sigma)),
        ("fixed_variance", fixed_variance),
        ("average_random_variance", random_variance),
        ("average_residual_variance", residual_variance),
    ]
    if total > 0.0:
        metrics.extend(
            [
                ("marginal_r2", fixed_variance / total),
                ("conditional_r2", (fixed_variance + random_variance) / total),
                (
                    "average_icc",
                    random_variance / (random_variance + residual_variance),
                ),
            ]
        )
    return PerformanceSummary(
        source="kamino",
        averaging_measure="equal over retained training rows",
        metrics=tuple(metrics),
    )


__all__ = [
    "AdjustmentMethod",
    "ContrastResult",
    "GridEstimate",
    "LinearFunctionBasis",
    "POSTFIT_CONTRACT_VERSION",
    "PerformanceSummary",
    "PostfitAnalysis",
    "ReferenceGridResult",
    "WeightMethod",
    "adapt_kamino",
    "adapt_statsmodels_mixedlm",
    "adapt_statsmodels_ols",
]
