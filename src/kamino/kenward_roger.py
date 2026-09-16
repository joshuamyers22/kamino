"""Kenward--Roger covariance adjustment and fixed-effect tests.

This module follows the matrix construction in pinned pbkrtest 0.5.5.  The
method is deliberately dense in observation space, so a preflight resource
gate is part of the public contract.
"""

# pyright: reportArgumentType=false, reportMissingTypeStubs=false
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnnecessaryIsInstance=false

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal, TypeAlias

import numpy as np
from scipy.stats import f as f_distribution

from kamino.errors import ModelSpecificationError, NumericalError, ResourceLimitError
from kamino.fit import _fit_design
from kamino.formula import GeneralDesign
from kamino.model import FloatArray, ObjectiveKind
from kamino.results import LinearMixedModelResult

KenwardRogerStatus: TypeAlias = Literal[
    "ok",
    "boundary",
    "unsupported_weights",
    "non_estimable",
    "inconsistent_hypothesis",
    "singular_information",
    "ill_conditioned",
    "invalid_adjustment",
    "numerical_failure",
]


def _readonly(value: object, *, ndim: int | None = None) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ModelSpecificationError(f"value must have {ndim} dimensions")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class KenwardRogerControl:
    """Conditioning and dense-resource limits for Kenward--Roger inference."""

    maximum_observations: int = 2_048
    maximum_covariance_parameters: int = 32
    maximum_dense_bytes: int = 512 * 1024 * 1024
    information_tolerance: float = 1e-10
    maximum_condition_number: float = 1e12
    hypothesis_rank_tolerance: float = float(np.sqrt(np.finfo(np.float64).eps))

    def __post_init__(self) -> None:
        integers = (
            self.maximum_observations,
            self.maximum_covariance_parameters,
            self.maximum_dense_bytes,
        )
        if not all(type(value) is int and value > 0 for value in integers):
            raise ModelSpecificationError(
                "Kenward-Roger resource limits must be positive integers"
            )
        tolerances = (
            self.information_tolerance,
            self.maximum_condition_number,
            self.hypothesis_rank_tolerance,
        )
        if not all(
            isinstance(value, Real)
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value)
            and value > 0.0
            for value in tolerances
        ):
            raise ModelSpecificationError(
                "Kenward-Roger tolerances must be finite and positive"
            )


@dataclass(frozen=True, slots=True)
class KenwardRogerTest:
    """Scaled Kenward--Roger F test for an estimable linear hypothesis."""

    contrast: FloatArray
    coefficient_names: tuple[str, ...]
    rhs: FloatArray
    estimable: bool
    available: bool
    status: KenwardRogerStatus
    message: str
    numerator_df: int | None
    denominator_df: float | None
    statistic: float | None
    scaling: float | None
    p_value: float | None
    unscaled_statistic: float | None
    unscaled_p_value: float | None
    auxiliary: FloatArray | None


@dataclass(frozen=True, slots=True)
class KenwardRogerAnalysis:
    """Reusable pbkrtest-compatible Kenward--Roger matrix state."""

    available: bool
    status: KenwardRogerStatus
    message: str
    source_kind: ObjectiveKind
    analysis_kind: ObjectiveKind
    reml_refit: bool
    reml_objective: float | None
    reml_theta: FloatArray | None
    reml_sigma: float | None
    covariance_parameter_names: tuple[str, ...]
    covariance: FloatArray | None
    adjusted_covariance: FloatArray | None
    covariance_parameter_information: FloatArray | None
    covariance_parameter_covariance: FloatArray | None
    derivative_matrices: tuple[FloatArray, ...]
    information_eigenvalues: FloatArray | None
    condition_number: float | None
    estimated_dense_bytes: int
    hypothesis_rank_tolerance: float
    _model: LinearMixedModelResult

    def test(
        self,
        contrast: Sequence[float] | Sequence[Sequence[float]] | FloatArray,
        *,
        rhs: Sequence[float] | FloatArray | float = 0.0,
    ) -> KenwardRogerTest:
        """Test one or more fixed-effect restrictions with KR scaling."""
        matrix = _validated_contrast(contrast, len(self._model.full_fixed_names))
        right = _validated_rhs(rhs, matrix.shape[0])
        estimable = all(self._model.fixed_rank.is_estimable(row) for row in matrix)
        if not estimable:
            return _unavailable_test(
                self,
                matrix,
                right,
                "non_estimable",
                "a contrast row is not estimable under the fitted fixed-design rank",
                estimable=False,
            )
        if not self.available:
            return _unavailable_test(
                self, matrix, right, self.status, self.message, estimable=True
            )

        retained = matrix[:, list(self._model.fixed_rank.retained_indices)]
        singular_values = np.linalg.svd(retained, compute_uv=False)
        if singular_values.size == 0 or singular_values[0] == 0.0:
            return _unavailable_test(
                self,
                matrix,
                right,
                "inconsistent_hypothesis",
                "hypothesis has numerical rank zero",
            )
        threshold = self.hypothesis_rank_tolerance * singular_values[0]
        rank = int(np.count_nonzero(singular_values > threshold))
        beta_h = np.linalg.pinv(retained, rcond=self.hypothesis_rank_tolerance) @ right
        residual = retained @ beta_h - right
        if np.max(np.abs(residual)) > threshold * max(
            1.0, float(np.linalg.norm(right))
        ):
            return _unavailable_test(
                self,
                matrix,
                right,
                "inconsistent_hypothesis",
                "redundant hypothesis rows have an inconsistent right-hand side",
            )
        _, _, right_vectors = np.linalg.svd(retained, full_matrices=False)
        reduced = right_vectors[:rank]
        reduced_rhs = reduced @ beta_h

        assert self.covariance is not None
        assert self.adjusted_covariance is not None
        assert self.covariance_parameter_covariance is not None
        phi = self.covariance
        phi_adjusted = self.adjusted_covariance
        try:
            hypothesis_covariance = reduced @ phi @ reduced.T
            theta = reduced.T @ np.linalg.solve(hypothesis_covariance, reduced)
            theta_phi = theta @ phi
            a1 = 0.0
            a2 = 0.0
            for left, p_left in enumerate(self.derivative_matrices):
                u_left = theta_phi @ p_left @ phi
                for right_index in range(left, len(self.derivative_matrices)):
                    p_right = self.derivative_matrices[right_index]
                    u_right = theta_phi @ p_right @ phi
                    multiplier = 1.0 if left == right_index else 2.0
                    weight = self.covariance_parameter_covariance[left, right_index]
                    a1 += multiplier * weight * np.trace(u_left) * np.trace(u_right)
                    a2 += multiplier * weight * np.sum(u_left * u_right.T)

            q = float(rank)
            b_value = (a1 + 6.0 * a2) / (2.0 * q)
            g_value = ((q + 1.0) * a1 - (q + 4.0) * a2) / ((q + 2.0) * a2)
            divisor = 3.0 * q + 2.0 * (1.0 - g_value)
            c1 = g_value / divisor
            c2 = (q - g_value) / divisor
            c3 = (q + 2.0 - g_value) / divisor
            v0 = 1.0 + c1 * b_value
            if abs(v0) < 1e-10:
                v0 = 0.0
            v1 = 1.0 - c2 * b_value
            v2 = 1.0 - c3 * b_value
            rho = ((1.0 - a2 / q) / v1) ** 2 * v0 / (q * v2)
            denominator_df = 4.0 + (q + 2.0) / (q * rho - 1.0)
            scaling = (
                1.0
                if abs(denominator_df - 2.0) < 1e-2
                else denominator_df * (1.0 - a2 / q) / (denominator_df - 2.0)
            )
            effect = reduced @ self._model_for_test().beta - reduced_rhs
            adjusted_hypothesis_covariance = reduced @ phi_adjusted @ reduced.T
            wald = float(
                effect @ np.linalg.solve(adjusted_hypothesis_covariance, effect)
            )
            unscaled = wald / q
            statistic = scaling * unscaled
        except (FloatingPointError, np.linalg.LinAlgError, ZeroDivisionError):
            return _unavailable_test(
                self,
                matrix,
                right,
                "invalid_adjustment",
                "Kenward-Roger test adjustment is numerically invalid",
            )
        values = np.array([a1, a2, v0, v1, v2, rho, scaling], dtype=np.float64)
        scalars = np.array(
            [denominator_df, scaling, unscaled, statistic, *values], dtype=np.float64
        )
        if (
            not np.isfinite(scalars).all()
            or denominator_df <= 0.0
            or scaling <= 0.0
            or statistic < 0.0
        ):
            return _unavailable_test(
                self,
                matrix,
                right,
                "invalid_adjustment",
                "Kenward-Roger scaling or degrees of freedom are invalid",
            )
        return KenwardRogerTest(
            contrast=_readonly(matrix),
            coefficient_names=self._model.full_fixed_names,
            rhs=_readonly(right),
            estimable=True,
            available=True,
            status="ok",
            message="Kenward-Roger test available",
            numerator_df=rank,
            denominator_df=float(denominator_df),
            statistic=float(statistic),
            scaling=float(scaling),
            p_value=float(f_distribution.sf(statistic, rank, denominator_df)),
            unscaled_statistic=float(unscaled),
            unscaled_p_value=float(f_distribution.sf(unscaled, rank, denominator_df)),
            auxiliary=_readonly(values),
        )

    def _model_for_test(self) -> LinearMixedModelResult:
        candidate = getattr(self, "_reml_model", None)
        return self._model if candidate is None else candidate


def _validated_contrast(value: object, width: int) -> FloatArray:
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


def _validated_rhs(value: object, rows: int) -> FloatArray:
    if np.isscalar(value):
        try:
            result = np.repeat(np.asarray(value, dtype=np.float64), rows)
        except (TypeError, ValueError) as error:
            raise ModelSpecificationError("rhs must be numeric") from error
    else:
        try:
            result = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ModelSpecificationError("rhs must be numeric") from error
    if result.shape != (rows,) or not np.isfinite(result).all():
        raise ModelSpecificationError("rhs must be finite with one value per row")
    return result


def _unavailable_test(
    analysis: KenwardRogerAnalysis,
    contrast: FloatArray,
    rhs: FloatArray,
    status: KenwardRogerStatus,
    message: str,
    *,
    estimable: bool = True,
) -> KenwardRogerTest:
    return KenwardRogerTest(
        contrast=_readonly(contrast),
        coefficient_names=analysis._model.full_fixed_names,
        rhs=_readonly(rhs),
        estimable=estimable,
        available=False,
        status=status,
        message=message,
        numerator_df=None,
        denominator_df=None,
        statistic=None,
        scaling=None,
        p_value=None,
        unscaled_statistic=None,
        unscaled_p_value=None,
        auxiliary=None,
    )


def _term_specs(model: LinearMixedModelResult):
    design = model._training_design
    if isinstance(design, GeneralDesign):
        return tuple(
            (term.group_indices, term.random_design) for term in design.spec.terms
        )
    result: list[tuple[object, FloatArray]] = []
    cursor = 0
    for size in design.spec.covariance_term_sizes:
        result.append(
            (
                design.spec.group_indices,
                design.spec.random_design[:, cursor : cursor + size],
            )
        )
        cursor += size
    return tuple(result)


def _covariance_components(
    model: LinearMixedModelResult,
) -> tuple[tuple[FloatArray, ...], FloatArray, tuple[str, ...]]:
    components: list[FloatArray] = []
    parameters: list[float] = []
    names: list[str] = []
    block_start = 0
    parameter_index = 1
    for indices_raw, random_design in _term_specs(model):
        indices = np.asarray(indices_raw)
        same_group = indices[:, None] == indices[None, :]
        size = random_design.shape[1]
        covariance = model.random_covariance[
            block_start : block_start + size, block_start : block_start + size
        ]
        for column in range(size):
            for row in range(column, size):
                component = np.outer(random_design[:, row], random_design[:, column])
                if row != column:
                    component += np.outer(
                        random_design[:, column], random_design[:, row]
                    )
                component *= same_group
                components.append(_readonly(component))
                parameters.append(float(covariance[row, column]))
                names.append(f"gamma[{parameter_index}]")
                parameter_index += 1
        block_start += size
    components.append(_readonly(np.eye(model._training_design.spec.n)))
    parameters.append(model.sigma2)
    names.append("residual_variance")
    return tuple(components), np.asarray(parameters), tuple(names)


def _unavailable_analysis(
    model: LinearMixedModelResult,
    status: KenwardRogerStatus,
    message: str,
    *,
    reml_model: LinearMixedModelResult | None = None,
    names: tuple[str, ...] = (),
    estimated_dense_bytes: int = 0,
) -> KenwardRogerAnalysis:
    active = model if reml_model is None else reml_model
    return KenwardRogerAnalysis(
        available=False,
        status=status,
        message=message,
        source_kind=model.kind,
        analysis_kind=ObjectiveKind.REML,
        reml_refit=model.kind is ObjectiveKind.ML,
        reml_objective=active.objective if reml_model is not None else None,
        reml_theta=_readonly(active.theta) if reml_model is not None else None,
        reml_sigma=active.sigma if reml_model is not None else None,
        covariance_parameter_names=names,
        covariance=None,
        adjusted_covariance=None,
        covariance_parameter_information=None,
        covariance_parameter_covariance=None,
        derivative_matrices=(),
        information_eigenvalues=None,
        condition_number=None,
        estimated_dense_bytes=estimated_dense_bytes,
        hypothesis_rank_tolerance=float(np.sqrt(np.finfo(np.float64).eps)),
        _model=model,
    )


def kenward_roger(
    model: LinearMixedModelResult,
    *,
    control: KenwardRogerControl | None = None,
) -> KenwardRogerAnalysis:
    """Compute a bounded pbkrtest-compatible Kenward--Roger adjustment."""
    if not isinstance(model, LinearMixedModelResult):
        raise ModelSpecificationError("Kenward-Roger inference requires a fitted model")
    if control is not None and not isinstance(control, KenwardRogerControl):
        raise ModelSpecificationError("control must be a KenwardRogerControl instance")
    active = control or KenwardRogerControl()
    spec = model._training_design.spec
    if not np.array_equal(spec.weights, np.ones(spec.n, dtype=np.float64)):
        return _unavailable_analysis(
            model,
            "unsupported_weights",
            "Kenward-Roger inference is unavailable for prior-weighted fits",
        )
    if spec.n > active.maximum_observations:
        raise ResourceLimitError(
            "observation count exceeds the Kenward-Roger dense resource limit"
        )

    reml_model = (
        model
        if model.kind is ObjectiveKind.REML
        else _fit_design(
            model._training_design,
            ObjectiveKind.REML,
            model._fit_control,
            model._sparse_limits,
        )
    )
    if reml_model.diagnostics.boundary:
        return _unavailable_analysis(
            model,
            "boundary",
            "Kenward-Roger inference is unavailable at a covariance boundary",
            reml_model=reml_model,
        )
    components, parameters, names = _covariance_components(reml_model)
    count = len(components)
    if count > active.maximum_covariance_parameters:
        raise ResourceLimitError(
            "covariance-parameter count exceeds the Kenward-Roger resource limit"
        )
    estimated_bytes = int((2 + 2 * count) * spec.n * spec.n * 8)
    if estimated_bytes > active.maximum_dense_bytes:
        raise ResourceLimitError(
            "Kenward-Roger dense workspace exceeds the configured byte limit"
        )

    try:
        covariance = sum(
            (
                value * component
                for value, component in zip(parameters, components, strict=True)
            ),
            start=np.zeros((spec.n, spec.n), dtype=np.float64),
        )
        chol = np.linalg.cholesky(covariance)
        identity = np.eye(spec.n, dtype=np.float64)
        covariance_inverse = np.linalg.solve(chol.T, np.linalg.solve(chol, identity))
        x = spec.x
        phi = np.asarray(reml_model.beta_covariance)
        tt = covariance_inverse @ x
        hh = tuple(component @ covariance_inverse for component in components)
        oo = tuple(value @ x for value in hh)
        derivatives = tuple(
            _readonly(-(value.T @ tt + (value.T @ tt).T) / 2.0) for value in oo
        )
        qq: dict[tuple[int, int], FloatArray] = {}
        for left in range(count):
            for right in range(left, count):
                qq[(left, right)] = oo[left].T @ covariance_inverse @ oo[right]
        trace = np.empty((count, count), dtype=np.float64)
        for left in range(count):
            for right in range(left, count):
                value = float(np.sum(hh[left].T * hh[right]))
                trace[left, right] = value
                trace[right, left] = value
        information = np.empty((count, count), dtype=np.float64)
        for left in range(count):
            phi_p = phi @ derivatives[left]
            for right in range(left, count):
                value = (
                    trace[left, right]
                    - 2.0 * np.sum(phi * qq[(left, right)])
                    + np.sum(phi_p * (derivatives[right] @ phi))
                )
                information[left, right] = value
                information[right, left] = value
        information = 0.5 * (information + information.T)
        eigenvalues = np.linalg.eigvalsh(information)
    except (MemoryError, np.linalg.LinAlgError, NumericalError) as error:
        return _unavailable_analysis(
            model,
            "numerical_failure",
            f"Kenward-Roger matrix construction failed: {error}",
            reml_model=reml_model,
            names=names,
            estimated_dense_bytes=estimated_bytes,
        )
    absolute = np.abs(eigenvalues)
    minimum_absolute = float(np.min(absolute))
    if minimum_absolute <= active.information_tolerance:
        return _unavailable_analysis(
            model,
            "singular_information",
            "Kenward-Roger covariance-parameter information is singular",
            reml_model=reml_model,
            names=names,
            estimated_dense_bytes=estimated_bytes,
        )
    condition = float(np.max(absolute) / minimum_absolute)
    if condition > active.maximum_condition_number:
        return _unavailable_analysis(
            model,
            "ill_conditioned",
            "Kenward-Roger covariance-parameter information exceeds the "
            "condition limit",
            reml_model=reml_model,
            names=names,
            estimated_dense_bytes=estimated_bytes,
        )
    parameter_covariance = 2.0 * np.linalg.inv(information)
    update = np.zeros_like(phi)
    for left in range(count):
        for right in range(left, count):
            value = qq[(left, right)] - derivatives[left] @ phi @ derivatives[right]
            multiplier = 1.0 if left == right else 2.0
            update += multiplier * parameter_covariance[left, right] * value
    adjusted = phi + 2.0 * phi @ update @ phi
    adjusted = 0.5 * (adjusted + adjusted.T)
    if not np.isfinite(adjusted).all() or np.linalg.eigvalsh(adjusted)[0] <= 0.0:
        return _unavailable_analysis(
            model,
            "invalid_adjustment",
            "Kenward-Roger adjusted covariance is not positive definite",
            reml_model=reml_model,
            names=names,
            estimated_dense_bytes=estimated_bytes,
        )
    result = KenwardRogerAnalysis(
        available=True,
        status="ok",
        message="Kenward-Roger covariance adjustment is available",
        source_kind=model.kind,
        analysis_kind=ObjectiveKind.REML,
        reml_refit=model.kind is ObjectiveKind.ML,
        reml_objective=reml_model.objective,
        reml_theta=_readonly(reml_model.theta),
        reml_sigma=reml_model.sigma,
        covariance_parameter_names=names,
        covariance=_readonly(phi),
        adjusted_covariance=_readonly(adjusted),
        covariance_parameter_information=_readonly(information),
        covariance_parameter_covariance=_readonly(parameter_covariance),
        derivative_matrices=derivatives,
        information_eigenvalues=_readonly(eigenvalues),
        condition_number=condition,
        estimated_dense_bytes=estimated_bytes,
        hypothesis_rank_tolerance=active.hypothesis_rank_tolerance,
        _model=model,
    )
    object.__setattr__(result, "_model", reml_model)
    return result


__all__ = [
    "KenwardRogerAnalysis",
    "KenwardRogerControl",
    "KenwardRogerTest",
    "kenward_roger",
]
