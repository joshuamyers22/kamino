"""Satterthwaite tests using full variance-parameter derivatives.

The implementation follows the construction used by the pinned lmerTest
companion oracle: derivatives are taken with respect to ``(theta, sigma)`` of
the *unprofiled* ML or REML deviance, and the covariance of those parameters is
twice the inverse deviance Hessian.
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
from scipy.stats import t as t_distribution

from kamino.block import (
    evaluate_prepared_single_group_block,
    prepare_single_group_block,
)
from kamino.errors import ModelSpecificationError, NumericalError
from kamino.formula import GeneralDesign
from kamino.model import FloatArray, ObjectiveKind
from kamino.results import LinearMixedModelResult
from kamino.sparse import evaluate_prepared_general_sparse, prepare_general_sparse

InferenceStatus: TypeAlias = Literal[
    "ok",
    "boundary",
    "non_estimable",
    "singular_hessian",
    "negative_curvature",
    "ill_conditioned",
    "unstable_derivatives",
    "invalid_variance",
    "inconsistent_hypothesis",
]


def _readonly(value: object, *, ndim: int | None = None) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ModelSpecificationError(f"value must have {ndim} dimensions")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SatterthwaiteControl:
    """Numerical and resource limits for derivative-based inference."""

    relative_step: float = 1e-4
    derivative_tolerance: float = 5e-3
    curvature_tolerance: float = 1e-8
    maximum_condition_number: float = 1e12
    maximum_variance_parameters: int = 32
    hypothesis_rank_tolerance: float = float(np.sqrt(np.finfo(np.float64).eps))

    def __post_init__(self) -> None:
        positive = (
            self.relative_step,
            self.derivative_tolerance,
            self.curvature_tolerance,
            self.maximum_condition_number,
            self.hypothesis_rank_tolerance,
        )
        if not all(
            isinstance(value, Real)
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value)
            and value > 0.0
            for value in positive
        ):
            raise ModelSpecificationError(
                "Satterthwaite tolerances must be finite and positive"
            )
        if (
            type(self.maximum_variance_parameters) is not int
            or self.maximum_variance_parameters <= 0
        ):
            raise ModelSpecificationError(
                "maximum_variance_parameters must be a positive integer"
            )


@dataclass(frozen=True, slots=True)
class SatterthwaiteTest:
    """One-degree-of-freedom fixed-effect test."""

    contrast: FloatArray
    coefficient_names: tuple[str, ...]
    rhs: float
    estimable: bool
    available: bool
    status: InferenceStatus
    message: str
    estimate: float | None
    standard_error: float | None
    denominator_df: float | None
    statistic: float | None
    p_value: float | None


@dataclass(frozen=True, slots=True)
class SatterthwaiteJointTest:
    """Multi-degree-of-freedom fixed-effect F test."""

    contrast: FloatArray
    coefficient_names: tuple[str, ...]
    rhs: FloatArray
    estimable: bool
    available: bool
    status: InferenceStatus
    message: str
    numerator_df: int | None
    denominator_df: float | None
    statistic: float | None
    p_value: float | None


@dataclass(frozen=True, slots=True)
class SatterthwaiteAnalysis:
    """Validated derivative state reusable across fixed-effect hypotheses."""

    available: bool
    status: InferenceStatus
    message: str
    parameter_names: tuple[str, ...]
    eta: FloatArray
    finite_difference_steps: FloatArray | None
    hessian: FloatArray | None
    variance_parameter_covariance: FloatArray | None
    beta_covariance_jacobian: FloatArray | None
    hessian_eigenvalues: FloatArray | None
    condition_number: float | None
    derivative_error: float | None
    hypothesis_rank_tolerance: float
    _model: LinearMixedModelResult

    def test(
        self, contrast: Sequence[float] | FloatArray, *, rhs: float = 0.0
    ) -> SatterthwaiteTest:
        """Test one estimable fixed-effect linear function."""
        values = _validated_vector(
            contrast, len(self._model.full_fixed_names), "contrast"
        )
        if (
            not isinstance(rhs, Real)
            or isinstance(rhs, (bool, np.bool_))
            or not np.isfinite(rhs)
        ):
            raise ModelSpecificationError("rhs must be finite")
        right = float(rhs)
        estimable = self._model.fixed_rank.is_estimable(values)
        if not estimable:
            return SatterthwaiteTest(
                contrast=_readonly(values),
                coefficient_names=self._model.full_fixed_names,
                rhs=right,
                estimable=False,
                available=False,
                status="non_estimable",
                message="contrast is not estimable under the fitted fixed-design rank",
                estimate=None,
                standard_error=None,
                denominator_df=None,
                statistic=None,
                p_value=None,
            )
        if not self.available:
            return SatterthwaiteTest(
                contrast=_readonly(values),
                coefficient_names=self._model.full_fixed_names,
                rhs=right,
                estimable=True,
                available=False,
                status=self.status,
                message=self.message,
                estimate=None,
                standard_error=None,
                denominator_df=None,
                statistic=None,
                p_value=None,
            )
        reduced = values[list(self._model.fixed_rank.retained_indices)]
        estimate = float(reduced @ self._model.beta)
        variance = float(reduced @ self._model.beta_covariance @ reduced)
        assert self.beta_covariance_jacobian is not None
        assert self.variance_parameter_covariance is not None
        gradient = np.einsum(
            "i,kij,j->k", reduced, self.beta_covariance_jacobian, reduced
        )
        denominator = float(gradient @ self.variance_parameter_covariance @ gradient)
        if (
            not np.isfinite(variance)
            or variance <= 0.0
            or not np.isfinite(denominator)
            or denominator <= 0.0
        ):
            return _invalid_test(self, values, right, estimate)
        standard_error = float(np.sqrt(variance))
        degrees = float(2.0 * variance * variance / denominator)
        statistic = float((estimate - right) / standard_error)
        if not np.isfinite(degrees) or degrees <= 0.0:
            return _invalid_test(self, values, right, estimate)
        probability = float(2.0 * t_distribution.sf(abs(statistic), degrees))
        return SatterthwaiteTest(
            contrast=_readonly(values),
            coefficient_names=self._model.full_fixed_names,
            rhs=right,
            estimable=True,
            available=True,
            status="ok",
            message="Satterthwaite test available",
            estimate=estimate,
            standard_error=standard_error,
            denominator_df=degrees,
            statistic=statistic,
            p_value=probability,
        )

    def joint_test(
        self,
        contrast: Sequence[Sequence[float]] | FloatArray,
        *,
        rhs: Sequence[float] | FloatArray | float = 0.0,
    ) -> SatterthwaiteJointTest:
        """Test a possibly redundant multi-row fixed-effect hypothesis."""
        matrix = _validated_matrix(
            contrast, len(self._model.full_fixed_names), "contrast"
        )
        right = _validated_rhs(rhs, matrix.shape[0])
        estimable = all(self._model.fixed_rank.is_estimable(row) for row in matrix)
        if not estimable:
            return _unavailable_joint(
                self, matrix, right, "non_estimable", "a contrast row is not estimable"
            )
        if not self.available:
            return _unavailable_joint(
                self, matrix, right, self.status, self.message, estimable=True
            )

        reduced = matrix[:, list(self._model.fixed_rank.retained_indices)]
        covariance = reduced @ self._model.beta_covariance @ reduced.T
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        if eigenvalues.size == 0 or eigenvalues[0] <= 0.0:
            return _unavailable_joint(
                self,
                matrix,
                right,
                "invalid_variance",
                "hypothesis covariance has no positive direction",
            )
        keep = eigenvalues > self._model_eigen_tolerance(eigenvalues[0])
        rank = int(np.count_nonzero(keep))
        if rank == 0:
            return _unavailable_joint(
                self,
                matrix,
                right,
                "invalid_variance",
                "hypothesis covariance has numerical rank zero",
            )
        effect = reduced @ self._model.beta - right
        null_projection = eigenvectors[:, ~keep].T @ effect
        scale = max(1.0, float(np.linalg.norm(effect)))
        if null_projection.size and np.max(np.abs(null_projection)) > (
            self._model_eigen_tolerance(eigenvalues[0]) * scale
        ):
            return _unavailable_joint(
                self,
                matrix,
                right,
                "inconsistent_hypothesis",
                "redundant hypothesis rows have an inconsistent right-hand side",
            )

        basis = eigenvectors[:, keep].T @ reduced
        transformed = eigenvectors[:, keep].T @ effect
        positive = eigenvalues[keep]
        t_squared = np.square(transformed) / positive
        f_statistic = float(np.sum(t_squared) / rank)
        assert self.beta_covariance_jacobian is not None
        assert self.variance_parameter_covariance is not None
        gradients = np.einsum(
            "qi,kij,qj->qk", basis, self.beta_covariance_jacobian, basis
        )
        denominators = np.einsum(
            "qi,ij,qj->q", gradients, self.variance_parameter_covariance, gradients
        )
        if not np.isfinite(denominators).all() or (denominators <= 0.0).any():
            return _unavailable_joint(
                self,
                matrix,
                right,
                "invalid_variance",
                "a component degrees-of-freedom denominator is not positive",
            )
        component_df = 2.0 * np.square(positive) / denominators
        denominator_df = _joint_denominator_df(component_df)
        if not np.isfinite(denominator_df) or denominator_df <= 0.0:
            return _unavailable_joint(
                self,
                matrix,
                right,
                "invalid_variance",
                "joint denominator degrees of freedom are invalid",
            )
        probability = float(f_distribution.sf(f_statistic, rank, denominator_df))
        return SatterthwaiteJointTest(
            contrast=_readonly(matrix),
            coefficient_names=self._model.full_fixed_names,
            rhs=_readonly(right),
            estimable=True,
            available=True,
            status="ok",
            message="Satterthwaite joint test available",
            numerator_df=rank,
            denominator_df=denominator_df,
            statistic=f_statistic,
            p_value=probability,
        )

    def _model_eigen_tolerance(self, largest: float) -> float:
        return self.hypothesis_rank_tolerance * largest


def _validated_vector(value: object, width: int, name: str) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelSpecificationError(f"{name} must be numeric") from error
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ModelSpecificationError(
            f"{name} must be a finite vector in the full coefficient space"
        )
    return result


def _validated_matrix(value: object, width: int, name: str) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelSpecificationError(f"{name} must be numeric") from error
    if (
        result.ndim != 2
        or result.shape[0] == 0
        or result.shape[1] != width
        or not np.isfinite(result).all()
    ):
        raise ModelSpecificationError(
            f"{name} must be a nonempty finite matrix in the full coefficient space"
        )
    return result


def _validated_rhs(value: object, rows: int) -> FloatArray:
    if np.isscalar(value):
        try:
            scalar = np.asarray(value, dtype=np.float64).reshape(1)
            result = np.repeat(scalar, rows)
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


def _invalid_test(
    analysis: SatterthwaiteAnalysis,
    contrast: FloatArray,
    rhs: float,
    estimate: float,
) -> SatterthwaiteTest:
    return SatterthwaiteTest(
        contrast=_readonly(contrast),
        coefficient_names=analysis._model.full_fixed_names,
        rhs=float(rhs),
        estimable=True,
        available=False,
        status="invalid_variance",
        message="contrast variance or degrees-of-freedom denominator is not positive",
        estimate=estimate,
        standard_error=None,
        denominator_df=None,
        statistic=None,
        p_value=None,
    )


def _unavailable_joint(
    analysis: SatterthwaiteAnalysis,
    contrast: FloatArray,
    rhs: FloatArray,
    status: InferenceStatus,
    message: str,
    *,
    estimable: bool = True,
) -> SatterthwaiteJointTest:
    return SatterthwaiteJointTest(
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
        p_value=None,
    )


def _joint_denominator_df(component_df: FloatArray) -> float:
    count = int(component_df.size)
    if count == 1:
        return float(component_df[0])
    if np.all(np.abs(np.diff(component_df)) < 1e-8):
        return float(np.mean(component_df))
    if (component_df <= 2.0).any():
        return 2.0
    expectation = float(np.sum(component_df / (component_df - 2.0)))
    return float(2.0 * expectation / (expectation - count))


class _Evaluator:
    def __init__(self, model: LinearMixedModelResult) -> None:
        self.model = model
        self.design = model._training_design
        if isinstance(self.design, GeneralDesign):
            self.workspace = prepare_general_sparse(
                self.design.spec, limits=model._sparse_limits
            )
        else:
            self.workspace = prepare_single_group_block(self.design.spec)

    def fixed(self, theta: FloatArray):
        if isinstance(self.design, GeneralDesign):
            return evaluate_prepared_general_sparse(
                self.workspace,
                theta,
                kind=self.model.kind,  # type: ignore[arg-type]
            )
        return evaluate_prepared_single_group_block(
            self.workspace,
            theta,
            kind=self.model.kind,  # type: ignore[arg-type]
        )

    def objective(self, eta: FloatArray) -> float:
        fixed = self.fixed(eta[:-1])
        sigma2 = float(eta[-1] * eta[-1])
        degrees = (
            self.design.spec.n
            if self.model.kind is ObjectiveKind.ML
            else self.design.spec.n - self.design.spec.p
        )
        value = fixed.logdet_c - fixed.logdet_weights
        if self.model.kind is ObjectiveKind.REML:
            value += fixed.logdet_s
        value += degrees * np.log(2.0 * np.pi * sigma2)
        value += fixed.penalized_residual_sum_squares / sigma2
        return float(value)

    def beta_covariance(self, eta: FloatArray) -> FloatArray:
        fixed = self.fixed(eta[:-1])
        return np.asarray(
            fixed.beta_covariance * (eta[-1] * eta[-1] / fixed.sigma2),
            dtype=np.float64,
        )


def _diagonal_indices(term_sizes: tuple[int, ...]) -> tuple[int, ...]:
    result: list[int] = []
    cursor = 0
    for size in term_sizes:
        for column in range(size):
            result.append(cursor)
            cursor += size - column
    return tuple(result)


def _steps(
    eta: FloatArray, relative_step: float, term_sizes: tuple[int, ...]
) -> FloatArray:
    result = relative_step * np.maximum(1.0, np.abs(eta))
    # Every diagonal theta and sigma is positive for an accepted regular fit.
    # Conservatively preserve the domain for central differences; off-diagonal
    # theta coordinates may freely cross zero.
    for index in (*_diagonal_indices(term_sizes), eta.size - 1):
        result[index] = min(result[index], 0.25 * eta[index])
    if (result <= 0.0).any() or not np.isfinite(result).all():
        raise NumericalError("finite-difference steps are not positive")
    return result


def _raw_hessian(
    evaluator: _Evaluator, eta: FloatArray, steps: FloatArray
) -> FloatArray:
    size = eta.size
    result = np.empty((size, size), dtype=np.float64)
    center = evaluator.objective(eta)
    for left in range(size):
        plus = eta.copy()
        minus = eta.copy()
        plus[left] += steps[left]
        minus[left] -= steps[left]
        result[left, left] = (
            evaluator.objective(plus) - 2.0 * center + evaluator.objective(minus)
        ) / (steps[left] * steps[left])
        for right in range(left):
            pp = eta.copy()
            pm = eta.copy()
            mp = eta.copy()
            mm = eta.copy()
            pp[left] += steps[left]
            pp[right] += steps[right]
            pm[left] += steps[left]
            pm[right] -= steps[right]
            mp[left] -= steps[left]
            mp[right] += steps[right]
            mm[left] -= steps[left]
            mm[right] -= steps[right]
            value = (
                evaluator.objective(pp)
                - evaluator.objective(pm)
                - evaluator.objective(mp)
                + evaluator.objective(mm)
            ) / (4.0 * steps[left] * steps[right])
            result[left, right] = value
            result[right, left] = value
    return result


def _raw_jacobian(
    evaluator: _Evaluator, eta: FloatArray, steps: FloatArray
) -> FloatArray:
    size = eta.size
    p = evaluator.model.beta.size
    result = np.empty((size, p, p), dtype=np.float64)
    for index in range(size):
        plus = eta.copy()
        minus = eta.copy()
        plus[index] += steps[index]
        minus[index] -= steps[index]
        result[index] = (
            evaluator.beta_covariance(plus) - evaluator.beta_covariance(minus)
        ) / (2.0 * steps[index])
    return result


def _relative_error(left: FloatArray, right: FloatArray) -> float:
    scale = max(1.0, float(np.linalg.norm(left, ord=np.inf)))
    return float(np.linalg.norm(left - right, ord=np.inf) / scale)


def _unavailable_analysis(
    model: LinearMixedModelResult,
    eta: FloatArray,
    names: tuple[str, ...],
    status: InferenceStatus,
    message: str,
    *,
    hessian: FloatArray | None = None,
    eigenvalues: FloatArray | None = None,
    condition_number: float | None = None,
    derivative_error: float | None = None,
    steps: FloatArray | None = None,
    hypothesis_rank_tolerance: float = float(np.sqrt(np.finfo(np.float64).eps)),
) -> SatterthwaiteAnalysis:
    return SatterthwaiteAnalysis(
        available=False,
        status=status,
        message=message,
        parameter_names=names,
        eta=_readonly(eta),
        finite_difference_steps=None if steps is None else _readonly(steps),
        hessian=None if hessian is None else _readonly(hessian),
        variance_parameter_covariance=None,
        beta_covariance_jacobian=None,
        hessian_eigenvalues=(None if eigenvalues is None else _readonly(eigenvalues)),
        condition_number=condition_number,
        derivative_error=derivative_error,
        hypothesis_rank_tolerance=hypothesis_rank_tolerance,
        _model=model,
    )


def satterthwaite(
    model: LinearMixedModelResult,
    *,
    control: SatterthwaiteControl | None = None,
) -> SatterthwaiteAnalysis:
    """Compute reusable full-variance-parameter Satterthwaite derivatives."""
    if not isinstance(model, LinearMixedModelResult):
        raise ModelSpecificationError("Satterthwaite inference requires a fitted model")
    if control is not None and not isinstance(control, SatterthwaiteControl):
        raise ModelSpecificationError("control must be a SatterthwaiteControl instance")
    active = control or SatterthwaiteControl()
    eta = np.concatenate((np.asarray(model.theta), [model.sigma]))
    names = tuple(f"theta[{index + 1}]" for index in range(model.theta.size)) + (
        "sigma",
    )
    if eta.size > active.maximum_variance_parameters:
        raise ModelSpecificationError(
            "variance-parameter count exceeds the Satterthwaite resource limit"
        )
    if model.diagnostics.boundary:
        return _unavailable_analysis(
            model,
            eta,
            names,
            "boundary",
            "Satterthwaite inference is unavailable at a covariance boundary",
        )

    evaluator = _Evaluator(model)
    steps = _steps(eta, active.relative_step, model.covariance_term_sizes)
    try:
        hessian_coarse = _raw_hessian(evaluator, eta, steps)
        hessian_half = _raw_hessian(evaluator, eta, steps / 2.0)
        hessian = hessian_half + (hessian_half - hessian_coarse) / 3.0
        jacobian_coarse = _raw_jacobian(evaluator, eta, steps)
        jacobian_half = _raw_jacobian(evaluator, eta, steps / 2.0)
        jacobian = jacobian_half + (jacobian_half - jacobian_coarse) / 3.0
    except (ModelSpecificationError, NumericalError, np.linalg.LinAlgError) as error:
        return _unavailable_analysis(
            model,
            eta,
            names,
            "unstable_derivatives",
            f"variance-parameter derivative evaluation failed: {error}",
            steps=steps,
        )
    hessian = 0.5 * (hessian + hessian.T)
    derivative_error = max(
        _relative_error(hessian, hessian_half),
        _relative_error(
            jacobian.reshape(eta.size, -1), jacobian_half.reshape(eta.size, -1)
        ),
    )
    if not np.isfinite(hessian).all() or not np.isfinite(jacobian).all():
        return _unavailable_analysis(
            model,
            eta,
            names,
            "unstable_derivatives",
            "variance-parameter derivatives contain non-finite values",
            derivative_error=derivative_error,
            steps=steps,
        )
    if derivative_error > active.derivative_tolerance:
        return _unavailable_analysis(
            model,
            eta,
            names,
            "unstable_derivatives",
            "variance-parameter derivatives fail the configured step-sensitivity gate",
            hessian=hessian,
            derivative_error=derivative_error,
            steps=steps,
        )

    eigenvalues = np.linalg.eigvalsh(hessian)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    threshold = active.curvature_tolerance * scale
    if eigenvalues[0] < -threshold:
        return _unavailable_analysis(
            model,
            eta,
            names,
            "negative_curvature",
            "unprofiled deviance Hessian has material negative curvature",
            hessian=hessian,
            eigenvalues=eigenvalues,
            derivative_error=derivative_error,
            steps=steps,
        )
    if eigenvalues[0] <= threshold:
        return _unavailable_analysis(
            model,
            eta,
            names,
            "singular_hessian",
            "unprofiled deviance Hessian is numerically singular",
            hessian=hessian,
            eigenvalues=eigenvalues,
            derivative_error=derivative_error,
            steps=steps,
        )
    condition = float(eigenvalues[-1] / eigenvalues[0])
    if condition > active.maximum_condition_number:
        return _unavailable_analysis(
            model,
            eta,
            names,
            "ill_conditioned",
            "unprofiled deviance Hessian exceeds the condition-number limit",
            hessian=hessian,
            eigenvalues=eigenvalues,
            condition_number=condition,
            derivative_error=derivative_error,
            steps=steps,
        )
    covariance = 2.0 * np.linalg.inv(hessian)
    return SatterthwaiteAnalysis(
        available=True,
        status="ok",
        message="full variance-parameter derivatives are available",
        parameter_names=names,
        eta=_readonly(eta),
        finite_difference_steps=_readonly(steps),
        hessian=_readonly(hessian),
        variance_parameter_covariance=_readonly(covariance),
        beta_covariance_jacobian=_readonly(jacobian),
        hessian_eigenvalues=_readonly(eigenvalues),
        condition_number=condition,
        derivative_error=derivative_error,
        hypothesis_rank_tolerance=active.hypothesis_rank_tolerance,
        _model=model,
    )


__all__ = [
    "SatterthwaiteAnalysis",
    "SatterthwaiteControl",
    "SatterthwaiteJointTest",
    "SatterthwaiteTest",
    "satterthwaite",
]
