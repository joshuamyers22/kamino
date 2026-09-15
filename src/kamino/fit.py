"""Public fitting entry point for the verified alpha model subset."""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnnecessaryIsInstance=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from kamino.block import (
    BACKEND_NAME,
    SingleGroupBlockResult,
    SingleGroupBlockWorkspace,
    evaluate_prepared_single_group_block,
    prepare_single_group_block,
)
from kamino.errors import ConvergenceError, ModelSpecificationError, NumericalError
from kamino.formula import (
    ContrastInput,
    DataInput,
    FrameVectorInput,
    GeneralDesign,
    NaAction,
    SingleGroupDesign,
    SubsetInput,
    build_model_design,
)
from kamino.model import ObjectiveKind
from kamino.results import LinearMixedModelResult, OptimizerDiagnostics
from kamino.sparse import (
    BACKEND_NAME as SPARSE_BACKEND_NAME,
)
from kamino.sparse import (
    GeneralSparseResult,
    GeneralSparseWorkspace,
    SparseBackendLimits,
    evaluate_prepared_general_sparse,
    prepare_general_sparse,
)


@dataclass(frozen=True, slots=True)
class FitControl:
    """Bounded controls for the first-alpha covariance optimizer."""

    initial_upper_bound: float = 1.0
    maximum_upper_bound: float = 1_048_576.0
    absolute_theta_tolerance: float = 1e-10
    maximum_evaluations: int = 1_000
    boundary_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        finite_positive = (
            self.initial_upper_bound,
            self.maximum_upper_bound,
            self.absolute_theta_tolerance,
            self.boundary_tolerance,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in finite_positive):
            raise ModelSpecificationError(
                "fit tolerances and bounds must be finite and positive"
            )
        if self.maximum_upper_bound <= self.initial_upper_bound:
            raise ModelSpecificationError(
                "maximum_upper_bound must exceed initial_upper_bound"
            )
        if self.maximum_evaluations <= 0:
            raise ModelSpecificationError("maximum_evaluations must be positive")


class _ScalarOptimizeResult(Protocol):
    success: bool
    fun: float
    x: float
    message: str


class _VectorOptimizeResult(Protocol):
    success: bool
    fun: float
    x: np.ndarray
    message: str


def _refine_scalar_minimum(
    theta: float,
    objective: _Objective,
    *,
    upper: float,
) -> float:
    """Remove bounded-solver square-root-epsilon stopping error.

    SciPy's bounded method includes a square-root machine-epsilon term in its
    stopping rule. A centered local parabola gives a materially more stable
    accepted theta across BLAS/OS combinations without changing the objective.
    """
    step = 1e-4 * max(1.0, abs(theta))
    if theta <= step or theta + step >= upper:
        return theta
    left = objective(theta - step)
    center = objective(theta)
    right = objective(theta + step)
    curvature = left - 2.0 * center + right
    if not np.isfinite(curvature) or curvature <= 0.0:
        return theta
    candidate = theta + 0.5 * step * (left - right) / curvature
    if not theta - step < candidate < theta + step:
        return theta
    candidate_objective = objective(candidate)
    roundoff = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(center))
    return candidate if candidate_objective <= center + roundoff else theta


class _Objective(Protocol):
    def __call__(self, theta: float) -> float: ...


def _fit_theta(
    design: SingleGroupDesign,
    kind: ObjectiveKind,
    control: FitControl,
    workspace: SingleGroupBlockWorkspace,
) -> tuple[np.ndarray, SingleGroupBlockResult, OptimizerDiagnostics]:
    cache: dict[float, float] = {}

    def objective(theta: float) -> float:
        theta_value = float(theta)
        if theta_value not in cache:
            if len(cache) >= control.maximum_evaluations:
                raise ConvergenceError("optimizer evaluation limit exceeded")
            cache[theta_value] = evaluate_prepared_single_group_block(
                workspace, [theta_value], kind=kind
            ).objective
        return cache[theta_value]

    upper = control.initial_upper_bound
    objective(0.0)
    objective(upper)
    while upper < control.maximum_upper_bound:
        midpoint = 0.5 * upper
        if objective(upper) >= objective(midpoint):
            break
        upper = min(2.0 * upper, control.maximum_upper_bound)
        objective(upper)
    if upper == control.maximum_upper_bound and objective(upper) < objective(
        0.5 * upper
    ):
        raise ConvergenceError(
            "theta optimum was not bracketed below the configured maximum"
        )

    try:
        optimum = cast(
            _ScalarOptimizeResult,
            minimize_scalar(
                objective,
                method="bounded",
                bounds=(0.0, upper),
                options={
                    "xatol": control.absolute_theta_tolerance,
                    "maxiter": control.maximum_evaluations,
                },
            ),
        )
    except ConvergenceError:
        raise
    except Exception as error:
        raise ConvergenceError(f"theta optimization failed: {error}") from error
    if not bool(optimum.success) and (
        len(cache) >= control.maximum_evaluations
        or "function evaluations" in str(optimum.message).lower()
    ):
        raise ConvergenceError("optimizer evaluation limit exceeded")
    if (
        not bool(optimum.success)
        or not np.isfinite(float(optimum.fun))
        or not np.isfinite(np.asarray(optimum.x, dtype=np.float64)).all()
    ):
        raise ConvergenceError(f"theta optimization failed: {optimum.message}")

    candidates = (0.0, float(optimum.x), upper)
    theta = min(candidates, key=objective)
    if theta <= control.boundary_tolerance:
        theta = 0.0
    elif theta < upper:
        theta = _refine_scalar_minimum(theta, objective, upper=upper)
    final = evaluate_prepared_single_group_block(workspace, [theta], kind=kind)
    message = str(optimum.message)
    if theta == 0.0:
        message = f"boundary optimum selected at theta=0; {message}"
    diagnostics = OptimizerDiagnostics(
        converged=True,
        message=message,
        evaluations=len(cache),
        boundary=theta == 0.0,
        lower_bound=0.0,
        search_upper_bound=upper,
        optimizer="scipy-bounded",
        parameter_count=1,
        backend=BACKEND_NAME,
        initial_upper_bound=control.initial_upper_bound,
        maximum_upper_bound=control.maximum_upper_bound,
        absolute_theta_tolerance=control.absolute_theta_tolerance,
        maximum_evaluations=control.maximum_evaluations,
        boundary_tolerance=control.boundary_tolerance,
    )
    theta_array = np.array([theta], dtype=np.float64)
    theta_array.setflags(write=False)
    return theta_array, final, diagnostics


def _diagonal_parameter_indices(term_sizes: tuple[int, ...]) -> tuple[int, ...]:
    indices: list[int] = []
    cursor = 0
    for size in term_sizes:
        for column in range(size):
            indices.append(cursor)
            cursor += size - column
    return tuple(indices)


def _fit_theta_vector(
    design: SingleGroupDesign,
    kind: ObjectiveKind,
    control: FitControl,
    workspace: SingleGroupBlockWorkspace,
) -> tuple[np.ndarray, SingleGroupBlockResult, OptimizerDiagnostics]:
    spec = design.spec
    parameter_count = spec.d
    diagonal_indices = _diagonal_parameter_indices(spec.covariance_term_sizes)
    diagonal_set = set(diagonal_indices)
    initial = np.zeros(parameter_count, dtype=np.float64)
    initial[list(diagonal_indices)] = 1.0
    bounds = [
        (0.0, None) if index in diagonal_set else (None, None)
        for index in range(parameter_count)
    ]
    cache: dict[tuple[float, ...], float] = {}

    def objective(theta: np.ndarray) -> float:
        values = np.array(theta, dtype=np.float64, copy=True)
        values[list(diagonal_indices)] = np.maximum(values[list(diagonal_indices)], 0.0)
        key = tuple(float(value) for value in values)
        if key not in cache:
            if len(cache) >= control.maximum_evaluations:
                raise ConvergenceError("optimizer evaluation limit exceeded")
            try:
                cache[key] = evaluate_prepared_single_group_block(
                    workspace, values, kind=kind
                ).objective
            except NumericalError:
                # Unbounded correlation-factor coordinates let Powell explore
                # finite but numerically singular trial points. Such a point is
                # infeasible; it is not a failure of the surrounding fit.
                cache[key] = 1e100
        return cache[key]

    try:
        optimum = cast(
            _VectorOptimizeResult,
            minimize(
                objective,
                initial,
                method="Powell",
                bounds=bounds,
                options={
                    "xtol": control.absolute_theta_tolerance,
                    "ftol": control.absolute_theta_tolerance,
                    "maxfev": control.maximum_evaluations,
                },
            ),
        )
    except ConvergenceError:
        raise
    except Exception as error:
        raise ConvergenceError(f"theta optimization failed: {error}") from error
    if not bool(optimum.success) and (
        len(cache) >= control.maximum_evaluations
        or "function evaluations" in str(optimum.message).lower()
    ):
        raise ConvergenceError("optimizer evaluation limit exceeded")
    if (
        not bool(optimum.success)
        or not np.isfinite(float(optimum.fun))
        or not np.isfinite(np.asarray(optimum.x, dtype=np.float64)).all()
    ):
        raise ConvergenceError(f"theta optimization failed: {optimum.message}")

    theta = np.array(optimum.x, dtype=np.float64, copy=True)
    for index in diagonal_indices:
        if theta[index] <= control.boundary_tolerance:
            theta[index] = 0.0
    final = evaluate_prepared_single_group_block(workspace, theta, kind=kind)
    boundary = any(theta[index] == 0.0 for index in diagonal_indices)
    message = str(optimum.message)
    if boundary:
        message = f"boundary optimum selected; {message}"
    diagnostics = OptimizerDiagnostics(
        converged=True,
        message=message,
        evaluations=len(cache),
        boundary=boundary,
        lower_bound=0.0,
        search_upper_bound=None,
        optimizer="scipy-powell",
        parameter_count=parameter_count,
        backend=BACKEND_NAME,
        initial_upper_bound=control.initial_upper_bound,
        maximum_upper_bound=control.maximum_upper_bound,
        absolute_theta_tolerance=control.absolute_theta_tolerance,
        maximum_evaluations=control.maximum_evaluations,
        boundary_tolerance=control.boundary_tolerance,
    )
    theta.setflags(write=False)
    return theta, final, diagnostics


def _fit_general_theta(
    design: GeneralDesign,
    kind: ObjectiveKind,
    control: FitControl,
    workspace: GeneralSparseWorkspace,
) -> tuple[np.ndarray, GeneralSparseResult, OptimizerDiagnostics]:
    parameter_count = design.spec.d
    diagonal_indices = _diagonal_parameter_indices(design.spec.covariance_term_sizes)
    diagonal_set = set(diagonal_indices)
    initial = np.zeros(parameter_count, dtype=np.float64)
    initial[list(diagonal_indices)] = 1.0
    bounds = [
        (0.0, None) if index in diagonal_set else (None, None)
        for index in range(parameter_count)
    ]
    cache: dict[tuple[float, ...], float] = {}

    def objective(theta: np.ndarray) -> float:
        values = np.array(theta, dtype=np.float64, copy=True)
        values[list(diagonal_indices)] = np.maximum(values[list(diagonal_indices)], 0.0)
        key = tuple(float(value) for value in values)
        if key not in cache:
            if len(cache) >= control.maximum_evaluations:
                raise ConvergenceError("optimizer evaluation limit exceeded")
            try:
                cache[key] = evaluate_prepared_general_sparse(
                    workspace, values, kind=kind
                ).objective
            except NumericalError:
                cache[key] = 1e100
        return cache[key]

    try:
        optimum = cast(
            _VectorOptimizeResult,
            minimize(
                objective,
                initial,
                method="Powell",
                bounds=bounds,
                options={
                    "xtol": control.absolute_theta_tolerance,
                    "ftol": control.absolute_theta_tolerance,
                    "maxfev": control.maximum_evaluations,
                },
            ),
        )
    except ConvergenceError:
        raise
    except Exception as error:
        raise ConvergenceError(f"theta optimization failed: {error}") from error
    if not bool(optimum.success) and (
        len(cache) >= control.maximum_evaluations
        or "function evaluations" in str(optimum.message).lower()
    ):
        raise ConvergenceError("optimizer evaluation limit exceeded")
    if (
        not bool(optimum.success)
        or not np.isfinite(float(optimum.fun))
        or not np.isfinite(np.asarray(optimum.x, dtype=np.float64)).all()
    ):
        raise ConvergenceError(f"theta optimization failed: {optimum.message}")

    theta = np.array(optimum.x, dtype=np.float64, copy=True)
    for index in diagonal_indices:
        if theta[index] <= control.boundary_tolerance:
            theta[index] = 0.0
    final = evaluate_prepared_general_sparse(workspace, theta, kind=kind)
    boundary = any(theta[index] == 0.0 for index in diagonal_indices)
    message = str(optimum.message)
    if boundary:
        message = f"boundary optimum selected; {message}"
    diagnostics = OptimizerDiagnostics(
        converged=True,
        message=message,
        evaluations=len(cache),
        boundary=boundary,
        lower_bound=0.0,
        search_upper_bound=None,
        optimizer="scipy-powell",
        parameter_count=parameter_count,
        backend=SPARSE_BACKEND_NAME,
        initial_upper_bound=control.initial_upper_bound,
        maximum_upper_bound=control.maximum_upper_bound,
        absolute_theta_tolerance=control.absolute_theta_tolerance,
        maximum_evaluations=control.maximum_evaluations,
        boundary_tolerance=control.boundary_tolerance,
    )
    theta.setflags(write=False)
    return theta, final, diagnostics


def lmer(
    formula: str,
    data: DataInput,
    *,
    reml: bool = True,
    weights: FrameVectorInput | None = None,
    offset: FrameVectorInput | None = None,
    contrasts: ContrastInput | None = None,
    na_action: NaAction = "error",
    subset: SubsetInput | None = None,
    control: FitControl | None = None,
    sparse_limits: SparseBackendLimits | None = None,
) -> LinearMixedModelResult:
    """Fit the verified Gaussian LMM subset.

    The fixed side accepts an intercept, additive numeric/categorical variables,
    distinct pairwise interactions, and formula offsets. The random side accepts
    either one random intercept, one correlated numeric random intercept/slope,
    independent numeric intercept and slope terms sharing one grouping factor,
    or ordinary nested/crossed random-intercept terms. Unsupported structures
    fail before optimization instead of being silently reinterpreted.
    """
    if not isinstance(reml, bool):
        raise ModelSpecificationError("reml must be a boolean")
    if control is not None and not isinstance(control, FitControl):
        raise ModelSpecificationError("control must be a FitControl instance")
    fit_control = control or FitControl()
    if sparse_limits is not None and not isinstance(sparse_limits, SparseBackendLimits):
        raise ModelSpecificationError(
            "sparse_limits must be a SparseBackendLimits instance"
        )
    design = build_model_design(
        formula,
        data,
        weights=weights,
        offset=offset,
        contrasts=contrasts,
        na_action=na_action,
        subset=subset,
    )
    kind = ObjectiveKind.REML if reml else ObjectiveKind.ML
    if isinstance(design, GeneralDesign):
        workspace = prepare_general_sparse(design.spec, limits=sparse_limits)
        theta, fixed, diagnostics = _fit_general_theta(
            design, kind, fit_control, workspace
        )
        contribution = np.zeros(design.spec.n, dtype=np.float64)
        effect_cursor = 0
        for term in design.spec.terms:
            effect_count = term.q
            effects = fixed.b[effect_cursor : effect_cursor + effect_count].reshape(
                term.group_count, term.k
            )
            effect_cursor += effect_count
            contribution += np.einsum(
                "nk,nk->n", term.random_design, effects[term.group_indices]
            )
        fitted = design.spec.offset + design.spec.x @ fixed.beta + contribution
        residuals = design.spec.y - fitted
        fitted.setflags(write=False)
        residuals.setflags(write=False)
        first_term = design.random_terms[0]
        return LinearMixedModelResult(
            formula=design.formula,
            kind=kind,
            objective=fixed.objective,
            log_likelihood=fixed.log_likelihood,
            theta=theta,
            beta=fixed.beta,
            beta_covariance=fixed.beta_covariance,
            sigma2=fixed.sigma2,
            random_variance=float(fixed.random_covariance[0, 0]),
            random_covariance=fixed.random_covariance,
            u=fixed.u,
            random_effects=fixed.b,
            fixed_names=design.spec.fixed_names,
            random_names=design.spec.random_names,
            group_name=first_term.group_name,
            group_levels=first_term.group_levels,
            random_coefficient_names=first_term.random_coefficient_names,
            covariance_term_sizes=design.spec.covariance_term_sizes,
            fixed_encoder=design.fixed_encoder,
            formula_offset_names=design.formula_offset_names,
            row_ids=design.spec.row_ids,
            omitted_row_ids=design.omitted_row_ids,
            excluded_row_ids=design.excluded_row_ids,
            na_action=design.na_action,
            fitted_values=fitted,
            residuals=residuals,
            diagnostics=diagnostics,
            _training_groups=first_term.training_groups,
            _training_offset=design.spec.offset,
            _predictor_name=None,
            _requires_explicit_offset=design.requires_explicit_offset,
            _training_fixed_design=design.spec.x,
            _training_random_design=design.spec.terms[0].random_design,
            random_terms=design.random_terms,
            _training_group_terms=tuple(
                term.training_groups for term in design.random_terms
            ),
            _training_random_design_terms=tuple(
                term.random_design for term in design.spec.terms
            ),
        )
    workspace = prepare_single_group_block(design.spec)
    if design.spec.d == 1:
        theta, fixed, diagnostics = _fit_theta(design, kind, fit_control, workspace)
    else:
        theta, fixed, diagnostics = _fit_theta_vector(
            design, kind, fit_control, workspace
        )
    effects = fixed.b.reshape(design.spec.group_count, design.spec.k)
    random_contribution = np.einsum(
        "nk,nk->n",
        design.spec.random_design,
        effects[design.group_indices],
    )
    fitted = design.spec.offset + design.spec.x @ fixed.beta + random_contribution
    residuals = design.spec.y - fitted
    fitted.setflags(write=False)
    residuals.setflags(write=False)
    return LinearMixedModelResult(
        formula=design.formula,
        kind=kind,
        objective=fixed.objective,
        log_likelihood=fixed.log_likelihood,
        theta=theta,
        beta=fixed.beta,
        beta_covariance=fixed.beta_covariance,
        sigma2=fixed.sigma2,
        random_variance=float(fixed.random_covariance[0, 0]),
        random_covariance=fixed.random_covariance,
        u=fixed.u,
        random_effects=fixed.b,
        fixed_names=design.spec.fixed_names,
        random_names=design.spec.random_names,
        group_name=design.group_name,
        group_levels=design.group_levels,
        random_coefficient_names=design.random_coefficient_names,
        covariance_term_sizes=design.spec.covariance_term_sizes,
        fixed_encoder=design.fixed_encoder,
        formula_offset_names=design.formula_offset_names,
        row_ids=design.spec.row_ids,
        omitted_row_ids=design.omitted_row_ids,
        excluded_row_ids=design.excluded_row_ids,
        na_action=design.na_action,
        fitted_values=fitted,
        residuals=residuals,
        diagnostics=diagnostics,
        _training_groups=design.training_groups,
        _training_offset=design.spec.offset,
        _predictor_name=design.predictor_name,
        _requires_explicit_offset=design.requires_explicit_offset,
        _training_fixed_design=design.spec.x,
        _training_random_design=design.spec.random_design,
    )


__all__ = ["FitControl", "lmer"]
