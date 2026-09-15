"""Public fitting entry point for the first verified alpha model."""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnnecessaryIsInstance=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from scipy.optimize import minimize_scalar

from kamino.block import (
    BACKEND_NAME,
    RandomInterceptBlockResult,
    evaluate_random_intercept_block,
)
from kamino.errors import ConvergenceError, ModelSpecificationError
from kamino.formula import (
    DataInput,
    RandomInterceptDesign,
    build_random_intercept_design,
)
from kamino.model import ObjectiveKind, VectorInput
from kamino.results import LinearMixedModelResult, OptimizerDiagnostics


@dataclass(frozen=True, slots=True)
class FitControl:
    """Bounded controls for the first-alpha scalar optimizer."""

    initial_upper_bound: float = 1.0
    maximum_upper_bound: float = 1_048_576.0
    absolute_theta_tolerance: float = 1e-10
    maximum_evaluations: int = 1_000
    boundary_tolerance: float = 1e-8

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
    design: RandomInterceptDesign, kind: ObjectiveKind, control: FitControl
) -> tuple[float, RandomInterceptBlockResult, OptimizerDiagnostics]:
    spec = design.spec
    cache: dict[float, float] = {}

    def objective(theta: float) -> float:
        theta_value = float(theta)
        if theta_value not in cache:
            if len(cache) >= control.maximum_evaluations:
                raise ConvergenceError("optimizer evaluation limit exceeded")
            cache[theta_value] = evaluate_random_intercept_block(
                spec, theta_value, kind=kind
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
    if not bool(optimum.success) or not np.isfinite(float(optimum.fun)):
        raise ConvergenceError(f"theta optimization failed: {optimum.message}")

    candidates = (0.0, float(optimum.x), upper)
    theta = min(candidates, key=objective)
    if theta <= control.boundary_tolerance:
        theta = 0.0
    elif theta < upper:
        theta = _refine_scalar_minimum(theta, objective, upper=upper)
    final = evaluate_random_intercept_block(spec, theta, kind=kind)
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
        backend=BACKEND_NAME,
    )
    return theta, final, diagnostics


def lmer(
    formula: str,
    data: DataInput,
    *,
    reml: bool = True,
    weights: VectorInput | None = None,
    offset: VectorInput | None = None,
    control: FitControl | None = None,
) -> LinearMixedModelResult:
    """Fit the verified random-intercept alpha subset of a Gaussian LMM.

    The accepted formula is exactly ``response ~ 1 + (1 | group)``. Unsupported
    structures fail before optimization instead of being silently reinterpreted.
    """
    if not isinstance(reml, bool):
        raise ModelSpecificationError("reml must be a boolean")
    if control is not None and not isinstance(control, FitControl):
        raise ModelSpecificationError("control must be a FitControl instance")
    fit_control = control or FitControl()
    design = build_random_intercept_design(
        formula, data, weights=weights, offset=offset
    )
    kind = ObjectiveKind.REML if reml else ObjectiveKind.ML
    theta, fixed, diagnostics = _fit_theta(design, kind, fit_control)
    fitted = (
        design.spec.offset + design.spec.x @ fixed.beta + fixed.b[design.group_indices]
    )
    residuals = design.spec.y - fitted
    theta_array = np.array([theta], dtype=np.float64)
    theta_array.setflags(write=False)
    fitted.setflags(write=False)
    residuals.setflags(write=False)
    return LinearMixedModelResult(
        formula=design.formula,
        kind=kind,
        objective=fixed.objective,
        log_likelihood=fixed.log_likelihood,
        theta=theta_array,
        beta=fixed.beta,
        beta_covariance=fixed.beta_covariance,
        sigma2=fixed.sigma2,
        random_variance=fixed.random_variance,
        u=fixed.u,
        random_effects=fixed.b,
        fixed_names=design.spec.fixed_names,
        random_names=design.spec.random_names,
        group_name=design.group_name,
        group_levels=design.group_levels,
        row_ids=design.spec.row_ids,
        fitted_values=fitted,
        residuals=residuals,
        diagnostics=diagnostics,
        _training_groups=design.training_groups,
        _training_offset=design.spec.offset,
    )


__all__ = ["FitControl", "lmer"]
