"""Named-target ML likelihood profiles with nuisance reoptimization."""

# pyright: reportArgumentType=false, reportMissingTypeStubs=false
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnnecessaryIsInstance=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from typing import Literal, TypeAlias, cast

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import OptimizeResult, minimize
from scipy.stats import chi2, norm

from kamino.block import (
    evaluate_prepared_single_group_block,
    prepare_single_group_block,
)
from kamino.errors import ModelSpecificationError, NumericalError, ResourceLimitError
from kamino.fit import _diagonal_parameter_indices, _fit_design
from kamino.formula import GeneralDesign
from kamino.model import FloatArray, ObjectiveKind
from kamino.results import LinearMixedModelResult
from kamino.sparse import evaluate_prepared_general_sparse, prepare_general_sparse

ProfileScale: TypeAlias = Literal["sdcor", "varcov"]
ProfileTargetKind: TypeAlias = Literal[
    "random_standard_deviation",
    "random_correlation",
    "random_variance",
    "random_covariance",
    "residual_scale",
    "residual_variance",
    "fixed_effect",
]
ProfileStatus: TypeAlias = Literal[
    "ok",
    "boundary_truncated",
    "unbounded",
    "nonmonotone",
    "optimizer_failure",
    "lower_baseline",
]


def _readonly(value: object) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ProfileControl:
    """Controls for bounded adaptive likelihood profiling."""

    alpha_maximum: float = 0.01
    maximum_points_per_target: int = 40
    delta: float = 0.4
    deviance_tolerance: float = 1e-8
    baseline_match_tolerance: float = 1e-5
    optimizer_tolerance: float = 1e-9
    maximum_optimizer_evaluations: int = 20_000
    maximum_targets: int = 32
    maximum_total_evaluations: int = 10_000_000
    minimum_scale: float = 1e-10
    monotonicity_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        probabilities = (self.alpha_maximum,)
        if not all(
            isinstance(value, Real)
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value)
            and 0.0 < value < 1.0
            for value in probabilities
        ):
            raise ModelSpecificationError(
                "profile alpha_maximum must be strictly between zero and one"
            )
        positive = (
            self.delta,
            self.deviance_tolerance,
            self.baseline_match_tolerance,
            self.optimizer_tolerance,
            self.minimum_scale,
            self.monotonicity_tolerance,
        )
        if not all(
            isinstance(value, Real)
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value)
            and value > 0.0
            for value in positive
        ):
            raise ModelSpecificationError(
                "profile numerical controls must be finite and positive"
            )
        integers = (
            self.maximum_points_per_target,
            self.maximum_optimizer_evaluations,
            self.maximum_targets,
            self.maximum_total_evaluations,
        )
        if not all(type(value) is int and value > 0 for value in integers):
            raise ModelSpecificationError(
                "profile resource controls must be positive integers"
            )
        if self.maximum_points_per_target < 3:
            raise ModelSpecificationError(
                "maximum_points_per_target must be at least three"
            )


@dataclass(frozen=True, slots=True)
class ProfilePoint:
    """One constrained optimum on a named likelihood profile."""

    target_value: float
    objective: float
    deviance_difference: float
    signed_root_deviance: float
    parameter_names: tuple[str, ...]
    parameters: FloatArray
    evaluations: int
    converged: bool
    message: str


@dataclass(frozen=True, slots=True)
class ProfileInterval:
    """A profile-likelihood confidence interval with endpoint status."""

    target: str
    level: float
    lower: float | None
    upper: float | None
    cutoff: float
    available: bool
    lower_status: Literal["ok", "boundary", "unbounded", "unavailable"]
    upper_status: Literal["ok", "boundary", "unbounded", "unavailable"]


@dataclass(frozen=True, slots=True)
class ProfileTrace:
    """Ordered constrained optima for one named target."""

    target: str
    kind: ProfileTargetKind
    estimate: float
    lower_bound: float
    upper_bound: float
    status: ProfileStatus
    message: str
    points: tuple[ProfilePoint, ...]

    def interval(self, *, level: float = 0.95) -> ProfileInterval:
        """Invert the signed-root profile with monotone cubic interpolation."""
        if not np.isfinite(level) or not 0.0 < level < 1.0:
            raise ModelSpecificationError(
                "profile interval level must be strictly between zero and one"
            )
        cutoff = float(norm.ppf(0.5 + level / 2.0))
        values = np.asarray([point.target_value for point in self.points])
        zeta = np.asarray([point.signed_root_deviance for point in self.points])
        lower, lower_status = _profile_endpoint(
            values,
            zeta,
            -cutoff,
            bound=self.lower_bound,
            side="lower",
        )
        upper, upper_status = _profile_endpoint(
            values,
            zeta,
            cutoff,
            bound=self.upper_bound,
            side="upper",
        )
        return ProfileInterval(
            target=self.target,
            level=float(level),
            lower=lower,
            upper=upper,
            cutoff=cutoff,
            available=lower is not None and upper is not None,
            lower_status=lower_status,
            upper_status=upper_status,
        )


@dataclass(frozen=True, slots=True)
class LikelihoodProfile:
    """One ML-baseline set of named likelihood-profile traces."""

    source_kind: ObjectiveKind
    baseline_kind: ObjectiveKind
    ml_refit: bool
    scale: ProfileScale
    baseline_objective: float
    baseline_theta: FloatArray
    baseline_sigma: float
    parameter_names: tuple[str, ...]
    target_order: tuple[str, ...]
    traces: tuple[ProfileTrace, ...]

    def trace(self, target: str) -> ProfileTrace:
        """Return a trace by its exact lme4-compatible target name."""
        for trace in self.traces:
            if trace.target == target:
                return trace
        raise ModelSpecificationError(f"profile target {target!r} was not computed")

    def interval(self, target: str, *, level: float = 0.95) -> ProfileInterval:
        """Invert one computed profile trace."""
        return self.trace(target).interval(level=level)


@dataclass(frozen=True, slots=True)
class _Target:
    name: str
    kind: ProfileTargetKind
    index: int
    lower: float
    upper: float


class _VarianceEvaluator:
    def __init__(self, model: LinearMixedModelResult, scale: ProfileScale) -> None:
        self.model = model
        self.scale = scale
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
                self.workspace, theta, kind=ObjectiveKind.ML
            )
        return evaluate_prepared_single_group_block(
            self.workspace, theta, kind=ObjectiveKind.ML
        )

    def evaluate(self, natural: FloatArray):
        theta, sigma = _natural_to_theta(
            natural,
            self.model.covariance_term_sizes,
            self.scale,
            minimum_scale=0.0,
        )
        if sigma <= 0.0:
            raise NumericalError("profile residual scale must be positive")
        fixed = self.fixed(theta)
        sigma2 = sigma * sigma
        objective = (
            fixed.logdet_c
            - fixed.logdet_weights
            + self.design.spec.n * np.log(2.0 * np.pi * sigma2)
            + fixed.penalized_residual_sum_squares / sigma2
        )
        if not np.isfinite(objective):
            raise NumericalError("profile objective is non-finite")
        return float(objective), fixed.beta


def _natural_from_result(
    model: LinearMixedModelResult, scale: ProfileScale
) -> FloatArray:
    values: list[float] = []
    block_start = 0
    for size in model.covariance_term_sizes:
        covariance = np.asarray(
            model.random_covariance[
                block_start : block_start + size, block_start : block_start + size
            ]
        )
        if scale == "sdcor":
            standard_deviations = np.sqrt(np.maximum(0.0, np.diag(covariance)))
            values.extend(float(value) for value in standard_deviations)
            for column in range(size):
                for row in range(column + 1, size):
                    denominator = standard_deviations[row] * standard_deviations[column]
                    values.append(
                        0.0
                        if denominator == 0.0
                        else float(covariance[row, column] / denominator)
                    )
        else:
            values.extend(float(value) for value in np.diag(covariance))
            for column in range(size):
                for row in range(column + 1, size):
                    values.append(float(covariance[row, column]))
        block_start += size
    values.append(model.sigma if scale == "sdcor" else model.sigma2)
    return np.asarray(values, dtype=np.float64)


def _psd_cholesky(covariance: FloatArray) -> FloatArray:
    size = covariance.shape[0]
    factor = np.zeros_like(covariance)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(covariance))))
    for row in range(size):
        diagonal = covariance[row, row] - factor[row, :row] @ factor[row, :row]
        if diagonal < -tolerance:
            raise NumericalError("profile covariance is not positive semidefinite")
        factor[row, row] = np.sqrt(max(0.0, diagonal))
        for target in range(row + 1, size):
            numerator = covariance[target, row] - (
                factor[target, :row] @ factor[row, :row]
            )
            if factor[row, row] == 0.0:
                if abs(numerator) > tolerance:
                    raise NumericalError(
                        "profile covariance has an unidentified boundary correlation"
                    )
                factor[target, row] = 0.0
            else:
                factor[target, row] = numerator / factor[row, row]
    return factor


def _natural_to_theta(
    natural: FloatArray,
    term_sizes: tuple[int, ...],
    scale: ProfileScale,
    *,
    minimum_scale: float,
) -> tuple[FloatArray, float]:
    cursor = 0
    factors: list[FloatArray] = []
    for size in term_sizes:
        diagonal = np.asarray(natural[cursor : cursor + size])
        cursor += size
        covariance = np.zeros((size, size), dtype=np.float64)
        if (diagonal < 0.0).any():
            raise NumericalError("profile variance or standard deviation is negative")
        if scale == "sdcor":
            np.fill_diagonal(covariance, diagonal * diagonal)
            for column in range(size):
                for row in range(column + 1, size):
                    correlation = float(natural[cursor])
                    cursor += 1
                    if abs(correlation) > 1.0 + 1e-12:
                        raise NumericalError("profile correlation lies outside [-1, 1]")
                    value = correlation * diagonal[row] * diagonal[column]
                    covariance[row, column] = value
                    covariance[column, row] = value
        else:
            np.fill_diagonal(covariance, diagonal)
            for column in range(size):
                for row in range(column + 1, size):
                    value = float(natural[cursor])
                    cursor += 1
                    covariance[row, column] = value
                    covariance[column, row] = value
        factors.append(_psd_cholesky(covariance))
    residual = float(natural[cursor])
    sigma = residual if scale == "sdcor" else float(np.sqrt(max(0.0, residual)))
    if sigma < minimum_scale:
        raise NumericalError("profile residual scale is below its numerical bound")
    theta: list[float] = []
    for factor in factors:
        for column in range(factor.shape[0]):
            theta.extend(float(value / sigma) for value in factor[column:, column])
    return np.asarray(theta, dtype=np.float64), sigma


def _targets(
    model: LinearMixedModelResult, scale: ProfileScale
) -> tuple[tuple[_Target, ...], tuple[str, ...]]:
    targets: list[_Target] = []
    covariance_index = 0
    display_index = 1
    for size in model.covariance_term_sizes:
        for _ in range(size):
            targets.append(
                _Target(
                    f".sig{display_index:02d}",
                    "random_standard_deviation"
                    if scale == "sdcor"
                    else "random_variance",
                    covariance_index,
                    0.0,
                    np.inf,
                )
            )
            covariance_index += 1
            display_index += 1
        for _column in range(size):
            for _row in range(_column + 1, size):
                targets.append(
                    _Target(
                        f".sig{display_index:02d}",
                        "random_correlation"
                        if scale == "sdcor"
                        else "random_covariance",
                        covariance_index,
                        -1.0 if scale == "sdcor" else -np.inf,
                        1.0 if scale == "sdcor" else np.inf,
                    )
                )
                covariance_index += 1
                display_index += 1
    targets.append(
        _Target(
            ".sigma",
            "residual_scale" if scale == "sdcor" else "residual_variance",
            covariance_index,
            0.0,
            np.inf,
        )
    )
    variance_names = tuple(target.name for target in targets)
    for fixed_index, name in enumerate(model.fixed_names):
        targets.append(_Target(name, "fixed_effect", fixed_index, -np.inf, np.inf))
    return tuple(targets), variance_names + model.fixed_names


def _bounds_for_variance_targets(
    targets: tuple[_Target, ...], variance_count: int, minimum_scale: float
) -> tuple[tuple[float | None, float | None], ...]:
    result: list[tuple[float | None, float | None]] = []
    for target in targets[:variance_count]:
        lower = target.lower
        if target.kind in ("residual_scale", "residual_variance"):
            lower = minimum_scale
        result.append(
            (
                None if not np.isfinite(lower) else lower,
                None if not np.isfinite(target.upper) else target.upper,
            )
        )
    return tuple(result)


def _optimize_variance_target(
    evaluator: _VarianceEvaluator,
    baseline_natural: FloatArray,
    target: _Target,
    value: float,
    bounds: tuple[tuple[float | None, float | None], ...],
    control: ProfileControl,
) -> tuple[float, FloatArray, FloatArray, int, str]:
    nuisance_indices = [
        index for index in range(baseline_natural.size) if index != target.index
    ]
    start = baseline_natural[nuisance_indices]
    nuisance_bounds = [bounds[index] for index in nuisance_indices]
    evaluations = 0

    def objective(nuisance: FloatArray) -> float:
        nonlocal evaluations
        evaluations += 1
        full = baseline_natural.copy()
        full[target.index] = value
        full[nuisance_indices] = nuisance
        try:
            result, _ = evaluator.evaluate(full)
            return result
        except (ModelSpecificationError, NumericalError, np.linalg.LinAlgError):
            return 1e100

    optimum = cast(
        OptimizeResult,
        minimize(
            objective,
            start,
            method="Powell",
            bounds=nuisance_bounds,
            options={
                "xtol": control.optimizer_tolerance,
                "ftol": control.optimizer_tolerance,
                "maxfev": control.maximum_optimizer_evaluations,
            },
        ),
    )
    if (
        not bool(optimum.success)
        or not np.isfinite(float(optimum.fun))
        or float(optimum.fun) >= 1e99
    ):
        raise NumericalError(f"profile nuisance optimization failed: {optimum.message}")
    full = baseline_natural.copy()
    full[target.index] = value
    full[nuisance_indices] = np.asarray(optimum.x, dtype=np.float64)
    objective_value, beta = evaluator.evaluate(full)
    return objective_value, full, beta, evaluations, str(optimum.message)


def _fixed_workspace(model: LinearMixedModelResult, fixed_index: int, value: float):
    design = model._training_design
    spec = design.spec
    keep = [index for index in range(spec.p) if index != fixed_index]
    offset = np.asarray(spec.offset) + np.asarray(spec.x[:, fixed_index]) * value
    updated_spec = replace(
        spec,
        x=_readonly(spec.x[:, keep]),
        offset=_readonly(offset),
        fixed_names=tuple(spec.fixed_names[index] for index in keep),
    )
    if isinstance(design, GeneralDesign):
        return updated_spec, prepare_general_sparse(
            updated_spec, limits=model._sparse_limits
        )
    return updated_spec, prepare_single_group_block(updated_spec)


def _optimize_fixed_target(
    model: LinearMixedModelResult,
    baseline: LinearMixedModelResult,
    target: _Target,
    value: float,
    scale: ProfileScale,
    control: ProfileControl,
) -> tuple[float, FloatArray, FloatArray, int, str]:
    _, workspace = _fixed_workspace(model, target.index, value)
    diagonal_indices = _diagonal_parameter_indices(model.covariance_term_sizes)
    diagonal_set = set(diagonal_indices)
    bounds = [
        (0.0, None) if index in diagonal_set else (None, None)
        for index in range(baseline.theta.size)
    ]
    evaluations = 0

    def objective(theta: FloatArray) -> float:
        nonlocal evaluations
        evaluations += 1
        try:
            if isinstance(model._training_design, GeneralDesign):
                return float(
                    evaluate_prepared_general_sparse(
                        workspace, theta, kind=ObjectiveKind.ML
                    ).objective
                )
            return float(
                evaluate_prepared_single_group_block(
                    workspace, theta, kind=ObjectiveKind.ML
                ).objective
            )
        except (ModelSpecificationError, NumericalError, np.linalg.LinAlgError):
            return 1e100

    optimum = cast(
        OptimizeResult,
        minimize(
            objective,
            baseline.theta,
            method="Powell",
            bounds=bounds,
            options={
                "xtol": control.optimizer_tolerance,
                "ftol": control.optimizer_tolerance,
                "maxfev": control.maximum_optimizer_evaluations,
            },
        ),
    )
    if (
        not bool(optimum.success)
        or not np.isfinite(float(optimum.fun))
        or float(optimum.fun) >= 1e99
    ):
        raise NumericalError(f"profile nuisance optimization failed: {optimum.message}")
    theta = np.asarray(optimum.x, dtype=np.float64)
    if isinstance(model._training_design, GeneralDesign):
        fitted = evaluate_prepared_general_sparse(
            workspace, theta, kind=ObjectiveKind.ML
        )
    else:
        fitted = evaluate_prepared_single_group_block(
            workspace, theta, kind=ObjectiveKind.ML
        )
    proxy = replace(
        baseline,
        theta=_readonly(theta),
        sigma2=fitted.sigma2,
        random_covariance=fitted.random_covariance,
    )
    natural = _natural_from_result(proxy, scale)
    beta = np.insert(np.asarray(fitted.beta), target.index, value)
    return float(fitted.objective), natural, beta, evaluations, str(optimum.message)


def _profile_point(
    model: LinearMixedModelResult,
    baseline: LinearMixedModelResult,
    evaluator: _VarianceEvaluator,
    baseline_natural: FloatArray,
    parameter_names: tuple[str, ...],
    target: _Target,
    value: float,
    bounds: tuple[tuple[float | None, float | None], ...],
    control: ProfileControl,
) -> ProfilePoint:
    if target.kind == "fixed_effect":
        objective, natural, beta, evaluations, message = _optimize_fixed_target(
            model, baseline, target, value, evaluator.scale, control
        )
    else:
        objective, natural, beta, evaluations, message = _optimize_variance_target(
            evaluator,
            baseline_natural,
            target,
            value,
            bounds,
            control,
        )
    difference = objective - baseline.objective
    if difference < -control.deviance_tolerance:
        raise NumericalError(
            "profiling found a materially lower objective than the ML baseline"
        )
    difference = max(0.0, difference)
    signed = np.sign(value - _target_estimate(target, baseline_natural, baseline.beta))
    signed_root = float(signed * np.sqrt(difference))
    return ProfilePoint(
        target_value=float(value),
        objective=float(objective),
        deviance_difference=float(difference),
        signed_root_deviance=signed_root,
        parameter_names=parameter_names,
        parameters=_readonly(np.concatenate((natural, beta))),
        evaluations=evaluations,
        converged=True,
        message=message,
    )


def _target_estimate(target: _Target, natural: FloatArray, beta: FloatArray) -> float:
    return float(
        beta[target.index] if target.kind == "fixed_effect" else natural[target.index]
    )


def _initial_step(
    target: _Target,
    estimate: float,
    baseline: LinearMixedModelResult,
    delta: float,
) -> float:
    if target.kind == "fixed_effect":
        return max(
            1e-8,
            delta
            * float(np.sqrt(baseline.beta_covariance[target.index, target.index])),
        )
    if target.kind in ("random_correlation",):
        return 0.05
    return max(1e-8, 0.05 * max(1.0, abs(estimate)))


def _adaptive_side(
    direction: float,
    model: LinearMixedModelResult,
    baseline: LinearMixedModelResult,
    evaluator: _VarianceEvaluator,
    baseline_natural: FloatArray,
    parameter_names: tuple[str, ...],
    target: _Target,
    bounds: tuple[tuple[float | None, float | None], ...],
    control: ProfileControl,
    cutoff: float,
) -> tuple[list[ProfilePoint], bool, bool, bool, bool]:
    estimate = _target_estimate(target, baseline_natural, baseline.beta)
    step = _initial_step(target, estimate, baseline, control.delta)
    previous_value = estimate
    previous_zeta = 0.0
    points: list[ProfilePoint] = []
    reached_bound = False
    reached_cutoff = False
    optimizer_failed = False
    lower_baseline = False
    maximum_side_points = max(1, (control.maximum_points_per_target - 1) // 2)
    for _ in range(maximum_side_points):
        proposed = previous_value + direction * step
        bound = target.lower if direction < 0 else target.upper
        if np.isfinite(bound) and (
            (direction < 0 and proposed < bound) or (direction > 0 and proposed > bound)
        ):
            proposed = bound
            reached_bound = True
        if proposed == previous_value:
            reached_bound = True
            break
        try:
            point = _profile_point(
                model,
                baseline,
                evaluator,
                baseline_natural,
                parameter_names,
                target,
                proposed,
                bounds,
                control,
            )
        except NumericalError as error:
            if "lower objective" in str(error):
                lower_baseline = True
            else:
                optimizer_failed = True
            break
        points.append(point)
        current_zeta = abs(point.signed_root_deviance)
        if current_zeta >= cutoff:
            reached_cutoff = True
            break
        if reached_bound:
            break
        zeta_increment = current_zeta - abs(previous_zeta)
        if zeta_increment > 1e-8:
            multiplier = min(10.0, max(0.25, control.delta / zeta_increment))
            step *= multiplier
        previous_value = proposed
        previous_zeta = point.signed_root_deviance
    return points, reached_bound, reached_cutoff, optimizer_failed, lower_baseline


def _trace(
    model: LinearMixedModelResult,
    baseline: LinearMixedModelResult,
    evaluator: _VarianceEvaluator,
    baseline_natural: FloatArray,
    parameter_names: tuple[str, ...],
    target: _Target,
    bounds: tuple[tuple[float | None, float | None], ...],
    control: ProfileControl,
    explicit_values: Sequence[float] | None,
) -> ProfileTrace:
    estimate = _target_estimate(target, baseline_natural, baseline.beta)
    baseline_point = ProfilePoint(
        target_value=estimate,
        objective=baseline.objective,
        deviance_difference=0.0,
        signed_root_deviance=0.0,
        parameter_names=parameter_names,
        parameters=_readonly(np.concatenate((baseline_natural, baseline.beta))),
        evaluations=0,
        converged=True,
        message="ML baseline",
    )
    optimizer_failed = False
    lower_baseline = False
    lower_bound_reached = False
    upper_bound_reached = False
    lower_cutoff_reached = False
    upper_cutoff_reached = False
    if explicit_values is not None:
        points = [baseline_point]
        for value in explicit_values:
            if not np.isfinite(value) or value < target.lower or value > target.upper:
                raise ModelSpecificationError(
                    f"explicit profile value for {target.name!r} is outside its bounds"
                )
            if np.isclose(value, estimate, rtol=0.0, atol=1e-14):
                continue
            try:
                points.append(
                    _profile_point(
                        model,
                        baseline,
                        evaluator,
                        baseline_natural,
                        parameter_names,
                        target,
                        float(value),
                        bounds,
                        control,
                    )
                )
            except NumericalError as error:
                lower_baseline = "lower objective" in str(error)
                optimizer_failed = not lower_baseline
                break
    else:
        total_parameters = baseline_natural.size + baseline.beta.size
        cutoff = float(np.sqrt(chi2.ppf(1.0 - control.alpha_maximum, total_parameters)))
        (
            lower,
            lower_bound_reached,
            lower_cutoff_reached,
            lower_failed,
            lower_base,
        ) = _adaptive_side(
            -1.0,
            model,
            baseline,
            evaluator,
            baseline_natural,
            parameter_names,
            target,
            bounds,
            control,
            cutoff,
        )
        (
            upper,
            upper_bound_reached,
            upper_cutoff_reached,
            upper_failed,
            upper_base,
        ) = _adaptive_side(
            1.0,
            model,
            baseline,
            evaluator,
            baseline_natural,
            parameter_names,
            target,
            bounds,
            control,
            cutoff,
        )
        optimizer_failed = lower_failed or upper_failed
        lower_baseline = lower_base or upper_base
        points = lower + [baseline_point] + upper
    points.sort(key=lambda point: point.target_value)
    zeta = np.asarray([point.signed_root_deviance for point in points])
    monotone = bool(np.all(np.diff(zeta) >= -control.monotonicity_tolerance))
    if lower_baseline:
        status: ProfileStatus = "lower_baseline"
        message = "profiling found a materially better ML objective"
    elif optimizer_failed:
        status = "optimizer_failure"
        message = "a nuisance optimization failed"
    elif not monotone:
        status = "nonmonotone"
        message = "signed-root profile is not monotone"
    elif explicit_values is None and (
        (not lower_bound_reached and not lower_cutoff_reached)
        or (not upper_bound_reached and not upper_cutoff_reached)
    ):
        status = "unbounded"
        message = "profile did not reach its cutoff on at least one side"
    elif lower_bound_reached or upper_bound_reached:
        status = "boundary_truncated"
        message = "profile reached a parameter boundary"
    else:
        status = "ok"
        message = "likelihood profile available"
    return ProfileTrace(
        target=target.name,
        kind=target.kind,
        estimate=estimate,
        lower_bound=target.lower,
        upper_bound=target.upper,
        status=status,
        message=message,
        points=tuple(points),
    )


def _profile_endpoint(
    values: FloatArray,
    zeta: FloatArray,
    cutoff: float,
    *,
    bound: float,
    side: Literal["lower", "upper"],
) -> tuple[float | None, Literal["ok", "boundary", "unbounded", "unavailable"]]:
    center = int(np.argmin(np.abs(zeta)))
    if side == "lower":
        side_values = values[: center + 1]
        side_zeta = zeta[: center + 1]
    else:
        side_values = values[center:]
        side_zeta = zeta[center:]
    if side_values.size < 2 or np.any(np.diff(side_zeta) <= 0.0):
        return None, "unavailable"
    if cutoff < side_zeta[0] or cutoff > side_zeta[-1]:
        endpoint = side_values[0] if side == "lower" else side_values[-1]
        if np.isfinite(bound) and np.isclose(endpoint, bound, rtol=0.0, atol=1e-10):
            return float(bound), "boundary"
        return None, "unbounded"
    interpolator = PchipInterpolator(side_zeta, side_values, extrapolate=False)
    return float(interpolator(cutoff)), "ok"


def likelihood_profile(
    model: LinearMixedModelResult,
    *,
    targets: Sequence[str] | None = None,
    scale: ProfileScale = "sdcor",
    values: Mapping[str, Sequence[float]] | None = None,
    control: ProfileControl | None = None,
) -> LikelihoodProfile:
    """Profile named covariance and fixed-effect targets on an ML baseline."""
    if not isinstance(model, LinearMixedModelResult):
        raise ModelSpecificationError("likelihood profiling requires a fitted model")
    if scale not in ("sdcor", "varcov"):
        raise ModelSpecificationError("profile scale must be 'sdcor' or 'varcov'")
    if control is not None and not isinstance(control, ProfileControl):
        raise ModelSpecificationError("control must be a ProfileControl instance")
    active = control or ProfileControl()
    baseline = (
        model
        if model.kind is ObjectiveKind.ML
        else _fit_design(
            model._training_design,
            ObjectiveKind.ML,
            model._fit_control,
            model._sparse_limits,
        )
    )
    available_targets, parameter_names = _targets(baseline, scale)
    by_name = {target.name: target for target in available_targets}
    selected_names = (
        tuple(by_name) if targets is None else tuple(str(target) for target in targets)
    )
    if not selected_names or len(set(selected_names)) != len(selected_names):
        raise ModelSpecificationError("profile targets must be nonempty and unique")
    unknown = [name for name in selected_names if name not in by_name]
    if unknown:
        raise ModelSpecificationError(
            f"unknown or non-estimable profile target {unknown[0]!r}"
        )
    if len(selected_names) > active.maximum_targets:
        raise ResourceLimitError("profile target count exceeds the configured limit")
    if values is not None and any(name not in selected_names for name in values):
        raise ModelSpecificationError(
            "explicit profile values must refer to selected targets"
        )
    explicit_count = 0 if values is None else sum(len(item) for item in values.values())
    worst_case_points = (
        len(selected_names) * active.maximum_points_per_target
        if values is None
        else explicit_count + len(selected_names)
    )
    if (
        worst_case_points * active.maximum_optimizer_evaluations
        > active.maximum_total_evaluations
    ):
        raise ResourceLimitError(
            "profile request exceeds the configured total evaluation limit"
        )

    baseline_natural = _natural_from_result(baseline, scale)
    variance_targets = tuple(
        target for target in available_targets if target.kind != "fixed_effect"
    )
    bounds = _bounds_for_variance_targets(
        available_targets, len(variance_targets), active.minimum_scale
    )
    evaluator = _VarianceEvaluator(baseline, scale)
    baseline_check, _ = evaluator.evaluate(baseline_natural)
    relative_difference = abs(baseline_check - baseline.objective) / max(
        1.0, abs(baseline.objective)
    )
    if relative_difference > active.baseline_match_tolerance:
        raise NumericalError(
            "profile parameterization does not reproduce the ML baseline objective"
        )
    traces = tuple(
        _trace(
            model,
            baseline,
            evaluator,
            baseline_natural,
            parameter_names,
            by_name[name],
            bounds,
            active,
            None if values is None else values.get(name),
        )
        for name in selected_names
    )
    if any(trace.status == "lower_baseline" for trace in traces):
        raise NumericalError("likelihood profiling invalidated the fitted ML baseline")
    return LikelihoodProfile(
        source_kind=model.kind,
        baseline_kind=ObjectiveKind.ML,
        ml_refit=model.kind is ObjectiveKind.REML,
        scale=scale,
        baseline_objective=baseline.objective,
        baseline_theta=_readonly(baseline.theta),
        baseline_sigma=baseline.sigma,
        parameter_names=parameter_names,
        target_order=selected_names,
        traces=traces,
    )


__all__ = [
    "LikelihoodProfile",
    "ProfileControl",
    "ProfileInterval",
    "ProfilePoint",
    "ProfileTrace",
    "likelihood_profile",
]
