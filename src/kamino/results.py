"""Immutable fitted-model and prediction results."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnnecessaryIsInstance=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd

from kamino.errors import PredictionError
from kamino.formula import ColumnInput, FixedEncoder, NaAction, RandomTermDesign
from kamino.model import FloatArray, ObjectiveKind, VectorInput

PredictionData: TypeAlias = pd.DataFrame | Mapping[str, ColumnInput]
PredictionMode: TypeAlias = Literal["population", "conditional"]
PredictionOffsetInput: TypeAlias = VectorInput | pd.Series


def _readonly(value: FloatArray) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _series_aligned(series: pd.Series, frame: pd.DataFrame) -> bool:
    series_index: object = series.index
    frame_index: object = frame.index
    return bool(series_index.equals(frame_index))  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class OptimizerDiagnostics:
    """Structured outcome from the covariance optimizer."""

    converged: bool
    message: str
    evaluations: int
    boundary: bool
    lower_bound: float
    search_upper_bound: float | None
    optimizer: str
    parameter_count: int
    backend: str
    initial_upper_bound: float
    maximum_upper_bound: float
    absolute_theta_tolerance: float
    maximum_evaluations: int
    boundary_tolerance: float


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Labeled predictions with explicit conditioning and new-group status."""

    values: FloatArray
    row_ids: tuple[str, ...]
    mode: PredictionMode
    new_group: tuple[bool, ...]


def _prediction_column(value: object, name: str) -> Sequence[object]:
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise PredictionError(f"prediction column {name!r} must be one-dimensional")
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    raise PredictionError(f"prediction column {name!r} must be a sequence")


def _prediction_offset(
    value: PredictionOffsetInput, n: int, data: PredictionData
) -> FloatArray:
    if (
        isinstance(data, pd.DataFrame)
        and isinstance(value, pd.Series)
        and not _series_aligned(value, data)
    ):
        raise PredictionError("prediction offset must align with data row identifiers")
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise PredictionError("prediction offset must be numeric") from error
    if result.ndim != 1 or result.shape != (n,):
        raise PredictionError("prediction offset must have one value per row")
    if not np.isfinite(result).all():
        raise PredictionError("prediction offset contains non-finite values")
    return result


def _numeric_prediction_column(data: PredictionData, name: str, n: int) -> FloatArray:
    try:
        raw = data[name]
    except (KeyError, TypeError) as error:
        raise PredictionError(f"prediction data requires column {name!r}") from error
    values = _prediction_column(raw, name)
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise PredictionError(f"prediction column {name!r} must be numeric") from error
    if result.shape != (n,):
        raise PredictionError(f"prediction column {name!r} must have one value per row")
    if not np.isfinite(result).all():
        raise PredictionError(f"prediction column {name!r} contains non-finite values")
    return result


def _group_label(value: object, group_name: str) -> str | None:
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    if not isinstance(value, str):
        raise PredictionError(
            f"grouping column {group_name!r} must contain string labels "
            "or missing values"
        )
    return value


def _prediction_rows(
    data: PredictionData, group_name: str, *, require_group: bool
) -> tuple[int, tuple[str, ...], Sequence[object] | None]:
    if isinstance(data, pd.DataFrame):
        n = len(data)
        row_ids = tuple(str(value) for value in data.index)
        groups: Sequence[object] | None = (
            data[group_name].tolist() if group_name in data else None
        )
    else:
        try:
            first = next(iter(data.values()))
        except StopIteration as error:
            raise PredictionError(
                "prediction data must contain at least one column"
            ) from error
        first_column = _prediction_column(first, "<first>")
        n = len(first_column)
        row_ids = tuple(str(index) for index in range(n))
        raw_groups = data.get(group_name)
        groups = (
            None if raw_groups is None else _prediction_column(raw_groups, group_name)
        )
        for name, values in data.items():
            column = _prediction_column(values, name)
            if len(column) != n:
                raise PredictionError("prediction columns must have equal lengths")
    if n == 0:
        raise PredictionError("prediction data must contain at least one row")
    if len(set(row_ids)) != n:
        raise PredictionError("prediction row identifiers must be unique")
    if require_group and groups is None:
        raise PredictionError(f"conditional prediction requires column {group_name!r}")
    return n, row_ids, groups


def _predict_new_data(
    *,
    data: PredictionData,
    mode: PredictionMode,
    allow_new_groups: bool,
    offset: PredictionOffsetInput | None,
    beta: FloatArray,
    fixed_encoder: FixedEncoder,
    random_effects: FloatArray,
    group_name: str,
    group_levels: tuple[str, ...],
    random_coefficient_names: tuple[str, ...],
    predictor_name: str | None,
    formula_offset_names: tuple[str, ...],
    requires_explicit_offset: bool,
    random_terms: tuple[RandomTermDesign, ...] = (),
) -> PredictionResult:
    n, row_ids, groups = _prediction_rows(
        data,
        group_name,
        require_group=mode == "conditional" and not random_terms,
    )
    formula_offset = np.zeros(n, dtype=np.float64)
    for name in formula_offset_names:
        formula_offset += _numeric_prediction_column(data, name, n)
    if offset is None:
        if requires_explicit_offset:
            raise PredictionError(
                "new data requires an explicit offset because the model "
                "was fitted with an argument-only offset"
            )
        prediction_offset = formula_offset
    else:
        prediction_offset = formula_offset + _prediction_offset(offset, n, data)

    for variable in fixed_encoder.variables:
        if variable.name not in data:
            raise PredictionError(f"prediction data requires column {variable.name!r}")
    try:
        fixed_design = fixed_encoder.evaluate(data, n)
    except Exception as error:
        if isinstance(error, PredictionError):
            raise
        raise PredictionError(f"fixed-effect encoding failed: {error}") from error
    random_design = (
        np.ones((n, 1), dtype=np.float64)
        if predictor_name is None
        else np.column_stack(
            (
                np.ones(n, dtype=np.float64),
                _numeric_prediction_column(data, predictor_name, n),
            )
        )
    )
    values = fixed_design @ beta + prediction_offset
    new_group = [False] * n
    if mode == "conditional" and random_terms:
        effect_cursor = 0
        for term in random_terms:
            term_groups: list[str | None] = []
            source_columns: list[Sequence[object]] = []
            for name in term.source_names:
                try:
                    source_columns.append(_prediction_column(data[name], name))
                except (KeyError, TypeError) as error:
                    raise PredictionError(
                        f"conditional prediction requires column {name!r}"
                    ) from error
            for components in zip(*source_columns, strict=True):
                labels = [_group_label(value, term.group_name) for value in components]
                term_groups.append(
                    None if any(label is None for label in labels) else ":".join(labels)  # type: ignore[arg-type]
                )
            coefficient_count = len(term.random_coefficient_names)
            effect_count = len(term.group_levels) * coefficient_count
            effect_matrix = random_effects[
                effect_cursor : effect_cursor + effect_count
            ].reshape(len(term.group_levels), coefficient_count)
            effect_cursor += effect_count
            effects = dict(zip(term.group_levels, effect_matrix, strict=True))
            random_design = (
                np.ones((n, 1), dtype=np.float64)
                if term.predictor_name is None
                else np.column_stack(
                    (
                        np.ones(n, dtype=np.float64),
                        _numeric_prediction_column(data, term.predictor_name, n),
                    )
                )
            )
            for index, label in enumerate(term_groups):
                if label not in effects:
                    if not allow_new_groups:
                        description = "missing" if label is None else repr(label)
                        raise PredictionError(
                            f"new or missing group {description} in {term.group_name!r}"
                        )
                    new_group[index] = True
                    continue
                values[index] += float(random_design[index] @ effects[label])
        return PredictionResult(
            values=_readonly(values),
            row_ids=row_ids,
            mode=mode,
            new_group=tuple(new_group),
        )
    if mode == "conditional":
        if groups is None:
            raise PredictionError(
                f"conditional prediction requires column {group_name!r}"
            )
        effect_matrix = random_effects.reshape(
            len(group_levels), len(random_coefficient_names)
        )
        effects = dict(zip(group_levels, effect_matrix, strict=True))
        for index, raw_group in enumerate(groups):
            label = _group_label(raw_group, group_name)
            if label not in effects:
                if not allow_new_groups:
                    description = "missing" if label is None else repr(label)
                    raise PredictionError(
                        f"new or missing group {description} in {group_name!r}"
                    )
                new_group[index] = True
                continue
            values[index] += float(random_design[index] @ effects[label])
    return PredictionResult(
        values=_readonly(values),
        row_ids=row_ids,
        mode=mode,
        new_group=tuple(new_group),
    )


@dataclass(frozen=True, slots=True)
class PredictionOnlyModel:
    """Validated, immutable model state loaded from a safe prediction bundle."""

    formula: str
    kind: ObjectiveKind
    objective: float
    log_likelihood: float
    theta: FloatArray
    beta: FloatArray
    beta_covariance: FloatArray
    sigma2: float
    random_variance: float
    random_covariance: FloatArray
    random_effects: FloatArray
    fixed_names: tuple[str, ...]
    random_names: tuple[str, ...]
    group_name: str
    group_levels: tuple[str, ...]
    random_coefficient_names: tuple[str, ...]
    covariance_term_sizes: tuple[int, ...]
    fixed_encoder: FixedEncoder
    formula_offset_names: tuple[str, ...]
    diagnostics: OptimizerDiagnostics
    predictor_name: str | None
    requires_explicit_offset: bool
    random_terms: tuple[RandomTermDesign, ...] = ()

    @property
    def sigma(self) -> float:
        """Residual standard deviation."""
        return float(np.sqrt(self.sigma2))

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Operations deliberately retained by a prediction-only bundle."""
        return ("population_prediction", "conditional_prediction")

    def predict(
        self,
        data: PredictionData | None = None,
        *,
        mode: PredictionMode,
        allow_new_groups: bool = False,
        offset: PredictionOffsetInput | None = None,
    ) -> PredictionResult:
        """Predict from explicit new data; training rows are not stored."""
        if mode not in ("population", "conditional"):
            raise PredictionError("mode must be 'population' or 'conditional'")
        if not isinstance(allow_new_groups, bool):
            raise PredictionError("allow_new_groups must be a boolean")
        if data is None:
            raise PredictionError(
                "prediction-only bundles do not store training rows; provide data"
            )
        return _predict_new_data(
            data=data,
            mode=mode,
            allow_new_groups=allow_new_groups,
            offset=offset,
            beta=self.beta,
            fixed_encoder=self.fixed_encoder,
            random_effects=self.random_effects,
            group_name=self.group_name,
            group_levels=self.group_levels,
            random_coefficient_names=self.random_coefficient_names,
            predictor_name=self.predictor_name,
            formula_offset_names=self.formula_offset_names,
            requires_explicit_offset=self.requires_explicit_offset,
            random_terms=self.random_terms,
        )


@dataclass(frozen=True, slots=True)
class LinearMixedModelResult:
    """Immutable first-alpha result for one independent grouping structure."""

    formula: str
    kind: ObjectiveKind
    objective: float
    log_likelihood: float
    theta: FloatArray
    beta: FloatArray
    beta_covariance: FloatArray
    sigma2: float
    random_variance: float
    random_covariance: FloatArray
    u: FloatArray
    random_effects: FloatArray
    fixed_names: tuple[str, ...]
    random_names: tuple[str, ...]
    group_name: str
    group_levels: tuple[str, ...]
    random_coefficient_names: tuple[str, ...]
    covariance_term_sizes: tuple[int, ...]
    fixed_encoder: FixedEncoder
    formula_offset_names: tuple[str, ...]
    row_ids: tuple[str, ...]
    omitted_row_ids: tuple[str, ...]
    excluded_row_ids: tuple[str, ...]
    na_action: NaAction
    fitted_values: FloatArray
    residuals: FloatArray
    diagnostics: OptimizerDiagnostics
    _training_groups: tuple[str, ...]
    _training_offset: FloatArray
    _predictor_name: str | None
    _requires_explicit_offset: bool
    _training_fixed_design: FloatArray
    _training_random_design: FloatArray
    random_terms: tuple[RandomTermDesign, ...] = ()
    _training_group_terms: tuple[tuple[str, ...], ...] = ()
    _training_random_design_terms: tuple[FloatArray, ...] = ()

    @property
    def sigma(self) -> float:
        """Residual standard deviation."""
        return float(np.sqrt(self.sigma2))

    @property
    def predictor_name(self) -> str | None:
        """Numeric predictor retained by the verified slope profile, if any."""
        return self._predictor_name

    @property
    def requires_explicit_offset(self) -> bool:
        """Whether prediction on new data requires an offset vector."""
        return self._requires_explicit_offset

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Atomically save a safe, prediction-only model bundle."""
        from kamino.bundle import save_model_bundle

        return save_model_bundle(self, path, overwrite=overwrite)

    def predict(
        self,
        data: PredictionData | None = None,
        *,
        mode: PredictionMode,
        allow_new_groups: bool = False,
        offset: PredictionOffsetInput | None = None,
    ) -> PredictionResult:
        """Predict conditionally or at the population level.

        Training predictions reuse the fitted total offset. New data reevaluates
        formula offsets and must supply a new argument offset whenever one was
        supplied during fitting.
        """
        if mode not in ("population", "conditional"):
            raise PredictionError("mode must be 'population' or 'conditional'")
        if not isinstance(allow_new_groups, bool):
            raise PredictionError("allow_new_groups must be a boolean")
        if data is None:
            if offset is not None:
                raise PredictionError("offset cannot be replaced for training data")
            row_ids = self.row_ids
            groups: Sequence[object] = self._training_groups
            n = len(row_ids)
            prediction_offset = self._training_offset
            fixed_design = self._training_fixed_design
            random_design = self._training_random_design
        else:
            return _predict_new_data(
                data=data,
                mode=mode,
                allow_new_groups=allow_new_groups,
                offset=offset,
                beta=self.beta,
                fixed_encoder=self.fixed_encoder,
                random_effects=self.random_effects,
                group_name=self.group_name,
                group_levels=self.group_levels,
                random_coefficient_names=self.random_coefficient_names,
                predictor_name=self._predictor_name,
                formula_offset_names=self.formula_offset_names,
                requires_explicit_offset=self.requires_explicit_offset,
                random_terms=self.random_terms,
            )

        values = fixed_design @ self.beta + prediction_offset
        new_group = [False] * n
        if mode == "conditional" and self.random_terms:
            effect_cursor = 0
            for term, term_groups, random_design in zip(
                self.random_terms,
                self._training_group_terms,
                self._training_random_design_terms,
                strict=True,
            ):
                coefficient_count = len(term.random_coefficient_names)
                effect_count = len(term.group_levels) * coefficient_count
                effect_matrix = self.random_effects[
                    effect_cursor : effect_cursor + effect_count
                ].reshape(len(term.group_levels), coefficient_count)
                effect_cursor += effect_count
                effects = dict(zip(term.group_levels, effect_matrix, strict=True))
                for index, label in enumerate(term_groups):
                    values[index] += float(random_design[index] @ effects[label])
        elif mode == "conditional":
            effect_matrix = self.random_effects.reshape(
                len(self.group_levels), len(self.random_coefficient_names)
            )
            effects = dict(zip(self.group_levels, effect_matrix, strict=True))
            for index, raw_group in enumerate(groups):
                label = _group_label(raw_group, self.group_name)
                if label not in effects:
                    if not allow_new_groups:
                        description = "missing" if label is None else repr(label)
                        raise PredictionError(
                            f"new or missing group {description} in {self.group_name!r}"
                        )
                    new_group[index] = True
                    continue
                values[index] += float(random_design[index] @ effects[label])
        values = _readonly(values)
        return PredictionResult(
            values=values,
            row_ids=row_ids,
            mode=mode,
            new_group=tuple(new_group),
        )


__all__ = [
    "LinearMixedModelResult",
    "OptimizerDiagnostics",
    "PredictionOnlyModel",
    "PredictionData",
    "PredictionMode",
    "PredictionResult",
]
