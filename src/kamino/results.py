"""Immutable fitted-model and prediction results."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnnecessaryIsInstance=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd

from kamino.errors import PredictionError
from kamino.formula import ColumnInput
from kamino.model import FloatArray, ObjectiveKind, VectorInput

PredictionData: TypeAlias = pd.DataFrame | Mapping[str, ColumnInput]
PredictionMode: TypeAlias = Literal["population", "conditional"]


def _readonly(value: FloatArray) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class OptimizerDiagnostics:
    """Structured outcome from the one-parameter reference optimizer."""

    converged: bool
    message: str
    evaluations: int
    boundary: bool
    lower_bound: float
    search_upper_bound: float
    backend: str


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Labeled predictions with explicit conditioning and new-group status."""

    values: FloatArray
    row_ids: tuple[str, ...]
    mode: PredictionMode
    new_group: tuple[bool, ...]


def _prediction_column(value: object, name: str) -> Sequence[object]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise PredictionError(f"prediction column {name!r} must be one-dimensional")
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    raise PredictionError(f"prediction column {name!r} must be a sequence")


def _prediction_offset(value: VectorInput, n: int) -> FloatArray:
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise PredictionError("prediction offset must be numeric") from error
    if result.ndim != 1 or result.shape != (n,):
        raise PredictionError("prediction offset must have one value per row")
    if not np.isfinite(result).all():
        raise PredictionError("prediction offset contains non-finite values")
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


@dataclass(frozen=True, slots=True)
class LinearMixedModelResult:
    """Immutable first-alpha result for one random-intercept model."""

    formula: str
    kind: ObjectiveKind
    objective: float
    log_likelihood: float
    theta: FloatArray
    beta: FloatArray
    beta_covariance: FloatArray
    sigma2: float
    random_variance: float
    u: FloatArray
    random_effects: FloatArray
    fixed_names: tuple[str, ...]
    random_names: tuple[str, ...]
    group_name: str
    group_levels: tuple[str, ...]
    row_ids: tuple[str, ...]
    fitted_values: FloatArray
    residuals: FloatArray
    diagnostics: OptimizerDiagnostics
    _training_groups: tuple[str, ...]
    _training_offset: FloatArray

    @property
    def sigma(self) -> float:
        """Residual standard deviation."""
        return float(np.sqrt(self.sigma2))

    def predict(
        self,
        data: PredictionData | None = None,
        *,
        mode: PredictionMode,
        allow_new_groups: bool = False,
        offset: VectorInput | None = None,
    ) -> PredictionResult:
        """Predict conditionally or at the population level.

        Training predictions reuse the fitted offset. New data must supply a new
        offset when the model was fitted with a nonzero argument-only offset.
        """
        if mode not in ("population", "conditional"):
            raise PredictionError("mode must be 'population' or 'conditional'")
        if not isinstance(allow_new_groups, bool):
            raise PredictionError("allow_new_groups must be a boolean")
        if data is None:
            if offset is not None:
                raise PredictionError("offset cannot be replaced for training data")
            row_ids = self.row_ids
            groups: Sequence[object] | None = self._training_groups
            n = len(row_ids)
            prediction_offset = self._training_offset
        else:
            n, row_ids, groups = _prediction_rows(
                data, self.group_name, require_group=mode == "conditional"
            )
            if offset is None:
                if np.any(self._training_offset != 0.0):
                    raise PredictionError(
                        "new data requires an explicit offset because the model "
                        "was fitted with an argument-only offset"
                    )
                prediction_offset = np.zeros(n, dtype=np.float64)
            else:
                prediction_offset = _prediction_offset(offset, n)

        values = np.full(n, self.beta[0], dtype=np.float64) + prediction_offset
        new_group = [False] * n
        if mode == "conditional":
            if groups is None:
                raise PredictionError(
                    f"conditional prediction requires column {self.group_name!r}"
                )
            effects = dict(zip(self.group_levels, self.random_effects, strict=True))
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
                values[index] += effects[label]
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
    "PredictionData",
    "PredictionMode",
    "PredictionResult",
]
