"""Restricted, labeled formula boundary for the first Phase 1 fit."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnnecessaryIsInstance=false

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from formulae import design_matrices

from kamino.errors import ModelSpecificationError, UnsupportedFormulaError
from kamino.model import SingleGroupSpec, VectorInput

ColumnInput: TypeAlias = Sequence[object] | np.ndarray[Any, Any]
DataInput: TypeAlias = pd.DataFrame | Mapping[str, ColumnInput]

_IDENTIFIER = r"[A-Za-z_]\w*"
_RANDOM_INTERCEPT = re.compile(
    rf"^\s*(?P<response>{_IDENTIFIER})\s*~\s*"
    rf"(?:1\s*\+\s*)?\(\s*1\s*\|\s*(?P<group>{_IDENTIFIER})\s*\)\s*$"
)
_RANDOM_SLOPE = re.compile(
    rf"^\s*(?P<response>{_IDENTIFIER})\s*~\s*"
    rf"(?:1\s*\+\s*)?(?P<predictor>{_IDENTIFIER})\s*\+\s*"
    rf"\(\s*1\s*\+\s*(?P=predictor)\s*\|\s*"
    rf"(?P<group>{_IDENTIFIER})\s*\)\s*$"
)


def _column(data: DataInput, name: str) -> Sequence[object]:
    try:
        value: Any = data[name]
    except (KeyError, TypeError) as error:
        raise ModelSpecificationError(f"data has no column {name!r}") from error
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ModelSpecificationError(f"column {name!r} must be one-dimensional")
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    raise ModelSpecificationError(f"column {name!r} must be a one-dimensional sequence")


def _row_ids(data: DataInput, n: int) -> tuple[str, ...]:
    if isinstance(data, pd.DataFrame):
        identifiers = tuple(str(value) for value in data.index)
    else:
        identifiers = tuple(str(index) for index in range(n))
    if len(set(identifiers)) != n:
        raise ModelSpecificationError("data row identifiers must be unique")
    return identifiers


def _group_levels(
    data: DataInput, name: str, values: Sequence[object]
) -> tuple[str, ...]:
    for value in values:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            raise ModelSpecificationError(
                f"grouping column {name!r} contains missing values"
            )
        if not isinstance(value, str):
            raise ModelSpecificationError(
                f"grouping column {name!r} must contain string labels"
            )
    labels = tuple(str(value) for value in values)
    if isinstance(data, pd.DataFrame) and isinstance(
        data[name].dtype, pd.CategoricalDtype
    ):
        categories = tuple(str(value) for value in data[name].cat.categories)
        observed = set(labels)
        levels = tuple(value for value in categories if value in observed)
    else:
        levels = tuple(sorted(set(labels)))
    if not levels:
        raise ModelSpecificationError(f"grouping column {name!r} has no levels")
    return levels


@dataclass(frozen=True, slots=True)
class SingleGroupDesign:
    """Canonical design and encoder state for one grouped covariance term."""

    spec: SingleGroupSpec
    formula: str
    response_name: str
    group_name: str
    group_levels: tuple[str, ...]
    training_groups: tuple[str, ...]
    predictor_name: str | None
    random_coefficient_names: tuple[str, ...]

    @property
    def group_indices(self) -> np.ndarray[Any, np.dtype[np.int64]]:
        """Return the compact observation-to-group map."""
        return self.spec.group_indices


def build_single_group_design(
    formula: str,
    data: DataInput,
    *,
    weights: VectorInput | None = None,
    offset: VectorInput | None = None,
) -> SingleGroupDesign:
    """Build an accepted single-group random-intercept or slope design."""
    if not isinstance(formula, str):
        raise UnsupportedFormulaError("formula must be a string")
    intercept_match = _RANDOM_INTERCEPT.fullmatch(formula)
    slope_match = _RANDOM_SLOPE.fullmatch(formula)
    match = intercept_match or slope_match
    if match is None:
        raise UnsupportedFormulaError(
            "the alpha fitter accepts only 'response ~ 1 + (1 | group)' or "
            "'response ~ predictor + (1 + predictor | group)'"
        )
    response_name = match.group("response")
    group_name = match.group("group")
    predictor_name = None if slope_match is None else match.group("predictor")
    response = _column(data, response_name)
    groups_raw = _column(data, group_name)
    predictor = None if predictor_name is None else _column(data, predictor_name)
    lengths = {len(response), len(groups_raw)}
    if predictor is not None:
        lengths.add(len(predictor))
    if len(lengths) != 1:
        raise ModelSpecificationError(
            "response, predictor, and grouping columns must align"
        )
    n = len(response)
    if n == 0:
        raise ModelSpecificationError("data must contain at least one row")
    levels = _group_levels(data, group_name, groups_raw)
    groups = tuple(str(value) for value in groups_raw)
    row_ids = _row_ids(data, n)
    frame_columns: dict[str, Sequence[object]] = {response_name: response}
    if predictor_name is not None and predictor is not None:
        frame_columns[predictor_name] = predictor
    frame = pd.DataFrame(frame_columns, index=row_ids)
    fixed_formula = (
        f"{response_name} ~ 1"
        if predictor_name is None
        else f"{response_name} ~ {predictor_name}"
    )
    try:
        matrices: Any = design_matrices(fixed_formula, frame, na_action="error")
    except Exception as error:
        raise ModelSpecificationError(f"formula evaluation failed: {error}") from error

    x = np.asarray(matrices.common).astype(np.float64, copy=False)
    response_array = (
        np.asarray(matrices.response).astype(np.float64, copy=False).reshape(-1)
    )
    expected_columns = 1 if predictor_name is None else 2
    if x.shape != (n, expected_columns) or not np.array_equal(x[:, 0], np.ones(n)):
        raise UnsupportedFormulaError(
            "the alpha fitter requires a fixed intercept and at most one "
            "numeric predictor"
        )
    if predictor_name is not None:
        try:
            predictor_array = np.asarray(predictor, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ModelSpecificationError(
                f"predictor column {predictor_name!r} must be numeric"
            ) from error
        if predictor_array.shape != (n,) or not np.array_equal(
            x[:, 1], predictor_array
        ):
            raise ModelSpecificationError(
                "formula backend returned an invalid numeric predictor design"
            )
    random_design = x
    fixed_names = (
        ("(Intercept)",) if predictor_name is None else ("(Intercept)", predictor_name)
    )
    random_coefficient_names = fixed_names
    indices = {level: index for index, level in enumerate(levels)}
    group_indices = np.fromiter(
        (indices[group] for group in groups), dtype=np.int64, count=n
    )

    spec = SingleGroupSpec.from_arrays(
        y=response_array,
        x=x,
        group_indices=group_indices,
        random_design=random_design,
        group_count=len(levels),
        weights=weights,
        offset=offset,
        row_ids=row_ids,
        fixed_names=fixed_names,
        random_names=tuple(
            f"{group_name}[{level}]:{coefficient}"
            for level in levels
            for coefficient in random_coefficient_names
        ),
    )
    canonical_fixed = "1" if predictor_name is None else f"1 + {predictor_name}"
    canonical_random = "1" if predictor_name is None else f"1 + {predictor_name}"
    return SingleGroupDesign(
        spec=spec,
        formula=(
            f"{response_name} ~ {canonical_fixed} + ({canonical_random} | {group_name})"
        ),
        response_name=response_name,
        group_name=group_name,
        group_levels=levels,
        training_groups=groups,
        predictor_name=predictor_name,
        random_coefficient_names=random_coefficient_names,
    )


def build_random_intercept_design(
    formula: str,
    data: DataInput,
    *,
    weights: VectorInput | None = None,
    offset: VectorInput | None = None,
) -> SingleGroupDesign:
    """Build the retained random-intercept-only development entry point."""
    design = build_single_group_design(formula, data, weights=weights, offset=offset)
    if design.spec.k != 1:
        raise UnsupportedFormulaError(
            "build_random_intercept_design accepts only a random intercept"
        )
    return design


__all__ = [
    "ColumnInput",
    "DataInput",
    "SingleGroupDesign",
    "build_random_intercept_design",
    "build_single_group_design",
]
