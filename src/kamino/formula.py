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
from kamino.model import ModelSpec, VectorInput

ColumnInput: TypeAlias = Sequence[object] | np.ndarray[Any, Any]
DataInput: TypeAlias = pd.DataFrame | Mapping[str, ColumnInput]

_IDENTIFIER = r"[A-Za-z_]\w*"
_RANDOM_INTERCEPT = re.compile(
    rf"^\s*(?P<response>{_IDENTIFIER})\s*~\s*"
    rf"(?:1\s*\+\s*)?\(\s*1\s*\|\s*(?P<group>{_IDENTIFIER})\s*\)\s*$"
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
class RandomInterceptDesign:
    """Canonical design and encoder state for one random-intercept term."""

    spec: ModelSpec
    formula: str
    response_name: str
    group_name: str
    group_levels: tuple[str, ...]
    training_groups: tuple[str, ...]


def build_random_intercept_design(
    formula: str,
    data: DataInput,
    *,
    weights: VectorInput | None = None,
    offset: VectorInput | None = None,
) -> RandomInterceptDesign:
    """Build the only formula profile currently accepted by the public fitter."""
    if not isinstance(formula, str):
        raise UnsupportedFormulaError("formula must be a string")
    match = _RANDOM_INTERCEPT.fullmatch(formula)
    if match is None:
        raise UnsupportedFormulaError(
            "the alpha fitter accepts only 'response ~ 1 + (1 | group)'"
        )
    response_name = match.group("response")
    group_name = match.group("group")
    response = _column(data, response_name)
    groups_raw = _column(data, group_name)
    if len(response) != len(groups_raw):
        raise ModelSpecificationError("response and grouping columns must align")
    n = len(response)
    if n == 0:
        raise ModelSpecificationError("data must contain at least one row")
    levels = _group_levels(data, group_name, groups_raw)
    groups = tuple(str(value) for value in groups_raw)
    row_ids = _row_ids(data, n)
    frame = pd.DataFrame(
        {
            response_name: response,
            group_name: pd.Categorical(groups, categories=levels, ordered=True),
        },
        index=row_ids,
    )
    try:
        matrices: Any = design_matrices(formula, frame, na_action="error")
    except Exception as error:
        raise ModelSpecificationError(f"formula evaluation failed: {error}") from error

    x = np.asarray(matrices.common).astype(np.float64, copy=False)
    response_array = (
        np.asarray(matrices.response).astype(np.float64, copy=False).reshape(-1)
    )
    if x.shape != (n, 1) or not np.array_equal(x, np.ones((n, 1))):
        raise UnsupportedFormulaError("the alpha fitter requires one fixed intercept")
    terms = list(matrices.group.terms.values())
    if len(terms) != 1 or terms[0].kind != "intercept":
        raise UnsupportedFormulaError("the alpha fitter requires one random intercept")
    raw_levels = tuple(str(value) for value in terms[0].groups)
    raw_z = np.asarray(matrices.group).astype(np.float64, copy=False)
    if set(raw_levels) != set(levels) or raw_z.shape != (n, len(levels)):
        raise ModelSpecificationError(
            "formula backend returned an invalid group design"
        )
    z = raw_z[:, [raw_levels.index(level) for level in levels]]
    expected_z = np.zeros((n, len(levels)), dtype=np.float64)
    indices = {level: index for index, level in enumerate(levels)}
    expected_z[np.arange(n), [indices[group] for group in groups]] = 1.0
    if not np.array_equal(z, expected_z):
        raise ModelSpecificationError("formula backend group design failed validation")

    spec = ModelSpec.from_arrays(
        y=response_array,
        x=x,
        z=z,
        weights=weights,
        offset=offset,
        row_ids=row_ids,
        fixed_names=("(Intercept)",),
        random_names=tuple(f"{group_name}[{level}]:(Intercept)" for level in levels),
    )
    return RandomInterceptDesign(
        spec=spec,
        formula=f"{response_name} ~ 1 + (1 | {group_name})",
        response_name=response_name,
        group_name=group_name,
        group_levels=levels,
        training_groups=groups,
    )


__all__ = [
    "ColumnInput",
    "DataInput",
    "RandomInterceptDesign",
    "build_random_intercept_design",
]
