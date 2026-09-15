"""Restricted, labeled formula and shared model-frame boundary."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnnecessaryIsInstance=false

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
from formulae import design_matrices

from kamino.errors import ModelSpecificationError, UnsupportedFormulaError
from kamino.model import (
    FloatArray,
    GeneralSparseSpec,
    SingleGroupSpec,
    SparseRandomTermSpec,
    VectorInput,
)

ColumnInput: TypeAlias = Sequence[object] | np.ndarray[Any, Any]
DataInput: TypeAlias = pd.DataFrame | Mapping[str, ColumnInput]
ContrastKind: TypeAlias = Literal["treatment", "sum"]
ContrastInput: TypeAlias = Mapping[str, ContrastKind]
NaAction: TypeAlias = Literal["error", "omit"]
SubsetInput: TypeAlias = Sequence[bool] | np.ndarray[Any, Any] | pd.Series
FrameVectorInput: TypeAlias = VectorInput | pd.Series

_IDENTIFIER = r"[A-Za-z_]\w*"
_RANDOM_INTERCEPT = re.compile(
    rf"^\s*(?P<response>{_IDENTIFIER})\s*~\s*"
    rf"(?:(?P<fixed>.+?)\s*\+\s*)?"
    rf"\(\s*1\s*\|\s*(?P<group>{_IDENTIFIER})\s*\)\s*$"
)
_RANDOM_SLOPE = re.compile(
    rf"^\s*(?P<response>{_IDENTIFIER})\s*~\s*"
    rf"(?P<fixed>.+?)\s*\+\s*"
    rf"\(\s*1\s*\+\s*(?P<random_predictor>{_IDENTIFIER})\s*\|\s*"
    rf"(?P<group>{_IDENTIFIER})\s*\)\s*$"
)
_INDEPENDENT_RANDOM_TERMS = re.compile(
    rf"^\s*(?P<response>{_IDENTIFIER})\s*~\s*"
    rf"(?P<fixed>.+?)\s*\+\s*"
    rf"\(\s*1\s*\|\s*(?P<group>{_IDENTIFIER})\s*\)\s*\+\s*"
    rf"\(\s*0\s*\+\s*(?P<random_predictor>{_IDENTIFIER})\s*\|\s*"
    rf"(?P=group)\s*\)\s*$"
)
_NUMERIC_DOUBLE_BAR = re.compile(
    rf"^\s*(?P<response>{_IDENTIFIER})\s*~\s*"
    rf"(?P<fixed>.+?)\s*\+\s*"
    rf"\(\s*1\s*\+\s*(?P<random_predictor>{_IDENTIFIER})\s*\|\|\s*"
    rf"(?P<group>{_IDENTIFIER})\s*\)\s*$"
)
_OFFSET = re.compile(rf"offset\(\s*(?P<name>{_IDENTIFIER})\s*\)")
_INTERACTION = re.compile(rf"(?P<left>{_IDENTIFIER})\s*\*\s*(?P<right>{_IDENTIFIER})")
_GENERAL_RANDOM_TERM = re.compile(
    rf"\(\s*1(?:\s*\+\s*(?P<random_predictor>{_IDENTIFIER}))?\s*\|\s*"
    rf"(?P<group>{_IDENTIFIER}(?:\s*[:/]\s*{_IDENTIFIER})?)\s*\)"
)

_RANK_TOLERANCE = 1e-7
_ESTIMABILITY_TOLERANCE = 1e-8


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


def _missing(value: object) -> bool:
    result = pd.isna(value)
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _series_aligned(series: pd.Series, frame: pd.DataFrame) -> bool:
    series_index: Any = series.index
    frame_index: Any = frame.index
    return bool(series_index.equals(frame_index))


def _row_ids(data: DataInput, n: int) -> tuple[str, ...]:
    if isinstance(data, pd.DataFrame):
        identifiers = tuple(str(value) for value in data.index)
    else:
        identifiers = tuple(str(index) for index in range(n))
    if len(set(identifiers)) != n:
        raise ModelSpecificationError("data row identifiers must be unique")
    return identifiers


def _categorical_levels(
    data: DataInput, name: str, values: Sequence[object]
) -> tuple[str, ...]:
    labels = tuple(str(value) for value in values)
    if isinstance(data, pd.DataFrame) and isinstance(
        data[name].dtype, pd.CategoricalDtype
    ):
        categories = tuple(str(value) for value in data[name].cat.categories)
        observed = set(labels)
        levels = tuple(value for value in categories if value in observed)
    else:
        levels = tuple(sorted(set(labels)))
    if len(set(levels)) != len(levels):
        raise ModelSpecificationError(
            f"categorical column {name!r} has ambiguous string labels"
        )
    if len(levels) < 2:
        raise ModelSpecificationError(
            f"categorical column {name!r} must have at least two observed levels"
        )
    return levels


def _group_levels(
    data: DataInput, name: str, values: Sequence[object]
) -> tuple[str, ...]:
    for value in values:
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
class FixedVariable:
    """Serializable encoding state for one fixed-effect variable."""

    name: str
    kind: Literal["numeric", "categorical"]
    levels: tuple[str, ...] = ()
    contrast: ContrastKind | None = None


@dataclass(frozen=True, slots=True)
class FixedEncoder:
    """Serializable fixed-design terms, levels, and contrast state."""

    variables: tuple[FixedVariable, ...]
    terms: tuple[tuple[str, ...], ...]

    @property
    def fixed_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._column_definitions())

    def _column_definitions(
        self,
    ) -> tuple[tuple[str, tuple[tuple[FixedVariable, int], ...]], ...]:
        by_name = {variable.name: variable for variable in self.variables}
        columns: list[tuple[str, tuple[tuple[FixedVariable, int], ...]]] = []
        for term in self.terms:
            if not term:
                columns.append(("(Intercept)", ()))
                continue
            components: list[list[tuple[str, FixedVariable, int]]] = []
            for name in term:
                variable = by_name[name]
                if variable.kind == "numeric":
                    components.append([(name, variable, 0)])
                elif variable.contrast == "treatment":
                    components.append(
                        [
                            (f"{name}{level}", variable, index)
                            for index, level in enumerate(variable.levels[1:], start=1)
                        ]
                    )
                else:
                    components.append(
                        [
                            (f"{name}{index + 1}", variable, index)
                            for index in range(len(variable.levels) - 1)
                        ]
                    )
            for combination in product(*components):
                columns.append(
                    (
                        ":".join(item[0] for item in combination),
                        tuple((item[1], item[2]) for item in combination),
                    )
                )
        return tuple(columns)

    def evaluate(self, data: DataInput, n: int) -> FloatArray:
        numeric: dict[str, FloatArray] = {}
        categorical: dict[str, np.ndarray[Any, np.dtype[np.str_]]] = {}
        for variable in self.variables:
            values = _column(data, variable.name)
            if len(values) != n:
                raise ModelSpecificationError(
                    "fixed-effect columns must align with the model frame"
                )
            if any(_missing(value) for value in values):
                raise ModelSpecificationError(
                    f"fixed-effect column {variable.name!r} contains missing or "
                    "non-finite values"
                )
            if variable.kind == "numeric":
                try:
                    converted = np.asarray(values, dtype=np.float64)
                except (TypeError, ValueError) as error:
                    raise ModelSpecificationError(
                        f"fixed-effect column {variable.name!r} must be numeric"
                    ) from error
                if converted.shape != (n,) or not np.isfinite(converted).all():
                    raise ModelSpecificationError(
                        f"fixed-effect column {variable.name!r} must be finite numeric"
                    )
                numeric[variable.name] = converted
            else:
                labels = np.asarray([str(value) for value in values], dtype=np.str_)
                unknown = sorted(set(labels) - set(variable.levels))
                if unknown:
                    raise ModelSpecificationError(
                        "fixed-effect column "
                        f"{variable.name!r} contains unknown levels "
                        f"{unknown!r}"
                    )
                categorical[variable.name] = labels

        matrix = np.empty((n, len(self.fixed_names)), dtype=np.float64)
        for column_index, (_, factors) in enumerate(self._column_definitions()):
            values = np.ones(n, dtype=np.float64)
            for variable, code in factors:
                if variable.kind == "numeric":
                    values *= numeric[variable.name]
                elif variable.contrast == "treatment":
                    values *= categorical[variable.name] == variable.levels[code]
                else:
                    labels = categorical[variable.name]
                    contrast_values = np.zeros(n, dtype=np.float64)
                    contrast_values[labels == variable.levels[code]] = 1.0
                    contrast_values[labels == variable.levels[-1]] = -1.0
                    values *= contrast_values
            matrix[:, column_index] = values
        return matrix


@dataclass(frozen=True, slots=True)
class FixedRank:
    """Coefficient-aligned lme4 rank reduction and estimability state."""

    full_names: tuple[str, ...]
    retained_indices: tuple[int, ...]
    dropped_indices: tuple[int, ...]
    pivot: tuple[int, ...]
    tolerance: float
    estimability_tolerance: float
    null_basis: FloatArray

    @property
    def rank(self) -> int:
        return len(self.retained_indices)

    @property
    def retained_names(self) -> tuple[str, ...]:
        return tuple(self.full_names[index] for index in self.retained_indices)

    @property
    def dropped_names(self) -> tuple[str, ...]:
        return tuple(self.full_names[index] for index in self.dropped_indices)

    def reduce(self, matrix: FloatArray) -> FloatArray:
        if matrix.ndim != 2 or matrix.shape[1] != len(self.full_names):
            raise ModelSpecificationError(
                "fixed design does not match the full coefficient space"
            )
        return np.asarray(matrix[:, self.retained_indices], dtype=np.float64)

    def is_estimable(self, contrast: Sequence[float] | FloatArray) -> bool:
        values = np.asarray(contrast, dtype=np.float64)
        if values.shape != (len(self.full_names),) or not np.isfinite(values).all():
            raise ModelSpecificationError(
                "contrast must be a finite vector in the full coefficient space"
            )
        if self.null_basis.shape[1] == 0:
            return True
        discrepancy = self.null_basis.T @ values
        scale = max(1.0, float(np.linalg.norm(values, ord=2)))
        return bool(np.max(np.abs(discrepancy)) <= self.estimability_tolerance * scale)


def _linpack_qr_pivot(
    matrix: FloatArray, *, tolerance: float
) -> tuple[tuple[int, ...], int]:
    """Reproduce the column-moving policy of R's non-LAPACK ``dqrdc2`` QR.

    Unlike a norm-pivoted LAPACK QR, ``dqrdc2`` preserves source order and moves
    columns whose remaining norm falls below ``tol`` times their original norm
    to the right edge.  This is the policy used by pinned lme4 2.0-6.
    """

    working = np.array(matrix, dtype=np.float64, order="F", copy=True)
    n, p = working.shape
    pivot = list(range(p))
    current_norm = np.linalg.norm(working, axis=0)
    original_norm = current_norm.copy()
    last = p - 1
    column = 0
    while column < min(n, p) and column <= last:
        while column <= last and (
            current_norm[column] == 0.0
            or current_norm[column] < original_norm[column] * tolerance
        ):
            moved = working[:, column].copy()
            working[:, column:last] = working[:, column + 1 : last + 1]
            working[:, last] = moved
            moved_current = float(current_norm[column])
            moved_original = float(original_norm[column])
            current_norm[column:last] = current_norm[column + 1 : last + 1]
            original_norm[column:last] = original_norm[column + 1 : last + 1]
            current_norm[last] = moved_current
            original_norm[last] = moved_original
            pivot.append(pivot.pop(column))
            last -= 1
        if column > last:
            break

        vector = working[column:, column]
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            column += 1
            continue
        if vector[0] != 0.0:
            norm = float(np.copysign(norm, vector[0]))
        vector /= norm
        vector[0] += 1.0
        for other in range(column + 1, p):
            projection = -float(vector @ working[column:, other]) / float(vector[0])
            working[column:, other] += projection * vector
            if other <= last and current_norm[other] != 0.0:
                ratio = abs(float(working[column, other])) / current_norm[other]
                reduced = max(0.0, 1.0 - ratio * ratio)
                stability = (
                    1.0
                    + 0.05 * reduced * (current_norm[other] / original_norm[other]) ** 2
                    if original_norm[other] != 0.0
                    else 1.0
                )
                if stability == 1.0:
                    current_norm[other] = float(
                        np.linalg.norm(working[column + 1 :, other])
                    )
                else:
                    current_norm[other] *= float(np.sqrt(reduced))
        working[column, column] = -norm
        column += 1
    return tuple(pivot), min(last + 1, n)


def _fixed_rank(matrix: FloatArray, names: tuple[str, ...]) -> FixedRank:
    pivot, rank = _linpack_qr_pivot(matrix, tolerance=_RANK_TOLERANCE)
    retained = tuple(pivot[:rank])
    dropped = tuple(pivot[rank:])
    reduced = np.asarray(matrix[:, retained], dtype=np.float64)
    _, reduced_rank = _linpack_qr_pivot(reduced, tolerance=_RANK_TOLERANCE)
    if reduced_rank != rank:
        raise ModelSpecificationError(
            "lme4-compatible rank dropping did not produce a full-rank design"
        )
    null_basis = np.zeros((len(names), len(dropped)), dtype=np.float64)
    if dropped:
        aliases = np.linalg.lstsq(reduced, matrix[:, dropped], rcond=None)[0]
        for column, dropped_index in enumerate(dropped):
            null_basis[list(retained), column] = -aliases[:, column]
            null_basis[dropped_index, column] = 1.0
        norms = np.linalg.norm(null_basis, axis=0)
        null_basis /= norms
        residual = matrix @ null_basis
        scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
        if float(np.max(np.abs(residual))) > _RANK_TOLERANCE * scale:
            raise ModelSpecificationError(
                "fixed-effect alias recovery failed its null-space invariant"
            )
    null_basis.setflags(write=False)
    return FixedRank(
        full_names=names,
        retained_indices=retained,
        dropped_indices=dropped,
        pivot=pivot,
        tolerance=_RANK_TOLERANCE,
        estimability_tolerance=_ESTIMABILITY_TOLERANCE,
        null_basis=null_basis,
    )


@dataclass(frozen=True, slots=True)
class _ParsedFormula:
    response_name: str
    group_name: str
    random_predictor_name: str | None
    independent_terms: bool
    fixed_terms: tuple[tuple[str, ...], ...]
    fixed_source: str
    formula_offset_names: tuple[str, ...]
    random_source: str


def _parse_formula(formula: str) -> _ParsedFormula:
    matches = (
        (_INDEPENDENT_RANDOM_TERMS.fullmatch(formula), "explicit"),
        (_NUMERIC_DOUBLE_BAR.fullmatch(formula), "double-bar"),
        (_RANDOM_SLOPE.fullmatch(formula), "correlated"),
        (_RANDOM_INTERCEPT.fullmatch(formula), "intercept"),
    )
    match: re.Match[str] | None = None
    random_source = ""
    for candidate, source in matches:
        if candidate is not None:
            match = candidate
            random_source = source
            break
    if match is None:
        raise UnsupportedFormulaError(
            "the alpha fitter accepts only one supported random structure and "
            "an allowlisted fixed-effects expression"
        )
    fixed_source = (match.groupdict().get("fixed") or "1").strip()
    offset_names: list[str] = []
    terms: list[tuple[str, ...]] = [()]
    seen: set[tuple[str, ...]] = {()}

    def append(term: tuple[str, ...]) -> None:
        if term not in seen:
            seen.add(term)
            terms.append(term)

    for raw_token in fixed_source.split("+"):
        token = raw_token.strip()
        if token == "1":
            continue
        offset_match = _OFFSET.fullmatch(token)
        if offset_match is not None:
            name = offset_match.group("name")
            if name in offset_names:
                raise UnsupportedFormulaError(
                    "the alpha fitter accepts each formula offset only once"
                )
            offset_names.append(name)
            continue
        if re.fullmatch(_IDENTIFIER, token):
            append((token,))
            continue
        interaction = _INTERACTION.fullmatch(token)
        if interaction is not None:
            left, right = interaction.group("left"), interaction.group("right")
            if left == right:
                raise UnsupportedFormulaError(
                    "the alpha fitter accepts only distinct two-variable fixed "
                    "interactions"
                )
            append((left,))
            append((right,))
            append((left, right))
            continue
        raise UnsupportedFormulaError(
            f"the alpha fitter accepts only allowlisted fixed-effects tokens; "
            f"got {token!r}"
        )

    random_predictor = match.groupdict().get("random_predictor")
    variables = {name for term in terms for name in term}
    if random_predictor is not None and random_predictor not in variables:
        raise UnsupportedFormulaError(
            "the fitter accepts only a supported random-slope predictor that "
            "is also a fixed effect"
        )
    return _ParsedFormula(
        response_name=match.group("response"),
        group_name=match.group("group"),
        random_predictor_name=random_predictor,
        independent_terms=random_source in ("explicit", "double-bar"),
        fixed_terms=tuple(terms),
        fixed_source=fixed_source,
        formula_offset_names=tuple(offset_names),
        random_source=random_source,
    )


def _subset_mask(
    subset: SubsetInput | None, n: int, data: DataInput
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    if subset is None:
        return np.ones(n, dtype=np.bool_)
    if (
        isinstance(data, pd.DataFrame)
        and isinstance(subset, pd.Series)
        and not _series_aligned(subset, data)
    ):
        raise ModelSpecificationError("subset must align with data row identifiers")
    values = np.asarray(subset)
    if values.shape != (n,) or values.dtype != np.dtype(np.bool_):
        raise ModelSpecificationError(
            "subset must be a boolean vector with one value per row"
        )
    return np.array(values, dtype=np.bool_, copy=True)


def _full_vector(
    value: FrameVectorInput | None,
    *,
    n: int,
    name: str,
    default: float,
    data: DataInput,
) -> Sequence[object]:
    if value is None:
        return [default] * n
    if (
        isinstance(data, pd.DataFrame)
        and isinstance(value, pd.Series)
        and not _series_aligned(value, data)
    ):
        raise ModelSpecificationError(f"{name} must align with data row identifiers")
    values = np.asarray(value, dtype=object)
    if values.shape != (n,):
        raise ModelSpecificationError(f"{name} must have one value per original row")
    return values.tolist()


def _fixed_encoder(
    parsed: _ParsedFormula,
    data: DataInput,
    retained: Mapping[str, Sequence[object]],
    contrasts: ContrastInput | None,
) -> FixedEncoder:
    requested = {} if contrasts is None else dict(contrasts)
    variable_names = tuple(
        dict.fromkeys(name for term in parsed.fixed_terms for name in term)
    )
    unknown_contrasts = set(requested) - set(variable_names)
    if unknown_contrasts:
        raise ModelSpecificationError(
            f"contrasts reference unknown fixed variables {sorted(unknown_contrasts)!r}"
        )
    variables: list[FixedVariable] = []
    for name in variable_names:
        values = retained[name]
        categorical = (
            isinstance(data, pd.DataFrame)
            and isinstance(data[name].dtype, pd.CategoricalDtype)
        ) or any(isinstance(value, str) for value in values)
        if categorical:
            contrast = requested.get(name, "treatment")
            if contrast not in ("treatment", "sum"):
                raise ModelSpecificationError(
                    f"contrast for {name!r} must be 'treatment' or 'sum'"
                )
            variables.append(
                FixedVariable(
                    name=name,
                    kind="categorical",
                    levels=_categorical_levels(data, name, values),
                    contrast=contrast,
                )
            )
        else:
            if name in requested:
                raise ModelSpecificationError(
                    "contrasts can be assigned only to categorical variables, "
                    f"not {name!r}"
                )
            variables.append(FixedVariable(name=name, kind="numeric"))
    return FixedEncoder(tuple(variables), parsed.fixed_terms)


def _random_variable(
    variable: FixedVariable, requested: Mapping[str, ContrastKind]
) -> FixedVariable:
    if variable.kind == "numeric":
        if variable.name in requested:
            raise ModelSpecificationError(
                "random contrasts can be assigned only to categorical variables, "
                f"not {variable.name!r}"
            )
        return variable
    contrast = requested.get(variable.name, "treatment")
    if contrast not in ("treatment", "sum"):
        raise ModelSpecificationError(
            f"random contrast for {variable.name!r} must be 'treatment' or 'sum'"
        )
    return FixedVariable(
        name=variable.name,
        kind="categorical",
        levels=variable.levels,
        contrast=contrast,
    )


@dataclass(frozen=True, slots=True)
class SingleGroupDesign:
    """Canonical design and encoder state for one grouped covariance structure."""

    spec: SingleGroupSpec
    formula: str
    response_name: str
    group_name: str
    group_levels: tuple[str, ...]
    training_groups: tuple[str, ...]
    predictor_name: str | None
    random_coefficient_names: tuple[str, ...]
    fixed_encoder: FixedEncoder
    fixed_rank: FixedRank
    random_encoder: FixedEncoder
    formula_offset_names: tuple[str, ...]
    requires_explicit_offset: bool
    omitted_row_ids: tuple[str, ...]
    excluded_row_ids: tuple[str, ...]
    na_action: NaAction

    @property
    def group_indices(self) -> np.ndarray[Any, np.dtype[np.int64]]:
        """Return the compact observation-to-group map."""
        return self.spec.group_indices


@dataclass(frozen=True, slots=True)
class RandomTermDesign:
    """Labeled prediction state for one sparse random-effects term."""

    group_name: str
    source_names: tuple[str, ...]
    group_levels: tuple[str, ...]
    training_groups: tuple[str, ...]
    random_coefficient_names: tuple[str, ...]
    predictor_name: str | None
    random_encoder: FixedEncoder


@dataclass(frozen=True, slots=True)
class GeneralDesign:
    """Canonical design for coupled grouping structures."""

    spec: GeneralSparseSpec
    formula: str
    response_name: str
    random_terms: tuple[RandomTermDesign, ...]
    fixed_encoder: FixedEncoder
    fixed_rank: FixedRank
    formula_offset_names: tuple[str, ...]
    requires_explicit_offset: bool
    omitted_row_ids: tuple[str, ...]
    excluded_row_ids: tuple[str, ...]
    na_action: NaAction


@dataclass(frozen=True, slots=True)
class _GeneralRandomExpression:
    group_name: str
    source_names: tuple[str, ...]
    predictor_name: str | None


def _parse_general_formula(
    formula: str,
) -> tuple[_ParsedFormula, tuple[_GeneralRandomExpression, ...]]:
    if formula.count("~") != 1:
        raise UnsupportedFormulaError("formula must contain exactly one '~'")
    response_source, rhs = formula.split("~", maxsplit=1)
    response_name = response_source.strip()
    if re.fullmatch(_IDENTIFIER, response_name) is None:
        raise UnsupportedFormulaError("response must be a simple identifier")
    matches = tuple(_GENERAL_RANDOM_TERM.finditer(rhs))
    expressions: list[_GeneralRandomExpression] = []
    for match in matches:
        group_source = re.sub(r"\s+", "", match.group("group"))
        predictor_name = match.group("random_predictor")
        if "/" in group_source:
            if predictor_name is not None:
                raise UnsupportedFormulaError(
                    "random slopes with slash-expanded grouping factors are unsupported"
                )
            parent, child = group_source.split("/", maxsplit=1)
            expressions.extend(
                (
                    _GeneralRandomExpression(parent, (parent,), None),
                    _GeneralRandomExpression(
                        f"{child}:{parent}", (child, parent), None
                    ),
                )
            )
        elif ":" in group_source:
            left, right = group_source.split(":", maxsplit=1)
            expressions.append(
                _GeneralRandomExpression(group_source, (left, right), predictor_name)
            )
        else:
            expressions.append(
                _GeneralRandomExpression(group_source, (group_source,), predictor_name)
            )
    if len(expressions) < 2:
        raise UnsupportedFormulaError(
            "the general sparse fitter requires at least two random-intercept terms"
        )
    names = [expression.group_name for expression in expressions]
    if len(set(names)) != len(names):
        raise UnsupportedFormulaError("duplicate random-effects terms are unsupported")
    fixed_rhs = rhs
    for match in reversed(matches):
        fixed_rhs = fixed_rhs[: match.start()] + fixed_rhs[match.end() :]
    fixed_tokens = [token.strip() for token in fixed_rhs.split("+") if token.strip()]
    fixed_source = " + ".join(fixed_tokens) if fixed_tokens else "1"
    parsed = _parse_formula(f"{response_name} ~ {fixed_source} + (1 | __kamino_group)")
    fixed_variables = {name for term in parsed.fixed_terms for name in term}
    missing_predictors = {
        expression.predictor_name
        for expression in expressions
        if expression.predictor_name is not None
        and expression.predictor_name not in fixed_variables
    }
    if missing_predictors:
        raise UnsupportedFormulaError(
            "random-slope predictors must also appear as fixed effects"
        )
    return parsed, tuple(expressions)


def _interaction_groups(
    retained: Mapping[str, Sequence[object]], expression: _GeneralRandomExpression
) -> tuple[str, ...]:
    source_columns = [retained[name] for name in expression.source_names]
    groups: list[str] = []
    seen_components: dict[str, tuple[str, ...]] = {}
    for values in zip(*source_columns, strict=True):
        if not all(isinstance(value, str) for value in values):
            raise ModelSpecificationError(
                f"grouping term {expression.group_name!r} must contain string labels"
            )
        components = tuple(str(value) for value in values)
        label = ":".join(components)
        previous = seen_components.setdefault(label, components)
        if previous != components:
            raise ModelSpecificationError(
                f"grouping term {expression.group_name!r} has ambiguous labels"
            )
        groups.append(label)
    return tuple(groups)


def build_general_design(
    formula: str,
    data: DataInput,
    *,
    weights: FrameVectorInput | None = None,
    offset: FrameVectorInput | None = None,
    contrasts: ContrastInput | None = None,
    random_contrasts: ContrastInput | None = None,
    na_action: NaAction = "error",
    subset: SubsetInput | None = None,
) -> GeneralDesign:
    """Build nested/crossed random-intercept terms through one shared frame."""
    if not isinstance(formula, str):
        raise UnsupportedFormulaError("formula must be a string")
    if na_action not in ("error", "omit"):
        raise ModelSpecificationError("na_action must be 'error' or 'omit'")
    if contrasts is not None and not isinstance(contrasts, Mapping):
        raise ModelSpecificationError("contrasts must be a mapping")
    if random_contrasts is not None and not isinstance(random_contrasts, Mapping):
        raise ModelSpecificationError("random_contrasts must be a mapping")
    parsed, expressions = _parse_general_formula(formula)
    response = _column(data, parsed.response_name)
    n = len(response)
    if n == 0:
        raise ModelSpecificationError("data must contain at least one row")
    row_ids = _row_ids(data, n)
    fixed_variables = tuple(
        dict.fromkeys(name for term in parsed.fixed_terms for name in term)
    )
    group_sources = tuple(
        dict.fromkeys(
            name for expression in expressions for name in expression.source_names
        )
    )
    required_names = tuple(
        dict.fromkeys(
            (parsed.response_name,)
            + fixed_variables
            + parsed.formula_offset_names
            + group_sources
        )
    )
    columns = {name: _column(data, name) for name in required_names}
    if any(len(values) != n for values in columns.values()):
        raise ModelSpecificationError("all model-frame columns must align")
    weight_values = _full_vector(weights, n=n, name="weights", default=1.0, data=data)
    argument_offset = _full_vector(offset, n=n, name="offset", default=0.0, data=data)
    selected = _subset_mask(subset, n, data)
    missing = np.zeros(n, dtype=np.bool_)
    for values in (*columns.values(), weight_values, argument_offset):
        missing |= np.fromiter(
            (_missing(value) for value in values), dtype=np.bool_, count=n
        )
    active_missing = selected & missing
    if active_missing.any() and na_action == "error":
        affected = tuple(row_ids[index] for index in np.flatnonzero(active_missing))
        raise ModelSpecificationError(
            f"model frame contains missing values in retained rows {affected!r}"
        )
    retained_mask = selected & ~missing
    if not retained_mask.any():
        raise ModelSpecificationError("model frame retains no observations")
    retained_indices = np.flatnonzero(retained_mask)
    retained = {
        name: [values[int(index)] for index in retained_indices]
        for name, values in columns.items()
    }
    retained_row_ids = tuple(row_ids[int(index)] for index in retained_indices)
    omitted_row_ids = tuple(
        row_ids[int(index)] for index in np.flatnonzero(active_missing)
    )
    excluded_row_ids = tuple(row_ids[int(index)] for index in np.flatnonzero(~selected))

    encoder = _fixed_encoder(parsed, data, retained, contrasts)
    requested_random_contrasts = (
        {} if random_contrasts is None else dict(random_contrasts)
    )
    random_predictor_names = {
        expression.predictor_name
        for expression in expressions
        if expression.predictor_name is not None
    }
    unknown_random_contrasts = set(requested_random_contrasts) - random_predictor_names
    if unknown_random_contrasts:
        raise ModelSpecificationError(
            "random_contrasts reference unknown random variables "
            f"{sorted(unknown_random_contrasts)!r}"
        )
    retained_frame: dict[str, Any] = {
        parsed.response_name: retained[parsed.response_name]
    }
    variable_by_name = {variable.name: variable for variable in encoder.variables}
    for variable in encoder.variables:
        values = retained[variable.name]
        retained_frame[variable.name] = (
            pd.Categorical(
                [str(value) for value in values],
                categories=variable.levels,
                ordered=True,
            )
            if variable.kind == "categorical"
            else values
        )
    encoded_frame = pd.DataFrame(retained_frame, index=retained_row_ids)
    formulae_terms: list[str] = []
    for term in parsed.fixed_terms:
        if not term:
            continue
        components = []
        for name in term:
            variable = variable_by_name[name]
            components.append(
                f"C({name}, Sum)"
                if variable.kind == "categorical" and variable.contrast == "sum"
                else name
            )
        formulae_terms.append(":".join(components))
    fixed_formula = (
        f"{parsed.response_name} ~ "
        f"{' + '.join(formulae_terms) if formulae_terms else '1'}"
    )
    try:
        matrices: Any = design_matrices(fixed_formula, encoded_frame, na_action="error")
    except Exception as error:
        raise ModelSpecificationError(f"formula evaluation failed: {error}") from error
    x_full = np.asarray(matrices.common).astype(np.float64, copy=False)
    owned_x = encoder.evaluate(encoded_frame, len(retained_indices))
    if x_full.shape != owned_x.shape or not np.array_equal(x_full, owned_x):
        raise ModelSpecificationError(
            "formula backend disagrees with the owned fixed-effect encoding"
        )
    fixed_rank = _fixed_rank(x_full, encoder.fixed_names)
    x = fixed_rank.reduce(x_full)
    response_array = (
        np.asarray(matrices.response).astype(np.float64, copy=False).reshape(-1)
    )
    try:
        weight_array = np.asarray(
            [weight_values[int(index)] for index in retained_indices], dtype=np.float64
        )
        total_offset = np.asarray(
            [argument_offset[int(index)] for index in retained_indices],
            dtype=np.float64,
        )
        for name in parsed.formula_offset_names:
            total_offset += np.asarray(retained[name], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelSpecificationError("weights and offsets must be numeric") from error

    pending_terms: list[
        tuple[int, int, _GeneralRandomExpression, tuple[str, ...], tuple[str, ...]]
    ] = []
    for position, expression in enumerate(expressions):
        groups = _interaction_groups(retained, expression)
        source_levels = tuple(
            _group_levels(data, name, retained[name])
            for name in expression.source_names
        )
        observed = set(groups)
        levels = tuple(
            ":".join(components)
            for components in product(*source_levels)
            if ":".join(components) in observed
        )
        pending_terms.append((len(levels), position, expression, groups, levels))
    pending_terms.sort(key=lambda item: (-item[0], item[1]))

    sparse_terms: list[SparseRandomTermSpec] = []
    random_terms: list[RandomTermDesign] = []
    random_names: list[str] = []
    for _, _, expression, groups, levels in pending_terms:
        level_indices = {level: index for index, level in enumerate(levels)}
        indices = np.fromiter(
            (level_indices[group] for group in groups),
            dtype=np.int64,
            count=len(groups),
        )
        if expression.predictor_name is None:
            random_encoder = FixedEncoder(variables=(), terms=((),))
        else:
            predictor_variable = _random_variable(
                variable_by_name[expression.predictor_name],
                requested_random_contrasts,
            )
            random_encoder = FixedEncoder(
                variables=(predictor_variable,),
                terms=((), (expression.predictor_name,)),
            )
        random_design = random_encoder.evaluate(encoded_frame, len(groups))
        random_coefficient_names = random_encoder.fixed_names
        sparse_terms.append(
            SparseRandomTermSpec.from_arrays(
                group_indices=indices,
                random_design=random_design,
                group_count=len(levels),
                n=len(groups),
            )
        )
        random_terms.append(
            RandomTermDesign(
                group_name=expression.group_name,
                source_names=expression.source_names,
                group_levels=levels,
                training_groups=groups,
                random_coefficient_names=random_coefficient_names,
                predictor_name=expression.predictor_name,
                random_encoder=random_encoder,
            )
        )
        random_names.extend(
            f"{expression.group_name}[{level}]:{coefficient}"
            for level in levels
            for coefficient in random_coefficient_names
        )
    spec = GeneralSparseSpec.from_arrays(
        y=response_array,
        x=x,
        terms=tuple(sparse_terms),
        weights=weight_array,
        offset=total_offset,
        row_ids=retained_row_ids,
        fixed_names=fixed_rank.retained_names,
        random_names=tuple(random_names),
    )
    canonical_fixed = parsed.fixed_source
    if not re.match(r"^\s*1(?:\s*\+|\s*$)", canonical_fixed):
        canonical_fixed = f"1 + {canonical_fixed}"
    canonical_random = " + ".join(
        (
            f"(1 | {term.group_name})"
            if term.predictor_name is None
            else f"(1 + {term.predictor_name} | {term.group_name})"
        )
        for term in random_terms
    )
    return GeneralDesign(
        spec=spec,
        formula=f"{parsed.response_name} ~ {canonical_fixed} + {canonical_random}",
        response_name=parsed.response_name,
        random_terms=tuple(random_terms),
        fixed_encoder=encoder,
        fixed_rank=fixed_rank,
        formula_offset_names=parsed.formula_offset_names,
        requires_explicit_offset=offset is not None,
        omitted_row_ids=omitted_row_ids,
        excluded_row_ids=excluded_row_ids,
        na_action=na_action,
    )


def build_single_group_design(
    formula: str,
    data: DataInput,
    *,
    weights: FrameVectorInput | None = None,
    offset: FrameVectorInput | None = None,
    contrasts: ContrastInput | None = None,
    random_contrasts: ContrastInput | None = None,
    na_action: NaAction = "error",
    subset: SubsetInput | None = None,
) -> SingleGroupDesign:
    """Build an accepted one-group design through one shared model frame."""
    if not isinstance(formula, str):
        raise UnsupportedFormulaError("formula must be a string")
    if na_action not in ("error", "omit"):
        raise ModelSpecificationError("na_action must be 'error' or 'omit'")
    if contrasts is not None and not isinstance(contrasts, Mapping):
        raise ModelSpecificationError("contrasts must be a mapping")
    if random_contrasts is not None and not isinstance(random_contrasts, Mapping):
        raise ModelSpecificationError("random_contrasts must be a mapping")
    parsed = _parse_formula(formula)
    response = _column(data, parsed.response_name)
    n = len(response)
    if n == 0:
        raise ModelSpecificationError("data must contain at least one row")
    row_ids = _row_ids(data, n)
    fixed_variables = tuple(
        dict.fromkeys(name for term in parsed.fixed_terms for name in term)
    )
    required_names = tuple(
        dict.fromkeys(
            (parsed.response_name, parsed.group_name)
            + fixed_variables
            + parsed.formula_offset_names
        )
    )
    columns = {name: _column(data, name) for name in required_names}
    if any(len(values) != n for values in columns.values()):
        raise ModelSpecificationError("all model-frame columns must align")
    weight_values = _full_vector(weights, n=n, name="weights", default=1.0, data=data)
    argument_offset = _full_vector(offset, n=n, name="offset", default=0.0, data=data)
    selected = _subset_mask(subset, n, data)
    missing = np.zeros(n, dtype=np.bool_)
    for values in (*columns.values(), weight_values, argument_offset):
        missing |= np.fromiter(
            (_missing(value) for value in values), dtype=np.bool_, count=n
        )
    active_missing = selected & missing
    if active_missing.any() and na_action == "error":
        affected = tuple(row_ids[index] for index in np.flatnonzero(active_missing))
        raise ModelSpecificationError(
            f"model frame contains missing values in retained rows {affected!r}"
        )
    retained_mask = selected & ~missing
    if not retained_mask.any():
        raise ModelSpecificationError("model frame retains no observations")
    retained_indices = np.flatnonzero(retained_mask)
    retained = {
        name: [values[int(index)] for index in retained_indices]
        for name, values in columns.items()
    }
    retained_row_ids = tuple(row_ids[int(index)] for index in retained_indices)
    omitted_row_ids = tuple(
        row_ids[int(index)] for index in np.flatnonzero(active_missing)
    )
    excluded_row_ids = tuple(row_ids[int(index)] for index in np.flatnonzero(~selected))

    encoder = _fixed_encoder(parsed, data, retained, contrasts)
    requested_random_contrasts = (
        {} if random_contrasts is None else dict(random_contrasts)
    )
    allowed_random_contrasts = (
        set()
        if parsed.random_predictor_name is None
        else {parsed.random_predictor_name}
    )
    unknown_random_contrasts = (
        set(requested_random_contrasts) - allowed_random_contrasts
    )
    if unknown_random_contrasts:
        raise ModelSpecificationError(
            "random_contrasts reference unknown random variables "
            f"{sorted(unknown_random_contrasts)!r}"
        )
    retained_frame: dict[str, Any] = {
        parsed.response_name: retained[parsed.response_name]
    }
    variable_by_name = {variable.name: variable for variable in encoder.variables}
    for variable in encoder.variables:
        values = retained[variable.name]
        retained_frame[variable.name] = (
            pd.Categorical(
                [str(value) for value in values],
                categories=variable.levels,
                ordered=True,
            )
            if variable.kind == "categorical"
            else values
        )
    encoded_frame = pd.DataFrame(retained_frame, index=retained_row_ids)
    formulae_terms: list[str] = []
    for term in parsed.fixed_terms:
        if not term:
            continue
        components = []
        for name in term:
            variable = variable_by_name[name]
            components.append(
                f"C({name}, Sum)"
                if variable.kind == "categorical" and variable.contrast == "sum"
                else name
            )
        formulae_terms.append(":".join(components))
    fixed_formula = (
        f"{parsed.response_name} ~ "
        f"{' + '.join(formulae_terms) if formulae_terms else '1'}"
    )
    try:
        matrices: Any = design_matrices(fixed_formula, encoded_frame, na_action="error")
    except Exception as error:
        raise ModelSpecificationError(f"formula evaluation failed: {error}") from error
    x_full = np.asarray(matrices.common).astype(np.float64, copy=False)
    owned_x = encoder.evaluate(encoded_frame, len(retained_indices))
    if x_full.shape != owned_x.shape or not np.array_equal(x_full, owned_x):
        raise ModelSpecificationError(
            "formula backend disagrees with the owned fixed-effect encoding"
        )
    fixed_rank = _fixed_rank(x_full, encoder.fixed_names)
    x = fixed_rank.reduce(x_full)
    response_array = (
        np.asarray(matrices.response).astype(np.float64, copy=False).reshape(-1)
    )

    try:
        weight_array = np.asarray(
            [weight_values[int(index)] for index in retained_indices], dtype=np.float64
        )
        total_offset = np.asarray(
            [argument_offset[int(index)] for index in retained_indices],
            dtype=np.float64,
        )
        for name in parsed.formula_offset_names:
            total_offset += np.asarray(retained[name], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelSpecificationError("weights and offsets must be numeric") from error

    groups_raw = retained[parsed.group_name]
    levels = _group_levels(data, parsed.group_name, groups_raw)
    groups = tuple(str(value) for value in groups_raw)
    level_indices = {level: index for index, level in enumerate(levels)}
    group_indices = np.fromiter(
        (level_indices[group] for group in groups), dtype=np.int64, count=len(groups)
    )
    if parsed.random_predictor_name is None:
        random_encoder = FixedEncoder(variables=(), terms=((),))
    else:
        predictor_variable = _random_variable(
            variable_by_name[parsed.random_predictor_name],
            requested_random_contrasts,
        )
        if predictor_variable.kind != "numeric" and parsed.independent_terms:
            raise UnsupportedFormulaError(
                "categorical double-bar and split random terms are unsupported"
            )
        random_encoder = FixedEncoder(
            variables=(predictor_variable,),
            terms=((), (parsed.random_predictor_name,)),
        )
    random_design = random_encoder.evaluate(encoded_frame, len(groups))
    random_coefficient_names = random_encoder.fixed_names

    spec = SingleGroupSpec.from_arrays(
        y=response_array,
        x=x,
        group_indices=group_indices,
        random_design=random_design,
        group_count=len(levels),
        covariance_term_sizes=(1, 1) if parsed.independent_terms else None,
        weights=weight_array,
        offset=total_offset,
        row_ids=retained_row_ids,
        fixed_names=fixed_rank.retained_names,
        random_names=tuple(
            f"{parsed.group_name}[{level}]:{coefficient}"
            for level in levels
            for coefficient in random_coefficient_names
        ),
    )
    if parsed.random_source == "intercept":
        canonical_random = f"(1 | {parsed.group_name})"
    elif parsed.random_source == "double-bar":
        canonical_random = (
            f"(1 + {parsed.random_predictor_name} || {parsed.group_name})"
        )
    elif parsed.random_source == "explicit":
        canonical_random = (
            f"(1 | {parsed.group_name}) + "
            f"(0 + {parsed.random_predictor_name} | {parsed.group_name})"
        )
    else:
        canonical_random = f"(1 + {parsed.random_predictor_name} | {parsed.group_name})"
    canonical_fixed = parsed.fixed_source
    if not re.match(r"^\s*1(?:\s*\+|\s*$)", canonical_fixed):
        canonical_fixed = f"1 + {canonical_fixed}"
    return SingleGroupDesign(
        spec=spec,
        formula=f"{parsed.response_name} ~ {canonical_fixed} + {canonical_random}",
        response_name=parsed.response_name,
        group_name=parsed.group_name,
        group_levels=levels,
        training_groups=groups,
        predictor_name=parsed.random_predictor_name,
        random_coefficient_names=random_coefficient_names,
        fixed_encoder=encoder,
        fixed_rank=fixed_rank,
        random_encoder=random_encoder,
        formula_offset_names=parsed.formula_offset_names,
        requires_explicit_offset=offset is not None,
        omitted_row_ids=omitted_row_ids,
        excluded_row_ids=excluded_row_ids,
        na_action=na_action,
    )


def build_random_intercept_design(
    formula: str,
    data: DataInput,
    *,
    weights: FrameVectorInput | None = None,
    offset: FrameVectorInput | None = None,
    contrasts: ContrastInput | None = None,
    random_contrasts: ContrastInput | None = None,
    na_action: NaAction = "error",
    subset: SubsetInput | None = None,
) -> SingleGroupDesign:
    """Build the retained random-intercept-only development entry point."""
    design = build_single_group_design(
        formula,
        data,
        weights=weights,
        offset=offset,
        contrasts=contrasts,
        random_contrasts=random_contrasts,
        na_action=na_action,
        subset=subset,
    )
    if design.spec.k != 1:
        raise UnsupportedFormulaError(
            "build_random_intercept_design accepts only a random intercept"
        )
    return design


def build_model_design(
    formula: str,
    data: DataInput,
    *,
    weights: FrameVectorInput | None = None,
    offset: FrameVectorInput | None = None,
    contrasts: ContrastInput | None = None,
    random_contrasts: ContrastInput | None = None,
    na_action: NaAction = "error",
    subset: SubsetInput | None = None,
) -> SingleGroupDesign | GeneralDesign:
    """Route an accepted formula to its compact structural design."""
    try:
        return build_single_group_design(
            formula,
            data,
            weights=weights,
            offset=offset,
            contrasts=contrasts,
            random_contrasts=random_contrasts,
            na_action=na_action,
            subset=subset,
        )
    except UnsupportedFormulaError as single_group_error:
        try:
            return build_general_design(
                formula,
                data,
                weights=weights,
                offset=offset,
                contrasts=contrasts,
                random_contrasts=random_contrasts,
                na_action=na_action,
                subset=subset,
            )
        except UnsupportedFormulaError:
            raise single_group_error from None


__all__ = [
    "ColumnInput",
    "ContrastInput",
    "DataInput",
    "FixedEncoder",
    "FixedRank",
    "FixedVariable",
    "FrameVectorInput",
    "GeneralDesign",
    "NaAction",
    "RandomTermDesign",
    "SingleGroupDesign",
    "SubsetInput",
    "build_general_design",
    "build_model_design",
    "build_random_intercept_design",
    "build_single_group_design",
]
