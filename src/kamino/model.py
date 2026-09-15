"""Immutable, validated numerical model specification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from kamino.errors import ModelSpecificationError

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
VectorInput = FloatArray | Sequence[float]
MatrixInput = FloatArray | Sequence[Sequence[float]]
IndexInput = IntArray | Sequence[int]


class ObjectiveKind(StrEnum):
    """Likelihood objective evaluated at a fixed covariance parameter."""

    ML = "ml"
    REML = "reml"


def _readonly_float64(
    value: VectorInput | MatrixInput, *, ndim: int, name: str
) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim:
        raise ModelSpecificationError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ModelSpecificationError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


def _readonly_group_indices(value: IndexInput, *, n: int, groups: int) -> IntArray:
    source = np.asarray(value)
    if source.ndim != 1 or source.shape != (n,):
        raise ModelSpecificationError("group_indices must have one value per row")
    if not np.issubdtype(source.dtype, np.integer) or np.issubdtype(
        source.dtype, np.bool_
    ):
        raise ModelSpecificationError("group_indices must contain integers")
    indices = np.array(source, dtype=np.int64, copy=True)
    if groups == 0 or (indices < 0).any() or (indices >= groups).any():
        raise ModelSpecificationError("group_indices contains an invalid group index")
    if groups > n:
        raise ModelSpecificationError("every random-intercept group must be observed")
    if not np.bincount(indices, minlength=groups).all():
        raise ModelSpecificationError("every random-intercept group must be observed")
    indices.setflags(write=False)
    return indices


def _validated_labels(
    *,
    n: int,
    p: int,
    q: int,
    row_ids: tuple[str, ...] | None,
    fixed_names: tuple[str, ...] | None,
    random_names: tuple[str, ...] | None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    rows = row_ids or tuple(str(index) for index in range(n))
    fixed = fixed_names or tuple(f"x{index}" for index in range(p))
    random = random_names or tuple(f"z{index}" for index in range(q))
    if len(rows) != n or len(set(rows)) != n:
        raise ModelSpecificationError("row_ids must be unique and match the row count")
    if len(fixed) != p or len(set(fixed)) != p:
        raise ModelSpecificationError("fixed_names must be unique and match x columns")
    if len(random) != q or len(set(random)) != q:
        raise ModelSpecificationError(
            "random_names must be unique and match random-effect columns"
        )
    return rows, fixed, random


def _validated_common_arrays(
    *,
    y: VectorInput,
    x: MatrixInput,
    weights: VectorInput | None,
    offset: VectorInput | None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    y_array = _readonly_float64(y, ndim=1, name="y")
    x_array = _readonly_float64(x, ndim=2, name="x")
    n = y_array.shape[0]
    if x_array.shape[0] != n:
        raise ModelSpecificationError("y and x must have the same row count")
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    if offset is None:
        offset = np.zeros(n, dtype=np.float64)
    weight_array = _readonly_float64(weights, ndim=1, name="weights")
    offset_array = _readonly_float64(offset, ndim=1, name="offset")
    if weight_array.shape != (n,) or offset_array.shape != (n,):
        raise ModelSpecificationError("weights and offset must have one value per row")
    if (weight_array <= 0.0).any():
        raise ModelSpecificationError("weights must be strictly positive")
    p = x_array.shape[1]
    if np.linalg.matrix_rank(x_array) != p:
        raise ModelSpecificationError("Phase 0 requires a full-rank fixed design")
    return y_array, x_array, weight_array, offset_array


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A direct-array Gaussian LMM specification for Phase 0."""

    y: FloatArray
    x: FloatArray
    z: FloatArray
    weights: FloatArray
    offset: FloatArray
    row_ids: tuple[str, ...]
    fixed_names: tuple[str, ...]
    random_names: tuple[str, ...]

    @classmethod
    def from_arrays(
        cls,
        *,
        y: VectorInput,
        x: MatrixInput,
        z: MatrixInput,
        weights: VectorInput | None = None,
        offset: VectorInput | None = None,
        row_ids: tuple[str, ...] | None = None,
        fixed_names: tuple[str, ...] | None = None,
        random_names: tuple[str, ...] | None = None,
    ) -> ModelSpec:
        y_array, x_array, weight_array, offset_array = _validated_common_arrays(
            y=y, x=x, weights=weights, offset=offset
        )
        z_array = _readonly_float64(z, ndim=2, name="z")
        n = y_array.shape[0]
        if z_array.shape[0] != n:
            raise ModelSpecificationError("y, x, and z must have the same row count")
        p = x_array.shape[1]
        q = z_array.shape[1]
        row_ids, fixed_names, random_names = _validated_labels(
            n=n,
            p=p,
            q=q,
            row_ids=row_ids,
            fixed_names=fixed_names,
            random_names=random_names,
        )
        return cls(
            y=y_array,
            x=x_array,
            z=z_array,
            weights=weight_array,
            offset=offset_array,
            row_ids=row_ids,
            fixed_names=fixed_names,
            random_names=random_names,
        )

    @property
    def n(self) -> int:
        return self.y.shape[0]

    @property
    def p(self) -> int:
        return self.x.shape[1]

    @property
    def q(self) -> int:
        return self.z.shape[1]


@dataclass(frozen=True, slots=True)
class SingleGroupSpec:
    """Compact model specification for one independent grouping structure.

    The random-effects design is encoded by one integer group index and ``k``
    random covariates per observation. No ``n``-by-``q`` matrix is constructed
    or stored.
    """

    y: FloatArray
    x: FloatArray
    group_indices: IntArray
    random_design: FloatArray
    group_count: int
    covariance_term_sizes: tuple[int, ...]
    weights: FloatArray
    offset: FloatArray
    row_ids: tuple[str, ...]
    fixed_names: tuple[str, ...]
    random_names: tuple[str, ...]

    @classmethod
    def from_arrays(
        cls,
        *,
        y: VectorInput,
        x: MatrixInput,
        group_indices: IndexInput,
        random_design: MatrixInput,
        group_count: int,
        covariance_term_sizes: tuple[int, ...] | None = None,
        weights: VectorInput | None = None,
        offset: VectorInput | None = None,
        row_ids: tuple[str, ...] | None = None,
        fixed_names: tuple[str, ...] | None = None,
        random_names: tuple[str, ...] | None = None,
    ) -> SingleGroupSpec:
        if group_count <= 0:
            raise ModelSpecificationError("group_count must be positive")
        y_array, x_array, weight_array, offset_array = _validated_common_arrays(
            y=y, x=x, weights=weights, offset=offset
        )
        n = y_array.shape[0]
        random_design_array = _readonly_float64(
            random_design, ndim=2, name="random_design"
        )
        if random_design_array.shape[0] != n:
            raise ModelSpecificationError(
                "y and random_design must have the same row count"
            )
        random_columns = random_design_array.shape[1]
        if random_columns == 0:
            raise ModelSpecificationError(
                "random_design must contain at least one column"
            )
        term_sizes = (
            (random_columns,)
            if covariance_term_sizes is None
            else covariance_term_sizes
        )
        if (
            not term_sizes
            or any(type(size) is not int or size <= 0 for size in term_sizes)
            or sum(term_sizes) != random_columns
        ):
            raise ModelSpecificationError(
                "covariance_term_sizes must be positive integers summing to "
                "the random-design column count"
            )
        indices = _readonly_group_indices(group_indices, n=n, groups=group_count)
        row_ids, fixed_names, random_names = _validated_labels(
            n=n,
            p=x_array.shape[1],
            q=group_count * random_columns,
            row_ids=row_ids,
            fixed_names=fixed_names,
            random_names=random_names,
        )
        return cls(
            y=y_array,
            x=x_array,
            group_indices=indices,
            random_design=random_design_array,
            group_count=group_count,
            covariance_term_sizes=term_sizes,
            weights=weight_array,
            offset=offset_array,
            row_ids=row_ids,
            fixed_names=fixed_names,
            random_names=random_names,
        )

    @property
    def n(self) -> int:
        return self.y.shape[0]

    @property
    def p(self) -> int:
        return self.x.shape[1]

    @property
    def q(self) -> int:
        return len(self.random_names)

    @property
    def k(self) -> int:
        """Number of random coefficients per group."""
        return self.random_design.shape[1]

    @property
    def d(self) -> int:
        """Number of covariance parameters across independent terms."""
        return sum(size * (size + 1) // 2 for size in self.covariance_term_sizes)
