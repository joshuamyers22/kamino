"""Immutable, validated numerical model specification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from kamino.errors import ModelSpecificationError

FloatArray = npt.NDArray[np.float64]
VectorInput = FloatArray | Sequence[float]
MatrixInput = FloatArray | Sequence[Sequence[float]]


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
        y_array = _readonly_float64(y, ndim=1, name="y")
        x_array = _readonly_float64(x, ndim=2, name="x")
        z_array = _readonly_float64(z, ndim=2, name="z")
        n = y_array.shape[0]
        if x_array.shape[0] != n or z_array.shape[0] != n:
            raise ModelSpecificationError("y, x, and z must have the same row count")
        if weights is None:
            weights = np.ones(n, dtype=np.float64)
        if offset is None:
            offset = np.zeros(n, dtype=np.float64)
        weight_array = _readonly_float64(weights, ndim=1, name="weights")
        offset_array = _readonly_float64(offset, ndim=1, name="offset")
        if weight_array.shape != (n,) or offset_array.shape != (n,):
            raise ModelSpecificationError(
                "weights and offset must have one value per row"
            )
        if (weight_array <= 0.0).any():
            raise ModelSpecificationError("weights must be strictly positive")
        p = x_array.shape[1]
        q = z_array.shape[1]
        row_ids = row_ids or tuple(str(index) for index in range(n))
        fixed_names = fixed_names or tuple(f"x{index}" for index in range(p))
        random_names = random_names or tuple(f"z{index}" for index in range(q))
        if len(row_ids) != n or len(set(row_ids)) != n:
            raise ModelSpecificationError(
                "row_ids must be unique and match the row count"
            )
        if len(fixed_names) != p or len(set(fixed_names)) != p:
            raise ModelSpecificationError(
                "fixed_names must be unique and match x columns"
            )
        if len(random_names) != q or len(set(random_names)) != q:
            raise ModelSpecificationError(
                "random_names must be unique and match z columns"
            )
        if np.linalg.matrix_rank(x_array) != p:
            raise ModelSpecificationError("Phase 0 requires a full-rank fixed design")
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
