# pyright: reportUnknownMemberType=false

from __future__ import annotations

import numpy as np
import pytest

from kamino.errors import ModelSpecificationError, ResourceLimitError
from kamino.model import (
    GeneralSparseSpec,
    ModelSpec,
    ObjectiveKind,
    SparseRandomTermSpec,
)
from kamino.pls import evaluate_fixed_theta
from kamino.sparse import (
    SparseBackendLimits,
    evaluate_general_sparse,
    evaluate_prepared_general_sparse,
    prepare_general_sparse,
)


def _crossed_spec() -> tuple[GeneralSparseSpec, np.ndarray]:
    first = np.repeat(np.arange(3, dtype=np.int64), 4)
    second = np.tile(np.arange(4, dtype=np.int64), 3)
    x_value = np.linspace(-1.0, 1.0, 12)
    x = np.column_stack((np.ones(12), x_value))
    weights = 0.8 + (np.arange(12) % 5) / 4.0
    offset = 0.03 * np.cos(np.arange(12))
    y = 1.2 + 0.4 * x_value + 0.2 * np.sin(first) - 0.1 * np.cos(second) + offset
    terms = tuple(
        SparseRandomTermSpec.from_arrays(
            group_indices=indices,
            random_design=np.ones((12, 1)),
            group_count=count,
            n=12,
        )
        for indices, count in ((first, 3), (second, 4))
    )
    spec = GeneralSparseSpec.from_arrays(
        y=y,
        x=x,
        terms=terms,
        weights=weights,
        offset=offset,
        fixed_names=("(Intercept)", "x"),
        random_names=tuple(f"z{index}" for index in range(7)),
    )
    z = np.zeros((12, 7), dtype=np.float64)
    z[np.arange(12), first] = 1.0
    z[np.arange(12), 3 + second] = 1.0
    return spec, z


@pytest.mark.parametrize("kind", [ObjectiveKind.ML, ObjectiveKind.REML])
@pytest.mark.parametrize("theta", [[0.7, 0.3], [0.0, 0.4], [0.0, 0.0]])
def test_general_sparse_matches_dense_pls(
    kind: ObjectiveKind, theta: list[float]
) -> None:
    spec, z = _crossed_spec()
    sparse_result = evaluate_general_sparse(spec, theta, kind=kind)
    dense_spec = ModelSpec.from_arrays(
        y=spec.y,
        x=spec.x,
        z=z,
        weights=spec.weights,
        offset=spec.offset,
        fixed_names=spec.fixed_names,
        random_names=spec.random_names,
    )
    dense_result = evaluate_fixed_theta(
        dense_spec, np.diag(np.repeat(theta, [3, 4])), kind=kind
    )

    assert sparse_result.objective == pytest.approx(dense_result.objective, abs=1e-11)
    assert sparse_result.logdet_c == pytest.approx(dense_result.logdet_c, abs=1e-12)
    assert sparse_result.logdet_s == pytest.approx(dense_result.logdet_s, abs=1e-12)
    assert sparse_result.sigma2 == pytest.approx(dense_result.sigma2, abs=1e-12)
    np.testing.assert_allclose(sparse_result.beta, dense_result.beta, atol=1e-12)
    np.testing.assert_allclose(sparse_result.u, dense_result.u, atol=1e-12)
    np.testing.assert_allclose(sparse_result.b, dense_result.b, atol=1e-12)


def test_sparse_workspace_retains_theta_independent_coupling_pattern() -> None:
    spec, _ = _crossed_spec()
    workspace = prepare_general_sparse(spec)
    assert workspace.structural_c_nonzeros == 31
    assert workspace.random_design.shape == (12, 7)
    assert workspace.random_design.nnz == 24

    zero = evaluate_general_sparse(spec, [0.0, 0.0])
    nonzero = evaluate_general_sparse(spec, [0.7, 0.3])
    assert zero.factor_nonzeros < nonzero.factor_nonzeros


def test_sparse_preflight_rejects_oversized_structure() -> None:
    spec, _ = _crossed_spec()
    with pytest.raises(ModelSpecificationError, match="random coefficient count"):
        prepare_general_sparse(
            spec,
            limits=SparseBackendLimits(maximum_random_coefficients=6),
        )

    with pytest.raises(ResourceLimitError, match="random-design nonzeros"):
        prepare_general_sparse(
            spec,
            limits=SparseBackendLimits(maximum_design_nonzeros=23),
        )
    with pytest.raises(ResourceLimitError, match="cross-product"):
        prepare_general_sparse(
            spec,
            limits=SparseBackendLimits(maximum_structural_c_nonzeros=30),
        )


def test_sparse_factor_fill_and_theta_validation_fail_closed() -> None:
    spec, _ = _crossed_spec()
    workspace = prepare_general_sparse(
        spec,
        limits=SparseBackendLimits(maximum_factor_nonzeros=1),
    )
    with pytest.raises(ResourceLimitError, match="factor fill"):
        evaluate_prepared_general_sparse(workspace, [0.7, 0.3])
    with pytest.raises(ModelSpecificationError, match="theta must contain"):
        evaluate_general_sparse(spec, [0.7])
    with pytest.raises(ModelSpecificationError, match="non-finite"):
        evaluate_general_sparse(spec, [0.7, np.nan])
    with pytest.raises(ModelSpecificationError, match="nonnegative"):
        evaluate_general_sparse(spec, [-0.7, 0.3])


def test_sparse_limit_values_and_term_count_are_validated() -> None:
    spec, _ = _crossed_spec()
    with pytest.raises(ModelSpecificationError, match="positive integers"):
        SparseBackendLimits(maximum_factor_nonzeros=0)
    with pytest.raises(ModelSpecificationError, match="at least two terms"):
        GeneralSparseSpec.from_arrays(
            y=spec.y,
            x=spec.x,
            terms=(spec.terms[0],),
            weights=spec.weights,
            offset=spec.offset,
        )
