import numpy as np
import pytest

from kamino.errors import ModelSpecificationError
from kamino.model import ModelSpec, SingleGroupSpec


def test_model_spec_copies_and_freezes_arrays() -> None:
    y = np.array([1.0, 2.0, 3.0])
    spec = ModelSpec.from_arrays(y=y, x=[[1.0], [1.0], [1.0]], z=np.eye(3))
    y[0] = 100.0
    assert spec.y[0] == 1.0
    assert not spec.y.flags.writeable


@pytest.mark.parametrize("weights", [[1.0, 0.0, 1.0], [1.0, -1.0, 1.0]])
def test_model_spec_rejects_nonpositive_weights(
    weights: list[float],
) -> None:
    with pytest.raises(ModelSpecificationError, match="strictly positive"):
        ModelSpec.from_arrays(
            y=[1.0, 2.0, 3.0],
            x=[[1.0], [1.0], [1.0]],
            z=np.eye(3),
            weights=weights,
        )


def test_model_spec_rejects_rank_deficiency_during_phase_zero() -> None:
    with pytest.raises(ModelSpecificationError, match="full-rank"):
        ModelSpec.from_arrays(
            y=[1.0, 2.0, 3.0],
            x=[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
            z=np.eye(3),
        )


def test_model_spec_rejects_invalid_shapes_and_values() -> None:
    with pytest.raises(ModelSpecificationError, match="y must have 1 dimensions"):
        ModelSpec.from_arrays(
            y=np.array([[1.0], [2.0], [3.0]]),
            x=[[1.0], [1.0], [1.0]],
            z=np.eye(3),
        )
    with pytest.raises(ModelSpecificationError, match="non-finite"):
        ModelSpec.from_arrays(
            y=[1.0, np.nan, 3.0],
            x=[[1.0], [1.0], [1.0]],
            z=np.eye(3),
        )
    with pytest.raises(ModelSpecificationError, match="same row count"):
        ModelSpec.from_arrays(
            y=[1.0, 2.0, 3.0],
            x=[[1.0], [1.0]],
            z=np.eye(3),
        )
    with pytest.raises(ModelSpecificationError, match="one value per row"):
        ModelSpec.from_arrays(
            y=[1.0, 2.0, 3.0],
            x=[[1.0], [1.0], [1.0]],
            z=np.eye(3),
            weights=[1.0, 1.0],
        )


def test_model_spec_rejects_misaligned_labels() -> None:
    with pytest.raises(ModelSpecificationError, match="row_ids"):
        ModelSpec.from_arrays(
            y=[1.0, 2.0, 3.0],
            x=[[1.0], [1.0], [1.0]],
            z=np.eye(3),
            row_ids=("a", "a", "c"),
        )
    with pytest.raises(ModelSpecificationError, match="fixed_names"):
        ModelSpec.from_arrays(
            y=[1.0, 2.0, 3.0],
            x=[[1.0], [1.0], [1.0]],
            z=np.eye(3),
            fixed_names=("a", "b"),
        )
    with pytest.raises(ModelSpecificationError, match="random_names"):
        ModelSpec.from_arrays(
            y=[1.0, 2.0, 3.0],
            x=[[1.0], [1.0], [1.0]],
            z=np.eye(3),
            random_names=("a", "b"),
        )


def test_random_intercept_spec_copies_and_freezes_compact_group_map() -> None:
    indices = np.array([0, 1, 0], dtype=np.int64)
    spec = SingleGroupSpec.from_arrays(
        y=[1.0, 2.0, 3.0],
        x=[[1.0], [1.0], [1.0]],
        group_indices=indices,
        random_design=[[1.0], [1.0], [1.0]],
        group_count=2,
    )
    indices[0] = 1

    np.testing.assert_array_equal(spec.group_indices, [0, 1, 0])
    assert not spec.group_indices.flags.writeable
    assert spec.q == 2
    assert not hasattr(spec, "z")


def test_single_group_spec_copies_and_freezes_random_covariates() -> None:
    random_design = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    spec = SingleGroupSpec.from_arrays(
        y=[1.0, 2.0, 3.0],
        x=random_design,
        group_indices=[0, 1, 0],
        random_design=random_design,
        group_count=2,
    )
    random_design[0, 1] = 99.0

    assert spec.random_design[0, 1] == 0.0
    assert not spec.random_design.flags.writeable
    assert spec.k == 2
    assert spec.q == 4


@pytest.mark.parametrize(
    ("random_design", "message"),
    [
        ([[1.0], [1.0]], "same row count"),
        ([list[float](), list[float](), list[float]()], "at least one column"),
        ([[1.0], [np.nan], [1.0]], "non-finite"),
    ],
)
def test_single_group_spec_rejects_invalid_random_covariates(
    random_design: list[list[float]], message: str
) -> None:
    with pytest.raises(ModelSpecificationError, match=message):
        SingleGroupSpec.from_arrays(
            y=[1.0, 2.0, 3.0],
            x=[[1.0], [1.0], [1.0]],
            group_indices=[0, 1, 0],
            random_design=random_design,
            group_count=2,
        )


@pytest.mark.parametrize(
    ("indices", "group_count", "message"),
    [
        ([0, 1], 2, "one value per row"),
        ([0.0, 1.0, 0.0], 2, "integers"),
        ([0, -1, 0], 2, "invalid group index"),
        ([0, 0, 0], 2, "every random-intercept group"),
        ([0, 1, 2], 4, "every random-intercept group"),
        ([0, 0, 0], 0, "positive"),
    ],
)
def test_random_intercept_spec_rejects_invalid_group_maps(
    indices: list[int] | list[float], group_count: int, message: str
) -> None:
    with pytest.raises(ModelSpecificationError, match=message):
        SingleGroupSpec.from_arrays(
            y=[1.0, 2.0, 3.0],
            x=[[1.0], [1.0], [1.0]],
            group_indices=indices,  # type: ignore[arg-type]
            random_design=[[1.0], [1.0], [1.0]],
            group_count=group_count,
        )
