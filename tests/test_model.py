import numpy as np
import pytest

from kamino.errors import ModelSpecificationError
from kamino.model import ModelSpec


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
