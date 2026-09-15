from __future__ import annotations

import numpy as np
import pytest

from kamino.errors import ModelSpecificationError
from kamino.model import ModelSpec, ObjectiveKind
from kamino.oracle import evaluate_dense_oracle
from kamino.pls import evaluate_fixed_theta


def example_spec(weight_multiplier: float = 1.0) -> ModelSpec:
    x = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ]
    )
    z = np.zeros((8, 4), dtype=np.float64)
    z[:4, :2] = x[:4]
    z[4:, 2:] = x[4:]
    return ModelSpec.from_arrays(
        y=[1.3, 2.1, 1.7, 4.4, 2.0, 2.4, 3.7, 4.1],
        x=x,
        z=z,
        weights=weight_multiplier * np.array([0.5, 1.0, 2.0, 3.0, 1.2, 0.8, 2.5, 0.7]),
        offset=[0.2, 0.1, -0.1, 0.3, 0.0, 0.2, -0.2, 0.1],
        fixed_names=("(Intercept)", "days"),
        random_names=("g0:int", "g0:days", "g1:int", "g1:days"),
    )


def repeated_factor(a: float, b: float, c: float) -> np.ndarray:
    block = np.array([[a, 0.0], [b, c]], dtype=np.float64)
    result = np.zeros((4, 4), dtype=np.float64)
    result[:2, :2] = block
    result[2:, 2:] = block
    return result


@pytest.mark.parametrize("kind", list(ObjectiveKind))
@pytest.mark.parametrize(
    "parameters",
    [
        (1.0, 0.0, 1.0),
        (0.8, 0.35, 0.4),
        (0.0, 0.4, 0.0),
        (0.0, 0.0, 0.0),
    ],
)
def test_pls_matches_independent_dense_oracle(
    kind: ObjectiveKind, parameters: tuple[float, float, float]
) -> None:
    spec = example_spec()
    lambda_ = repeated_factor(*parameters)
    pls = evaluate_fixed_theta(spec, lambda_, kind=kind)
    oracle = evaluate_dense_oracle(spec, lambda_, kind=kind)
    assert pls.objective == pytest.approx(oracle.objective, abs=1e-11, rel=1e-11)
    assert pls.penalized_residual_sum_squares == pytest.approx(
        oracle.penalized_residual_sum_squares, abs=1e-11, rel=1e-11
    )
    np.testing.assert_allclose(pls.beta, oracle.beta, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(
        pls.beta_covariance,
        oracle.beta_covariance,
        atol=1e-11,
        rtol=1e-11,
    )


@pytest.mark.parametrize("kind", list(ObjectiveKind))
def test_uniform_weight_scaling_preserves_absolute_model(
    kind: ObjectiveKind,
) -> None:
    scale = 7.0
    lambda_ = repeated_factor(0.8, 0.35, 0.4)
    original = evaluate_fixed_theta(example_spec(), lambda_, kind=kind)
    rescaled = evaluate_fixed_theta(
        example_spec(weight_multiplier=scale),
        lambda_ / np.sqrt(scale),
        kind=kind,
    )
    assert rescaled.objective == pytest.approx(original.objective, abs=1e-11)
    assert rescaled.sigma2 == pytest.approx(scale * original.sigma2, rel=1e-11)
    np.testing.assert_allclose(rescaled.beta, original.beta, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(
        rescaled.random_covariance,
        original.random_covariance,
        atol=1e-11,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        rescaled.beta_covariance,
        original.beta_covariance,
        atol=1e-11,
        rtol=1e-11,
    )


@pytest.mark.parametrize("kind", list(ObjectiveKind))
def test_offset_equals_centering_the_response(
    kind: ObjectiveKind,
) -> None:
    spec = example_spec()
    lambda_ = repeated_factor(0.8, 0.35, 0.4)
    centered = ModelSpec.from_arrays(
        y=spec.y - spec.offset,
        x=spec.x,
        z=spec.z,
        weights=spec.weights,
        fixed_names=spec.fixed_names,
        random_names=spec.random_names,
    )
    with_offset = evaluate_fixed_theta(spec, lambda_, kind=kind)
    without_offset = evaluate_fixed_theta(centered, lambda_, kind=kind)
    assert with_offset.objective == pytest.approx(without_offset.objective, abs=1e-12)
    np.testing.assert_array_equal(with_offset.beta, without_offset.beta)


def test_dense_oracle_refuses_unbounded_allocation() -> None:
    spec = example_spec()
    with pytest.raises(ModelSpecificationError, match="refuses"):
        evaluate_dense_oracle(spec, repeated_factor(1.0, 0.0, 1.0), maximum_n=7)


@pytest.mark.parametrize("use_oracle", [False, True])
@pytest.mark.parametrize(
    "lambda_",
    [np.eye(3), np.full((4, 4), np.nan)],
)
def test_evaluators_reject_invalid_covariance_factor(
    use_oracle: bool, lambda_: np.ndarray
) -> None:
    evaluator = evaluate_dense_oracle if use_oracle else evaluate_fixed_theta
    with pytest.raises(ModelSpecificationError, match="lambda_"):
        evaluator(example_spec(), lambda_)


@pytest.mark.parametrize("use_oracle", [False, True])
def test_reml_rejects_saturated_fixed_design(use_oracle: bool) -> None:
    spec = ModelSpec.from_arrays(
        y=[1.0, 2.0],
        x=np.eye(2),
        z=np.zeros((2, 1)),
    )
    evaluator = evaluate_dense_oracle if use_oracle else evaluate_fixed_theta
    with pytest.raises(ModelSpecificationError, match="REML requires"):
        evaluator(spec, np.zeros((1, 1)))


def test_result_is_immutable_and_likelihood_is_labeled() -> None:
    result = evaluate_fixed_theta(
        example_spec(),
        repeated_factor(0.8, 0.35, 0.4),
        kind=ObjectiveKind.ML,
    )
    assert result.kind is ObjectiveKind.ML
    assert result.log_likelihood == -0.5 * result.objective
    assert not result.beta.flags.writeable
