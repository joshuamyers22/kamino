from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kamino.model import ModelSpec, ObjectiveKind
from kamino.pls import evaluate_fixed_theta

FIXTURE_PATH = (
    Path(__file__).parents[1] / "oracle" / "fixtures" / "v1" / "fixed_theta.json"
)
OPTIMIZED_FIXTURE_PATH = FIXTURE_PATH.with_name("optimized.json")


def load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def covariance_factor(theta: list[float], groups: int = 3) -> np.ndarray:
    block = np.array([[theta[0], 0.0], [theta[1], theta[2]]])
    return np.kron(np.eye(groups), block)


def model_spec() -> ModelSpec:
    data = load_fixture()["data"]
    return ModelSpec.from_arrays(
        y=data["y"],
        x=data["X"],
        z=data["Z"],
        weights=data["weights"],
        offset=data["offset"],
        row_ids=tuple(data["row_ids"]),
        fixed_names=tuple(data["fixed_names"]),
        random_names=tuple(data["random_names"]),
    )


@pytest.mark.parametrize("case", load_fixture()["cases"], ids=lambda case: case["id"])
def test_fixed_theta_matches_pinned_lme4(case: dict[str, Any]) -> None:
    result = evaluate_fixed_theta(
        model_spec(),
        covariance_factor(case["theta"]),
        kind=ObjectiveKind(case["kind"]),
    )

    assert result.objective == pytest.approx(case["objective"], abs=1e-12)
    assert result.logdet_c == pytest.approx(case["ldL2"], abs=1e-12)
    assert result.logdet_s == pytest.approx(case["ldRX2"], abs=1e-12)
    assert result.weighted_residual_sum_squares == pytest.approx(
        case["wrss"], abs=1e-12
    )
    assert result.penalized_residual_sum_squares == pytest.approx(
        case["pwrss"], abs=1e-12
    )
    np.testing.assert_allclose(result.beta, case["beta"], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(result.u, case["u"], atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    "fit",
    json.loads(OPTIMIZED_FIXTURE_PATH.read_text(encoding="utf-8"))["fits"],
    ids=lambda fit: fit["id"],
)
def test_evaluation_at_pinned_lme4_optimum(fit: dict[str, Any]) -> None:
    result = evaluate_fixed_theta(
        model_spec(),
        covariance_factor(fit["theta"]),
        kind=ObjectiveKind(fit["kind"]),
    )

    assert result.objective == pytest.approx(fit["objective"], abs=1e-11)
    assert np.sqrt(result.sigma2) == pytest.approx(fit["sigma"], abs=1e-12)
    np.testing.assert_allclose(result.beta, fit["beta"], atol=1e-11, rtol=0.0)
    np.testing.assert_allclose(result.u, fit["u"], atol=1e-11, rtol=0.0)
