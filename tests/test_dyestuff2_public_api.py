# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from kamino import ObjectiveKind, lmer
from kamino.formula import build_random_intercept_design
from kamino.oracle import evaluate_dense_oracle

FIXTURE_PATH = (
    Path(__file__).parents[1] / "oracle" / "fixtures" / "v1" / "dyestuff2.json"
)


def fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def dyestuff2_frame() -> pd.DataFrame:
    data = fixture()["data"]
    return pd.DataFrame(
        {"Yield": data["response"], "Batch": data["groups"]},
        index=data["row_ids"],
    )


def expected_fit(kind: ObjectiveKind) -> dict[str, Any]:
    return next(item for item in fixture()["fits"] if item["kind"] == kind.value)


@pytest.mark.parametrize("kind", list(ObjectiveKind))
def test_public_block_fit_matches_pinned_lme4_dyestuff2_boundary(
    kind: ObjectiveKind,
) -> None:
    expected = expected_fit(kind)
    result = lmer(
        "Yield ~ 1 + (1 | Batch)",
        dyestuff2_frame(),
        reml=kind is ObjectiveKind.REML,
    )

    assert result.kind is kind
    assert result.objective == pytest.approx(expected["objective"], abs=1e-12)
    assert result.log_likelihood == pytest.approx(expected["log_likelihood"], abs=1e-12)
    np.testing.assert_allclose(result.theta, expected["theta"], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(result.beta, expected["beta"], atol=1e-12, rtol=0.0)
    assert result.sigma == pytest.approx(expected["sigma"], abs=1e-12)
    assert result.sigma2 == pytest.approx(expected["residual_variance"], abs=1e-12)
    assert result.random_variance == expected["random_variance"] == 0.0
    np.testing.assert_allclose(
        result.beta_covariance, expected["beta_covariance"], atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(result.u, expected["u"], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(result.random_effects, expected["b"], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        result.fitted_values, expected["fitted"], atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        result.residuals, expected["residuals"], atol=1e-12, rtol=0.0
    )
    assert result.diagnostics.converged
    assert result.diagnostics.boundary
    assert "boundary optimum selected at theta=0" in result.diagnostics.message
    assert result.diagnostics.backend == "single-group-block-cholesky"


@pytest.mark.parametrize("kind", list(ObjectiveKind))
def test_dyestuff2_boundary_matches_dense_and_closed_form_oracles(
    kind: ObjectiveKind,
) -> None:
    frame = dyestuff2_frame()
    result = lmer("Yield ~ 1 + (1 | Batch)", frame, reml=kind is ObjectiveKind.REML)
    design = build_random_intercept_design("Yield ~ 1 + (1 | Batch)", frame)
    dense = evaluate_dense_oracle(
        design.spec,
        np.zeros((design.spec.q, design.spec.q), dtype=np.float64),
        kind=kind,
    )
    centered = design.spec.y - design.spec.offset
    mean = float(np.mean(centered))
    degrees = design.spec.n if kind is ObjectiveKind.ML else design.spec.n - 1
    closed_form_sigma2 = float(np.sum((centered - mean) ** 2) / degrees)

    assert result.objective == pytest.approx(dense.objective, abs=1e-12)
    assert result.beta[0] == pytest.approx(mean, abs=1e-12)
    assert result.sigma2 == pytest.approx(closed_form_sigma2, abs=1e-12)


@pytest.mark.parametrize("kind", list(ObjectiveKind))
def test_dyestuff2_predictions_match_pinned_lme4_at_boundary(
    kind: ObjectiveKind,
) -> None:
    expected = expected_fit(kind)["predictions"]
    result = lmer(
        "Yield ~ 1 + (1 | Batch)",
        dyestuff2_frame(),
        reml=kind is ObjectiveKind.REML,
    )

    training_population = result.predict(mode="population")
    training_conditional = result.predict(mode="conditional")
    np.testing.assert_allclose(
        training_population.values,
        expected["training_population"],
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        training_conditional.values,
        expected["training_conditional"],
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        training_conditional.values, training_population.values
    )

    existing = {"Batch": expected["existing_levels"]}
    conditional = result.predict(existing, mode="conditional")
    np.testing.assert_allclose(
        conditional.values, expected["existing_conditional"], atol=1e-12, rtol=0.0
    )
    new_group = result.predict(
        {"Batch": ["new"]}, mode="conditional", allow_new_groups=True
    )
    assert new_group.values[0] == pytest.approx(
        expected["new_level_conditional"], abs=1e-12
    )
    assert new_group.new_group == (True,)
