# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import kamino.fit as fit_module
import kamino.formula as formula_module
from kamino import FitControl, ObjectiveKind, lmer
from kamino.block import evaluate_single_group_block
from kamino.errors import (
    ConvergenceError,
    NumericalError,
    PredictionError,
    UnsupportedFormulaError,
)
from kamino.formula import build_single_group_design
from kamino.oracle import evaluate_dense_oracle

FIXTURE_PATH = (
    Path(__file__).parents[1] / "oracle" / "fixtures" / "v1" / "sleepstudy.json"
)
SYNTHETIC_FIXED_PATH = FIXTURE_PATH.with_name("fixed_theta.json")
SYNTHETIC_OPTIMIZED_PATH = FIXTURE_PATH.with_name("optimized.json")


def fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def sleepstudy_frame() -> pd.DataFrame:
    data = fixture()["data"]
    return pd.DataFrame(
        {
            "Reaction": data["response"],
            "Days": data["predictor"],
            "Subject": pd.Categorical(
                data["groups"], categories=data["group_levels"], ordered=True
            ),
        },
        index=data["row_ids"],
    )


def expected_fit(kind: ObjectiveKind) -> dict[str, Any]:
    return next(item for item in fixture()["fits"] if item["kind"] == kind.value)


def covariance_factor(theta: list[float], groups: int) -> np.ndarray:
    factor = np.array([[theta[0], 0.0], [theta[1], theta[2]]], dtype=np.float64)
    return np.kron(np.eye(groups), factor)


def test_sleepstudy_formula_has_exact_compact_design_and_labels() -> None:
    expected = fixture()["data"]
    design = build_single_group_design(
        "Reaction ~ Days + (1 + Days | Subject)", sleepstudy_frame()
    )

    assert design.formula == "Reaction ~ 1 + Days + (1 + Days | Subject)"
    assert design.predictor_name == "Days"
    assert design.random_coefficient_names == ("(Intercept)", "Days")
    assert design.group_levels == tuple(expected["group_levels"])
    assert design.spec.fixed_names == tuple(expected["fixed_names"])
    assert design.spec.random_names == tuple(expected["random_names"])
    np.testing.assert_array_equal(design.group_indices, expected["group_indices"])
    np.testing.assert_allclose(design.spec.x, expected["X"], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        design.spec.random_design,
        expected["random_design"],
        atol=0.0,
        rtol=0.0,
    )
    assert design.spec.k == 2
    assert design.spec.q == 36
    assert not hasattr(design.spec, "z")


def test_formulae_never_receives_the_sleepstudy_grouped_term(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, bool]] = []
    original = cast(Callable[..., Any], formula_module.design_matrices)

    def recording_design_matrices(
        formula: str, data: pd.DataFrame, **kwargs: Any
    ) -> Any:
        calls.append((formula, data.shape[1], "Subject" in data.columns))
        return original(formula, data, **kwargs)

    monkeypatch.setattr(formula_module, "design_matrices", recording_design_matrices)
    build_single_group_design(
        "Reaction ~ Days + (1 + Days | Subject)", sleepstudy_frame()
    )

    assert calls == [("Reaction ~ Days", 2, False)]


@pytest.mark.parametrize(
    "formula",
    [
        "Reaction ~ Days + (0 + Days | Subject)",
        "Reaction ~ Days + (1 + Other | Subject)",
        "Reaction ~ Days + (1 + Days | Subject) + (1 | Batch)",
    ],
)
def test_public_slope_fit_rejects_unverified_structures(formula: str) -> None:
    with pytest.raises(UnsupportedFormulaError, match="accepts only"):
        lmer(formula, sleepstudy_frame())


@pytest.mark.parametrize(
    "case", fixture()["fixed_theta_cases"], ids=lambda case: case["id"]
)
def test_sleepstudy_fixed_theta_matches_lme4_and_dense_marginal_oracle(
    case: dict[str, Any],
) -> None:
    design = build_single_group_design(
        "Reaction ~ Days + (1 + Days | Subject)", sleepstudy_frame()
    )
    kind = ObjectiveKind(case["kind"])
    result = evaluate_single_group_block(design.spec, case["theta"], kind=kind)
    dense = evaluate_dense_oracle(
        design.spec,
        covariance_factor(case["theta"], design.spec.group_count),
        kind=kind,
    )
    components = case["components"]

    assert result.objective == pytest.approx(case["objective"], abs=1e-9)
    assert result.objective == pytest.approx(dense.objective, abs=1e-9)
    assert result.logdet_c == pytest.approx(components["ldL2"], abs=1e-10)
    assert result.logdet_s == pytest.approx(components["ldRX2"], abs=1e-10)
    assert result.weighted_residual_sum_squares == pytest.approx(
        components["wrss"], abs=1e-8
    )
    assert result.penalized_residual_sum_squares == pytest.approx(
        components["pwrss"], abs=1e-8
    )
    np.testing.assert_allclose(result.beta, components["beta"], atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(result.u, components["u"], atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(result.b, components["b"], atol=1e-9, rtol=0.0)


@pytest.mark.parametrize("kind", list(ObjectiveKind))
def test_public_sleepstudy_fit_matches_pinned_lme4(kind: ObjectiveKind) -> None:
    expected = expected_fit(kind)
    result = lmer(
        "Reaction ~ Days + (1 + Days | Subject)",
        sleepstudy_frame(),
        reml=kind is ObjectiveKind.REML,
    )

    assert result.kind is kind
    assert result.objective == pytest.approx(expected["objective"], abs=1e-6)
    assert result.log_likelihood == pytest.approx(expected["log_likelihood"], abs=5e-7)
    np.testing.assert_allclose(result.theta, expected["theta"], atol=5e-5, rtol=0.0)
    np.testing.assert_allclose(result.beta, expected["beta"], atol=1e-8, rtol=0.0)
    assert result.sigma == pytest.approx(expected["sigma"], abs=2e-4)
    assert result.sigma2 == pytest.approx(expected["residual_variance"], abs=1e-2)
    np.testing.assert_allclose(
        result.random_covariance,
        expected["random_covariance"],
        atol=1e-3,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        result.beta_covariance,
        expected["beta_covariance"],
        atol=1e-3,
        rtol=1e-4,
    )
    np.testing.assert_allclose(result.u, expected["u"], atol=1e-3, rtol=0.0)
    np.testing.assert_allclose(
        result.random_effects, expected["b"], atol=1e-3, rtol=0.0
    )
    np.testing.assert_allclose(
        result.fitted_values, expected["fitted"], atol=1e-3, rtol=0.0
    )
    np.testing.assert_allclose(
        result.residuals, expected["residuals"], atol=1e-3, rtol=0.0
    )
    assert result.random_variance == result.random_covariance[0, 0]
    assert result.random_coefficient_names == ("(Intercept)", "Days")
    assert result.diagnostics.converged
    assert not result.diagnostics.boundary
    assert result.diagnostics.optimizer == "scipy-powell"
    assert result.diagnostics.parameter_count == 3
    assert result.diagnostics.search_upper_bound is None
    assert result.diagnostics.backend == "single-group-block-cholesky"
    for array in (result.theta, result.random_covariance, result.random_effects):
        assert not array.flags.writeable


@pytest.mark.parametrize("kind", list(ObjectiveKind))
def test_sleepstudy_predictions_match_pinned_lme4(kind: ObjectiveKind) -> None:
    expected = expected_fit(kind)["predictions"]
    result = lmer(
        "Reaction ~ Days + (1 + Days | Subject)",
        sleepstudy_frame(),
        reml=kind is ObjectiveKind.REML,
    )

    np.testing.assert_allclose(
        result.predict(mode="population").values,
        expected["training_population"],
        atol=1e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.predict(mode="conditional").values,
        expected["training_conditional"],
        atol=1e-3,
        rtol=0.0,
    )
    existing = pd.DataFrame(
        {
            "Days": expected["existing_days"],
            "Subject": expected["existing_levels"],
        },
        index=["existing-1", "existing-2", "existing-3"],
    )
    np.testing.assert_allclose(
        result.predict(existing, mode="population").values,
        expected["existing_population"],
        atol=1e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.predict(existing, mode="conditional").values,
        expected["existing_conditional"],
        atol=1e-3,
        rtol=0.0,
    )
    unseen = {"Days": expected["new_days"], "Subject": ["new"] * 3}
    with pytest.raises(PredictionError, match="new or missing group"):
        result.predict(unseen, mode="conditional")
    allowed = result.predict(unseen, mode="conditional", allow_new_groups=True)
    np.testing.assert_allclose(
        allowed.values,
        expected["new_level_conditional"],
        atol=1e-8,
        rtol=0.0,
    )
    assert allowed.new_group == (True, True, True)


def test_sleepstudy_prediction_validates_numeric_predictor() -> None:
    result = lmer("Reaction ~ Days + (1 + Days | Subject)", sleepstudy_frame())
    with pytest.raises(PredictionError, match="requires column 'Days'"):
        result.predict({"Subject": ["308"]}, mode="conditional")
    with pytest.raises(PredictionError, match="must be numeric"):
        result.predict({"Days": ["late"], "Subject": ["308"]}, mode="conditional")
    with pytest.raises(PredictionError, match="non-finite"):
        result.predict({"Days": [np.nan], "Subject": ["308"]}, mode="conditional")


def test_vector_optimizer_has_structured_evaluation_limit_failure() -> None:
    with pytest.raises(ConvergenceError, match="evaluation limit"):
        lmer(
            "Reaction ~ Days + (1 + Days | Subject)",
            sleepstudy_frame(),
            control=FitControl(maximum_evaluations=2),
        )


def test_vector_optimizer_treats_a_numerically_invalid_probe_as_infeasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = fit_module.evaluate_prepared_single_group_block
    evaluations = 0

    def intermittently_invalid(*args: Any, **kwargs: Any) -> Any:
        nonlocal evaluations
        evaluations += 1
        if evaluations == 2:
            raise NumericalError("synthetic invalid trial point")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        fit_module, "evaluate_prepared_single_group_block", intermittently_invalid
    )
    result = lmer(
        "Reaction ~ Days + (1 + Days | Subject)", sleepstudy_frame(), reml=False
    )

    assert result.diagnostics.converged
    assert evaluations > 2


@pytest.mark.parametrize("kind", list(ObjectiveKind))
def test_public_random_slope_fit_accepts_a_singular_boundary(
    kind: ObjectiveKind,
) -> None:
    fixed = json.loads(SYNTHETIC_FIXED_PATH.read_text(encoding="utf-8"))["data"]
    expected = next(
        item
        for item in json.loads(SYNTHETIC_OPTIMIZED_PATH.read_text(encoding="utf-8"))[
            "fits"
        ]
        if item["kind"] == kind.value
    )
    result = lmer(
        "y ~ x + (1 + x | g)",
        {
            "y": fixed["y"],
            "x": [row[1] for row in fixed["X"]],
            "g": ["beta"] * 4 + ["alpha"] * 4 + ["gamma"] * 4,
        },
        reml=kind is ObjectiveKind.REML,
        weights=fixed["weights"],
        offset=fixed["offset"],
    )

    assert result.objective == pytest.approx(expected["objective"], abs=1e-8)
    np.testing.assert_allclose(result.theta[:2], expected["theta"][:2], atol=1e-5)
    assert result.theta[2] == 0.0
    assert result.diagnostics.boundary
    assert "boundary optimum selected" in result.diagnostics.message
