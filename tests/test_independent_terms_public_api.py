# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import kamino.formula as formula_module
from kamino import ObjectiveKind, lmer
from kamino.block import evaluate_single_group_block, prepare_single_group_block
from kamino.errors import ModelSpecificationError
from kamino.formula import build_single_group_design
from kamino.oracle import evaluate_dense_oracle

FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "oracle"
    / "fixtures"
    / "v1"
    / "sleepstudy_independent.json"
)


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


def test_double_bar_and_explicit_terms_have_the_same_compact_design() -> None:
    frame = sleepstudy_frame()
    double_bar = build_single_group_design(
        "Reaction ~ Days + (1 + Days || Subject)", frame
    )
    explicit = build_single_group_design(
        "Reaction ~ Days + (1 | Subject) + (0 + Days | Subject)", frame
    )
    expected = fixture()["data"]

    assert double_bar.formula == "Reaction ~ 1 + Days + (1 + Days || Subject)"
    assert explicit.formula == (
        "Reaction ~ 1 + Days + (1 | Subject) + (0 + Days | Subject)"
    )
    assert double_bar.spec.covariance_term_sizes == (1, 1)
    assert double_bar.spec.d == 2
    assert double_bar.spec.k == 2
    assert double_bar.spec.q == 36
    assert double_bar.random_coefficient_names == ("(Intercept)", "Days")
    assert double_bar.spec.random_names == tuple(expected["random_names"])
    np.testing.assert_array_equal(double_bar.spec.x, explicit.spec.x)
    np.testing.assert_array_equal(
        double_bar.spec.random_design, explicit.spec.random_design
    )
    np.testing.assert_array_equal(double_bar.group_indices, explicit.group_indices)
    assert not hasattr(double_bar.spec, "z")
    workspace = prepare_single_group_block(double_bar.spec)
    assert np.any(np.abs(workspace.random_cross[:, 0, 1]) > 0.0)

    z = np.zeros((double_bar.spec.n, double_bar.spec.q), dtype=np.float64)
    rows = np.arange(double_bar.spec.n)
    for column in range(double_bar.spec.k):
        z[rows, double_bar.group_indices * double_bar.spec.k + column] = (
            double_bar.spec.random_design[:, column]
        )
    canonical = np.array(z, dtype="<f8", order="F", copy=True)
    canonical[canonical == 0.0] = 0.0
    assert (
        hashlib.sha256(canonical.tobytes(order="F")).hexdigest()
        == expected["Z_group_major_sha256"]
    )


@pytest.mark.parametrize(
    "formula",
    [
        "Reaction ~ Days + (1 + Days || Subject)",
        "Reaction ~ Days + (1 | Subject) + (0 + Days | Subject)",
    ],
)
def test_formulae_receives_only_the_fixed_formula_for_independent_terms(
    formula: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, bool]] = []
    original = cast(Callable[..., Any], formula_module.design_matrices)

    def recording_design_matrices(
        fixed_formula: str, data: pd.DataFrame, **kwargs: Any
    ) -> Any:
        calls.append((fixed_formula, "Subject" in data.columns))
        return original(fixed_formula, data, **kwargs)

    monkeypatch.setattr(formula_module, "design_matrices", recording_design_matrices)
    build_single_group_design(formula, sleepstudy_frame())

    assert calls == [("Reaction ~ Days", False)]


def test_categorical_double_bar_remains_fail_closed() -> None:
    frame = sleepstudy_frame().assign(Shift=pd.Categorical(["day", "night"] * 90))

    with pytest.raises(ModelSpecificationError, match="must be numeric"):
        lmer("Reaction ~ Shift + (1 + Shift || Subject)", frame)


@pytest.mark.parametrize(
    "case", fixture()["fixed_theta_cases"], ids=lambda case: case["id"]
)
def test_independent_terms_fixed_theta_matches_lme4_and_dense_oracle(
    case: dict[str, Any],
) -> None:
    design = build_single_group_design(
        "Reaction ~ Days + (1 + Days || Subject)", sleepstudy_frame()
    )
    kind = ObjectiveKind(case["kind"])
    result = evaluate_single_group_block(design.spec, case["theta"], kind=kind)
    per_group_factor = np.zeros((design.spec.k, design.spec.k), dtype=np.float64)
    np.fill_diagonal(per_group_factor, np.asarray(case["theta"], dtype=np.float64))
    dense_factor = np.asarray(
        np.kron(np.eye(design.spec.group_count), per_group_factor),
        dtype=np.float64,
    )
    dense = evaluate_dense_oracle(
        design.spec,
        dense_factor,
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
    np.testing.assert_allclose(
        result.u, components["u_group_major"], atol=1e-9, rtol=0.0
    )
    np.testing.assert_allclose(
        result.b, components["b_group_major"], atol=1e-9, rtol=0.0
    )


@pytest.mark.parametrize("kind", list(ObjectiveKind))
def test_public_independent_sleepstudy_fit_matches_pinned_lme4(
    kind: ObjectiveKind,
) -> None:
    expected = expected_fit(kind)
    result = lmer(
        "Reaction ~ Days + (1 + Days || Subject)",
        sleepstudy_frame(),
        reml=kind is ObjectiveKind.REML,
    )

    assert result.objective == pytest.approx(expected["objective"], abs=1e-6)
    assert result.log_likelihood == pytest.approx(expected["log_likelihood"], abs=5e-7)
    np.testing.assert_allclose(result.theta, expected["theta"], atol=2e-5, rtol=0.0)
    np.testing.assert_allclose(result.beta, expected["beta"], atol=1e-8, rtol=0.0)
    assert result.sigma == pytest.approx(expected["sigma"], abs=2e-4)
    np.testing.assert_allclose(
        result.random_covariance,
        expected["random_covariance_group_major"],
        atol=2e-2,
        rtol=1e-4,
    )
    assert result.random_covariance[0, 1] == 0.0
    assert result.random_covariance[1, 0] == 0.0
    np.testing.assert_allclose(
        result.random_effects, expected["b_group_major"], atol=1e-3, rtol=0.0
    )
    np.testing.assert_allclose(
        result.fitted_values, expected["fitted"], atol=1e-3, rtol=0.0
    )
    assert result.covariance_term_sizes == (1, 1)
    assert result.diagnostics.parameter_count == 2
    assert result.diagnostics.optimizer == "scipy-powell"


@pytest.mark.parametrize("kind", list(ObjectiveKind))
def test_explicit_independent_terms_and_double_bar_fit_identically(
    kind: ObjectiveKind,
) -> None:
    frame = sleepstudy_frame()
    double_bar = lmer(
        "Reaction ~ Days + (1 + Days || Subject)",
        frame,
        reml=kind is ObjectiveKind.REML,
    )
    explicit = lmer(
        "Reaction ~ Days + (1 | Subject) + (0 + Days | Subject)",
        frame,
        reml=kind is ObjectiveKind.REML,
    )

    assert explicit.objective == double_bar.objective
    np.testing.assert_array_equal(explicit.theta, double_bar.theta)
    np.testing.assert_array_equal(explicit.beta, double_bar.beta)
    np.testing.assert_array_equal(explicit.random_effects, double_bar.random_effects)
    assert explicit.covariance_term_sizes == (1, 1)


@pytest.mark.parametrize("kind", list(ObjectiveKind))
def test_independent_sleepstudy_predictions_match_pinned_lme4(
    kind: ObjectiveKind,
) -> None:
    expected = expected_fit(kind)["predictions"]
    result = lmer(
        "Reaction ~ Days + (1 + Days || Subject)",
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
    existing = {
        "Days": expected["existing_days"],
        "Subject": expected["existing_levels"],
    }
    np.testing.assert_allclose(
        result.predict(existing, mode="conditional").values,
        expected["existing_conditional"],
        atol=1e-3,
        rtol=0.0,
    )
    unseen = {"Days": expected["new_days"], "Subject": ["new"] * 3}
    np.testing.assert_allclose(
        result.predict(unseen, mode="conditional", allow_new_groups=True).values,
        expected["new_level_conditional"],
        atol=1e-8,
        rtol=0.0,
    )


@pytest.mark.parametrize("kind", list(ObjectiveKind))
def test_independent_slope_optimizer_returns_exact_boundary(
    kind: ObjectiveKind,
) -> None:
    boundary = fixture()["synthetic_boundary"]
    data = boundary["data"]
    expected = next(item for item in boundary["fits"] if item["kind"] == kind.value)
    frame = pd.DataFrame(
        {
            "y": data["y"],
            "x": data["x"],
            "g": pd.Categorical(
                data["groups"], categories=data["group_levels"], ordered=True
            ),
        }
    )
    result = lmer(
        "y ~ x + (1 + x || g)",
        frame,
        reml=kind is ObjectiveKind.REML,
        weights=data["weights"],
        offset=data["offset"],
    )

    assert result.objective == pytest.approx(expected["objective"], abs=1e-8)
    assert result.theta[0] == pytest.approx(expected["theta"][0], abs=1e-5)
    assert result.theta[1] == 0.0
    assert result.random_covariance[1, 1] == 0.0
    assert result.diagnostics.boundary
