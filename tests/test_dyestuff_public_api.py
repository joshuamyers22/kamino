# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import kamino.fit as fit_module
import kamino.formula as formula_module
from kamino import FitControl, ObjectiveKind, lmer
from kamino.errors import (
    ConvergenceError,
    ModelSpecificationError,
    PredictionError,
    UnsupportedFormulaError,
)
from kamino.formula import build_random_intercept_design
from kamino.oracle import evaluate_dense_oracle

FIXTURE_PATH = (
    Path(__file__).parents[1] / "oracle" / "fixtures" / "v1" / "dyestuff.json"
)


def fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def dyestuff_frame() -> pd.DataFrame:
    data = fixture()["data"]
    return pd.DataFrame(
        {"Yield": data["response"], "Batch": data["groups"]},
        index=data["row_ids"],
    )


def expected_fit(kind: ObjectiveKind) -> dict[str, Any]:
    return next(item for item in fixture()["fits"] if item["kind"] == kind.value)


@pytest.mark.parametrize("kind", list(ObjectiveKind))
def test_public_fit_matches_pinned_lme4_dyestuff(kind: ObjectiveKind) -> None:
    expected = expected_fit(kind)
    result = lmer(
        "Yield ~ 1 + (1 | Batch)",
        dyestuff_frame(),
        reml=kind is ObjectiveKind.REML,
    )

    assert result.kind is kind
    assert result.objective == pytest.approx(expected["objective"], abs=1e-8)
    assert result.log_likelihood == pytest.approx(expected["log_likelihood"], abs=1e-8)
    np.testing.assert_allclose(result.theta, expected["theta"], atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(result.beta, expected["beta"], atol=1e-8, rtol=0.0)
    assert result.sigma == pytest.approx(expected["sigma"], abs=1e-6)
    assert result.sigma2 == pytest.approx(expected["residual_variance"], abs=1e-4)
    assert result.random_variance == pytest.approx(
        expected["random_variance"], abs=1e-4
    )
    np.testing.assert_allclose(
        result.beta_covariance, expected["beta_covariance"], atol=1e-4, rtol=0.0
    )
    np.testing.assert_allclose(result.u, expected["u"], atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(
        result.random_effects, expected["b"], atol=1e-5, rtol=0.0
    )
    np.testing.assert_allclose(
        result.fitted_values, expected["fitted"], atol=1e-5, rtol=0.0
    )
    np.testing.assert_allclose(
        result.residuals, expected["residuals"], atol=1e-5, rtol=0.0
    )
    assert result.fixed_names == tuple(fixture()["data"]["fixed_names"])
    assert result.random_names == tuple(fixture()["data"]["random_names"])
    assert result.group_levels == tuple(fixture()["data"]["group_levels"])
    assert result.row_ids == tuple(fixture()["data"]["row_ids"])
    assert result.diagnostics.converged
    assert not result.diagnostics.boundary
    assert result.diagnostics.backend == "random-intercept-block-cholesky"


@pytest.mark.parametrize("kind", list(ObjectiveKind))
def test_dyestuff_public_fit_matches_independent_dense_oracle(
    kind: ObjectiveKind,
) -> None:
    frame = dyestuff_frame()
    result = lmer("Yield ~ 1 + (1 | Batch)", frame, reml=kind is ObjectiveKind.REML)
    design = build_random_intercept_design("Yield ~ 1 + (1 | Batch)", frame)
    dense = evaluate_dense_oracle(
        design.spec,
        np.eye(design.spec.q, dtype=np.float64) * result.theta[0],
        kind=kind,
    )

    assert result.objective == pytest.approx(dense.objective, abs=1e-10)
    np.testing.assert_allclose(result.beta, dense.beta, atol=1e-10, rtol=0.0)
    assert result.sigma2 == pytest.approx(dense.sigma2, abs=1e-10)


@pytest.mark.parametrize(
    ("kind", "expected_random_variance"),
    [
        (ObjectiveKind.ML, 1388.3333333333333),
        (ObjectiveKind.REML, 1764.05),
    ],
)
def test_balanced_dyestuff_variance_reaches_the_scalar_minimum(
    kind: ObjectiveKind, expected_random_variance: float
) -> None:
    result = lmer(
        "Yield ~ 1 + (1 | Batch)",
        dyestuff_frame(),
        reml=kind is ObjectiveKind.REML,
    )
    assert result.random_variance == pytest.approx(expected_random_variance, abs=5e-5)


@pytest.mark.parametrize("kind", list(ObjectiveKind))
def test_dyestuff_predictions_match_pinned_lme4(kind: ObjectiveKind) -> None:
    expected = expected_fit(kind)["predictions"]
    result = lmer(
        "Yield ~ 1 + (1 | Batch)",
        dyestuff_frame(),
        reml=kind is ObjectiveKind.REML,
    )

    training_population = result.predict(mode="population")
    training_conditional = result.predict(mode="conditional")
    np.testing.assert_allclose(
        training_population.values,
        expected["training_population"],
        atol=1e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        training_conditional.values,
        expected["training_conditional"],
        atol=1e-5,
        rtol=0.0,
    )

    existing = pd.DataFrame(
        {"Batch": expected["existing_levels"]},
        index=[f"prediction-{index}" for index in range(6)],
    )
    existing_population = result.predict(existing, mode="population")
    existing_conditional = result.predict(existing, mode="conditional")
    np.testing.assert_allclose(
        existing_population.values,
        expected["existing_population"],
        atol=1e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        existing_conditional.values,
        expected["existing_conditional"],
        atol=1e-5,
        rtol=0.0,
    )
    assert existing_conditional.row_ids == tuple(existing.index)
    assert existing_conditional.new_group == (False,) * 6

    new_group = pd.DataFrame({"Batch": ["new"]}, index=["unseen"])
    with pytest.raises(PredictionError, match="new or missing group"):
        result.predict(new_group, mode="conditional")
    allowed = result.predict(new_group, mode="conditional", allow_new_groups=True)
    assert allowed.values[0] == pytest.approx(
        expected["new_level_conditional"], abs=1e-5
    )
    assert allowed.new_group == (True,)


def test_population_prediction_does_not_require_group_column() -> None:
    result = lmer("Yield ~ 1 + (1 | Batch)", dyestuff_frame())
    prediction = result.predict(
        pd.DataFrame({"occasion": [1, 2]}, index=["a", "b"]), mode="population"
    )
    np.testing.assert_array_equal(prediction.values, np.repeat(result.beta[0], 2))
    assert prediction.new_group == (False, False)


def test_result_and_predictions_are_immutable() -> None:
    result = lmer("Yield ~ 1 + (1 | Batch)", dyestuff_frame())
    prediction = result.predict(mode="conditional")
    for array in (
        result.theta,
        result.beta,
        result.beta_covariance,
        result.u,
        result.random_effects,
        result.fitted_values,
        result.residuals,
        prediction.values,
    ):
        assert not array.flags.writeable


@pytest.mark.parametrize(
    "formula",
    [
        "Yield ~ 0 + (1 | Batch)",
        "Yield ~ 1 + Days + (1 | Batch)",
        "Yield ~ 1 + (1 + Days | Batch)",
        "Yield ~ 1 + (1 | Batch) + (1 | Other)",
        "Yield ~ 1 + (1 | Batch/Other)",
        "Yield ~ 1 + (1 || Batch)",
    ],
)
def test_public_fit_rejects_unadvertised_formulas(formula: str) -> None:
    with pytest.raises(UnsupportedFormulaError, match="accepts only"):
        lmer(formula, dyestuff_frame())


def test_public_fit_validates_frame_and_controls() -> None:
    frame = dyestuff_frame()
    frame.loc["dyestuff-01", "Batch"] = None
    with pytest.raises(ModelSpecificationError, match="missing"):
        lmer("Yield ~ 1 + (1 | Batch)", frame)
    with pytest.raises(ModelSpecificationError, match="strictly positive"):
        lmer(
            "Yield ~ 1 + (1 | Batch)",
            dyestuff_frame(),
            weights=[1.0] * 29 + [0.0],
        )
    with pytest.raises(ModelSpecificationError, match="finite and positive"):
        FitControl(absolute_theta_tolerance=0.0)


def test_zero_between_group_variance_is_a_valid_boundary_fit() -> None:
    frame = pd.DataFrame(
        {
            "y": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "g": ["a", "a", "b", "b", "c", "c"],
        }
    )
    result = lmer("y ~ 1 + (1 | g)", frame)
    assert result.theta[0] == 0.0
    assert result.random_variance == 0.0
    assert result.diagnostics.boundary


def test_categorical_group_order_is_preserved() -> None:
    frame = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "g": pd.Categorical(
                ["A", "B", "C"], categories=["C", "A", "B"], ordered=True
            ),
        }
    )
    design = build_random_intercept_design("y ~ 1 + (1 | g)", frame)
    assert design.group_levels == ("C", "A", "B")
    assert design.spec.random_names == (
        "g[C]:(Intercept)",
        "g[A]:(Intercept)",
        "g[B]:(Intercept)",
    )
    np.testing.assert_array_equal(
        design.group_indices,
        np.array([1, 2, 0], dtype=np.int64),
    )


def test_formula_backend_never_receives_the_grouped_term(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original = cast(Any, formula_module.design_matrices)

    def recording_design_matrices(
        formula: str, data: pd.DataFrame, **kwargs: Any
    ) -> Any:
        calls.append(formula)
        assert "|" not in formula
        assert data.shape[1] == 1
        assert "y" in data.columns
        return original(formula, data, **kwargs)

    monkeypatch.setattr(formula_module, "design_matrices", recording_design_matrices)
    design = build_random_intercept_design(
        "y ~ 1 + (1 | g)", {"y": [1.0, 2.0], "g": ["b", "a"]}
    )

    assert calls == ["y ~ 1"]
    assert not hasattr(design.spec, "z")


def test_high_cardinality_formula_design_has_linear_compact_storage() -> None:
    n = 4_096
    groups = [f"g-{index:04d}" for index in range(n)]
    design = build_random_intercept_design(
        "y ~ 1 + (1 | g)", {"y": np.arange(n, dtype=np.float64), "g": groups}
    )

    assert design.spec.q == n
    assert design.group_indices.shape == (n,)
    assert design.group_indices.nbytes == n * np.dtype(np.int64).itemsize
    assert not hasattr(design.spec, "z")


def test_mapping_input_has_canonical_rows_and_sorted_groups() -> None:
    design = build_random_intercept_design(
        "y ~ (1 | g)", {"y": [1.0, 2.0, 3.0], "g": ["z", "a", "z"]}
    )
    assert design.formula == "y ~ 1 + (1 | g)"
    assert design.spec.row_ids == ("0", "1", "2")
    assert design.group_levels == ("a", "z")


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"y": [1.0], "g": 1}, "one-dimensional sequence"),
        ({"y": np.ones((2, 1)), "g": ["a", "b"]}, "one-dimensional"),
        ({"y": [1.0, 2.0], "g": ["a"]}, "must align"),
        ({"y": [], "g": []}, "at least one row"),
        ({"y": [1.0, 2.0], "g": [1, 2]}, "string labels"),
    ],
)
def test_formula_frame_rejects_invalid_columns(data: Any, message: str) -> None:
    with pytest.raises(ModelSpecificationError, match=message):
        build_random_intercept_design("y ~ 1 + (1 | g)", data)


def test_formula_frame_rejects_missing_columns_and_duplicate_rows() -> None:
    with pytest.raises(ModelSpecificationError, match="no column"):
        build_random_intercept_design("y ~ 1 + (1 | g)", {"y": [1.0]})
    duplicate_rows = pd.DataFrame(
        {"y": [1.0, 2.0], "g": ["a", "b"]}, index=["same", "same"]
    )
    with pytest.raises(ModelSpecificationError, match="identifiers"):
        build_random_intercept_design("y ~ 1 + (1 | g)", duplicate_rows)
    with pytest.raises(UnsupportedFormulaError, match="formula must be a string"):
        build_random_intercept_design(42, duplicate_rows)  # type: ignore[arg-type]


def test_formula_frame_wraps_vendor_missing_response_error() -> None:
    frame = pd.DataFrame({"y": [1.0, np.nan], "g": ["a", "b"]})
    with pytest.raises(ModelSpecificationError, match="formula evaluation failed"):
        build_random_intercept_design("y ~ 1 + (1 | g)", frame)


def test_prediction_mapping_validation_and_missing_groups() -> None:
    result = lmer("Yield ~ 1 + (1 | Batch)", dyestuff_frame())
    known = result.predict({"Batch": np.array(["A", "B"])}, mode="conditional")
    assert known.row_ids == ("0", "1")
    with pytest.raises(PredictionError, match="mode must"):
        result.predict(mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(PredictionError, match="at least one column"):
        result.predict({}, mode="population")
    with pytest.raises(PredictionError, match="must be a sequence"):
        result.predict({"Batch": "A"}, mode="conditional")  # type: ignore[dict-item]
    with pytest.raises(PredictionError, match="one-dimensional"):
        result.predict({"Batch": np.array([["A"]])}, mode="conditional")
    with pytest.raises(PredictionError, match="string labels"):
        result.predict({"Batch": [1]}, mode="conditional")
    with pytest.raises(PredictionError, match="equal lengths"):
        result.predict({"Batch": ["A", "B"], "other": [1]}, mode="conditional")
    with pytest.raises(PredictionError, match="at least one row"):
        result.predict({"Batch": []}, mode="conditional")
    with pytest.raises(PredictionError, match="requires column"):
        result.predict({"other": [1]}, mode="conditional")
    missing = result.predict(
        {"Batch": [None]}, mode="conditional", allow_new_groups=True
    )
    assert missing.new_group == (True,)
    assert missing.values[0] == result.beta[0]
    with pytest.raises(PredictionError, match="must be a boolean"):
        result.predict(
            {"Batch": ["new"]},
            mode="conditional",
            allow_new_groups=1,  # type: ignore[arg-type]
        )


def test_prediction_applies_offsets_without_recycling_training_values() -> None:
    frame = dyestuff_frame()
    training_offset = np.linspace(-1.0, 1.0, len(frame))
    result = lmer("Yield ~ 1 + (1 | Batch)", frame, offset=training_offset)
    population = result.predict(mode="population")
    np.testing.assert_allclose(
        population.values, result.beta[0] + training_offset, atol=0.0, rtol=0.0
    )
    with pytest.raises(PredictionError, match="requires an explicit offset"):
        result.predict({"Batch": ["A"]}, mode="conditional")
    prediction = result.predict({"Batch": ["A"]}, mode="conditional", offset=[2.0])
    assert prediction.values[0] == pytest.approx(
        result.beta[0] + result.random_effects[0] + 2.0
    )
    with pytest.raises(PredictionError, match="cannot be replaced"):
        result.predict(mode="population", offset=training_offset)
    with pytest.raises(PredictionError, match="one value per row"):
        result.predict({"Batch": ["A"]}, mode="conditional", offset=[1.0, 2.0])
    with pytest.raises(PredictionError, match="non-finite"):
        result.predict({"Batch": ["A"]}, mode="conditional", offset=[np.nan])


def test_prediction_rejects_duplicate_dataframe_rows() -> None:
    result = lmer("Yield ~ 1 + (1 | Batch)", dyestuff_frame())
    frame = pd.DataFrame({"Batch": ["A", "B"]}, index=["same", "same"])
    with pytest.raises(PredictionError, match="identifiers"):
        result.predict(frame, mode="conditional")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"initial_upper_bound": 2.0, "maximum_upper_bound": 1.0},
            "must exceed",
        ),
        ({"maximum_evaluations": 0}, "must be positive"),
    ],
)
def test_invalid_fit_controls_are_rejected_during_construction(
    arguments: dict[str, Any], message: str
) -> None:
    with pytest.raises(ModelSpecificationError, match=message):
        FitControl(**arguments)


def test_optimizer_expands_its_bracket_for_large_group_variance() -> None:
    frame = pd.DataFrame(
        {
            "y": [0.0, 0.1, -0.1, 100.0, 100.1, 99.9, 200.0, 200.1, 199.9],
            "g": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
        }
    )
    result = lmer("y ~ 1 + (1 | g)", frame)
    assert result.diagnostics.search_upper_bound > 1.0
    assert result.theta[0] > 1.0


def test_optimizer_rejects_unbracketed_and_over_budget_fits() -> None:
    with pytest.raises(ConvergenceError, match="not bracketed"):
        lmer(
            "Yield ~ 1 + (1 | Batch)",
            dyestuff_frame(),
            control=FitControl(initial_upper_bound=0.1, maximum_upper_bound=0.2),
        )
    with pytest.raises(ConvergenceError, match="evaluation limit"):
        lmer(
            "Yield ~ 1 + (1 | Batch)",
            dyestuff_frame(),
            control=FitControl(maximum_evaluations=2),
        )


def test_optimizer_wraps_failures_and_rejects_unsuccessful_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("optimizer broke")

    monkeypatch.setattr(fit_module, "minimize_scalar", fail)
    with pytest.raises(ConvergenceError, match="optimizer broke"):
        lmer("Yield ~ 1 + (1 | Batch)", dyestuff_frame())

    def unsuccessful(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            success=False, fun=1.0, x=0.5, message="did not converge"
        )

    monkeypatch.setattr(fit_module, "minimize_scalar", unsuccessful)
    with pytest.raises(ConvergenceError, match="did not converge"):
        lmer("Yield ~ 1 + (1 | Batch)", dyestuff_frame())


def test_reml_flag_must_be_boolean() -> None:
    with pytest.raises(ModelSpecificationError, match="boolean"):
        lmer(
            "Yield ~ 1 + (1 | Batch)",
            dyestuff_frame(),
            reml="yes",  # type: ignore[arg-type]
        )
    with pytest.raises(ModelSpecificationError, match="FitControl"):
        lmer(
            "Yield ~ 1 + (1 | Batch)",
            dyestuff_frame(),
            control={"maximum_evaluations": 5},  # type: ignore[arg-type]
        )
