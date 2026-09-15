# pyright: reportArgumentType=false, reportMissingTypeStubs=false
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from kamino import BundleError, ObjectiveKind, lmer, load_model_bundle
from kamino.errors import (
    ModelSpecificationError,
    PredictionError,
    UnsupportedFormulaError,
)
from kamino.formula import build_single_group_design
from kamino.oracle import evaluate_dense_oracle

FIXTURES = Path(__file__).parents[1] / "oracle" / "fixtures" / "v1"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _frame() -> pd.DataFrame:
    source = _fixture("model_frame.json")["data"]
    return pd.DataFrame(
        {
            "y": source["response"],
            "x": source["predictor"],
            "f": pd.Categorical(
                source["fixed_factor"],
                categories=source["fixed_factor_levels"],
                ordered=True,
            ),
            "g": pd.Categorical(
                source["groups"],
                categories=source["group_levels"],
                ordered=True,
            ),
            "w": source["weights"],
            "o": source["formula_offset"],
            "a": source["argument_offset"],
        },
        index=source["row_ids"],
    )


def _matrix_sha256(matrix: np.ndarray[Any, Any]) -> str:
    canonical = np.array(matrix, dtype="<f8", order="F", copy=True)
    canonical[canonical == 0.0] = 0.0
    return hashlib.sha256(canonical.tobytes(order="F")).hexdigest()


@pytest.mark.parametrize("contrast", ["treatment", "sum"])
@pytest.mark.parametrize("reml", [False, True], ids=["ml", "reml"])
def test_weighted_offset_categorical_fit_and_predictions_match_lme4(
    contrast: str, reml: bool
) -> None:
    fixture = _fixture("model_frame.json")
    expected = next(
        fit
        for fit in fixture["fits"]
        if fit["contrast"] == contrast and (fit["kind"] == "reml") == reml
    )
    frame = _frame()
    result = lmer(
        "y ~ x + f + offset(o) + (1 | g)",
        frame,
        reml=reml,
        weights=frame["w"],
        offset=frame["a"],
        contrasts={"f": contrast},  # type: ignore[dict-item]
    )
    design = build_single_group_design(
        "y ~ x + f + offset(o) + (1 | g)",
        frame,
        weights=frame["w"],
        offset=frame["a"],
        contrasts={"f": contrast},  # type: ignore[dict-item]
    )

    assert result.fixed_names == tuple(expected["fixed_names"])
    assert _matrix_sha256(design.spec.x) == expected["X_sha256"]
    assert result.objective == pytest.approx(expected["objective"], abs=1e-8)
    assert result.log_likelihood == pytest.approx(expected["log_likelihood"], abs=5e-9)
    np.testing.assert_allclose(result.theta, expected["theta"], atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(result.beta, expected["beta"], atol=1e-7, rtol=0.0)
    assert result.sigma == pytest.approx(expected["sigma"], abs=1e-7)
    np.testing.assert_allclose(
        result.random_covariance,
        expected["random_covariance"],
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.beta_covariance, expected["beta_covariance"], atol=1e-6, rtol=0.0
    )
    np.testing.assert_allclose(result.u, expected["u"], atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(
        result.random_effects, expected["b"], atol=1e-6, rtol=0.0
    )
    np.testing.assert_allclose(
        result.fitted_values, expected["fitted"], atol=1e-6, rtol=0.0
    )
    np.testing.assert_allclose(
        result.residuals, expected["residuals"], atol=1e-6, rtol=0.0
    )
    kind = ObjectiveKind.REML if reml else ObjectiveKind.ML
    relative_factor = np.eye(design.spec.q, dtype=np.float64) * result.theta[0]
    dense = evaluate_dense_oracle(design.spec, relative_factor, kind=kind)
    assert result.objective == pytest.approx(dense.objective, abs=1e-10)
    np.testing.assert_allclose(result.beta, dense.beta, atol=1e-10, rtol=0.0)
    assert result.sigma2 == pytest.approx(dense.sigma2, abs=1e-10)

    for name in ("existing", "unseen"):
        prediction = expected["predictions"][name]
        new_data = {
            "x": np.atleast_1d(prediction["x"]),
            "f": np.atleast_1d(prediction["f"]),
            "g": np.atleast_1d(prediction["g"]),
            "o": np.atleast_1d(prediction["formula_offset"]),
        }
        argument_offset = np.atleast_1d(prediction["argument_offset"])
        for mode in ("population", "conditional"):
            actual = result.predict(
                new_data,
                mode=mode,
                allow_new_groups=name == "unseen",
                offset=argument_offset,
            )
            np.testing.assert_allclose(
                actual.values,
                np.atleast_1d(prediction[mode]),
                atol=1e-6,
                rtol=0.0,
            )
            expected_new_group = name == "unseen" and mode == "conditional"
            assert actual.new_group == (expected_new_group,) * len(actual.values)


def test_shared_model_frame_matches_lme4_subset_missing_weights_and_offsets() -> None:
    expected = _fixture("model_frame.json")["shared_frame"]
    frame = _frame()
    frame.loc[["r02", "r03"], ["x", "f", "w", "a"]] = np.nan
    frame.loc[["r06", "r03"], ["y", "g", "o"]] = np.nan

    design = build_single_group_design(
        "y ~ x + f + offset(o) + (1 | g)",
        frame,
        weights=frame["w"],
        offset=frame["a"],
        subset=expected["subset"],
        na_action="omit",
    )
    assert design.spec.row_ids == tuple(expected["retained_row_ids"])
    assert design.omitted_row_ids == ("r02", "r06")
    assert design.excluded_row_ids == ("r03",)
    assert design.spec.fixed_names == tuple(expected["fixed_names"])
    assert _matrix_sha256(design.spec.x) == expected["X_sha256"]
    assert design.training_groups == tuple(expected["groups"])
    np.testing.assert_array_equal(design.spec.weights, expected["weights"])
    np.testing.assert_array_equal(design.spec.offset, expected["offset"])

    result = lmer(
        "y ~ x + f + offset(o) + (1 | g)",
        frame,
        weights=frame["w"],
        offset=frame["a"],
        subset=expected["subset"],
        na_action="omit",
    )
    assert result.row_ids == tuple(expected["retained_row_ids"])
    assert result.omitted_row_ids == ("r02", "r06")
    assert result.excluded_row_ids == ("r03",)
    assert result.na_action == "omit"

    with pytest.raises(ModelSpecificationError, match="r02.*r06"):
        lmer(
            "y ~ x + f + offset(o) + (1 | g)",
            frame,
            weights=frame["w"],
            offset=frame["a"],
            subset=expected["subset"],
            na_action="error",
        )


def test_interaction_expansion_matches_pinned_lme4_matrix_contract() -> None:
    expected = next(
        case
        for case in _fixture("formula_contract.json")["cases"]
        if case["id"] == "fixed_interaction"
    )
    design = build_single_group_design("y ~ x * f + (1 | g)", _frame())
    assert design.spec.fixed_names == tuple(expected["fixed_names"])
    assert design.spec.x.shape == tuple(expected["shape_X"])
    assert _matrix_sha256(design.spec.x) == expected["X_sha256"]


def test_subset_drops_unused_fixed_and_group_levels_in_declared_order() -> None:
    frame = _frame()
    selected = (frame["f"] != "low") & (frame["g"] != "beta")
    design = build_single_group_design(
        "y ~ x + f + (1 | g)", frame, subset=selected.to_numpy()
    )
    assert design.fixed_encoder.variables[1].levels == ("middle", "high")
    assert design.spec.fixed_names == ("(Intercept)", "x", "fhigh")
    assert design.group_levels == ("gamma", "alpha")
    assert design.excluded_row_ids == tuple(frame.index[~selected])


def test_prediction_reapplies_fixed_encoder_and_both_offset_sources() -> None:
    frame = _frame()
    result = lmer(
        "y ~ x + f + offset(o) + (1 | g)",
        frame,
        weights=frame["w"],
        offset=frame["a"],
    )
    with pytest.raises(PredictionError, match="requires column 'o'"):
        result.predict(
            {"x": [0.0], "f": ["middle"], "g": ["gamma"]},
            mode="population",
            offset=[0.0],
        )
    with pytest.raises(PredictionError, match="requires an explicit offset"):
        result.predict(
            {"x": [0.0], "f": ["middle"], "g": ["gamma"], "o": [0.0]},
            mode="population",
        )
    with pytest.raises(PredictionError, match="unknown levels.*other"):
        result.predict(
            {"x": [0.0], "f": ["other"], "g": ["gamma"], "o": [0.0]},
            mode="population",
            offset=[0.0],
        )


def test_categorical_prediction_bundle_round_trip(tmp_path: Path) -> None:
    frame = _frame()
    fitted = lmer(
        "y ~ x + f + offset(o) + (1 | g)",
        frame,
        weights=frame["w"],
        offset=frame["a"],
        contrasts={"f": "sum"},
    )
    loaded = load_model_bundle(fitted.save(tmp_path / "categorical.kamino"))
    assert loaded.fixed_encoder == fitted.fixed_encoder
    assert loaded.formula_offset_names == ("o",)
    new_data = {"x": [0.25], "f": ["high"], "g": ["alpha"], "o": [-0.1]}
    expected = fitted.predict(new_data, mode="conditional", offset=[0.03])
    actual = loaded.predict(new_data, mode="conditional", offset=[0.03])
    np.testing.assert_array_equal(actual.values, expected.values)
    with pytest.raises(BundleError):
        load_model_bundle(tmp_path / "missing.kamino")


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ f + (1 + f || g)",
        "y ~ x + offset(o) + offset(o) + (1 | g)",
    ],
)
def test_unsupported_formula_structures_fail_closed(formula: str) -> None:
    frame = _frame().assign(h=["one", "two"] * 6)
    with pytest.raises(UnsupportedFormulaError):
        lmer(formula, frame)


def test_model_frame_controls_are_validated_before_optimization() -> None:
    frame = _frame()
    with pytest.raises(ModelSpecificationError, match="na_action"):
        lmer("y ~ x + (1 | g)", frame, na_action="drop")  # type: ignore[arg-type]
    with pytest.raises(ModelSpecificationError, match="boolean vector"):
        lmer("y ~ x + (1 | g)", frame, subset=[1] * len(frame))  # type: ignore[list-item]
    with pytest.raises(ModelSpecificationError, match="row identifiers"):
        lmer("y ~ x + (1 | g)", frame, weights=frame["w"].iloc[::-1])
    with pytest.raises(ModelSpecificationError, match="row identifiers"):
        lmer(
            "y ~ x + (1 | g)",
            frame,
            subset=pd.Series(True, index=frame.index[::-1]),
        )
    with pytest.raises(ModelSpecificationError, match="categorical variables"):
        lmer("y ~ x + (1 | g)", frame, contrasts={"x": "sum"})
    with pytest.raises(ModelSpecificationError, match="unknown fixed variables"):
        lmer("y ~ x + (1 | g)", frame, contrasts={"f": "sum"})
    aliased = lmer("y ~ x + duplicate + (1 | g)", frame.assign(duplicate=frame["x"]))
    assert aliased.fixed_names == ("(Intercept)", "x")
    assert aliased.dropped_fixed_names == ("duplicate",)


def test_prediction_offset_series_must_share_new_data_row_identity() -> None:
    result = lmer("y ~ x + (1 | g)", _frame(), offset=[0.0] * 12)
    new_data = pd.DataFrame(
        {"x": [0.0, 1.0], "g": ["gamma", "alpha"]}, index=["one", "two"]
    )
    with pytest.raises(PredictionError, match="row identifiers"):
        result.predict(
            new_data,
            mode="population",
            offset=pd.Series([0.0, 0.0], index=["two", "one"]),
        )
