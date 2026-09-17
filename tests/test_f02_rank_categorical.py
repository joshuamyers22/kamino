# pyright: reportArgumentType=false, reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from kamino import ObjectiveKind, lmer, load_model_bundle
from kamino.block import evaluate_single_group_block
from kamino.errors import ModelSpecificationError, PredictionError
from kamino.formula import (
    GeneralDesign,
    build_general_design,
    build_single_group_design,
)
from kamino.model import ModelSpec
from kamino.oracle import evaluate_dense_oracle
from kamino.pls import evaluate_fixed_theta
from kamino.sparse import evaluate_prepared_general_sparse, prepare_general_sparse

FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "oracle"
    / "fixtures"
    / "v1"
    / "f02_rank_categorical.json"
)


def fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def categorical_frame() -> pd.DataFrame:
    source = fixture()["data"]
    return pd.DataFrame(
        {
            "y": source["response"],
            "f": pd.Categorical(
                source["fixed_factor"], categories=source["fixed_factor_levels"]
            ),
            "g": pd.Categorical(source["groups"], categories=source["group_levels"]),
            "h": pd.Categorical(
                source["second_groups"],
                categories=source["second_group_levels"],
            ),
        },
        index=source["row_ids"],
    )


def _matrix_sha256(matrix: np.ndarray[Any, Any]) -> str:
    canonical = np.array(matrix, dtype="<f8", order="F", copy=True)
    canonical[canonical == 0.0] = 0.0
    return hashlib.sha256(canonical.tobytes(order="F")).hexdigest()


def _factor(theta: list[float], size: int) -> np.ndarray[Any, np.dtype[np.float64]]:
    result = np.zeros((size, size), dtype=np.float64)
    cursor = 0
    for column in range(size):
        width = size - column
        result[column:, column] = theta[cursor : cursor + width]
        cursor += width
    return result


def _dense_general(design: GeneralDesign) -> np.ndarray[Any, np.dtype[np.float64]]:
    matrix = np.zeros((design.spec.n, design.spec.q), dtype=np.float64)
    rows = np.arange(design.spec.n)
    offset = 0
    for term in design.spec.terms:
        for coefficient in range(term.k):
            matrix[
                rows,
                offset + term.group_indices * term.k + coefficient,
            ] = term.random_design[:, coefficient]
        offset += term.q
    return matrix


def test_exact_alias_matches_lme4_drop_map_and_preserves_estimability() -> None:
    expected = fixture()["rank_cases"]["exact_alias"]
    frame = categorical_frame().assign(x=expected["x"], duplicate=expected["duplicate"])
    design = build_single_group_design(expected["formula"], frame)

    full_x = design.fixed_encoder.evaluate(frame, len(frame))
    assert design.fixed_rank.full_names == tuple(expected["full_names"])
    assert design.spec.fixed_names == tuple(expected["retained_names"])
    assert design.fixed_rank.pivot == tuple(
        index - 1 for index in expected["qr_pivot_one_based"]
    )
    assert design.fixed_rank.dropped_indices == (
        expected["dropped_indices_one_based"] - 1,
    )
    assert _matrix_sha256(full_x) == expected["full_X_sha256"]
    assert _matrix_sha256(design.spec.x) == expected["retained_X_sha256"]
    np.testing.assert_allclose(
        full_x @ design.fixed_rank.null_basis,
        0.0,
        atol=1e-14,
        rtol=0.0,
    )

    result = lmer(expected["formula"], frame, reml=False)
    np.testing.assert_array_equal(np.isnan(result.full_beta), [False, False, True])
    assert result.dropped_fixed_names == ("duplicate",)
    estimable = result.linear_function([0.0, 1.0, 2.0])
    assert estimable.estimable
    assert estimable.estimate is not None
    assert estimable.standard_error is not None
    nonestimable = result.linear_function([0.0, 1.0, 0.0])
    assert not nonestimable.estimable
    assert nonestimable.estimate is None
    assert nonestimable.standard_error is None


@pytest.mark.parametrize("reml", [False, True], ids=["ml", "reml"])
def test_rank_deficient_fit_and_estimable_predictions_match_lme4(reml: bool) -> None:
    expected_case = fixture()["rank_cases"]["exact_alias"]
    expected = next(
        item for item in expected_case["fits"] if (item["kind"] == "reml") == reml
    )
    frame = categorical_frame().assign(
        x=expected_case["x"], duplicate=expected_case["duplicate"]
    )
    result = lmer(expected_case["formula"], frame, reml=reml)

    assert result.objective == pytest.approx(expected["objective"], abs=1e-10)
    np.testing.assert_allclose(
        result.theta, np.atleast_1d(expected["theta"]), atol=2e-8
    )
    np.testing.assert_allclose(result.beta, expected["beta"], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        result.beta_covariance, expected["beta_covariance"], atol=2e-9, rtol=0.0
    )

    compatible = {
        "x": [-0.75, 0.4],
        "duplicate": [-1.5, 0.8],
        "g": ["g1", "new"],
    }
    for mode in ("population", "conditional"):
        prediction = result.predict(compatible, mode=mode, allow_new_groups=True)
        np.testing.assert_allclose(
            prediction.values, expected[mode], atol=1e-8, rtol=0.0
        )
        assert prediction.estimable == (True, True)

    incompatible = dict(compatible)
    incompatible["duplicate"] = [-1.4, 0.8]
    prediction = result.predict(incompatible, mode="population", allow_new_groups=True)
    assert prediction.estimable == (False, True)
    assert np.isnan(prediction.values[0])
    assert np.isfinite(prediction.values[1])


def test_empty_interaction_and_threshold_cases_match_lme4_qr_policy() -> None:
    cases = fixture()["rank_cases"]
    near = cases["near_alias"]
    near_frame = categorical_frame().assign(x=near["x"], duplicate=near["duplicate"])
    near_design = build_single_group_design(near["formula"], near_frame)
    assert near_design.spec.fixed_names == tuple(near["retained_names"])
    assert near_design.fixed_rank.dropped_indices == (
        near["dropped_indices_one_based"] - 1,
    )
    assert near_design.fixed_rank.pivot == tuple(
        index - 1 for index in near["qr_pivot_one_based"]
    )

    empty = cases["empty_interaction"]
    empty_frame = pd.DataFrame(
        {
            "y": empty["response"],
            "a": pd.Categorical(empty["a"], categories=empty["a_levels"]),
            "b": pd.Categorical(empty["b"], categories=empty["b_levels"]),
            "g": pd.Categorical(empty["groups"]),
        }
    )
    empty_design = build_single_group_design(empty["formula"], empty_frame)
    assert empty_design.fixed_rank.full_names == tuple(empty["full_names"])
    assert empty_design.spec.fixed_names == tuple(empty["retained_names"])
    assert empty_design.fixed_rank.dropped_indices == (
        empty["dropped_indices_one_based"] - 1,
    )
    assert empty_design.fixed_rank.pivot == tuple(
        index - 1 for index in empty["qr_pivot_one_based"]
    )
    assert (
        _matrix_sha256(
            empty_design.fixed_encoder.evaluate(empty_frame, len(empty_frame))
        )
        == empty["full_X_sha256"]
    )

    threshold = cases["threshold_sensitive"]
    alias = cases["exact_alias"]
    threshold_frame = categorical_frame().assign(x=alias["x"], tiny=threshold["tiny"])
    threshold_design = build_single_group_design(threshold["formula"], threshold_frame)
    assert threshold_design.fixed_rank.full_names == tuple(threshold["full_names"])
    assert threshold_design.spec.fixed_names == tuple(threshold["retained_names"])
    assert threshold_design.fixed_rank.dropped_indices == ()
    assert threshold_design.fixed_rank.pivot == tuple(
        index - 1 for index in threshold["qr_pivot_one_based"]
    )


@pytest.mark.parametrize(
    "case", fixture()["categorical_fixed_theta_cases"], ids=lambda case: case["id"]
)
def test_categorical_random_fixed_theta_matches_lme4_and_dense_oracle(
    case: dict[str, Any],
) -> None:
    design = build_single_group_design(
        "y ~ f + (1 + f | g)",
        categorical_frame(),
        contrasts={"f": case["fixed_contrast"]},  # type: ignore[dict-item]
        random_contrasts={"f": case["random_contrast"]},  # type: ignore[dict-item]
    )
    kind = ObjectiveKind(case["kind"])
    actual = evaluate_single_group_block(design.spec, case["theta"], kind=kind)
    local_factor = _factor(case["theta"], design.spec.k)
    dense_factor = np.kron(np.eye(design.spec.group_count), local_factor)
    dense = evaluate_dense_oracle(design.spec, dense_factor, kind=kind)
    components = case["components"]

    assert design.random_coefficient_names == tuple(case["random_coefficient_names"])
    assert _matrix_sha256(design.spec.x) == case["X_sha256"]
    assert actual.objective == pytest.approx(case["objective"], abs=1e-10)
    assert actual.objective == pytest.approx(dense.objective, abs=1e-10)
    assert actual.logdet_c == pytest.approx(components["ldL2"], abs=1e-10)
    assert actual.logdet_s == pytest.approx(components["ldRX2"], abs=1e-10)
    np.testing.assert_allclose(actual.beta, components["beta"], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(actual.u, components["u"], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(actual.b, components["b"], atol=1e-10, rtol=0.0)


@pytest.mark.parametrize(
    ("fixed_contrast", "random_contrast"),
    [("treatment", "treatment"), ("sum", "sum"), ("sum", "treatment")],
)
@pytest.mark.parametrize("reml", [False, True], ids=["ml", "reml"])
def test_categorical_random_final_fit_and_new_data_match_lme4(
    fixed_contrast: str, random_contrast: str, reml: bool
) -> None:
    expected = next(
        item
        for item in fixture()["categorical_fits"]
        if item["fixed_contrast"] == fixed_contrast
        and item["random_contrast"] == random_contrast
        and (item["kind"] == "reml") == reml
    )
    result = lmer(
        "y ~ f + (1 + f | g)",
        categorical_frame(),
        reml=reml,
        contrasts={"f": fixed_contrast},  # type: ignore[dict-item]
        random_contrasts={"f": random_contrast},  # type: ignore[dict-item]
    )

    assert result.objective <= expected["objective"] + 1e-6
    assert result.objective == pytest.approx(expected["objective"], abs=1e-5)
    np.testing.assert_allclose(result.beta, expected["beta"], atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(
        result.random_covariance,
        expected["random_covariance"],
        atol=1e-4,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.random_effects, expected["b"], atol=2e-4, rtol=0.0
    )
    np.testing.assert_allclose(
        result.fitted_values, expected["fitted"], atol=2e-4, rtol=0.0
    )
    assert result.diagnostics.optimizer == "scipy-l-bfgs-b"

    prediction_cases = (
        (
            "existing",
            {"f": ["a", "b", "c"], "g": ["g1", "g3", "g8"]},
            False,
        ),
        ("unseen", {"f": ["a", "c"], "g": ["new", "g2"]}, True),
    )
    for name, data, allow_new in prediction_cases:
        for mode in ("population", "conditional"):
            actual = result.predict(data, mode=mode, allow_new_groups=allow_new)
            np.testing.assert_allclose(
                actual.values,
                expected["predictions"][f"{name}_{mode}"],
                atol=1e-4,
                rtol=0.0,
            )
            assert actual.estimable == (True,) * len(actual.values)

    with pytest.raises(PredictionError, match="unknown levels"):
        result.predict({"f": ["unknown"], "g": ["g1"]}, mode="population")


@pytest.mark.parametrize(
    "case", fixture()["crossed"]["fixed_theta_cases"], ids=lambda case: case["id"]
)
def test_crossed_categorical_slope_matches_lme4_and_dense_pls(
    case: dict[str, Any],
) -> None:
    expected_fixture = fixture()["crossed"]
    design = build_general_design(expected_fixture["formula"], categorical_frame())
    workspace = prepare_general_sparse(design.spec)
    kind = ObjectiveKind(case["kind"])
    actual = evaluate_prepared_general_sparse(workspace, case["theta"], kind=kind)
    z = _dense_general(design)
    dense_spec = ModelSpec.from_arrays(
        y=design.spec.y,
        x=design.spec.x,
        z=z,
        weights=design.spec.weights,
        offset=design.spec.offset,
        row_ids=design.spec.row_ids,
        fixed_names=design.spec.fixed_names,
        random_names=design.spec.random_names,
    )
    blocks: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    cursor = 0
    for term in design.spec.terms:
        width = term.d
        local = _factor(case["theta"][cursor : cursor + width], term.k)
        blocks.extend([local] * term.group_count)
        cursor += width
    q = design.spec.q
    covariance_factor = np.zeros((q, q), dtype=np.float64)
    start = 0
    for block in blocks:
        stop = start + block.shape[0]
        covariance_factor[start:stop, start:stop] = block
        start = stop
    dense = evaluate_fixed_theta(dense_spec, covariance_factor, kind=kind)
    components = case["components"]

    assert tuple(term.random_coefficient_names for term in design.random_terms) == (
        ("(Intercept)", "fb", "fc"),
        ("(Intercept)",),
    )
    assert _matrix_sha256(z) == expected_fixture["Z_sha256"]
    assert actual.objective == pytest.approx(case["objective"], abs=1e-10)
    assert actual.objective == pytest.approx(dense.objective, abs=1e-10)
    np.testing.assert_allclose(actual.beta, components["beta"], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(actual.u, components["u"], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(actual.b, components["b"], atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("reml", [False, True], ids=["ml", "reml"])
def test_crossed_categorical_slope_final_fit_and_predictions_match_lme4(
    reml: bool,
    tmp_path: Path,
) -> None:
    expected_fixture = fixture()["crossed"]
    expected = next(
        item for item in expected_fixture["fits"] if (item["kind"] == "reml") == reml
    )
    result = lmer(expected_fixture["formula"], categorical_frame(), reml=reml)
    loaded = load_model_bundle(
        result.save(tmp_path / f"crossed-categorical-{reml}.kamino")
    )

    assert result.objective <= expected["objective"] + 1e-6
    assert result.objective == pytest.approx(expected["objective"], abs=3e-6)
    np.testing.assert_allclose(result.beta, expected["beta"], atol=1e-9, rtol=0.0)
    expected_covariance = np.zeros((4, 4), dtype=np.float64)
    expected_covariance[:3, :3] = expected["random_covariances"][0]
    expected_covariance[3:, 3:] = expected["random_covariances"][1]
    np.testing.assert_allclose(
        result.random_covariance, expected_covariance, atol=2e-4, rtol=0.0
    )
    np.testing.assert_allclose(
        result.random_effects, expected["b"], atol=2e-4, rtol=0.0
    )
    for name in ("existing", "unseen"):
        data = {key: [value] for key, value in expected_fixture[name].items()}
        for mode in ("population", "conditional"):
            prediction = result.predict(
                data, mode=mode, allow_new_groups=name == "unseen"
            )
            recovered = loaded.predict(
                data, mode=mode, allow_new_groups=name == "unseen"
            )
            np.testing.assert_allclose(
                prediction.values,
                np.atleast_1d(expected["predictions"][f"{name}_{mode}"]),
                atol=2e-4,
                rtol=0.0,
            )
            np.testing.assert_array_equal(recovered.values, prediction.values)
            assert recovered.new_group == prediction.new_group


def test_f02_bundle_recovers_rank_and_categorical_random_artifacts(
    tmp_path: Path,
) -> None:
    rank_case = fixture()["rank_cases"]["exact_alias"]
    rank_result = lmer(
        rank_case["formula"],
        categorical_frame().assign(x=rank_case["x"], duplicate=rank_case["duplicate"]),
    )
    rank_loaded = load_model_bundle(rank_result.save(tmp_path / "rank.kamino"))
    assert rank_loaded.fixed_rank.full_names == rank_result.fixed_rank.full_names
    assert rank_loaded.fixed_rank.dropped_indices == (2,)
    rank_data = {
        "x": [-0.75, 0.4],
        "duplicate": [-1.5, 0.8],
        "g": ["g1", "new"],
    }
    rank_bad = {**rank_data, "duplicate": [-1.4, 0.8]}
    for data in (rank_data, rank_bad):
        for mode in ("population", "conditional"):
            expected = rank_result.predict(
                data,
                mode=mode,
                allow_new_groups=True,  # type: ignore[arg-type]
            )
            actual = rank_loaded.predict(
                data,
                mode=mode,
                allow_new_groups=True,  # type: ignore[arg-type]
            )
            np.testing.assert_equal(actual.values, expected.values)
            assert actual.estimable == expected.estimable
            assert actual.new_group == expected.new_group

    categorical_data = {"f": ["a", "c"], "g": ["g1", "new"]}
    for contrast in ("treatment", "sum"):
        categorical_result = lmer(
            "y ~ f + (1 + f | g)",
            categorical_frame(),
            reml=False,
            contrasts={"f": contrast},  # type: ignore[dict-item]
            random_contrasts={"f": contrast},  # type: ignore[dict-item]
        )
        categorical_loaded = load_model_bundle(
            categorical_result.save(tmp_path / f"categorical-{contrast}.kamino")
        )
        assert categorical_loaded.random_encoder == categorical_result.random_encoder
        for mode in ("population", "conditional"):
            expected = categorical_result.predict(
                categorical_data,
                mode=mode,
                allow_new_groups=True,  # type: ignore[arg-type]
            )
            actual = categorical_loaded.predict(
                categorical_data,
                mode=mode,
                allow_new_groups=True,  # type: ignore[arg-type]
            )
            np.testing.assert_array_equal(actual.values, expected.values)
            assert actual.new_group == expected.new_group
    with pytest.raises(ModelSpecificationError, match="finite vector"):
        rank_result.linear_function([1.0, 2.0])


def test_categorical_new_data_reordering_and_missing_group_are_explicit() -> None:
    result = lmer("y ~ f + (1 + f | g)", categorical_frame(), reml=False)
    ordinary = result.predict({"f": ["a", "c"], "g": ["g1", "g2"]}, mode="conditional")
    reordered = pd.DataFrame(
        {
            "f": pd.Categorical(["a", "c"], categories=["c", "b", "a"]),
            "g": pd.Categorical(
                ["g1", "g2"], categories=list(reversed(result.group_levels))
            ),
        }
    )
    np.testing.assert_array_equal(
        result.predict(reordered, mode="conditional").values, ordinary.values
    )

    missing = pd.DataFrame({"f": ["b"], "g": [None]})
    with pytest.raises(PredictionError, match="missing group"):
        result.predict(missing, mode="conditional")
    allowed = result.predict(missing, mode="conditional", allow_new_groups=True)
    assert allowed.new_group == (True,)
    assert allowed.estimable == (True,)

    with pytest.raises(ModelSpecificationError, match="unknown random variables"):
        lmer(
            "y ~ f + (1 + f | g)",
            categorical_frame(),
            random_contrasts={"other": "sum"},
        )
