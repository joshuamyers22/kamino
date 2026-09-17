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
from kamino.formula import GeneralDesign, build_general_design
from kamino.model import ModelSpec
from kamino.pls import evaluate_fixed_theta
from kamino.sparse import evaluate_prepared_general_sparse, prepare_general_sparse

FIXTURES = Path(__file__).parents[1] / "oracle" / "fixtures" / "v1"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _case(name: str) -> tuple[dict[str, Any], str, dict[str, object]]:
    fixture = _fixture(name)
    response_name = fixture["response_name"]
    data: dict[str, object] = {response_name: fixture["data"]["response"]}
    for group_name in fixture["group_columns"]:
        data[group_name] = fixture["data"][group_name]
    return fixture, response_name, data


def _dense_z(design: GeneralDesign) -> np.ndarray[Any, np.dtype[np.float64]]:
    z = np.zeros((design.spec.n, design.spec.q), dtype=np.float64)
    rows = np.arange(design.spec.n)
    offset = 0
    for term in design.spec.terms:
        for coefficient in range(term.k):
            columns = offset + term.group_indices * term.k + coefficient
            z[rows, columns] = term.random_design[:, coefficient]
        offset += term.q
    return z


def _matrix_sha256(matrix: np.ndarray[Any, Any]) -> str:
    canonical = np.array(matrix, dtype="<f8", order="F", copy=True)
    canonical[canonical == 0.0] = 0.0
    return hashlib.sha256(canonical.tobytes(order="F")).hexdigest()


@pytest.mark.parametrize(
    "fixture_name", ["pastes_sparse.json", "penicillin_sparse.json"]
)
def test_general_formula_preserves_lme4_term_order_and_sparse_z(
    fixture_name: str,
) -> None:
    fixture, _, data = _case(fixture_name)
    design = build_general_design(fixture["formula"], data)
    expected_terms = fixture["data"]["random_terms"]

    assert tuple(term.group_name for term in design.random_terms) == tuple(
        term["grouping"] for term in expected_terms
    )
    assert tuple(term.group_levels for term in design.random_terms) == tuple(
        tuple(term["levels"]) for term in expected_terms
    )
    assert _matrix_sha256(_dense_z(design)) == fixture["data"]["Z_sha256"]


@pytest.mark.parametrize(
    "fixture_name", ["pastes_sparse.json", "penicillin_sparse.json"]
)
def test_general_sparse_fixed_theta_matches_lme4_and_dense_pls(
    fixture_name: str,
) -> None:
    fixture, _, data = _case(fixture_name)
    design = build_general_design(fixture["formula"], data)
    workspace = prepare_general_sparse(design.spec)
    z = _dense_z(design)
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
    repeats = [term.group_count for term in design.spec.terms]

    for case in fixture["fixed_theta_cases"]:
        kind = ObjectiveKind(case["kind"])
        theta = np.asarray(case["theta"], dtype=np.float64)
        actual = evaluate_prepared_general_sparse(workspace, theta, kind=kind)
        dense = evaluate_fixed_theta(
            dense_spec,
            np.diag(np.repeat(theta, repeats)),
            kind=kind,
        )
        expected = case["components"]
        assert actual.objective == pytest.approx(case["objective"], abs=1e-10)
        assert actual.objective == pytest.approx(dense.objective, abs=1e-10)
        assert actual.logdet_c == pytest.approx(expected["ldL2"], abs=1e-10)
        assert actual.logdet_s == pytest.approx(expected["ldRX2"], abs=1e-10)
        assert actual.weighted_residual_sum_squares == pytest.approx(
            expected["wrss"], abs=1e-10
        )
        assert actual.penalized_residual_sum_squares == pytest.approx(
            expected["pwrss"], abs=1e-10
        )
        np.testing.assert_allclose(
            actual.beta, np.atleast_1d(expected["beta"]), atol=1e-10, rtol=0.0
        )
        np.testing.assert_allclose(actual.u, expected["u"], atol=1e-10, rtol=0.0)
        np.testing.assert_allclose(actual.b, expected["b"], atol=1e-10, rtol=0.0)


@pytest.mark.parametrize(
    "fixture_name", ["pastes_sparse.json", "penicillin_sparse.json"]
)
@pytest.mark.parametrize("reml", [False, True], ids=["ml", "reml"])
def test_general_sparse_final_fit_and_predictions_match_lme4(
    fixture_name: str, reml: bool
) -> None:
    fixture, _, data = _case(fixture_name)
    expected = next(fit for fit in fixture["fits"] if (fit["kind"] == "reml") == reml)
    result = lmer(fixture["formula"], data, reml=reml)

    assert result.diagnostics.backend == "scipy-superlu-symmetric-sparse"
    assert result.objective <= expected["objective"] + 1e-7
    assert result.objective == pytest.approx(expected["objective"], abs=1e-7)
    assert result.log_likelihood == pytest.approx(expected["log_likelihood"], abs=5e-8)
    np.testing.assert_allclose(result.theta, expected["theta"], atol=2e-4, rtol=0.0)
    np.testing.assert_allclose(
        result.beta, np.atleast_1d(expected["beta"]), atol=1e-8, rtol=0.0
    )
    assert result.sigma == pytest.approx(expected["sigma"], abs=2e-6)
    expected_covariance = np.diag(
        [covariance[0][0] for covariance in expected["random_covariances"]]
    )
    np.testing.assert_allclose(
        result.random_covariance, expected_covariance, atol=5e-4, rtol=0.0
    )
    np.testing.assert_allclose(result.u, expected["u"], atol=3e-5, rtol=0.0)
    np.testing.assert_allclose(
        result.random_effects, expected["b"], atol=3e-5, rtol=0.0
    )
    np.testing.assert_allclose(
        result.fitted_values, expected["fitted"], atol=3e-5, rtol=0.0
    )
    np.testing.assert_allclose(
        result.residuals, expected["residuals"], atol=3e-5, rtol=0.0
    )
    np.testing.assert_allclose(
        result.predict(mode="population").values,
        expected["predictions"]["training_population"],
        atol=3e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.predict(mode="conditional").values,
        expected["predictions"]["training_conditional"],
        atol=3e-5,
        rtol=0.0,
    )
    for name in ("existing", "unseen"):
        prediction_data = {
            key: np.atleast_1d(value) for key, value in fixture[name].items()
        }
        population = result.predict(
            prediction_data,
            mode="population",
            allow_new_groups=name == "unseen",
        )
        conditional = result.predict(
            prediction_data,
            mode="conditional",
            allow_new_groups=name == "unseen",
        )
        np.testing.assert_allclose(
            population.values,
            np.atleast_1d(expected["predictions"][f"{name}_population"]),
            atol=3e-5,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            conditional.values,
            np.atleast_1d(expected["predictions"][f"{name}_conditional"]),
            atol=3e-5,
            rtol=0.0,
        )
        assert conditional.new_group == (name == "unseen",)


@pytest.mark.parametrize(
    "fixture_name", ["pastes_sparse.json", "penicillin_sparse.json"]
)
def test_general_sparse_bundle_recovers_nested_and_crossed_terms(
    fixture_name: str,
    tmp_path: Path,
) -> None:
    fixture, _, data = _case(fixture_name)
    result = lmer(fixture["formula"], data)
    loaded = load_model_bundle(result.save(tmp_path / f"{fixture_name}.kamino"))
    assert tuple(term.group_name for term in loaded.random_terms) == tuple(
        term.group_name for term in result.random_terms
    )
    assert tuple(term.source_names for term in loaded.random_terms) == tuple(
        term.source_names for term in result.random_terms
    )
    for name in ("existing", "unseen"):
        prediction_data = {
            key: np.atleast_1d(value) for key, value in fixture[name].items()
        }
        for mode in ("population", "conditional"):
            expected = result.predict(
                prediction_data,
                mode=mode,
                allow_new_groups=name == "unseen",
            )
            actual = loaded.predict(
                prediction_data,
                mode=mode,
                allow_new_groups=name == "unseen",
            )
            np.testing.assert_array_equal(actual.values, expected.values)
            assert actual.new_group == expected.new_group


@pytest.mark.parametrize("reml", [False, True], ids=["ml", "reml"])
def test_insteval_crossed_scale_matches_lme4(reml: bool, tmp_path: Path) -> None:
    fixture = _fixture("insteval_sparse.json")
    source = fixture["data"]
    frame = pd.DataFrame(
        {
            "y": source["response"],
            "s": pd.Categorical(source["s"], categories=source["s_levels"]),
            "d": pd.Categorical(source["d"], categories=source["d_levels"]),
            "service": pd.Categorical(
                source["service"], categories=source["service_levels"]
            ),
            "dept": pd.Categorical(source["dept"], categories=source["dept_levels"]),
        }
    )
    expected = next(fit for fit in fixture["fits"] if (fit["kind"] == "reml") == reml)
    result = lmer(fixture["formula"], frame, reml=reml)

    assert len(frame) == source["n"] == 73_421
    assert len(result.random_effects) == source["q"] == 4_100
    assert len(result.beta) == source["p"] == 28
    assert result.objective == pytest.approx(expected["objective"], abs=1e-6)
    np.testing.assert_allclose(result.theta, expected["theta"], atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(result.beta, expected["beta"], atol=1e-6, rtol=0.0)
    assert result.sigma == pytest.approx(expected["sigma"], abs=1e-7)
    loaded = load_model_bundle(result.save(tmp_path / f"insteval-{reml}.kamino"))
    prediction_data: dict[str, list[object]] = {
        name: [source[name][0]] for name in ("s", "d", "service", "dept")
    }
    for mode in ("population", "conditional"):
        live_prediction = result.predict(prediction_data, mode=mode)
        recovered_prediction = loaded.predict(prediction_data, mode=mode)
        np.testing.assert_array_equal(
            recovered_prediction.values, live_prediction.values
        )
        assert recovered_prediction.new_group == live_prediction.new_group
