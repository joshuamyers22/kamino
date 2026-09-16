# pyright: reportArgumentType=false, reportMissingTypeStubs=false
# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from kamino import (
    SatterthwaiteAnalysis,
    SatterthwaiteControl,
    lmer,
    satterthwaite,
)
from kamino.errors import ModelSpecificationError

FIXTURES = Path(__file__).parents[1] / "oracle" / "fixtures" / "v1"
SATTERTHWAITE_MODULE = import_module("kamino.satterthwaite")


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _dyestuff(name: str = "dyestuff.json") -> pd.DataFrame:
    data = _fixture(name)["data"]
    return pd.DataFrame(
        {"Yield": data["response"], "Batch": data["groups"]},
        index=data["row_ids"],
    )


def _sleepstudy() -> pd.DataFrame:
    data = _fixture("sleepstudy.json")["data"]
    return pd.DataFrame(
        {
            "Reaction": data["response"],
            "Days": data["predictor"],
            "Subject": data["groups"],
        },
        index=data["row_ids"],
    )


def _fit_oracle_case(case: dict[str, Any]):
    if case["dataset"] == "Dyestuff":
        return lmer(
            "Yield ~ 1 + (1 | Batch)",
            _dyestuff(),
            reml=case["kind"] == "reml",
        )
    return lmer(
        "Reaction ~ Days + (1 + Days | Subject)",
        _sleepstudy(),
        reml=case["kind"] == "reml",
    )


@pytest.mark.parametrize(
    "case",
    _fixture("i02_satterthwaite.json")["cases"],
    ids=lambda case: case["id"],
)
def test_full_derivatives_and_tests_match_pinned_lmertest(
    case: dict[str, Any],
) -> None:
    model = _fit_oracle_case(case)
    analysis = model.satterthwaite()

    assert analysis.available
    assert analysis.status == "ok"
    assert analysis.hessian is not None
    assert analysis.variance_parameter_covariance is not None
    assert analysis.beta_covariance_jacobian is not None
    np.testing.assert_allclose(analysis.eta, case["eta"], atol=1.5e-4, rtol=0.0)
    np.testing.assert_allclose(
        analysis.hessian,
        case["variance_parameter_hessian"],
        atol=2e-4,
        rtol=3e-3,
    )
    np.testing.assert_allclose(
        analysis.variance_parameter_covariance,
        case["variance_parameter_covariance"],
        atol=2e-4,
        rtol=3e-3,
    )
    np.testing.assert_allclose(
        analysis.beta_covariance_jacobian,
        case["beta_covariance_jacobian"],
        atol=2e-4,
        rtol=3e-3,
    )

    expected = case["one_df"]
    actual = analysis.test(np.atleast_1d(expected["contrast"]), rhs=expected["rhs"])
    assert actual.available and actual.estimable
    assert actual.estimate == pytest.approx(expected["estimate"], abs=1e-8)
    assert actual.standard_error == pytest.approx(expected["standard_error"], abs=2e-5)
    assert actual.denominator_df == pytest.approx(expected["denominator_df"], abs=2e-3)
    assert actual.statistic == pytest.approx(expected["statistic"], abs=2e-4)
    assert actual.p_value == pytest.approx(expected["p_value"], abs=2e-7)

    expected_joint = case["joint"]
    joint = analysis.joint_test(
        np.atleast_2d(expected_joint["contrast"]), rhs=expected_joint["rhs"]
    )
    assert joint.available and joint.estimable
    assert joint.numerator_df == expected_joint["numerator_df"]
    assert joint.denominator_df == pytest.approx(
        expected_joint["denominator_df"], abs=3e-3
    )
    assert joint.statistic == pytest.approx(expected_joint["statistic"], rel=1e-4)
    assert joint.p_value == pytest.approx(expected_joint["p_value"], abs=2e-7)


def test_boundary_fit_refuses_degrees_of_freedom() -> None:
    model = lmer("Yield ~ 1 + (1 | Batch)", _dyestuff("dyestuff2.json"))
    analysis = model.satterthwaite()

    assert not analysis.available
    assert analysis.status == "boundary"
    result = analysis.test([1.0])
    assert result.estimable
    assert not result.available
    assert result.denominator_df is None
    assert result.p_value is None


def test_joint_test_reduces_redundant_rows_and_rejects_inconsistent_rhs() -> None:
    model = lmer("Reaction ~ Days + (1 + Days | Subject)", _sleepstudy(), reml=True)
    analysis = model.satterthwaite()
    single = analysis.test([0.0, 1.0])
    duplicate = analysis.joint_test([[0.0, 1.0], [0.0, 2.0]])

    assert single.available and duplicate.available
    assert single.statistic is not None
    assert single.denominator_df is not None
    assert duplicate.numerator_df == 1
    assert duplicate.statistic == pytest.approx(single.statistic**2, rel=1e-10)
    assert duplicate.denominator_df == pytest.approx(single.denominator_df, rel=1e-8)

    inconsistent = analysis.joint_test([[0.0, 1.0], [0.0, 2.0]], rhs=[0.0, 1.0])
    assert not inconsistent.available
    assert inconsistent.status == "inconsistent_hypothesis"


def test_rank_deficient_hypotheses_preserve_full_coordinate_estimability() -> None:
    fixture = _fixture("f02_rank_categorical.json")
    source = fixture["data"]
    rank_case = fixture["rank_cases"]["exact_alias"]
    frame = pd.DataFrame(
        {
            "y": source["response"],
            "g": source["groups"],
            "x": rank_case["x"],
            "duplicate": rank_case["duplicate"],
        }
    )
    analysis = lmer(rank_case["formula"], frame).satterthwaite()

    assert analysis.test([0.0, 1.0, 2.0]).available
    rejected = analysis.test([0.0, 1.0, 0.0])
    assert not rejected.estimable
    assert rejected.status == "non_estimable"


def test_general_sparse_backend_and_public_function_share_the_same_contract() -> None:
    fixture = _fixture("pastes_sparse.json")
    source = fixture["data"]
    frame = pd.DataFrame(
        {
            "strength": source["response"],
            "batch": source["batch"],
            "cask": source["cask"],
        }
    )
    model = lmer(fixture["formula"], frame)
    analysis = satterthwaite(model)

    assert analysis.available
    assert analysis.test([1.0]).available
    assert analysis.parameter_names == ("theta[1]", "theta[2]", "sigma")


def test_controls_inputs_and_derivative_arrays_are_defensive() -> None:
    model = lmer("Yield ~ 1 + (1 | Batch)", _dyestuff())
    analysis = model.satterthwaite()

    assert analysis.hessian is not None
    assert not analysis.hessian.flags.writeable
    assert analysis.beta_covariance_jacobian is not None
    assert not analysis.beta_covariance_jacobian.flags.writeable
    assert analysis.finite_difference_steps is not None
    assert not analysis.finite_difference_steps.flags.writeable
    with pytest.raises(ValueError):
        analysis.hessian[0, 0] = 0.0
    with pytest.raises(ModelSpecificationError, match="full coefficient space"):
        analysis.test([1.0, 0.0])
    with pytest.raises(ModelSpecificationError, match="rhs must be finite"):
        analysis.test([1.0], rhs=True)
    with pytest.raises(ModelSpecificationError, match="positive integer"):
        SatterthwaiteControl(maximum_variance_parameters=0)
    with pytest.raises(ModelSpecificationError, match="resource limit"):
        model.satterthwaite(control=SatterthwaiteControl(maximum_variance_parameters=1))


def test_step_sensitivity_and_condition_limits_fail_closed() -> None:
    model = lmer("Yield ~ 1 + (1 | Batch)", _dyestuff())

    unstable = model.satterthwaite(
        control=SatterthwaiteControl(derivative_tolerance=1e-12)
    )
    assert not unstable.available
    assert unstable.status == "unstable_derivatives"

    ill_conditioned = model.satterthwaite(
        control=SatterthwaiteControl(maximum_condition_number=1.0)
    )
    assert not ill_conditioned.available
    assert ill_conditioned.status == "ill_conditioned"


def test_material_negative_curvature_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = lmer("Yield ~ 1 + (1 | Batch)", _dyestuff())

    def negative_hessian(*args: object) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.diag([-1.0, 1.0])

    monkeypatch.setattr(SATTERTHWAITE_MODULE, "_raw_hessian", negative_hessian)
    analysis = model.satterthwaite()
    assert not analysis.available
    assert analysis.status == "negative_curvature"


def test_reml_sigma_only_limit_recovers_residual_degrees_of_freedom() -> None:
    model = lmer("Yield ~ 1 + (1 | Batch)", _dyestuff())
    sigma = model.sigma
    residual_df = model._training_design.spec.n - model._training_design.spec.p
    covariance = model.beta_covariance
    analysis = SatterthwaiteAnalysis(
        available=True,
        status="ok",
        message="analytical sigma-only limit",
        parameter_names=("sigma",),
        eta=np.array([sigma]),
        finite_difference_steps=np.array([1e-4]),
        hessian=np.array([[4.0 * residual_df / model.sigma2]]),
        variance_parameter_covariance=np.array([[model.sigma2 / (2.0 * residual_df)]]),
        beta_covariance_jacobian=np.array([2.0 * covariance / sigma]),
        hessian_eigenvalues=np.array([4.0 * residual_df / model.sigma2]),
        condition_number=1.0,
        derivative_error=0.0,
        hypothesis_rank_tolerance=np.sqrt(np.finfo(np.float64).eps),
        _model=model,
    )

    result = analysis.test([1.0])
    assert result.available
    assert result.denominator_df == pytest.approx(residual_df, abs=1e-12)
