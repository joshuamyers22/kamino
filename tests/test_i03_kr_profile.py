# pyright: reportArgumentType=false, reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from kamino import (
    KenwardRogerControl,
    ObjectiveKind,
    ProfileControl,
    ProfilePoint,
    ProfileTrace,
    kenward_roger,
    likelihood_profile,
    lmer,
)
from kamino.errors import ModelSpecificationError, ResourceLimitError

FIXTURES = Path(__file__).parents[1] / "oracle" / "fixtures" / "v1"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _frame(dataset: str) -> pd.DataFrame:
    if dataset == "Dyestuff":
        data = _fixture("dyestuff.json")["data"]
        return pd.DataFrame({"Yield": data["response"], "Batch": data["groups"]})
    data = _fixture("sleepstudy.json")["data"]
    return pd.DataFrame(
        {
            "Reaction": data["response"],
            "Days": data["predictor"],
            "Subject": data["groups"],
        }
    )


def _fit(case: dict[str, Any]):
    formula = (
        "Yield ~ 1 + (1 | Batch)"
        if case["dataset"] == "Dyestuff"
        else "Reaction ~ Days + (1 + Days | Subject)"
    )
    return lmer(formula, _frame(case["dataset"]), reml=case["input_kind"] == "reml")


@pytest.mark.parametrize(
    "case",
    _fixture("i03_kr_profile.json")["kenward_roger"],
    ids=lambda case: case["id"],
)
def test_kenward_roger_intermediates_and_tests_match_pinned_pbkrtest(
    case: dict[str, Any],
) -> None:
    analysis = _fit(case).kenward_roger()

    assert analysis.available
    assert analysis.status == "ok"
    assert analysis.analysis_kind is ObjectiveKind.REML
    assert analysis.reml_refit is (case["input_kind"] == "ml")
    assert analysis.reml_objective == pytest.approx(case["reml_objective"], abs=1e-6)
    assert analysis.reml_theta is not None
    np.testing.assert_allclose(analysis.reml_theta, case["reml_theta"], atol=5e-5)
    assert analysis.reml_sigma == pytest.approx(case["reml_sigma"], abs=2e-4)
    assert analysis.covariance is not None
    assert analysis.adjusted_covariance is not None
    assert analysis.covariance_parameter_information is not None
    assert analysis.covariance_parameter_covariance is not None
    np.testing.assert_allclose(
        analysis.covariance, case["covariance"], atol=1e-3, rtol=1e-4
    )
    np.testing.assert_allclose(
        analysis.adjusted_covariance,
        case["adjusted_covariance"],
        atol=1e-3,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        analysis.covariance_parameter_information,
        case["covariance_parameter_information"],
        atol=5e-8,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        analysis.covariance_parameter_covariance,
        case["covariance_parameter_covariance"],
        atol=2.0,
        rtol=3e-5,
    )
    assert len(analysis.derivative_matrices) == len(case["derivative_matrices"])
    for actual, expected in zip(
        analysis.derivative_matrices, case["derivative_matrices"], strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=5e-8, rtol=2e-5)

    for key in ("one_df", "joint"):
        expected = case[key]
        result = analysis.test(expected["contrast"], rhs=expected["rhs"])
        assert result.available and result.estimable
        assert result.numerator_df == expected["numerator_df"]
        assert result.denominator_df == pytest.approx(
            expected["denominator_df"], abs=5e-8
        )
        assert result.scaling == pytest.approx(expected["scaling"], abs=5e-8)
        assert result.statistic == pytest.approx(
            expected["statistic"], rel=2e-5, abs=1e-4
        )
        assert result.p_value == pytest.approx(expected["p_value"], rel=2e-5)
        assert result.unscaled_statistic == pytest.approx(
            expected["unscaled_statistic"], rel=2e-5, abs=1e-4
        )
        assert result.auxiliary is not None
        np.testing.assert_allclose(result.auxiliary, expected["auxiliary"], atol=5e-8)


def test_kenward_roger_redundancy_estimability_and_fail_closed_gates() -> None:
    sleep = lmer(
        "Reaction ~ Days + (1 + Days | Subject)", _frame("sleepstudy"), reml=True
    )
    analysis = sleep.kenward_roger()
    one = analysis.test([0.0, 1.0])
    duplicate = analysis.test([[0.0, 1.0], [0.0, 2.0]])
    assert one.available and duplicate.available
    assert duplicate.numerator_df == 1
    assert duplicate.statistic == pytest.approx(one.statistic, rel=1e-10)
    inconsistent = analysis.test([[0.0, 1.0], [0.0, 2.0]], rhs=[0.0, 1.0])
    assert not inconsistent.available
    assert inconsistent.status == "inconsistent_hypothesis"

    boundary = lmer(
        "Yield ~ 1 + (1 | Batch)",
        pd.DataFrame(
            {
                "Yield": _fixture("dyestuff2.json")["data"]["response"],
                "Batch": _fixture("dyestuff2.json")["data"]["groups"],
            }
        ),
    ).kenward_roger()
    assert not boundary.available and boundary.status == "boundary"

    weighted_frame = _frame("Dyestuff")
    weighted = lmer(
        "Yield ~ 1 + (1 | Batch)",
        weighted_frame,
        weights=np.linspace(0.5, 1.5, len(weighted_frame)),
    ).kenward_roger()
    assert not weighted.available and weighted.status == "unsupported_weights"

    ill = lmer("Yield ~ 1 + (1 | Batch)", _frame("Dyestuff")).kenward_roger(
        control=KenwardRogerControl(maximum_condition_number=1.0)
    )
    assert not ill.available and ill.status == "ill_conditioned"
    with pytest.raises(ResourceLimitError, match="observation count"):
        sleep.kenward_roger(control=KenwardRogerControl(maximum_observations=10))


def test_kenward_roger_general_sparse_and_immutable_state() -> None:
    fixture = _fixture("pastes_sparse.json")
    source = fixture["data"]
    model = lmer(
        fixture["formula"],
        pd.DataFrame(
            {
                "strength": source["response"],
                "batch": source["batch"],
                "cask": source["cask"],
            }
        ),
    )
    analysis = model.kenward_roger()
    assert analysis.available
    assert analysis.test([1.0]).available
    assert analysis.adjusted_covariance is not None
    assert not analysis.adjusted_covariance.flags.writeable
    assert all(not value.flags.writeable for value in analysis.derivative_matrices)
    with pytest.raises(ValueError):
        analysis.adjusted_covariance[0, 0] = 0.0


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in _fixture("i03_kr_profile.json")["profiles"]
        if case["input_kind"] == "ml"
    ],
    ids=lambda case: case["id"],
)
def test_adaptive_profiles_and_intervals_match_pinned_lme4(
    case: dict[str, Any],
) -> None:
    profile = _fit(case).profile(targets=case["target_order"])

    assert profile.baseline_kind is ObjectiveKind.ML
    assert not profile.ml_refit
    assert profile.baseline_objective == pytest.approx(
        case["baseline_objective"], abs=1e-6
    )
    assert profile.target_order == tuple(case["target_order"])
    for trace, expected in zip(profile.traces, case["traces"], strict=True):
        assert trace.target == expected["target"]
        assert trace.status in ("ok", "boundary_truncated")
        interval = trace.interval(level=expected["interval_level"])
        assert interval.available
        tolerance = 0.1 if case["dataset"] == "Dyestuff" else 2e-3
        assert interval.lower == pytest.approx(expected["interval"][0], abs=tolerance)
        assert interval.upper == pytest.approx(expected["interval"][1], abs=tolerance)
        assert all(not point.parameters.flags.writeable for point in trace.points)


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in _fixture("i03_kr_profile.json")["profiles"]
        if case["input_kind"] == "reml"
    ],
    ids=lambda case: case["id"],
)
def test_reml_profiles_use_a_separate_ml_baseline(case: dict[str, Any]) -> None:
    target = case["target_order"][-1]
    expected_trace = case["traces"][-1]
    selected = [
        expected_trace["points"][0]["parameters"][-1],
        expected_trace["points"][-1]["parameters"][-1],
    ]
    if target.startswith("."):
        target_index = expected_trace["parameter_names"].index(target)
        selected = [
            expected_trace["points"][0]["parameters"][target_index],
            expected_trace["points"][-1]["parameters"][target_index],
        ]
    profile = _fit(case).profile(targets=[target], values={target: selected})
    assert profile.source_kind is ObjectiveKind.REML
    assert profile.baseline_kind is ObjectiveKind.ML
    assert profile.ml_refit
    assert profile.baseline_objective == pytest.approx(
        case["baseline_objective"], abs=1e-6
    )


def test_explicit_profile_points_match_lme4_nuisance_optimization() -> None:
    fixture = _fixture("i03_kr_profile.json")
    case = next(item for item in fixture["profiles"] if item["id"] == "sleepstudy_ml")
    requested: dict[str, list[float]] = {}
    expected_zeta: dict[tuple[str, float], float] = {}
    for trace in case["traces"]:
        target_index = trace["parameter_names"].index(trace["target"])
        selected = (trace["points"][0], trace["points"][-1])
        requested[trace["target"]] = []
        for point in selected:
            value = float(point["parameters"][target_index])
            requested[trace["target"]].append(value)
            expected_zeta[(trace["target"], value)] = point["signed_root_deviance"]

    profile = _fit(case).profile(targets=case["target_order"], values=requested)
    for trace in profile.traces:
        for point in trace.points:
            if point.target_value == pytest.approx(trace.estimate, abs=1e-12):
                continue
            expected = expected_zeta[(trace.target, point.target_value)]
            assert point.signed_root_deviance == pytest.approx(expected, abs=3e-3)


def test_profile_varcov_controls_validation_and_limits() -> None:
    model = lmer("Yield ~ 1 + (1 | Batch)", _frame("Dyestuff"), reml=True)
    profile = model.profile(
        targets=[".sig01", ".sigma"],
        scale="varcov",
        values={".sig01": [1000.0, 2000.0], ".sigma": [2000.0, 3000.0]},
    )
    assert profile.ml_refit
    assert profile.scale == "varcov"
    assert profile.trace(".sig01").kind == "random_variance"
    assert profile.trace(".sigma").kind == "residual_variance"
    with pytest.raises(ModelSpecificationError, match="unknown or non-estimable"):
        model.profile(targets=["missing"])
    with pytest.raises(ModelSpecificationError, match="outside its bounds"):
        model.profile(targets=[".sig01"], values={".sig01": [-1.0]})
    with pytest.raises(ResourceLimitError, match="target count"):
        model.profile(control=ProfileControl(maximum_targets=1))
    with pytest.raises(ModelSpecificationError, match="at least three"):
        ProfileControl(maximum_points_per_target=2)


@pytest.mark.parametrize(
    "control",
    [
        lambda: KenwardRogerControl(maximum_observations=0),
        lambda: KenwardRogerControl(maximum_covariance_parameters=True),
        lambda: KenwardRogerControl(information_tolerance=0.0),
        lambda: KenwardRogerControl(maximum_condition_number=np.inf),
        lambda: ProfileControl(alpha_maximum=1.0),
        lambda: ProfileControl(delta=0.0),
        lambda: ProfileControl(maximum_targets=True),
    ],
)
def test_i03_control_validation(control: Any) -> None:
    with pytest.raises(ModelSpecificationError):
        control()


def test_i03_public_entrypoint_and_hypothesis_validation() -> None:
    model = lmer("Yield ~ 1 + (1 | Batch)", _frame("Dyestuff"), reml=True)
    with pytest.raises(ModelSpecificationError, match="fitted model"):
        kenward_roger(object())
    with pytest.raises(ModelSpecificationError, match="KenwardRogerControl"):
        model.kenward_roger(control=object())
    with pytest.raises(ModelSpecificationError, match="fitted model"):
        likelihood_profile(object())
    with pytest.raises(ModelSpecificationError, match="'sdcor' or 'varcov'"):
        model.profile(scale="bad")
    with pytest.raises(ModelSpecificationError, match="ProfileControl"):
        model.profile(control=object())

    analysis = model.kenward_roger()
    for contrast in ("bad", [1.0, 2.0], [np.nan]):
        with pytest.raises(ModelSpecificationError, match="contrast"):
            analysis.test(contrast)
    for rhs in ("bad", [0.0, 1.0], [np.nan]):
        with pytest.raises(ModelSpecificationError, match="rhs"):
            analysis.test([1.0], rhs=rhs)
    zero = analysis.test([0.0])
    assert not zero.available and zero.status == "inconsistent_hypothesis"


def test_kenward_roger_rank_and_resource_failure_contracts() -> None:
    rank_fixture = _fixture("f02_rank_categorical.json")
    source = rank_fixture["data"]
    case = rank_fixture["rank_cases"]["exact_alias"]
    frame = pd.DataFrame(
        {
            "y": source["response"],
            "g": source["groups"],
            "x": case["x"],
            "duplicate": case["duplicate"],
        }
    )
    rank_analysis = lmer(case["formula"], frame, reml=True).kenward_roger()
    nonestimable = rank_analysis.test([0.0, 1.0, 0.0])
    assert not nonestimable.available
    assert not nonestimable.estimable
    assert nonestimable.status == "non_estimable"

    model = lmer("Yield ~ 1 + (1 | Batch)", _frame("Dyestuff"), reml=True)
    singular = model.kenward_roger(
        control=KenwardRogerControl(information_tolerance=1e100)
    )
    assert not singular.available and singular.status == "singular_information"
    with pytest.raises(ResourceLimitError, match="parameter count"):
        model.kenward_roger(
            control=KenwardRogerControl(maximum_covariance_parameters=1)
        )
    with pytest.raises(ResourceLimitError, match="byte limit"):
        model.kenward_roger(control=KenwardRogerControl(maximum_dense_bytes=1))

    boundary_frame = pd.DataFrame(
        {
            "Yield": _fixture("dyestuff2.json")["data"]["response"],
            "Batch": _fixture("dyestuff2.json")["data"]["groups"],
        }
    )
    boundary = lmer("Yield ~ 1 + (1 | Batch)", boundary_frame).kenward_roger()
    test = boundary.test([1.0])
    assert not test.available and test.status == "boundary"


def test_profile_request_validation_and_endpoint_statuses() -> None:
    model = lmer("Yield ~ 1 + (1 | Batch)", _frame("Dyestuff"), reml=False)
    with pytest.raises(ModelSpecificationError, match="nonempty and unique"):
        model.profile(targets=[])
    with pytest.raises(ModelSpecificationError, match="nonempty and unique"):
        model.profile(targets=[".sigma", ".sigma"])
    with pytest.raises(ModelSpecificationError, match="selected targets"):
        model.profile(targets=[".sigma"], values={".sig01": [1.0]})
    with pytest.raises(ResourceLimitError, match="total evaluation"):
        model.profile(
            targets=[".sigma"],
            control=ProfileControl(
                maximum_points_per_target=3,
                maximum_optimizer_evaluations=10,
                maximum_total_evaluations=29,
            ),
        )
    short = model.profile(
        targets=["(Intercept)"],
        control=ProfileControl(
            alpha_maximum=1e-12,
            maximum_points_per_target=3,
        ),
    )
    assert short.trace("(Intercept)").status == "unbounded"

    def point(value: float, zeta: float) -> ProfilePoint:
        return ProfilePoint(
            target_value=value,
            objective=zeta * zeta,
            deviance_difference=zeta * zeta,
            signed_root_deviance=zeta,
            parameter_names=("x",),
            parameters=np.asarray([value]),
            evaluations=0,
            converged=True,
            message="synthetic",
        )

    complete = ProfileTrace(
        target="x",
        kind="fixed_effect",
        estimate=1.0,
        lower_bound=-np.inf,
        upper_bound=np.inf,
        status="ok",
        message="synthetic",
        points=(point(0.0, -3.0), point(1.0, 0.0), point(2.0, 3.0)),
    )
    interval = complete.interval()
    assert interval.available
    assert interval.lower_status == interval.upper_status == "ok"
    with pytest.raises(ModelSpecificationError, match="strictly between"):
        complete.interval(level=1.0)

    boundary = ProfileTrace(
        target="x",
        kind="random_standard_deviation",
        estimate=1.0,
        lower_bound=0.0,
        upper_bound=np.inf,
        status="boundary_truncated",
        message="synthetic",
        points=(point(0.0, -1.0), point(1.0, 0.0), point(2.0, 3.0)),
    ).interval()
    assert boundary.available
    assert boundary.lower == 0.0 and boundary.lower_status == "boundary"

    unavailable = ProfileTrace(
        target="x",
        kind="fixed_effect",
        estimate=1.0,
        lower_bound=-np.inf,
        upper_bound=np.inf,
        status="unbounded",
        message="synthetic",
        points=(point(1.0, 0.0), point(2.0, 1.0)),
    ).interval()
    assert not unavailable.available
    assert unavailable.lower_status == "unavailable"
    assert unavailable.upper_status == "unbounded"

    profile = model.profile(targets=[".sigma"], values={".sigma": [model.sigma]})
    with pytest.raises(ModelSpecificationError, match="was not computed"):
        profile.trace("missing")
