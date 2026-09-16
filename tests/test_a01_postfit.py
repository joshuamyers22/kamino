# pyright: reportArgumentType=false, reportCallIssue=false
# pyright: reportMissingTypeStubs=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from kamino import (
    LinearFunctionBasis,
    PostfitError,
    adapt_kamino,
    adapt_statsmodels_mixedlm,
    adapt_statsmodels_ols,
    lmer,
)
from kamino.errors import ModelSpecificationError

FIXTURES = Path(__file__).parents[1] / "oracle" / "fixtures" / "v1"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _factorial_frame() -> pd.DataFrame:
    cells = (("a", "u", 8), ("a", "v", 4), ("b", "u", 3), ("b", "v", 9))
    rows: list[dict[str, object]] = []
    group_effects = (-0.7, 0.4, 0.9, -0.2, 0.3, -0.5)
    residual_pattern = (-0.2, 0.1, 0.15, -0.05)
    cursor = 0
    for f_value, h_value, count in cells:
        for cell_index in range(count):
            x = -1.5 + 3.0 * ((cursor % 7) / 6.0)
            group_index = cursor % len(group_effects)
            mean = (
                10.0
                + (1.7 if f_value == "b" else 0.0)
                + (-0.8 if h_value == "v" else 0.0)
                + (1.1 if f_value == "b" and h_value == "v" else 0.0)
                + 0.6 * x
                + group_effects[group_index]
            )
            rows.append(
                {
                    "y": mean + residual_pattern[cell_index % 4],
                    "x": x,
                    "f": f_value,
                    "h": h_value,
                    "g": f"g{group_index + 1}",
                    "prior": 0.75 + 0.25 * (cursor % 3),
                    "o": 0.1 * (cursor % 5),
                }
            )
            cursor += 1
    frame = pd.DataFrame(rows)
    frame["f"] = pd.Categorical(frame["f"], categories=["a", "b"])
    frame["h"] = pd.Categorical(frame["h"], categories=["u", "v"])
    frame["g"] = pd.Categorical(
        frame["g"], categories=[f"g{index + 1}" for index in range(6)]
    )
    return frame


def _dyestuff_frame() -> pd.DataFrame:
    source = _fixture("dyestuff.json")["data"]
    return pd.DataFrame({"Yield": source["response"], "Batch": source["groups"]})


def test_pinned_emmeans_oracle_for_weights_labels_and_pairwise_tests() -> None:
    oracle = _fixture("a01_postfit.json")
    frame = _factorial_frame()
    fit = lmer(oracle["formula"], frame, reml=False)
    analysis = fit.postfit(data=frame)

    assert list(fit.fixed_names) == oracle["fixed_names"]
    np.testing.assert_allclose(fit.beta, oracle["beta"], atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(
        fit.beta_covariance, oracle["covariance"], atol=1e-8, rtol=0.0
    )
    for weighting, expected in oracle["marginal_means"].items():
        grid = analysis.reference_grid(specs=["f"], at={"x": [0.0]}, weights=weighting)
        assert [dict(item.levels) for item in grid.estimates] == expected["labels"]
        np.testing.assert_allclose(
            [item.linear_function for item in grid.estimates],
            expected["linear_functions"],
            atol=1e-14,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            [item.result.estimate for item in grid.estimates],
            expected["estimate"],
            atol=1e-8,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            [item.result.standard_error for item in grid.estimates],
            expected["standard_error"],
            atol=1e-8,
            rtol=0.0,
        )
        assert all(item.result.distribution == "normal" for item in grid.estimates)
        assert all(item.result.denominator_df is None for item in grid.estimates)

    cells = analysis.reference_grid(specs=["f", "h"], at={"x": [0.0]})
    for adjustment, expected in oracle["pairwise"].items():
        observed = cells.pairwise(adjustment=adjustment)
        np.testing.assert_allclose(
            sorted(float(item.adjusted_p_value) for item in observed),
            sorted(expected["p_value"]),
            atol=2e-8,
            rtol=2e-7,
        )


def test_kamino_basis_inference_pairing_and_tidy() -> None:
    frame = _dyestuff_frame()
    fit = lmer("Yield ~ 1 + (1 | Batch)", frame, reml=True)

    asymptotic = fit.postfit(data=frame)
    assert asymptotic.basis.source == "kamino"
    assert asymptotic.basis.covariance_kind == "model_based"
    assert asymptotic.basis.inference_method == "asymptotic"
    result = asymptotic.contrast([1.0], rhs=1500.0)
    assert result.available and result.distribution == "normal"
    assert result.estimate == pytest.approx(fit.beta[0])

    satterthwaite = fit.postfit(inference="satterthwaite", data=frame)
    satt = satterthwaite.tidy()[0]
    assert satt.available and satt.distribution == "t"
    assert satt.denominator_df == pytest.approx(5.0, abs=1e-5)

    kr = fit.postfit(inference="kenward_roger", data=frame)
    kr_term = kr.tidy()[0]
    assert kr.basis.covariance_kind == "kenward_roger_adjusted"
    assert kr_term.available and kr_term.distribution == "f"
    assert kr_term.denominator_df == pytest.approx(5.0, abs=1e-8)
    with pytest.raises(ModelSpecificationError, match="unsupported Kamino"):
        adapt_kamino(fit, inference="residual_t")


def test_reference_grid_weighting_matches_explicit_linear_functions() -> None:
    frame = _factorial_frame()
    fit = lmer("y ~ x + f * h + (1 | g)", frame, reml=False)
    analysis = fit.postfit(data=frame)

    expected_weights = {
        "equal": ((0.5, 0.5), (0.5, 0.5)),
        "proportional": ((11 / 24, 13 / 24), (11 / 24, 13 / 24)),
        "outer": ((11 / 24, 13 / 24), (11 / 24, 13 / 24)),
        "cells": ((8 / 12, 4 / 12), (3 / 12, 9 / 12)),
        "flat": ((0.5, 0.5), (0.5, 0.5)),
    }
    for method, expected in expected_weights.items():
        grid = analysis.reference_grid(specs=["f"], at={"x": [0.0]}, weights=method)
        assert len(grid.estimates) == 2
        for estimate, weights in zip(grid.estimates, expected, strict=True):
            np.testing.assert_allclose(estimate.grid_weights, weights, atol=1e-14)
            reduced = estimate.linear_function[list(fit.fixed_rank.retained_indices)]
            assert estimate.result.estimate == pytest.approx(
                reduced @ fit.beta + estimate.offset, abs=1e-12
            )

    user = analysis.reference_grid(
        specs=["f"], at={"x": [0.0]}, weights=[1.0, 3.0, 2.0, 2.0]
    )
    np.testing.assert_allclose(user.estimates[0].grid_weights, [0.25, 0.75])
    np.testing.assert_allclose(user.estimates[1].grid_weights, [0.5, 0.5])

    by_h = analysis.reference_grid(specs=["f"], by=["h"], at={"x": [0.0]})
    assert len(by_h.estimates) == 4
    assert len(by_h.pairwise()) == 2


def test_reference_grid_offsets_pairs_and_adjustments() -> None:
    frame = _factorial_frame()
    fit = lmer(
        "y ~ f + offset(o) + (1 | g)",
        frame,
        offset=np.repeat(0.25, len(frame)),
        reml=False,
    )
    analysis = fit.postfit(data=frame)
    with pytest.raises(ModelSpecificationError, match="explicit reference-grid offset"):
        analysis.reference_grid(specs=["f"], at={"o": [2.0]})
    grid = analysis.reference_grid(
        specs=["f"], at={"o": [2.0]}, offset=0.25, weights="equal"
    )
    assert all(estimate.offset == pytest.approx(2.25) for estimate in grid.estimates)
    unadjusted = grid.pairwise(adjustment="none")
    bonferroni = grid.pairwise(adjustment="bonferroni")
    sidak = grid.pairwise(adjustment="sidak")
    holm = grid.pairwise(adjustment="holm")
    assert len(unadjusted) == 1
    assert bonferroni[0].adjusted_p_value == unadjusted[0].p_value
    assert sidak[0].adjusted_p_value == pytest.approx(unadjusted[0].p_value)
    assert holm[0].adjusted_p_value == unadjusted[0].p_value
    with pytest.raises(ModelSpecificationError, match="adjustment"):
        grid.pairwise(adjustment="tukey")


def test_rank_deficient_postfit_preserves_full_labels_and_estimability() -> None:
    fixture = _fixture("f02_rank_categorical.json")
    source = fixture["data"]
    case = fixture["rank_cases"]["exact_alias"]
    frame = pd.DataFrame(
        {
            "y": source["response"],
            "g": source["groups"],
            "x": case["x"],
            "duplicate": case["duplicate"],
        }
    )
    fit = lmer(case["formula"], frame, reml=False)
    analysis = fit.postfit(data=frame)
    assert analysis.basis.full_coefficient_names == (
        "(Intercept)",
        "x",
        "duplicate",
    )
    tidy = analysis.tidy()
    assert tidy[0].available
    assert not tidy[1].available and not tidy[1].estimable
    combined = analysis.contrast([0.0, 1.0, 2.0])
    assert combined.available


def test_kamino_performance_uses_rowwise_random_slope_variance() -> None:
    source = _fixture("sleepstudy.json")["data"]
    frame = pd.DataFrame(
        {
            "Reaction": source["response"],
            "Days": source["predictor"],
            "Subject": source["groups"],
        }
    )
    fit = lmer("Reaction ~ Days + (1 + Days | Subject)", frame, reml=False)
    performance = fit.postfit(data=frame).performance()
    z = np.column_stack((np.ones(len(frame)), frame["Days"].to_numpy()))
    expected_random = float(
        np.mean(np.einsum("ij,jk,ik->i", z, fit.random_covariance, z))
    )
    assert performance.averaging_measure == "equal over retained training rows"
    assert performance.get("average_random_variance") == pytest.approx(
        expected_random, abs=1e-10
    )
    assert (
        0.0
        <= performance.get("marginal_r2")
        <= performance.get("conditional_r2")
        <= 1.0
    )
    with pytest.raises(ModelSpecificationError, match="unavailable"):
        performance.get("aicc")


def test_statsmodels_ols_wls_and_robust_covariance_contracts() -> None:
    frame = _factorial_frame()
    ols_result = smf.ols("y ~ x + f * h", frame).fit()
    ols = adapt_statsmodels_ols(ols_result)
    assert ols.basis.source == "statsmodels_ols"
    assert ols.basis.inference_method == "residual_t"
    np.testing.assert_allclose(ols.basis.coefficients, ols_result.params)
    np.testing.assert_allclose(ols.basis.covariance, ols_result.cov_params())
    slope = ols.contrast([0.0, 0.0, 0.0, 1.0, 0.0])
    direct = ols_result.t_test([0.0, 0.0, 0.0, 1.0, 0.0])
    assert slope.statistic == pytest.approx(
        float(np.asarray(direct.tvalue).item()), abs=1e-12
    )
    assert slope.p_value == pytest.approx(
        float(np.asarray(direct.pvalue).item()), abs=1e-12
    )

    wls_result = smf.wls("y ~ x + f * h", frame, weights=frame["prior"]).fit()
    wls = adapt_statsmodels_ols(wls_result)
    assert wls.basis.source == "statsmodels_wls"
    assert wls.basis.inference_method == "residual_t"

    robust_result = ols_result.get_robustcov_results(cov_type="HC1", use_t=True)
    robust = adapt_statsmodels_ols(robust_result)
    assert robust.basis.covariance_kind == "statsmodels_hc1"
    assert robust.basis.inference_method == "asymptotic"
    assert robust.contrast([0.0, 0.0, 0.0, 1.0, 0.0]).distribution == "normal"
    assert robust.performance().get("rsquared") == pytest.approx(ols_result.rsquared)


def test_statsmodels_reference_grid_and_pairwise_match_direct_predictions() -> None:
    frame = _factorial_frame()
    result = smf.ols("y ~ x + f * h", frame).fit()
    analysis = adapt_statsmodels_ols(result)
    grid = analysis.reference_grid(specs=["f"], at={"x": [0.0]}, weights="cells")
    expected = []
    for f_value, h_weights in (("a", (8 / 12, 4 / 12)), ("b", (3 / 12, 9 / 12))):
        new = pd.DataFrame(
            {
                "x": [0.0, 0.0],
                "f": pd.Categorical([f_value, f_value], categories=["a", "b"]),
                "h": pd.Categorical(["u", "v"], categories=["u", "v"]),
            }
        )
        expected.append(float(np.asarray(result.predict(new)) @ h_weights))
    np.testing.assert_allclose(
        [estimate.result.estimate for estimate in grid.estimates], expected, atol=1e-12
    )
    pair = grid.pairwise(adjustment="holm")[0]
    assert pair.estimate == pytest.approx(expected[0] - expected[1], abs=1e-12)
    bad = frame.copy()
    bad["f"] = bad["f"].cat.add_categories(["unknown"])
    with pytest.raises(ModelSpecificationError, match="cannot be encoded"):
        analysis.reference_grid(
            specs=["f"], at={"f": ["unknown"], "x": [0.0]}, data=bad
        )


def test_statsmodels_mixedlm_adapter_preserves_fixed_block_identity() -> None:
    frame = _factorial_frame()
    result = smf.mixedlm("y ~ x + f", frame, groups=frame["g"]).fit(
        reml=False, method="lbfgs"
    )
    analysis = adapt_statsmodels_mixedlm(result)
    assert analysis.basis.source == "statsmodels_mixedlm"
    assert analysis.basis.inference_method == "asymptotic"
    np.testing.assert_allclose(analysis.basis.coefficients, result.fe_params)
    np.testing.assert_allclose(
        analysis.basis.covariance,
        np.asarray(result.cov_params())[: result.model.k_fe, : result.model.k_fe],
    )
    grid = analysis.reference_grid(specs=["f"], at={"x": [0.0]})
    expected = result.predict(
        pd.DataFrame(
            {
                "x": [0.0, 0.0],
                "f": pd.Categorical(["a", "b"], categories=["a", "b"]),
            }
        )
    )
    np.testing.assert_allclose(
        [estimate.result.estimate for estimate in grid.estimates], expected, atol=1e-10
    )
    assert np.isfinite(analysis.performance().get("log_likelihood"))


def test_external_adapters_fail_closed_on_wrong_or_nonformula_results() -> None:
    frame = _factorial_frame()
    with pytest.raises(ModelSpecificationError, match="OLS or WLS"):
        adapt_statsmodels_ols(object())
    with pytest.raises(ModelSpecificationError, match="MixedLM"):
        adapt_statsmodels_mixedlm(object())
    raw = sm.OLS(frame["y"].to_numpy(), sm.add_constant(frame["x"].to_numpy())).fit()
    with pytest.raises(ModelSpecificationError, match="formula fit"):
        adapt_statsmodels_ols(raw)


def test_statsmodels_rank_deficiency_is_not_silently_treated_as_estimable() -> None:
    frame = _factorial_frame().assign(duplicate=lambda value: 2.0 * value["x"])
    result = smf.ols("y ~ x + duplicate", frame).fit()
    analysis = adapt_statsmodels_ols(result)
    assert analysis.basis.null_basis.shape == (3, 1)
    assert not analysis.contrast([0.0, 1.0, 0.0]).available
    assert analysis.contrast([0.0, 1.0, 2.0]).available


def test_basis_and_request_validation_fail_closed() -> None:
    with pytest.raises(PostfitError, match="residual degrees"):
        LinearFunctionBasis(
            contract_version="1.0.0",
            source="statsmodels_ols",
            full_coefficient_names=("x",),
            retained_indices=(0,),
            coefficients=np.asarray([1.0]),
            covariance=np.asarray([[1.0]]),
            null_basis=np.empty((1, 0)),
            covariance_kind="statsmodels_hc1",
            inference_method="residual_t",
            denominator_df=10.0,
        )
    frame = _factorial_frame()
    analysis = lmer("y ~ x + f + (1 | g)", frame).postfit(data=frame)
    with pytest.raises(ModelSpecificationError, match="specs"):
        analysis.reference_grid(specs="f")
    with pytest.raises(ModelSpecificationError, match="user weights"):
        analysis.reference_grid(specs=["f"], weights=[1.0])
    with pytest.raises(ModelSpecificationError, match="confidence level"):
        analysis.contrast([1.0, 0.0, 0.0], level=1.0)


def test_basis_identity_validation_and_immutable_full_coordinates() -> None:
    valid: dict[str, Any] = {
        "contract_version": "1.0.0",
        "source": "statsmodels_ols",
        "full_coefficient_names": ("intercept", "x"),
        "retained_indices": (0, 1),
        "coefficients": np.asarray([1.0, 2.0]),
        "covariance": np.eye(2),
        "null_basis": np.empty((2, 0)),
        "covariance_kind": "model_based",
        "inference_method": "asymptotic",
        "denominator_df": None,
    }

    def basis(**changes: object) -> LinearFunctionBasis:
        return LinearFunctionBasis(**(valid | changes))

    result = basis(retained_indices=(0,), coefficients=[1.0], covariance=[[1.0]])
    assert result.coefficient_names == ("intercept",)
    np.testing.assert_equal(result.full_coefficients, [1.0, np.nan])
    assert not result.full_coefficients.flags.writeable

    invalid = (
        ({"contract_version": "0"}, "contract version"),
        ({"source": "unknown"}, "adapter source"),
        ({"inference_method": "unknown"}, "inference method"),
        ({"full_coefficient_names": ("x", "x")}, "coefficient names"),
        ({"retained_indices": (0, 0)}, "indices"),
        ({"coefficients": [1.0]}, "shapes"),
        ({"null_basis": np.empty((3, 0))}, "null basis"),
        ({"null_basis": [[np.nan], [0.0]]}, "null basis must be finite"),
        ({"null_basis": [[2.0], [0.0]]}, "orthonormal"),
        ({"coefficients": [1.0, np.nan]}, "finite"),
        ({"covariance": [[1.0, 0.5], [0.0, 1.0]]}, "symmetric"),
        ({"covariance": [[1.0, 0.0], [0.0, -1.0]]}, "semidefinite"),
        ({"covariance_kind": ""}, "kind"),
        (
            {"inference_method": "residual_t", "denominator_df": 0.0},
            "positive degrees",
        ),
        ({"denominator_df": 3.0}, "only residual-t"),
    )
    for changes, message in invalid:
        with pytest.raises(PostfitError, match=message):
            basis(**changes)


def test_joint_wald_contracts_and_reference_grid_validation() -> None:
    frame = _factorial_frame()
    fitted = smf.ols("y ~ x + f * h", frame).fit()
    analysis = adapt_statsmodels_ols(fitted)
    joint_rows = np.asarray([[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]])
    joint = analysis.contrast(joint_rows)
    assert joint.available and joint.distribution == "f"
    assert joint.numerator_df == 2
    assert joint.denominator_df == pytest.approx(fitted.df_resid)

    robust = adapt_statsmodels_ols(fitted.get_robustcov_results(cov_type="HC1"))
    robust_joint = robust.contrast(joint_rows)
    assert robust_joint.available and robust_joint.distribution == "chi_square"
    assert robust_joint.denominator_df is None
    zero = analysis.contrast(np.zeros(5))
    assert not zero.available and zero.status == "invalid_variance"
    inconsistent = analysis.contrast(
        np.asarray([[0.0, 1.0, 0.0, 0.0, 0.0]] * 2), rhs=[0.0, 1.0]
    )
    assert (
        not inconsistent.available and inconsistent.status == "inconsistent_hypothesis"
    )

    kamino_fit = lmer("y ~ x + f + (1 | g)", frame)
    missing_data = kamino_fit.postfit()
    with pytest.raises(ModelSpecificationError, match="original model-frame"):
        missing_data.reference_grid(specs=["f"])
    grid_analysis = kamino_fit.postfit(data=frame)
    requests = (
        ({"specs": ["f"], "by": ["f"]}, "disjoint"),
        ({"specs": ["unknown"]}, "not a fixed predictor"),
        ({"specs": ["f"], "at": {"unknown": [1]}}, "at variable"),
        ({"specs": ["f"], "at": {"x": []}}, "nonempty"),
        ({"specs": ["f"], "weights": "tukey"}, "weight method"),
        ({"specs": ["f"], "weights": [-1.0, 1.0]}, "finite and nonnegative"),
        ({"specs": ["f"], "weights": [0.0, 0.0]}, "positive denominator"),
        ({"specs": ["f"], "offset": [0.0]}, "one value per grid row"),
    )
    for request, message in requests:
        with pytest.raises(ModelSpecificationError, match=message):
            grid_analysis.reference_grid(**request)
