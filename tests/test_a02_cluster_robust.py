# pyright: reportArgumentType=false, reportMissingTypeStubs=false
# pyright: reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownVariableType=false

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from kamino import ClusterRobustControl, cluster_robust, lmer
from kamino.errors import ModelSpecificationError, PostfitError, ResourceLimitError

FIXTURES = Path(__file__).parents[1] / "oracle" / "fixtures" / "v1"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _synthetic_frame() -> pd.DataFrame:
    source = _fixture("a02_cluster_robust.json")["synthetic_data"]
    return pd.DataFrame(
        {
            "y": source["response"],
            "x": source["x"],
            "z": source["z"],
            "school": source["school"],
            "district": source["district"],
        },
        index=source["row_ids"],
    )


def _sleepstudy_frame() -> pd.DataFrame:
    source = _fixture("sleepstudy.json")["data"]
    return pd.DataFrame(
        {
            "Reaction": source["response"],
            "Days": source["predictor"],
            "Subject": source["groups"],
        },
        index=source["row_ids"],
    )


def _fit_case(case: dict[str, Any]):
    if case["id"].startswith("synthetic"):
        frame = _synthetic_frame()
        return (
            lmer(
                "y ~ x + z + (1 | school)",
                frame,
                reml=case["kind"] == "reml",
            ),
            frame["district"],
        )
    return (
        lmer(
            "Reaction ~ Days + (1 + Days | Subject)",
            _sleepstudy_frame(),
            reml=True,
        ),
        "Subject",
    )


@pytest.mark.parametrize(
    "case",
    _fixture("a02_cluster_robust.json")["cases"],
    ids=lambda case: case["id"],
)
def test_cr_covariance_adjustments_and_tests_match_pinned_clubsandwich(
    case: dict[str, Any],
) -> None:
    model, cluster = _fit_case(case)
    for covariance_type in ("CR0", "CR1", "CR2"):
        analysis = model.cluster_robust(cluster, covariance_type=covariance_type)
        np.testing.assert_allclose(
            analysis.covariance,
            case["covariance"][covariance_type],
            atol=2e-8,
            rtol=2e-8,
        )

    analysis = cluster_robust(model, cluster=cluster, covariance_type="CR2")
    np.testing.assert_allclose(analysis.bread, case["bread"], atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(
        analysis.marginal_residuals,
        case["marginal_residuals"],
        atol=1e-8,
        rtol=1e-9,
    )
    for actual, expected in zip(
        analysis.working_targets, case["working_targets"], strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=2e-6)
    for actual, expected in zip(analysis.adjustments, case["adjustments"], strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-8)
    for actual, expected in zip(
        analysis.estimating_matrices,
        case["estimating_matrices"],
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=2e-6)
    np.testing.assert_allclose(
        analysis.score_contributions,
        case["score_contributions"],
        atol=1e-3,
        rtol=2e-6,
    )

    expected_tests = case["coefficient_tests"]
    for index in range(len(case["beta"])):
        contrast = np.eye(len(case["beta"]))[index]
        result = analysis.test(contrast)
        assert result.available and result.estimable
        assert result.method == "CR2 Satterthwaite"
        assert result.distribution == "t"
        assert result.estimate == pytest.approx(
            expected_tests["estimate"][index], abs=1e-8
        )
        assert result.standard_error == pytest.approx(
            expected_tests["standard_error"][index], abs=2e-8
        )
        assert result.statistic == pytest.approx(
            expected_tests["statistic"][index], abs=2e-7
        )
        assert result.denominator_df == pytest.approx(
            expected_tests["denominator_df"][index], abs=2e-7
        )
        assert result.p_value == pytest.approx(
            expected_tests["p_value"][index], abs=2e-9
        )

    expected_joint = case["joint"]
    joint = analysis.test(expected_joint["contrast"])
    assert joint.available and joint.method == "CR2 HTZ"
    assert joint.distribution == "f"
    assert joint.numerator_df == expected_joint["numerator_df"]
    assert joint.statistic == pytest.approx(expected_joint["statistic"], rel=2e-7)
    assert joint.scale == pytest.approx(expected_joint["scale"], abs=2e-9)
    assert joint.denominator_df == pytest.approx(
        expected_joint["denominator_df"], abs=2e-7
    )
    assert joint.p_value == pytest.approx(expected_joint["p_value"], abs=2e-9)


def test_cr0_cr1_normalization_and_marginal_score_contract() -> None:
    frame = _synthetic_frame()
    model = lmer("y ~ x + z + (1 | school)", frame, reml=False)
    cr0 = model.cluster_robust(frame["district"], covariance_type="CR0")
    cr1 = model.cluster_robust(frame["district"], covariance_type="CR1")
    clusters = len(cr0.cluster_levels)

    np.testing.assert_allclose(
        cr0.covariance,
        cr0.bread @ (cr0.score_contributions @ cr0.score_contributions.T) @ cr0.bread,
        atol=1e-14,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        cr1.covariance,
        cr0.covariance * clusters / (clusters - 1),
        atol=1e-14,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        cr0.marginal_residuals,
        frame["y"].to_numpy() - model._training_fixed_design @ model.beta,
        atol=1e-14,
    )
    one = cr0.test([0.0, 1.0, 0.0])
    joint = cr1.test([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert one.available and one.distribution == "normal"
    assert one.denominator_df is None
    assert joint.available and joint.distribution == "chi_square"
    assert joint.denominator_df is None


def test_offsets_are_removed_but_conditional_random_effects_are_not() -> None:
    frame = _synthetic_frame()
    offset = np.linspace(-0.5, 0.5, len(frame))
    shifted = frame.assign(o=offset, shifted_y=frame["y"] + offset)
    base = lmer("y ~ x + z + (1 | school)", frame, reml=True)
    with_offset = lmer(
        "shifted_y ~ x + z + offset(o) + (1 | school)",
        shifted,
        reml=True,
    )
    base_analysis = base.cluster_robust(frame["district"])
    offset_analysis = with_offset.cluster_robust(shifted["district"])

    np.testing.assert_allclose(
        offset_analysis.marginal_residuals,
        shifted["shifted_y"].to_numpy()
        - offset
        - with_offset._training_fixed_design @ with_offset.beta,
        atol=1e-14,
    )
    assert not np.allclose(offset_analysis.marginal_residuals, with_offset.residuals)
    np.testing.assert_allclose(
        offset_analysis.covariance, base_analysis.covariance, atol=2e-12, rtol=2e-10
    )


def test_cluster_series_aligns_by_retained_row_identity_and_permutation() -> None:
    frame = _synthetic_frame()
    subset = np.ones(len(frame), dtype=np.bool_)
    subset[[2, 17, 44]] = False
    original = lmer("y ~ x + z + (1 | school)", frame, subset=subset, reml=False)
    original_robust = original.cluster_robust(frame["district"])

    shuffled = frame.sample(frac=1.0, random_state=20260916)
    shuffled_subset = shuffled.index.isin(original.row_ids)
    permuted = lmer(
        "y ~ x + z + (1 | school)",
        shuffled,
        subset=shuffled_subset,
        reml=False,
    )
    permuted_robust = permuted.cluster_robust(frame["district"])
    np.testing.assert_allclose(
        permuted_robust.covariance,
        original_robust.covariance,
        atol=2e-10,
        rtol=2e-9,
    )


def test_prior_weights_and_non_nested_crossed_clusters_fail_closed() -> None:
    frame = _synthetic_frame()
    weighted = lmer("y ~ x + z + (1 | school)", frame, weights=np.ones(len(frame)))
    with pytest.raises(PostfitError, match="prior weights"):
        weighted.cluster_robust(frame["district"])

    fixture = _fixture("penicillin_sparse.json")
    source = fixture["data"]
    crossed = lmer(
        fixture["formula"],
        {
            fixture["response_name"]: source["response"],
            "plate": source["plate"],
            "sample": source["sample"],
        },
    )
    with pytest.raises(ModelSpecificationError, match="must be nested"):
        crossed.cluster_robust("plate")


def test_nested_random_terms_use_the_outer_independent_cluster_by_default() -> None:
    fixture = _fixture("pastes_sparse.json")
    source = fixture["data"]
    model = lmer(
        fixture["formula"],
        {
            fixture["response_name"]: source["response"],
            "batch": source["batch"],
            "cask": source["cask"],
        },
    )
    robust = model.cluster_robust()
    assert robust.cluster_name == "batch"
    assert robust.cluster_sizes == (6,) * 10
    assert robust.test([1.0]).available


def test_rank_estimability_resources_and_defensive_state() -> None:
    rank = _fixture("f02_rank_categorical.json")
    source = rank["data"]
    case = rank["rank_cases"]["exact_alias"]
    frame = pd.DataFrame(
        {
            "y": source["response"],
            "g": source["groups"],
            "x": case["x"],
            "duplicate": case["duplicate"],
        }
    )
    model = lmer(case["formula"], frame)
    robust = model.cluster_robust("g")
    assert robust.test([0.0, 1.0, 2.0]).available
    rejected = robust.test([0.0, 1.0, 0.0])
    assert not rejected.available and not rejected.estimable
    assert rejected.status == "non_estimable"
    assert not robust.covariance.flags.writeable
    assert not robust.score_contributions.flags.writeable
    assert all(not matrix.flags.writeable for matrix in robust.adjustments)
    with pytest.raises(AttributeError, match="immutable"):
        robust.covariance = np.eye(2)  # type: ignore[misc]

    with pytest.raises(ResourceLimitError, match="observation-space"):
        model.cluster_robust("g", control=ClusterRobustControl(maximum_cluster_size=1))
    with pytest.raises(ModelSpecificationError, match="linearly independent"):
        robust.test([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
    with pytest.raises(ModelSpecificationError, match="one value"):
        model.cluster_robust(["only-one"])
    with pytest.raises(ModelSpecificationError, match="nonempty strings"):
        model.cluster_robust([1] * len(frame))


def test_invalid_controls_clusters_and_hypotheses_fail_before_results() -> None:
    frame = _synthetic_frame()
    model = lmer("y ~ x + z + (1 | school)", frame)
    robust = model.cluster_robust(frame["district"])

    with pytest.raises(ModelSpecificationError, match="CR0, CR1, or CR2"):
        model.cluster_robust(covariance_type="CR3")  # type: ignore[arg-type]
    with pytest.raises(ModelSpecificationError, match="ClusterRobustControl"):
        model.cluster_robust(control=object())  # type: ignore[arg-type]
    with pytest.raises(ModelSpecificationError, match="at least two"):
        model.cluster_robust(["one"] * len(frame))
    with pytest.raises(ModelSpecificationError, match="not the fitted"):
        model.cluster_robust("district")
    with pytest.raises(ModelSpecificationError, match="every retained"):
        model.cluster_robust(frame["district"].iloc[:-1])
    with pytest.raises(ResourceLimitError, match="design"):
        model.cluster_robust(
            frame["district"],
            control=ClusterRobustControl(maximum_observations=1),
        )
    with pytest.raises(ModelSpecificationError, match="full coefficient space"):
        robust.test([1.0])
    with pytest.raises(ModelSpecificationError, match="rhs must be finite"):
        robust.test([1.0, 0.0, 0.0], rhs=True)
    with pytest.raises(ModelSpecificationError, match="between zero and one"):
        robust.test([1.0, 0.0, 0.0], level=1.0)
    with pytest.raises(ModelSpecificationError, match="positive integer"):
        ClusterRobustControl(maximum_bytes=0)
    with pytest.raises(ModelSpecificationError, match="finite and positive"):
        ClusterRobustControl(maximum_condition_number=np.inf)
    with pytest.raises(ModelSpecificationError, match="live fitted result"):
        cluster_robust(object())  # type: ignore[arg-type]
