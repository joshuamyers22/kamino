# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import kamino.block as block_module
from kamino.block import BACKEND_NAME, evaluate_random_intercept_block
from kamino.errors import ModelSpecificationError
from kamino.formula import build_random_intercept_design
from kamino.model import ObjectiveKind
from kamino.oracle import evaluate_dense_oracle
from kamino.pls import evaluate_fixed_theta

FIXTURE_PATH = (
    Path(__file__).parents[1] / "oracle" / "fixtures" / "v1" / "dyestuff.json"
)


def dyestuff_design() -> Any:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    data = fixture["data"]
    frame = pd.DataFrame(
        {"Yield": data["response"], "Batch": data["groups"]},
        index=data["row_ids"],
    )
    return build_random_intercept_design("Yield ~ 1 + (1 | Batch)", frame)


@pytest.mark.parametrize("kind", list(ObjectiveKind))
@pytest.mark.parametrize("theta", [0.0, 0.25, 0.75, 2.0])
def test_block_matches_dense_pls_and_marginal_oracles(
    kind: ObjectiveKind, theta: float
) -> None:
    design = dyestuff_design()
    block = evaluate_random_intercept_block(
        design.spec, design.group_indices, theta, kind=kind
    )
    lambda_ = theta * np.eye(design.spec.q, dtype=np.float64)
    pls = evaluate_fixed_theta(design.spec, lambda_, kind=kind)
    dense = evaluate_dense_oracle(design.spec, lambda_, kind=kind)

    assert block.objective == pytest.approx(pls.objective, abs=1e-12)
    assert block.objective == pytest.approx(dense.objective, abs=1e-12)
    assert block.log_likelihood == pytest.approx(pls.log_likelihood, abs=1e-12)
    assert block.sigma2 == pytest.approx(pls.sigma2, abs=1e-12)
    assert block.random_variance == pytest.approx(pls.sigma2 * theta * theta, abs=1e-12)
    assert block.penalized_residual_sum_squares == pytest.approx(
        pls.penalized_residual_sum_squares, abs=1e-10
    )
    assert block.weighted_residual_sum_squares == pytest.approx(
        pls.weighted_residual_sum_squares, abs=1e-10
    )
    assert block.random_effect_penalty == pytest.approx(
        pls.random_effect_penalty, abs=1e-10
    )
    assert block.logdet_c == pytest.approx(pls.logdet_c, abs=1e-12)
    assert block.logdet_s == pytest.approx(pls.logdet_s, abs=1e-12)
    np.testing.assert_allclose(block.beta, pls.beta, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(block.u, pls.u, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(block.b, pls.b, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(
        block.beta_covariance, pls.beta_covariance, atol=1e-10, rtol=0.0
    )


def test_block_handles_unsorted_groups_weights_and_offsets() -> None:
    frame = pd.DataFrame(
        {
            "y": [1.2, 3.0, 1.0, 5.2, 2.8, 4.9, 1.1, 3.2],
            "g": ["a", "b", "a", "c", "b", "c", "a", "b"],
        }
    )
    design = build_random_intercept_design(
        "y ~ 1 + (1 | g)",
        frame,
        weights=[0.5, 2.0, 1.5, 0.75, 1.25, 2.5, 1.0, 1.75],
        offset=[0.1, -0.1, 0.0, 0.2, -0.2, 0.1, -0.05, 0.05],
    )
    theta = 0.8
    block = evaluate_random_intercept_block(
        design.spec, design.group_indices, theta, kind=ObjectiveKind.REML
    )
    pls = evaluate_fixed_theta(
        design.spec,
        theta * np.eye(design.spec.q, dtype=np.float64),
        kind=ObjectiveKind.REML,
    )
    assert block.objective == pytest.approx(pls.objective, abs=1e-12)
    np.testing.assert_allclose(block.b, pls.b, atol=1e-12, rtol=0.0)


def test_block_factorizes_only_the_fixed_effect_schur_complement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = dyestuff_design()
    shapes: list[tuple[int, ...]] = []
    original = np.linalg.cholesky

    def recording_cholesky(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        shapes.append(value.shape)
        return original(value)

    monkeypatch.setattr(block_module.np.linalg, "cholesky", recording_cholesky)
    evaluate_random_intercept_block(design.spec, design.group_indices, 0.75)
    assert shapes == [(design.spec.p, design.spec.p)]
    assert design.spec.q > design.spec.p


@pytest.mark.parametrize("theta", [-1.0, np.inf, np.nan])
def test_block_rejects_invalid_theta(theta: float) -> None:
    design = dyestuff_design()
    with pytest.raises(ModelSpecificationError, match="finite and nonnegative"):
        evaluate_random_intercept_block(design.spec, design.group_indices, theta)


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        (np.array([0], dtype=np.int64), "one value per row"),
        (np.full(30, -1, dtype=np.int64), "invalid group index"),
        (np.zeros(30, dtype=np.int64), "every random-intercept group"),
    ],
)
def test_block_rejects_invalid_group_map(
    indices: np.ndarray[Any, np.dtype[np.int64]], message: str
) -> None:
    design = dyestuff_design()
    with pytest.raises(ModelSpecificationError, match=message):
        evaluate_random_intercept_block(design.spec, indices, 0.5)


def test_public_fitter_reports_the_block_backend() -> None:
    design = dyestuff_design()
    from kamino import lmer

    result = lmer(
        "Yield ~ 1 + (1 | Batch)",
        {
            "Yield": design.spec.y,
            "Batch": list(design.training_groups),
        },
    )
    assert result.diagnostics.backend == BACKEND_NAME
