"""Phase 0 formula-to-ML/REML walking skeleton against the R fixture."""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from formula_spike import formulae_random_labels
from formulae import design_matrices
from scipy.optimize import minimize

from kamino.model import ModelSpec, ObjectiveKind
from kamino.pls import evaluate_fixed_theta

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "oracle" / "fixtures" / "v1" / "fixed_theta.json"
OPTIMIZED_FIXTURE = ROOT / "oracle" / "fixtures" / "v1" / "optimized.json"


def main() -> None:
    fixture: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    optimized_fixture: dict[str, Any] = json.loads(
        OPTIMIZED_FIXTURE.read_text(encoding="utf-8")
    )
    data = fixture["data"]
    frame = pd.DataFrame(
        {
            "y": data["y"],
            "x": [row[1] for row in data["X"]],
            "g": pd.Categorical(
                ["beta"] * 4 + ["alpha"] * 4 + ["gamma"] * 4,
                categories=["gamma", "alpha", "beta"],
            ),
        },
        index=data["row_ids"],
    )

    # The experimental adapter has already extracted offset(o) from the user's
    # formula and passes only the design-producing expression to Formulae.
    matrices: Any = design_matrices("y ~ x + (1 + x | g)", frame)
    raw_z = np.asarray(matrices.group).astype(np.float64, copy=False)
    raw_labels = formulae_random_labels(matrices.group)
    random_names = list(data["random_names"])
    z = raw_z[:, [raw_labels.index(label) for label in random_names]]
    spec = ModelSpec.from_arrays(
        y=np.asarray(matrices.response),
        x=np.asarray(matrices.common),
        z=z,
        weights=data["weights"],
        offset=data["offset"],
        row_ids=tuple(data["row_ids"]),
        fixed_names=tuple(data["fixed_names"]),
        random_names=tuple(random_names),
    )

    def lambda_from_theta(theta: np.ndarray) -> np.ndarray:
        block = np.array([[theta[0], 0.0], [theta[1], theta[2]]], dtype=np.float64)
        return np.kron(np.eye(3), block).astype(np.float64, copy=False)

    evidence: dict[str, Any] = {"schema_version": "1.0.0", "results": {}}
    for kind in ObjectiveKind:
        fixed_expected = next(
            case
            for case in fixture["cases"]
            if case["kind"] == kind.value and case["theta"] == [0.8, 0.35, 0.4]
        )
        fixed_result = evaluate_fixed_theta(
            spec, lambda_from_theta(np.array([0.8, 0.35, 0.4])), kind=kind
        )
        optimized_expected = next(
            fit for fit in optimized_fixture["fits"] if fit["kind"] == kind.value
        )

        def criterion(theta: np.ndarray, objective_kind: ObjectiveKind = kind) -> float:
            return evaluate_fixed_theta(
                spec, lambda_from_theta(theta), kind=objective_kind
            ).objective

        optimum = minimize(
            criterion,
            np.array([1.0, 0.0, 1.0]),
            method="Powell",
            bounds=[(0.0, None), (None, None), (0.0, None)],
            options={"maxiter": 100_000, "xtol": 1e-10, "ftol": 1e-10},
        )
        evidence["results"][kind.value] = {
            "fixed_theta_absolute_error": abs(
                fixed_result.objective - fixed_expected["objective"]
            ),
            "optimized_objective": float(optimum.fun),
            "lme4_optimized_objective": optimized_expected["objective"],
            "optimized_objective_absolute_error": abs(
                float(optimum.fun) - optimized_expected["objective"]
            ),
            "optimized_theta": optimum.x.tolist(),
            "lme4_theta": optimized_expected["theta"],
            "evaluations": int(optimum.nfev),
            "converged": bool(optimum.success),
        }
    evidence["all_checks_pass"] = all(
        result["converged"]
        and result["fixed_theta_absolute_error"] <= 1e-12
        and result["optimized_objective_absolute_error"] <= 1e-8
        for result in evidence["results"].values()
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not evidence["all_checks_pass"]:
        raise SystemExit("walking skeleton failed")


if __name__ == "__main__":
    main()
