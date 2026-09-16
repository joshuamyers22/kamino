"""Run the locked I01 simulation/refit assessment."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import scipy

from kamino import LinearMixedModelResult, lmer

ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "statistical" / "i01_plan.json"
DYESTUFF_PATH = ROOT / "oracle" / "fixtures" / "v1" / "dyestuff.json"
I01_ORACLE_PATH = ROOT / "oracle" / "fixtures" / "v1" / "i01_bootstrap.json"


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _model() -> LinearMixedModelResult:
    data = _load(DYESTUFF_PATH)["data"]
    frame = pd.DataFrame(
        {"Yield": data["response"], "Batch": data["groups"]},
        index=data["row_ids"],
    )
    return lmer("Yield ~ 1 + (1 | Batch)", frame, reml=False)


def _maximum_same_group_covariance_error(
    values: np.ndarray[Any, np.dtype[np.float64]], expected: float
) -> float:
    errors: list[float] = []
    for group in range(6):
        left = group * 5
        actual = float(np.cov(values[:, left], values[:, left + 1])[0, 1])
        errors.append(abs(actual - expected) / expected)
    return max(errors)


def _maximum_independent_group_covariance_z(
    values: np.ndarray[Any, np.dtype[np.float64]], variance: float
) -> float:
    standard_error = variance / np.sqrt(values.shape[0])
    scores: list[float] = []
    for left_group in range(5):
        for right_group in range(left_group + 1, 6):
            left = left_group * 5
            right = right_group * 5
            actual = float(np.cov(values[:, left], values[:, right])[0, 1])
            scores.append(abs(actual) / standard_error)
    return max(scores)


def _lme4_metrics(model: LinearMixedModelResult) -> dict[str, float | int]:
    fixture = _load(I01_ORACLE_PATH)
    maxima = {
        "objective": 0.0,
        "theta": 0.0,
        "beta": 0.0,
        "sigma": 0.0,
        "random_variance": 0.0,
    }
    for case in fixture["cases"]:
        baseline = (
            model
            if case["kind"] == "ml"
            else lmer(
                "Yield ~ 1 + (1 | Batch)",
                {
                    "Yield": _load(DYESTUFF_PATH)["data"]["response"],
                    "Batch": _load(DYESTUFF_PATH)["data"]["groups"],
                },
                reml=True,
            )
        )
        fitted = baseline.refit(case["response"])
        expected = case["fit"]
        maxima["objective"] = max(
            maxima["objective"], abs(fitted.objective - expected["objective"])
        )
        maxima["theta"] = max(
            maxima["theta"],
            float(np.max(np.abs(fitted.theta - expected["theta"]))),
        )
        maxima["beta"] = max(
            maxima["beta"],
            float(np.max(np.abs(fitted.beta - expected["beta"]))),
        )
        maxima["sigma"] = max(maxima["sigma"], abs(fitted.sigma - expected["sigma"]))
        maxima["random_variance"] = max(
            maxima["random_variance"],
            abs(fitted.random_variance - expected["random_variance"]),
        )
    return {"cases": len(fixture["cases"]), **maxima}


def assess() -> dict[str, object]:
    plan_payload = PLAN_PATH.read_bytes()
    plan = cast(dict[str, Any], json.loads(plan_payload))
    gates = plan["primary_gates"]
    replicates = plan["replicates"]
    rng = plan["rng"]
    model = _model()
    started = time.perf_counter()

    moment_count = int(replicates["moment_assessment"])
    unconditional = model.simulate(
        moment_count,
        seed=int(rng["assessment_moment_seed"]),
        mode="unconditional",
    )
    conditional = model.simulate(
        moment_count,
        seed=int(rng["assessment_moment_seed"]),
        mode="conditional",
    )
    unconditional_values = np.stack([draw.response for draw in unconditional.draws])
    conditional_values = np.stack([draw.response for draw in conditional.draws])
    population = model.predict(mode="population").values
    conditional_mean = model.predict(mode="conditional").values
    unconditional_variance = model.random_variance + model.sigma2
    unconditional_mean_z = np.max(
        np.abs(unconditional_values.mean(axis=0) - population)
        / np.sqrt(unconditional_variance / moment_count)
    )
    conditional_mean_z = np.max(
        np.abs(conditional_values.mean(axis=0) - conditional_mean)
        / np.sqrt(model.sigma2 / moment_count)
    )
    relative_variance_error = max(
        float(
            np.max(
                np.abs(
                    np.var(unconditional_values, axis=0, ddof=1)
                    - unconditional_variance
                )
                / unconditional_variance
            )
        ),
        float(
            np.max(
                np.abs(np.var(conditional_values, axis=0, ddof=1) - model.sigma2)
                / model.sigma2
            )
        ),
    )
    same_group_error = _maximum_same_group_covariance_error(
        unconditional_values, model.random_variance
    )
    independent_group_z = _maximum_independent_group_covariance_z(
        unconditional_values, unconditional_variance
    )

    refits = model.parametric_bootstrap(
        int(replicates["refit_assessment"]),
        seed=int(rng["assessment_refit_seed"]),
        workers=4,
        allow_incomplete=True,
    )
    accounting = refits.failure_accounting()
    usable = np.array([record.usable for record in refits.records], dtype=np.bool_)
    beta_values = refits.estimates[usable, 0]
    beta_standard_error = float(np.std(beta_values, ddof=1) / np.sqrt(beta_values.size))
    beta_bias_z = (
        abs(float(np.mean(beta_values)) - float(model.beta[0])) / beta_standard_error
    )

    worker_count = int(replicates["worker_equivalence"])
    serial = model.parametric_bootstrap(
        worker_count,
        seed=int(rng["assessment_worker_seed"]),
        workers=1,
    )
    parallel = model.parametric_bootstrap(
        worker_count,
        seed=int(rng["assessment_worker_seed"]),
        workers=4,
    )
    worker_response_equal = tuple(
        record.response_sha256 for record in serial.records
    ) == tuple(record.response_sha256 for record in parallel.records)
    worker_statistic_error = float(
        np.max(np.abs(serial.estimates - parallel.estimates))
    )
    lme4 = _lme4_metrics(model)

    metrics = {
        "maximum_absolute_standardized_response_mean": float(
            max(unconditional_mean_z, conditional_mean_z)
        ),
        "maximum_relative_response_variance_error": relative_variance_error,
        "maximum_relative_same_group_covariance_error": same_group_error,
        "maximum_absolute_standardized_independent_group_covariance": (
            independent_group_z
        ),
        "absolute_standardized_beta_bias": beta_bias_z,
        "observed_refit_failure_rate": accounting.observed_failure_rate,
        "one_sided_95_percent_failure_rate": accounting.upper_failure_rate,
        "refit_statuses": dict(accounting.statuses),
        "worker_response_identity": worker_response_equal,
        "worker_maximum_statistic_error": worker_statistic_error,
        "lme4": lme4,
    }
    checks = {
        "response_mean": metrics["maximum_absolute_standardized_response_mean"]
        <= gates["maximum_absolute_standardized_response_mean"],
        "response_variance": metrics["maximum_relative_response_variance_error"]
        <= gates["maximum_relative_response_variance_error"],
        "same_group_covariance": metrics["maximum_relative_same_group_covariance_error"]
        <= gates["maximum_relative_same_group_covariance_error"],
        "independent_group_covariance": metrics[
            "maximum_absolute_standardized_independent_group_covariance"
        ]
        <= gates["maximum_absolute_standardized_independent_group_covariance"],
        "beta_bias": metrics["absolute_standardized_beta_bias"]
        <= gates["maximum_absolute_standardized_beta_bias"],
        "observed_failure_rate": metrics["observed_refit_failure_rate"]
        <= gates["maximum_observed_refit_failure_rate"],
        "bounded_failure_rate": metrics["one_sided_95_percent_failure_rate"]
        <= gates["maximum_one_sided_95_percent_failure_rate"],
        "worker_identity": worker_response_equal
        and worker_statistic_error <= gates["worker_response_and_statistic_tolerance"],
        "lme4_objective": lme4["objective"]
        <= gates["lme4_objective_absolute_tolerance"],
        "lme4_parameters": max(
            lme4["theta"],
            lme4["beta"],
            lme4["sigma"],
            lme4["random_variance"],
        )
        <= gates["lme4_parameter_absolute_tolerance"],
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "schema_version": "1.0.0",
        "plan_id": plan["id"],
        "plan_sha256": hashlib.sha256(plan_payload).hexdigest(),
        "status": "pass" if all(checks.values()) else "fail",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "threadpoolctl": version("threadpoolctl"),
        },
        "replicates": replicates,
        "metrics": metrics,
        "checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
        "scope_note": plan["interval_scope"]["release_claim"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    arguments = parser.parse_args()
    report = assess()
    if report["status"] != "pass":
        raise SystemExit(json.dumps(report, indent=2, sort_keys=True))
    if arguments.verify is not None:
        expected = _load(arguments.verify)
        if expected["plan_sha256"] != report["plan_sha256"]:
            raise SystemExit("tracked I01 report does not match the locked plan")
        if expected["status"] != "pass":
            raise SystemExit("tracked I01 report is not passing")
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
