"""Run the locked I03 Kenward--Roger and profile calibration."""

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
from scipy.stats import chi2

from kamino import lmer

ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "statistical" / "i03_plan.json"


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return float(center - half), float(center + half)


def assess() -> dict[str, object]:
    plan_payload = PLAN_PATH.read_bytes()
    plan = cast(dict[str, Any], json.loads(plan_payload))
    specification = plan["data_generating_model"]
    gates = plan["primary_gates"]
    replicates = int(plan["replicates"])
    alpha = float(plan["hypotheses"]["alpha"])
    cutoff = float(chi2.ppf(1.0 - alpha, 1))
    rng = np.random.Generator(np.random.PCG64DXSM(int(plan["rng"]["assessment_seed"])))
    groups = int(specification["groups"])
    x_pattern = np.asarray(specification["x_within_group"], dtype=np.float64)
    x = np.tile(x_pattern, groups)
    group_index = np.repeat(np.arange(groups), x_pattern.size)
    labels = np.repeat([f"g{index + 1}" for index in range(groups)], x_pattern.size)
    fixed = np.asarray(specification["fixed_effects"], dtype=np.float64)
    kr_available = 0
    profile_available = 0
    kr_rejections = 0
    profile_rejections = 0
    kr_statuses: dict[str, int] = {}
    profile_statuses: dict[str, int] = {}
    started = time.perf_counter()

    for _ in range(replicates):
        random_intercepts = rng.normal(
            0.0, float(specification["random_intercept_sd"]), groups
        )
        response = (
            fixed[0]
            + fixed[1] * x
            + random_intercepts[group_index]
            + rng.normal(0.0, float(specification["residual_sd"]), x.size)
        )
        try:
            model = lmer(
                "y ~ x + (1 | group)",
                pd.DataFrame({"y": response, "x": x, "group": labels}),
                reml=True,
            )
        except Exception as error:
            status = type(error).__name__
            kr_statuses[status] = kr_statuses.get(status, 0) + 1
            profile_statuses[status] = profile_statuses.get(status, 0) + 1
            continue

        try:
            analysis = model.kenward_roger()
            test = analysis.test([0.0, 1.0], rhs=0.0)
            if test.available:
                kr_available += 1
                kr_rejections += int(cast(float, test.p_value) < alpha)
            else:
                kr_statuses[test.status] = kr_statuses.get(test.status, 0) + 1
        except Exception as error:
            status = type(error).__name__
            kr_statuses[status] = kr_statuses.get(status, 0) + 1

        try:
            profiled = model.profile(targets=["x"], values={"x": [0.0]})
            trace = profiled.trace("x")
            point = next(
                item for item in trace.points if abs(item.target_value) < 1e-14
            )
            if trace.status in ("ok", "boundary_truncated") and point.converged:
                profile_available += 1
                profile_rejections += int(point.deviance_difference > cutoff)
            else:
                profile_statuses[trace.status] = (
                    profile_statuses.get(trace.status, 0) + 1
                )
        except Exception as error:
            status = type(error).__name__
            profile_statuses[status] = profile_statuses.get(status, 0) + 1

    kr_rate = kr_rejections / kr_available
    profile_rate = profile_rejections / profile_available
    profile_coverage = 1.0 - profile_rate
    kr_interval = _wilson_interval(kr_rejections, kr_available)
    profile_interval = _wilson_interval(profile_rejections, profile_available)
    kr_unavailable_rate = (replicates - kr_available) / replicates
    profile_unavailable_rate = (replicates - profile_available) / replicates
    checks = {
        "kr_minimum_available": kr_available
        >= gates["minimum_available_replicates_per_method"],
        "profile_minimum_available": profile_available
        >= gates["minimum_available_replicates_per_method"],
        "kr_maximum_unavailable_rate": kr_unavailable_rate
        <= gates["maximum_unavailable_rate_per_method"],
        "profile_maximum_unavailable_rate": profile_unavailable_rate
        <= gates["maximum_unavailable_rate_per_method"],
        "kr_rejection_rate": gates["minimum_rejection_rate"]
        <= kr_rate
        <= gates["maximum_rejection_rate"],
        "profile_rejection_rate": gates["minimum_rejection_rate"]
        <= profile_rate
        <= gates["maximum_rejection_rate"],
        "kr_wilson_contains_nominal": kr_interval[0] <= alpha <= kr_interval[1],
        "profile_wilson_contains_nominal": profile_interval[0]
        <= alpha
        <= profile_interval[1],
        "profile_coverage": gates["minimum_profile_coverage"]
        <= profile_coverage
        <= gates["maximum_profile_coverage"],
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
        "metrics": {
            "kr_available": kr_available,
            "kr_unavailable": replicates - kr_available,
            "kr_unavailable_rate": kr_unavailable_rate,
            "kr_unavailable_statuses": kr_statuses,
            "kr_rejections": kr_rejections,
            "kr_rejection_rate": kr_rate,
            "kr_wilson_95_percent": kr_interval,
            "profile_available": profile_available,
            "profile_unavailable": replicates - profile_available,
            "profile_unavailable_rate": profile_unavailable_rate,
            "profile_unavailable_statuses": profile_statuses,
            "profile_rejections": profile_rejections,
            "profile_rejection_rate": profile_rate,
            "profile_wilson_95_percent": profile_interval,
            "profile_coverage": profile_coverage,
            "profile_likelihood_ratio_cutoff": cutoff,
        },
        "checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
        "scope_note": plan["release_claim"],
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
            raise SystemExit("tracked I03 report does not match the locked plan")
        if expected["status"] != "pass":
            raise SystemExit("tracked I03 report is not passing")
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
