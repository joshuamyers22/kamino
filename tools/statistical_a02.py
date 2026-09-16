"""Run the locked A02 CR2 Satterthwaite/HTZ null calibration."""

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

from kamino import lmer

ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "statistical" / "a02_plan.json"


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
    hypotheses = plan["hypotheses"]
    replicates = int(plan["replicates"])
    rng = np.random.Generator(np.random.PCG64DXSM(int(plan["rng"]["assessment_seed"])))
    clusters = int(specification["independent_clusters"])
    schools_per = int(specification["schools_per_cluster"])
    observations_per = int(specification["observations_per_school"])
    schools = clusters * schools_per
    x_pattern = np.asarray(specification["x_within_school"], dtype=np.float64)
    x = np.tile(x_pattern, schools)
    school_index = np.repeat(np.arange(schools), observations_per)
    cluster_index = np.repeat(np.arange(clusters), schools_per * observations_per)
    within = np.tile(np.arange(observations_per), schools)
    z = ((school_index + 2 * within) % 7 - 3.0) / 3.0
    school_labels = np.asarray([f"s{index + 1}" for index in school_index])
    cluster_labels = np.asarray([f"d{index + 1}" for index in cluster_index])
    fixed = np.asarray(specification["fixed_effects"], dtype=np.float64)
    alpha = float(hypotheses["alpha"])
    one_rejections = 0
    joint_rejections = 0
    available = 0
    statuses: dict[str, int] = {}
    started = time.perf_counter()

    for _ in range(replicates):
        school_effect = rng.normal(
            0.0, float(specification["school_random_intercept_sd"]), schools
        )
        cluster_effect = rng.normal(
            0.0, float(specification["cluster_random_intercept_sd"]), clusters
        )
        cluster_slope = rng.normal(
            0.0, float(specification["cluster_random_slope_sd"]), clusters
        )
        residual_sd = float(specification["residual_sd_base"]) + float(
            specification["residual_sd_x_multiplier"]
        ) * np.abs(x)
        response = (
            fixed[0]
            + fixed[1] * x
            + fixed[2] * z
            + school_effect[school_index]
            + cluster_effect[cluster_index]
            + cluster_slope[cluster_index] * x
            + rng.normal(0.0, residual_sd)
        )
        try:
            model = lmer(
                "y ~ x + z + (1 | school)",
                pd.DataFrame(
                    {
                        "y": response,
                        "x": x,
                        "z": z,
                        "school": school_labels,
                    }
                ),
                reml=True,
            )
            robust = model.cluster_robust(cluster_labels)
            one = robust.test([0.0, 1.0, 0.0], rhs=float(fixed[1]))
            joint = robust.test([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], rhs=fixed[1:])
            if one.available and joint.available:
                available += 1
                one_rejections += int(cast(float, one.p_value) < alpha)
                joint_rejections += int(cast(float, joint.p_value) < alpha)
            else:
                status = one.status if not one.available else joint.status
                statuses[status] = statuses.get(status, 0) + 1
        except Exception as error:  # every failed replicate remains accounted for
            status = type(error).__name__
            statuses[status] = statuses.get(status, 0) + 1

    one_rate = one_rejections / available
    joint_rate = joint_rejections / available
    unavailable_rate = (replicates - available) / replicates
    one_interval = _wilson_interval(one_rejections, available)
    joint_interval = _wilson_interval(joint_rejections, available)
    checks = {
        "minimum_available_replicates": available
        >= gates["minimum_available_replicates"],
        "maximum_unavailable_rate": unavailable_rate
        <= gates["maximum_unavailable_rate"],
        "one_df_rejection_rate": gates["minimum_rejection_rate"]
        <= one_rate
        <= gates["maximum_rejection_rate"],
        "joint_rejection_rate": gates["minimum_rejection_rate"]
        <= joint_rate
        <= gates["maximum_rejection_rate"],
        "one_df_wilson_contains_nominal": one_interval[0] <= alpha <= one_interval[1],
        "joint_wilson_contains_nominal": joint_interval[0]
        <= alpha
        <= joint_interval[1],
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
            "available": available,
            "unavailable": replicates - available,
            "unavailable_rate": unavailable_rate,
            "unavailable_statuses": statuses,
            "one_df_rejections": one_rejections,
            "one_df_rejection_rate": one_rate,
            "one_df_wilson_95_percent": one_interval,
            "joint_rejections": joint_rejections,
            "joint_rejection_rate": joint_rate,
            "joint_wilson_95_percent": joint_interval,
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
            raise SystemExit("tracked A02 report does not match the locked plan")
        if expected["status"] != "pass":
            raise SystemExit("tracked A02 report is not passing")
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
