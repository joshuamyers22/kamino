"""Run the locked I02 Satterthwaite type-I-error assessment."""

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
PLAN_PATH = ROOT / "statistical" / "i02_plan.json"


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
    groups = int(specification["groups"])
    x_pattern = np.asarray(specification["x_within_group"], dtype=np.float64)
    x = np.tile(x_pattern, groups)
    labels = np.repeat([f"g{index + 1}" for index in range(groups)], x_pattern.size)
    fixed = np.asarray(specification["fixed_effects"], dtype=np.float64)
    alpha = float(hypotheses["alpha"])
    one_rejections = 0
    joint_rejections = 0
    available = 0
    statuses: dict[str, int] = {}
    started = time.perf_counter()

    for _ in range(replicates):
        random_intercepts = rng.normal(
            0.0, float(specification["random_intercept_sd"]), groups
        )
        response = (
            fixed[0]
            + fixed[1] * x
            + random_intercepts[np.repeat(np.arange(groups), x_pattern.size)]
            + rng.normal(0.0, float(specification["residual_sd"]), x.size)
        )
        try:
            model = lmer(
                "y ~ x + (1 | group)",
                pd.DataFrame({"y": response, "x": x, "group": labels}),
                reml=True,
            )
            analysis = model.satterthwaite()
            one = analysis.test([0.0, 1.0], rhs=float(fixed[1]))
            joint = analysis.joint_test(np.eye(2), rhs=fixed)
            if one.available and joint.available:
                available += 1
                one_rejections += int(cast(float, one.p_value) < alpha)
                joint_rejections += int(cast(float, joint.p_value) < alpha)
            else:
                status = one.status if not one.available else joint.status
                statuses[status] = statuses.get(status, 0) + 1
        except Exception as error:  # assessment ledger retains every failure
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
            raise SystemExit("tracked I02 report does not match the locked plan")
        if expected["status"] != "pass":
            raise SystemExit("tracked I02 report is not passing")
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
