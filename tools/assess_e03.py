"""Build the E03 correctness/resource assessment from immutable raw reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parents[1]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index(
    cases: list[dict[str, Any]], *, sparse: bool = False
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (("insteval" if sparse else str(case["scenario"])), str(case["kind"])): case
        for case in cases
    }


def _maximum_absolute(left: list[float], right: list[float]) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
            )
        )
    )


def _native_median(case: dict[str, Any], *, sparse: bool) -> float:
    key = "end_to_end" if sparse else "public_fit_all"
    return float(case["timings"][key]["median_seconds"])


def _dimensions_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("n", "p", "q", "d", "random_design_nonzeros")
    return all(int(left[key]) == int(right[key]) for key in keys)


def build_assessment(
    single_path: Path,
    sparse_path: Path,
    lme4_path: Path,
    risks_path: Path,
    hardware_path: Path,
) -> dict[str, Any]:
    single = _load(single_path)
    sparse = _load(sparse_path)
    reference = _load(lme4_path)
    single_manifest = _load(ROOT / single["manifest"])
    sparse_manifest = _load(ROOT / sparse["manifest"])
    revisions = {single["revision"], sparse["revision"], reference["revision"]}
    if len(revisions) != 1:
        raise ValueError(f"raw report revisions differ: {sorted(revisions)}")

    native = _index(single["cases"])
    native.update(_index(sparse["cases"], sparse=True))
    lme4_cases = _index(reference["cases"])
    if native.keys() != lme4_cases.keys():
        raise ValueError("Kamino and lme4 case identities differ")

    comparisons: list[dict[str, Any]] = []
    all_fidelity_pass = True
    for identity in sorted(native):
        scenario, kind = identity
        candidate = native[identity]
        oracle = lme4_cases[identity]
        is_sparse = scenario == "insteval"
        tolerances = (
            sparse_manifest["tolerances"]
            if is_sparse
            else single_manifest["comparison_tolerances"]
        )
        fit = candidate["fit"]
        reference_fit = oracle["fit"]
        errors = {
            "objective_absolute": abs(
                float(fit["objective"]) - float(reference_fit["objective"])
            ),
            "theta_max_absolute": _maximum_absolute(
                fit["theta"], reference_fit["theta"]
            ),
            "beta_max_absolute": _maximum_absolute(fit["beta"], reference_fit["beta"]),
            "sigma_absolute": abs(float(fit["sigma"]) - float(reference_fit["sigma"])),
        }
        fidelity_pass = bool(
            errors["objective_absolute"] <= float(tolerances["objective_absolute"])
            and errors["theta_max_absolute"] <= float(tolerances["theta_absolute"])
            and errors["beta_max_absolute"] <= float(tolerances["beta_absolute"])
            and errors["sigma_absolute"] <= float(tolerances["sigma_absolute"])
        )
        all_fidelity_pass = all_fidelity_pass and fidelity_pass
        candidate_seconds = _native_median(candidate, sparse=is_sparse)
        oracle_seconds = float(oracle["timings"]["end_to_end"]["median_seconds"])
        comparisons.append(
            {
                "scenario": scenario,
                "kind": kind,
                "dimensions_match": _dimensions_match(
                    candidate["dimensions"], oracle["dimensions"]
                ),
                "fidelity_errors": errors,
                "fidelity_tolerances": {
                    "objective_absolute": float(tolerances["objective_absolute"]),
                    "theta_absolute": float(tolerances["theta_absolute"]),
                    "beta_absolute": float(tolerances["beta_absolute"]),
                    "sigma_absolute": float(tolerances["sigma_absolute"]),
                },
                "fidelity_pass": fidelity_pass,
                "kamino_median_seconds": candidate_seconds,
                "lme4_median_seconds": oracle_seconds,
                "observed_kamino_over_lme4_ratio": candidate_seconds / oracle_seconds,
                "ratio_interpretation": "observational-only",
            }
        )

    dimensions_pass = all(item["dimensions_match"] for item in comparisons)
    native_correctness = bool(
        single["correctness_pass"] and sparse["all_correctness_pass"]
    )
    native_resources = bool(single["resource_pass"] and sparse["all_resource_pass"])
    oracle_correctness = bool(reference["correctness_pass"])
    gate_pass = bool(
        all_fidelity_pass
        and dimensions_pass
        and native_correctness
        and native_resources
        and oracle_correctness
    )
    return {
        "schema_version": "1.0.0",
        "assessment_id": "e03-performance-and-risk-v1",
        "protocol_revision": revisions.pop(),
        "inputs": {
            "kamino_single_group": {
                "path": str(single_path.relative_to(ROOT)),
                "sha256": _sha256(single_path),
            },
            "kamino_general_sparse": {
                "path": str(sparse_path.relative_to(ROOT)),
                "sha256": _sha256(sparse_path),
            },
            "lme4": {
                "path": str(lme4_path.relative_to(ROOT)),
                "sha256": _sha256(lme4_path),
            },
            "risk_register": {
                "path": str(risks_path.relative_to(ROOT)),
                "sha256": _sha256(risks_path),
            },
            "hardware_profile": {
                "path": str(hardware_path.relative_to(ROOT)),
                "sha256": _sha256(hardware_path),
            },
        },
        "comparisons": comparisons,
        "checks": {
            "dimensions_pass": dimensions_pass,
            "cross_runtime_fidelity_pass": all_fidelity_pass,
            "kamino_internal_correctness_pass": native_correctness,
            "kamino_resource_pass": native_resources,
            "lme4_correctness_pass": oracle_correctness,
            "e03_gate_pass": gate_pass,
        },
        "decision": {
            "comparative_performance_claim": "not-claimed",
            "insteval_within_2x": "unverified",
            "reason": (
                "The runtimes share physical hardware and one-thread policy but "
                "differ in OS/BLAS stack and optimizer algorithm."
            ),
            "timing_regression_threshold_fraction": 0.2,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--lme4", type=Path, required=True)
    parser.add_argument(
        "--risks", type=Path, default=ROOT / "benchmarks" / "e03_risks_v1.json"
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=ROOT / "benchmarks" / "hardware" / "apple_m1_pro_8c_16gb_2026_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    assessment = build_assessment(
        arguments.single.resolve(),
        arguments.sparse.resolve(),
        arguments.lme4.resolve(),
        arguments.risks.resolve(),
        arguments.hardware.resolve(),
    )
    if not assessment["checks"]["e03_gate_pass"]:
        raise SystemExit("E03 correctness/resource assessment failed")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(assessment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(assessment, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
