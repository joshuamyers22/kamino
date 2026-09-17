"""Verify committed E03 reports, identities, decisions, and risk ownership."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "benchmarks" / "results"
EXPECTED_IMAGE = (
    "ghcr.io/joshuamyers22/kamino-oracle@sha256:"
    "17e45268be294316967064d0600a727463d738d7dfb0c68ec43769baebe8eb4d"
)
EXPECTED_CASES = {
    (scenario, kind)
    for scenario in (
        "random-intercept",
        "fixed-categorical",
        "correlated-random-slope",
        "independent-random-terms",
        "insteval",
    )
    for kind in ("ml", "reml")
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _verify_samples(report: dict[str, Any], minimum: int, key: str) -> None:
    for case in report["cases"]:
        samples = case["timings"][key]["samples_seconds"]
        _require(
            len(samples) >= minimum,
            f"too few samples for {case.get('scenario', 'insteval')} {case['kind']}",
        )
        _require(all(float(value) > 0 for value in samples), "timings must be positive")


def main() -> None:
    single_path = RESULTS / "e03_kamino_single_group_v1.json"
    sparse_path = RESULTS / "e03_kamino_general_sparse_v1.json"
    lme4_path = RESULTS / "e03_lme4_v1.json"
    assessment_path = RESULTS / "e03_assessment_v1.json"
    for path in (single_path, sparse_path, lme4_path, assessment_path):
        _require(path.is_file(), f"missing E03 artifact: {path.relative_to(ROOT)}")

    single = _load(single_path)
    sparse = _load(sparse_path)
    lme4 = _load(lme4_path)
    assessment = _load(assessment_path)
    risks = _load(ROOT / "benchmarks" / "e03_risks_v1.json")
    hardware = _load(
        ROOT / "benchmarks" / "hardware" / "apple_m1_pro_8c_16gb_2026_v1.json"
    )
    single_manifest_path = ROOT / single["manifest"]
    sparse_manifest_path = ROOT / sparse["manifest"]

    _require(
        single["manifest_sha256"] == _sha256(single_manifest_path),
        "single-group manifest hash drifted",
    )
    _require(
        sparse["manifest_sha256"] == _sha256(sparse_manifest_path),
        "sparse manifest hash drifted",
    )
    _require(
        lme4["manifests"]["single_group"]["sha256"] == _sha256(single_manifest_path),
        "lme4 single-group manifest hash drifted",
    )
    _require(
        lme4["manifests"]["general_sparse"]["sha256"] == _sha256(sparse_manifest_path),
        "lme4 sparse manifest hash drifted",
    )
    revisions = {
        single["revision"],
        sparse["revision"],
        lme4["revision"],
        assessment["protocol_revision"],
    }
    _require(
        len(revisions) == 1
        and re.fullmatch(r"[0-9a-f]{40}", revisions.pop()) is not None,
        "E03 protocol revisions differ or are invalid",
    )

    profile = hardware["hardware_profile"]
    _require(
        single["environment"]["hardware_profile"] == profile,
        "single-group hardware profile differs",
    )
    _require(
        sparse["environment"]["hardware_profile"] == profile,
        "sparse hardware profile differs",
    )
    _require(
        lme4["environment"]["hardware_profile"] == profile,
        "lme4 hardware profile differs",
    )
    _require(
        lme4["environment"]["oracle_image"] == EXPECTED_IMAGE,
        "lme4 image digest differs",
    )
    _require(
        single["environment"]["blas_threads"] == 1, "single-group thread count differs"
    )
    _require(sparse["environment"]["blas_threads"] == 1, "sparse thread count differs")
    _require(lme4["environment"]["blas_threads"] == 1, "lme4 thread count differs")

    _verify_samples(single, 6, "public_fit_all")
    _verify_samples(sparse, 5, "end_to_end")
    _verify_samples(lme4, 5, "end_to_end")
    identities = {(case["scenario"], case["kind"]) for case in lme4["cases"]}
    _require(identities == EXPECTED_CASES, "lme4 E03 case coverage differs")
    _require(
        single["correctness_pass"] and single["resource_pass"],
        "single-group E03 gate failed",
    )
    _require(
        sparse["all_correctness_pass"] and sparse["all_resource_pass"],
        "sparse E03 gate failed",
    )
    _require(lme4["correctness_pass"], "lme4 E03 correctness gate failed")

    for item in assessment["inputs"].values():
        path = ROOT / item["path"]
        _require(
            item["sha256"] == _sha256(path), f"assessment input drifted: {item['path']}"
        )
    _require(assessment["checks"]["e03_gate_pass"], "committed E03 assessment failed")
    _require(
        assessment["decision"]["comparative_performance_claim"] == "not-claimed",
        "invalid comparative claim",
    )
    _require(
        assessment["decision"]["insteval_within_2x"] == "unverified",
        "2x aspiration must remain unverified",
    )
    _require(
        all(
            item["ratio_interpretation"] == "observational-only"
            for item in assessment["comparisons"]
        ),
        "timing ratio is not observational",
    )

    required_risks = {f"E03-R{index:02d}" for index in range(1, 7)}
    _require(
        {item["id"] for item in risks["risks"]} == required_risks,
        "E03 risk set differs",
    )
    for risk in risks["risks"]:
        _require(bool(risk["owner"]), f"risk has no owner: {risk['id']}")
        _require(
            re.fullmatch(r"\d{4}-\d{2}-\d{2}", risk["review_on"]) is not None,
            f"risk has invalid review date: {risk['id']}",
        )
        _require(
            bool(risk["control"] and risk["closure_evidence"]),
            f"risk control is incomplete: {risk['id']}",
        )
    print(
        "verified E03 raw reports, fidelity/resource gate, claim boundary, "
        "and owned risks"
    )


if __name__ == "__main__":
    main()
