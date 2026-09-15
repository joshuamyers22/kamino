from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST = Path(__file__).parents[1] / "benchmarks" / "single_group_v1.json"
SPARSE_MANIFEST = Path(__file__).parents[1] / "benchmarks" / "general_sparse_v1.json"


def test_single_group_benchmark_manifest_covers_complete_public_scope() -> None:
    manifest: dict[str, Any] = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "1.0.0"
    assert manifest["generator"] == {
        "id": "deterministic-balanced-single-group-v1",
        "observations": 1_000_000,
        "groups": 10_000,
        "observations_per_group": 100,
        "contains_caller_data": False,
    }
    assert {item["id"] for item in manifest["scenarios"]} == {
        "random-intercept",
        "fixed-categorical",
        "correlated-random-slope",
        "independent-random-terms",
    }
    assert manifest["objective_kinds"] == ["ml", "reml"]
    assert manifest["measurement"]["end_to_end_repetitions"] >= 6
    assert manifest["measurement"]["fixed_theta_warm_repetitions"] >= 5
    assert manifest["regression_policy"]["timing_decisions_require_dedicated_hardware"]
    assert manifest["oracle"]["comparative_timing_status"] == "not-comparable"


def test_general_sparse_manifest_gates_insteval_crossed_scale() -> None:
    manifest: dict[str, Any] = json.loads(SPARSE_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "1.0.0"
    assert manifest["fixture"] == "oracle/fixtures/v1/insteval_sparse.json"
    assert manifest["objective_kinds"] == ["ml", "reml"]
    assert manifest["ceilings"]["peak_rss_megabytes"] <= 1500.0
    assert manifest["ceilings"]["end_to_end_seconds_per_fit"] <= 180.0
    assert manifest["oracle"] == {
        "profile": "lme4-2.0.6-unstructured-gaussian-v1",
        "dataset": "InstEval",
        "comparative_timing_status": "not-comparable",
    }
