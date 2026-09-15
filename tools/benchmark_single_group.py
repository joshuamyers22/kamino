"""Manifested Phase 1 single-group time and peak-memory benchmark."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportPrivateUsage=false

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "single_group_v1.json"
DEFAULT_OUTPUT = ROOT / ".work" / "benchmarks" / "single_group_v1.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scenario", help=argparse.SUPPRESS)
    parser.add_argument("--kind", choices=("ml", "reml"), help=argparse.SUPPRESS)
    return parser


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0.0":
        raise ValueError("unsupported benchmark manifest schema")
    return manifest


def _percentile(samples: Sequence[float], percentile: float) -> float:
    if not samples:
        raise ValueError("at least one benchmark sample is required")
    ordered = sorted(float(value) for value in samples)
    rank = max(0, int(np.ceil(percentile * len(ordered))) - 1)
    return ordered[rank]


def _summary(samples: Sequence[float]) -> dict[str, Any]:
    return {
        "samples_seconds": [float(value) for value in samples],
        "median_seconds": float(statistics.median(samples)),
        "p95_seconds": _percentile(samples, 0.95),
    }


def _peak_rss_megabytes() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    byte_count = value if sys.platform == "darwin" else value * 1024.0
    return byte_count / 1_000_000.0


def _problem(
    *, observations: int, groups: int, scenario: str
) -> tuple[pd.DataFrame, np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    if observations <= 0 or groups <= 0 or observations % groups:
        raise ValueError("benchmark observations must divide evenly across groups")
    group_indices = np.repeat(np.arange(groups, dtype=np.int64), observations // groups)
    row = np.arange(observations, dtype=np.float64)
    group_signal = 0.8 * np.sin(np.arange(groups, dtype=np.float64) * 0.013)
    residual = 0.2 * np.sin(row * 0.17) + 0.1 * np.cos(row * 0.071)
    offset = 0.05 * np.cos(row * 0.031)
    weights = 0.75 + (np.arange(observations, dtype=np.int64) % 11) / 10.0
    categories = [f"group-{index:05d}" for index in range(groups)]
    group_values = pd.Categorical.from_codes(group_indices, categories=categories)
    if scenario == "random-intercept":
        y = 2.5 + group_signal[group_indices] + residual + offset
        frame = pd.DataFrame({"y": y, "group": group_values})
    elif scenario == "correlated-random-slope":
        x = np.tile(
            np.linspace(-1.0, 1.0, observations // groups, dtype=np.float64),
            groups,
        )
        slope_signal = 0.3 * np.cos(np.arange(groups, dtype=np.float64) * 0.019)
        y = (
            2.5
            + 0.7 * x
            + group_signal[group_indices]
            + slope_signal[group_indices] * x
            + residual
            + offset
        )
        frame = pd.DataFrame({"y": y, "x": x, "group": group_values})
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    return frame, weights, offset


def _timed(function: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    value = function()
    return value, time.perf_counter() - started


def _worker(manifest: dict[str, Any], scenario_id: str, kind_value: str) -> None:
    from kamino import ObjectiveKind, lmer
    from kamino.block import (
        evaluate_prepared_single_group_block,
        prepare_single_group_block,
    )
    from kamino.fit import FitControl, _fit_theta, _fit_theta_vector
    from kamino.formula import build_single_group_design

    scenario = next(item for item in manifest["scenarios"] if item["id"] == scenario_id)
    generator = manifest["generator"]
    observations = int(generator["observations"])
    groups = int(generator["groups"])
    if bool(manifest.get("smoke")):
        groups = 20
        observations = 2_000
    frame, weights, offset = _problem(
        observations=observations, groups=groups, scenario=scenario_id
    )
    kind = ObjectiveKind(kind_value)
    reml = kind is ObjectiveKind.REML
    formula = str(scenario["formula"])
    repetitions = int(manifest["measurement"]["end_to_end_repetitions"])
    total_samples: list[float] = []
    final_result: Any = None
    for _ in range(repetitions):
        final_result, elapsed = _timed(
            lambda: lmer(formula, frame, reml=reml, weights=weights, offset=offset)
        )
        total_samples.append(elapsed)
        gc.collect()

    prediction, prediction_seconds = _timed(
        lambda: final_result.predict(mode="conditional")
    )
    prediction_error = float(
        np.max(np.abs(prediction.values - final_result.fitted_values))
    )

    design, encoding_seconds = _timed(
        lambda: build_single_group_design(
            formula, frame, weights=weights, offset=offset
        )
    )
    workspace, assembly_seconds = _timed(
        lambda: prepare_single_group_block(design.spec)
    )
    theta = scenario["theta"]
    _, fixed_cold_seconds = _timed(
        lambda: evaluate_prepared_single_group_block(workspace, theta, kind=kind)
    )
    fixed_warm_samples: list[float] = []
    for _ in range(int(manifest["measurement"]["fixed_theta_warm_repetitions"])):
        _, elapsed = _timed(
            lambda: evaluate_prepared_single_group_block(workspace, theta, kind=kind)
        )
        fixed_warm_samples.append(elapsed)
    control = FitControl()
    if design.spec.k == 1:
        optimized, optimization_seconds = _timed(
            lambda: _fit_theta(design, kind, control, workspace)
        )
    else:
        optimized, optimization_seconds = _timed(
            lambda: _fit_theta_vector(design, kind, control, workspace)
        )
    optimized_theta, optimized_result, diagnostics = optimized

    peak_rss = _peak_rss_megabytes()
    dense_z_bytes = observations * design.spec.q * 8
    compact_array_bytes = sum(
        array.nbytes
        for array in (
            design.spec.y,
            design.spec.x,
            design.spec.group_indices,
            design.spec.random_design,
            design.spec.weights,
            design.spec.offset,
        )
    )
    workspace_array_bytes = sum(
        array.nbytes
        for array in (
            workspace.centered_response,
            workspace.random_cross,
            workspace.random_fixed_cross,
            workspace.random_response_cross,
            workspace.fixed_cross,
            workspace.fixed_response_cross,
        )
    )
    objective_error = abs(float(final_result.objective) - optimized_result.objective)
    theta_error = float(np.max(np.abs(final_result.theta - optimized_theta)))
    ceiling_rss = float(manifest["ceilings"]["peak_rss_megabytes"])
    ceiling_seconds = float(manifest["ceilings"]["end_to_end_seconds_per_fit"])
    correctness_pass = bool(
        prediction_error <= 1e-10
        and objective_error <= 1e-10
        and theta_error <= 1e-10
        and np.isfinite(optimized_result.objective)
        and np.isfinite(optimized_theta).all()
        and diagnostics.converged
        and diagnostics.backend == "single-group-block-cholesky"
        and not hasattr(design.spec, "z")
    )
    resource_pass = bool(
        peak_rss <= ceiling_rss and max(total_samples) <= ceiling_seconds
    )
    report = {
        "scenario": scenario_id,
        "kind": kind_value,
        "dimensions": {
            "n": observations,
            "p": design.spec.p,
            "q": design.spec.q,
            "k": design.spec.k,
            "d": len(optimized_theta),
            "random_design_nnz_upper_bound": observations * design.spec.k,
            "forbidden_dense_z_bytes": dense_z_bytes,
            "compact_spec_array_bytes": compact_array_bytes,
            "workspace_array_bytes": workspace_array_bytes,
        },
        "timings": {
            "public_fit": _summary(total_samples),
            "parse_and_encoding_seconds": encoding_seconds,
            "block_assembly_seconds": assembly_seconds,
            "symbolic_analysis_seconds": 0.0,
            "symbolic_analysis_status": "not-applicable-independent-blocks",
            "fixed_theta_cold_seconds": fixed_cold_seconds,
            "fixed_theta_warm": _summary(fixed_warm_samples),
            "optimization_seconds_preassembled": optimization_seconds,
            "prediction_seconds": prediction_seconds,
            "inference_status": "unavailable-in-phase1",
        },
        "optimizer": {
            "name": diagnostics.optimizer,
            "evaluations": diagnostics.evaluations,
            "boundary": diagnostics.boundary,
        },
        "peak_rss_megabytes": peak_rss,
        "checks": {
            "correctness_pass": correctness_pass,
            "resource_pass": resource_pass,
            "conditional_prediction_max_absolute_error": prediction_error,
            "staged_objective_absolute_error": objective_error,
            "staged_theta_max_absolute_error": theta_error,
            "dense_z_absent": not hasattr(design.spec, "z"),
        },
    }
    print(json.dumps(report, sort_keys=True))
    if not correctness_pass:
        raise SystemExit("benchmark correctness gate failed")
    if not resource_pass:
        raise SystemExit("benchmark resource ceiling exceeded")


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _parent(manifest_path: Path, output: Path, smoke: bool) -> None:
    manifest = _load_manifest(manifest_path)
    if smoke:
        manifest["smoke"] = True
        manifest["measurement"]["end_to_end_repetitions"] = 1
        manifest["measurement"]["fixed_theta_warm_repetitions"] = 2
    environment = os.environ.copy()
    threads = str(manifest["measurement"]["blas_threads"])
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = threads
    worker_manifest = output.with_suffix(".manifest.json")
    worker_manifest.parent.mkdir(parents=True, exist_ok=True)
    worker_manifest.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    cases: list[dict[str, Any]] = []
    try:
        for scenario in manifest["scenarios"]:
            for kind in manifest["objective_kinds"]:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--manifest",
                    str(worker_manifest),
                    "--scenario",
                    str(scenario["id"]),
                    "--kind",
                    str(kind),
                ]
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=float(manifest["measurement"]["worker_timeout_seconds"]),
                )
                cases.append(json.loads(completed.stdout))
    finally:
        worker_manifest.unlink(missing_ok=True)
    report = {
        "schema_version": "1.0.0",
        "benchmark_id": manifest["benchmark_id"],
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "revision": _git_revision(),
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "blas_threads": int(threads),
            "uv_lock_sha256": hashlib.sha256(
                (ROOT / "uv.lock").read_bytes()
            ).hexdigest(),
            "shared_ci_timing_regression_authoritative": False,
        },
        "oracle_comparison": manifest["oracle"],
        "cases": cases,
        "correctness_pass": all(case["checks"]["correctness_pass"] for case in cases),
        "resource_pass": all(case["checks"]["resource_pass"] for case in cases),
        "smoke": smoke,
    }
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    arguments = _parser().parse_args()
    manifest = _load_manifest(arguments.manifest)
    if arguments.worker:
        if arguments.scenario is None or arguments.kind is None:
            raise SystemExit("worker mode requires scenario and kind")
        _worker(manifest, arguments.scenario, arguments.kind)
    else:
        _parent(
            arguments.manifest.resolve(), arguments.output.resolve(), arguments.smoke
        )


if __name__ == "__main__":
    main()
