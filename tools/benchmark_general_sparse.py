"""Manifested crossed general-sparse resource benchmark."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "general_sparse_v1.json"
DEFAULT_OUTPUT = ROOT / ".work" / "benchmarks" / "general_sparse_v1.json"


def _peak_rss_megabytes() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    byte_count = value if sys.platform == "darwin" else value * 1024.0
    return byte_count / 1_000_000.0


def _frame(source: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": source["response"],
            "s": pd.Categorical(source["s"], categories=source["s_levels"]),
            "d": pd.Categorical(source["d"], categories=source["d_levels"]),
            "service": pd.Categorical(
                source["service"], categories=source["service_levels"]
            ),
            "dept": pd.Categorical(source["dept"], categories=source["dept_levels"]),
        }
    )


def _worker(manifest: dict[str, Any], kind: str) -> None:
    from kamino import ObjectiveKind, lmer
    from kamino.formula import build_general_design
    from kamino.sparse import (
        evaluate_prepared_general_sparse,
        prepare_general_sparse,
    )

    fixture: dict[str, Any] = json.loads(
        (ROOT / manifest["fixture"]).read_text(encoding="utf-8")
    )
    source = fixture["data"]
    frame = _frame(source)
    expected = next(item for item in fixture["fits"] if item["kind"] == kind)
    reml = kind == "reml"
    started = time.perf_counter()
    result = lmer(fixture["formula"], frame, reml=reml)
    elapsed = time.perf_counter() - started
    prediction_error = float(
        np.max(np.abs(result.predict(mode="conditional").values - result.fitted_values))
    )
    design = build_general_design(fixture["formula"], frame)
    workspace = prepare_general_sparse(design.spec)
    staged = evaluate_prepared_general_sparse(
        workspace,
        result.theta,
        kind=ObjectiveKind(kind),
    )
    objective_error = abs(result.objective - float(expected["objective"]))
    theta_error = float(
        np.max(np.abs(result.theta - np.asarray(expected["theta"], dtype=np.float64)))
    )
    beta_error = float(
        np.max(np.abs(result.beta - np.asarray(expected["beta"], dtype=np.float64)))
    )
    sigma_error = abs(result.sigma - float(expected["sigma"]))
    staged_error = abs(result.objective - staged.objective)
    tolerances = manifest["tolerances"]
    correctness_pass = bool(
        objective_error <= tolerances["objective_absolute"]
        and theta_error <= tolerances["theta_absolute"]
        and beta_error <= tolerances["beta_absolute"]
        and sigma_error <= tolerances["sigma_absolute"]
        and staged_error <= tolerances["staged_objective_absolute"]
        and prediction_error <= tolerances["prediction_absolute"]
        and result.diagnostics.backend == "scipy-superlu-symmetric-sparse"
        and not hasattr(design.spec, "z")
    )
    peak_rss = _peak_rss_megabytes()
    resource_pass = bool(
        elapsed <= manifest["ceilings"]["end_to_end_seconds_per_fit"]
        and peak_rss <= manifest["ceilings"]["peak_rss_megabytes"]
    )
    report = {
        "kind": kind,
        "dimensions": {
            "n": design.spec.n,
            "p": design.spec.p,
            "q": design.spec.q,
            "d": design.spec.d,
            "random_design_nonzeros": int(workspace.random_design.nnz),
            "structural_c_nonzeros": workspace.structural_c_nonzeros,
            "accepted_factor_nonzeros": staged.factor_nonzeros,
            "forbidden_dense_z_bytes": design.spec.n * design.spec.q * 8,
        },
        "elapsed_seconds": elapsed,
        "peak_rss_megabytes": peak_rss,
        "optimizer_evaluations": result.diagnostics.evaluations,
        "checks": {
            "correctness_pass": correctness_pass,
            "resource_pass": resource_pass,
            "objective_absolute_error": objective_error,
            "theta_max_absolute_error": theta_error,
            "beta_max_absolute_error": beta_error,
            "sigma_absolute_error": sigma_error,
            "staged_objective_absolute_error": staged_error,
            "prediction_max_absolute_error": prediction_error,
            "dense_z_absent": not hasattr(design.spec, "z"),
        },
    }
    print(json.dumps(report, sort_keys=True))
    if not correctness_pass:
        raise SystemExit("general sparse correctness gate failed")
    if not resource_pass:
        raise SystemExit("general sparse resource ceiling exceeded")


def _parent(manifest_path: Path, output: Path) -> None:
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0.0":
        raise ValueError("unsupported benchmark manifest schema")
    environment = os.environ.copy()
    threads = str(manifest["measurement"]["blas_threads"])
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = threads
    cases: list[dict[str, Any]] = []
    for kind in manifest["objective_kinds"]:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--manifest",
                str(manifest_path),
                "--kind",
                kind,
            ],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=float(manifest["measurement"]["worker_timeout_seconds"]),
        )
        cases.append(json.loads(completed.stdout))
    report = {
        "schema_version": "1.0.0",
        "benchmark_id": manifest["benchmark_id"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "cases": cases,
        "maximum_peak_rss_megabytes": max(case["peak_rss_megabytes"] for case in cases),
        "all_correctness_pass": all(
            case["checks"]["correctness_pass"] for case in cases
        ),
        "all_resource_pass": all(case["checks"]["resource_pass"] for case in cases),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--kind", choices=("ml", "reml"), help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    manifest: dict[str, Any] = json.loads(
        arguments.manifest.read_text(encoding="utf-8")
    )
    if arguments.worker:
        if arguments.kind is None:
            raise SystemExit("worker requires --kind")
        _worker(manifest, arguments.kind)
    else:
        _parent(arguments.manifest, arguments.output)


if __name__ == "__main__":
    main()
