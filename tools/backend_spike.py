"""Phase 0 feasibility benchmark for block and general sparse PLS solves."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy
from scipy import sparse
from scipy.sparse.linalg import splu

from kamino.model import ModelSpec, ObjectiveKind
from kamino.pls import evaluate_fixed_theta

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Problem:
    y: FloatArray
    x: FloatArray
    random_design: FloatArray
    group: npt.NDArray[np.int64]
    weights: FloatArray
    offset: FloatArray
    block: FloatArray


@dataclass(frozen=True)
class Evaluation:
    objective: float
    beta: FloatArray
    u: FloatArray
    pwrss: float


def make_problem(groups: int, observations_per_group: int = 4) -> Problem:
    group = np.repeat(np.arange(groups, dtype=np.int64), observations_per_group)
    time_value = np.tile(
        np.linspace(-1.5, 1.5, observations_per_group, dtype=np.float64), groups
    )
    x = np.column_stack((np.ones(group.size), time_value))
    random_design = x.copy()
    group_signal = 0.3 * np.sin(group * 0.37)
    residual = 0.08 * np.cos(np.arange(group.size) * 0.71)
    offset = 0.05 * np.sin(np.arange(group.size) * 0.19)
    y = 1.7 + 0.55 * time_value + group_signal + residual + offset
    weights = 0.75 + (np.arange(group.size) % 7) / 5.0
    block = np.array([[0.8, 0.0], [0.25, 0.4]], dtype=np.float64)
    return Problem(y, x, random_design, group, weights, offset, block)


def evaluate_block(problem: Problem) -> Evaluation:
    sqrt_weights = np.sqrt(problem.weights)
    response = sqrt_weights * (problem.y - problem.offset)
    design = sqrt_weights[:, None] * problem.x
    random = sqrt_weights[:, None] * (problem.random_design @ problem.block)
    schur = design.T @ design
    target = design.T @ response
    logdet_c = 0.0
    saved: list[tuple[npt.NDArray[np.bool_], FloatArray, FloatArray, FloatArray]] = []

    for group_id in range(int(problem.group.max()) + 1):
        rows = problem.group == group_id
        a = random[rows]
        x_group = design[rows]
        y_group = response[rows]
        c = np.eye(2) + a.T @ a
        chol = np.linalg.cholesky(c)
        d = a.T @ x_group
        f = a.T @ y_group
        c_inv_d = np.linalg.solve(chol.T, np.linalg.solve(chol, d))
        c_inv_f = np.linalg.solve(chol.T, np.linalg.solve(chol, f))
        schur -= d.T @ c_inv_d
        target -= d.T @ c_inv_f
        logdet_c += float(2.0 * np.log(chol.diagonal()).sum())
        saved.append((rows, chol, d, f))

    beta = np.linalg.solve(schur, target)
    u = np.empty(problem.group.size // 2, dtype=np.float64)
    fitted_random = np.empty(problem.y.size, dtype=np.float64)
    for group_id, (rows, chol, d, f) in enumerate(saved):
        group_u = np.linalg.solve(chol.T, np.linalg.solve(chol, f - d @ beta))
        u[group_id * 2 : group_id * 2 + 2] = group_u
        fitted_random[rows] = random[rows] @ group_u

    residual = response - design @ beta - fitted_random
    pwrss = float(residual @ residual + u @ u)
    degrees = problem.y.size - problem.x.shape[1]
    logdet_s = float(np.linalg.slogdet(schur)[1])
    value = (
        logdet_c
        - float(np.log(problem.weights).sum())
        + logdet_s
        + degrees * (1.0 + np.log(2.0 * np.pi * pwrss / degrees))
    )
    return Evaluation(value, beta, u, pwrss)


def sparse_random_matrix(problem: Problem) -> sparse.csr_matrix:
    weighted = np.sqrt(problem.weights)[:, None] * (
        problem.random_design @ problem.block
    )
    rows = np.repeat(np.arange(problem.group.size), 2)
    columns = (problem.group[:, None] * 2 + np.arange(2)).reshape(-1)
    return sparse.csr_matrix(
        (weighted.reshape(-1), (rows, columns)),
        shape=(problem.group.size, (int(problem.group.max()) + 1) * 2),
    )


def evaluate_sparse(problem: Problem) -> Evaluation:
    sqrt_weights = np.sqrt(problem.weights)
    response = sqrt_weights * (problem.y - problem.offset)
    design = sqrt_weights[:, None] * problem.x
    a = sparse_random_matrix(problem)
    q = (int(problem.group.max()) + 1) * 2
    c = (sparse.eye(q, format="csc") + a.T @ a).tocsc()
    factor = splu(c)
    d = np.asarray(a.T @ design)
    f = np.asarray(a.T @ response)
    c_inv_d = factor.solve(d)
    c_inv_f = factor.solve(f)
    schur = design.T @ design - d.T @ c_inv_d
    target = design.T @ response - d.T @ c_inv_f
    beta = np.linalg.solve(schur, target)
    u = factor.solve(f - d @ beta)
    residual = response - design @ beta - a @ u
    pwrss = float(residual @ residual + u @ u)
    degrees = problem.y.size - problem.x.shape[1]
    logdet_c = float(np.log(np.abs(factor.U.diagonal())).sum())
    logdet_s = float(np.linalg.slogdet(schur)[1])
    value = (
        logdet_c
        - float(np.log(problem.weights).sum())
        + logdet_s
        + degrees * (1.0 + np.log(2.0 * np.pi * pwrss / degrees))
    )
    return Evaluation(value, beta, u, pwrss)


def evaluate_production(problem: Problem) -> Evaluation:
    groups = int(problem.group.max()) + 1
    z = np.zeros((problem.group.size, groups * 2), dtype=np.float64)
    for row, group_id in enumerate(problem.group):
        z[row, group_id * 2 : group_id * 2 + 2] = problem.random_design[row]
    lambda_ = np.kron(np.eye(groups), problem.block).astype(np.float64, copy=False)
    spec = ModelSpec.from_arrays(
        y=problem.y,
        x=problem.x,
        z=z,
        weights=problem.weights,
        offset=problem.offset,
    )
    result = evaluate_fixed_theta(spec, lambda_, kind=ObjectiveKind.REML)
    return Evaluation(
        result.objective,
        result.beta,
        result.u,
        result.penalized_residual_sum_squares,
    )


def median_seconds(function: Callable[[], Any], repeats: int = 3) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main() -> None:
    check_problem = make_problem(8)
    production = evaluate_production(check_problem)
    block = evaluate_block(check_problem)
    general_sparse = evaluate_sparse(check_problem)

    benchmark: list[dict[str, float | int]] = []
    for groups in (100, 1_000, 5_000):
        problem = make_problem(groups)
        q = groups * 2
        sparse_c = sparse_random_matrix(problem).T @ sparse_random_matrix(problem)
        benchmark.append(
            {
                "groups": groups,
                "n": int(problem.y.size),
                "q": q,
                "block_median_seconds": median_seconds(
                    lambda problem=problem: evaluate_block(problem)
                ),
                "sparse_lu_median_seconds": median_seconds(
                    lambda problem=problem: evaluate_sparse(problem)
                ),
                "dense_c_megabytes": q * q * 8 / 1_000_000,
                "sparse_c_nnz": int(sparse_c.nnz + q),
            }
        )

    maximum_correctness_error = max(
        abs(block.objective - production.objective),
        abs(general_sparse.objective - production.objective),
        float(np.max(np.abs(block.beta - production.beta))),
        float(np.max(np.abs(general_sparse.beta - production.beta))),
    )
    evidence = {
        "schema_version": "1.0.0",
        "correctness_pass": bool(maximum_correctness_error <= 1e-10),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "correctness_at_8_groups": {
            "block_objective_absolute_error": abs(
                block.objective - production.objective
            ),
            "sparse_objective_absolute_error": abs(
                general_sparse.objective - production.objective
            ),
            "block_beta_max_absolute_error": float(
                np.max(np.abs(block.beta - production.beta))
            ),
            "sparse_beta_max_absolute_error": float(
                np.max(np.abs(general_sparse.beta - production.beta))
            ),
        },
        "benchmark": benchmark,
        "decision": {
            "phase_1": "owned independent-group block Cholesky",
            "phase_2_candidate": "SciPy sparse LU only as a feasibility baseline",
            "reason": (
                "The block solve preserves the model structure, has bounded 2x2 "
                "factorizations here, and avoids a q-by-q dense allocation. General "
                "sparse support still needs a verified SPD Cholesky backend, ordering "
                "controls, symbolic reuse, wheel coverage, and log-determinant tests."
            ),
        },
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not evidence["correctness_pass"]:
        raise SystemExit("backend correctness comparison failed")


if __name__ == "__main__":
    main()
