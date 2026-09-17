# E03 reproducible performance and remaining-risk evidence

Status: complete on 2026-09-16 for protocol commit
[`4db3174`](https://github.com/joshuamyers22/kamino/commit/4db31746c979a7b0924f3f46008c24df666663e3).

This record follows the production-project-template performance-experiment
format at template commit `6526db479fced81463d1a97e25ff3ceefbe9eee2`.

## Decision and hypothesis

- Decision: whether E03 has reproducible fit-time/peak-memory evidence and owned
  residual risks; whether any lme4-relative performance claim is justified.
- Product path: formula-to-fit for four single-group structures and the crossed
  InstEval general-sparse model, under ML and REML.
- Hypothesis: all manifested fits preserve pinned lme4 fidelity and stay within
  Kamino resource ceilings. Cross-runtime timing ratios are exploratory.
- Control: digest-pinned lme4 2.0-6 with fixed NLopt BOBYQA controls.
- Invariants: no tolerance weakening, no dense random indicator, one BLAS thread,
  exact protocol revision, bounded workers, and no comparative claim from an
  unmatched runtime stack.

## Artifact and environment identity

- Protocol revision: `4db31746c979a7b0924f3f46008c24df666663e3`.
- Kamino: CPython 3.12.14, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5,
  `uv.lock` SHA-256 `421c0016…8fe6edb`.
- lme4: R 4.5.2, lme4 2.0.6, Matrix 1.7-4, nloptr 2.2.1, immutable
  Linux/ARM64 image `sha256:17e45268…eb4d`.
- Host: Apple M1 Pro, 8 cores (6 performance/2 efficiency), 16 GB, macOS
  15.1 (24B83). Docker VM: Linux ARM64, 8 logical CPUs, 8.32 GB memory.
  The non-sensitive machine record is
  [`apple_m1_pro_8c_16gb_2026_v1.json`](../../benchmarks/hardware/apple_m1_pro_8c_16gb_2026_v1.json).
- Threads: `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
  `VECLIB_MAXIMUM_THREADS` fixed to 1.
- Workloads: deterministic 100,000-row/1,000-group single-group generator and
  pinned 73,421-row InstEval fixture with 4,100 random coefficients and 146,842
  stored random-design nonzeros.
- Measurement limitations: CPU affinity, frequency, thermal state, and all
  background activity were not isolated. Kamino used native macOS/Accelerate;
  lme4 used Docker Linux/OpenBLAS. Optimizers differ.

## Measurement design

Each case runs in a fresh worker. The first fit is recorded as cold and later
fits as normally warm. Single-group cases use six total fits and five warm
fixed-theta evaluations; InstEval uses five total fits and five fixed-theta
evaluations. Reports retain every sample, median, p95, peak RSS, n/p/q/d/nnz,
optimizer/evaluation identity, parsing/encoding, assembly, fixed-theta,
optimization status, symbolic-analysis status, prediction where available, and
the fact that inference was outside this fit benchmark.

Python uses `time.perf_counter()` and `getrusage(RUSAGE_SELF).ru_maxrss`; R uses
elapsed process timing and Linux `VmHWM`. Setup and data construction occur in
each worker but outside the isolated staged timings; public end-to-end fits
include formula parsing, assembly, and optimization. SuperLU symbolic work is
included in factorization because SciPy exposes no supported split phase.

Raw reports and hashes are committed as:

- [Kamino single-group](../../benchmarks/results/e03_kamino_single_group_v1.json)
- [Kamino InstEval](../../benchmarks/results/e03_kamino_general_sparse_v1.json)
- [lme4 control](../../benchmarks/results/e03_lme4_v1.json)
- [hashed assessment](../../benchmarks/results/e03_assessment_v1.json)

## Results

| Workload | Kind | Kamino median / p95 (s) | Kamino peak RSS (MB) | lme4 median / p95 (s) | Observed ratio |
|---|---|---:|---:|---:|---:|
| Random intercept | ML | 0.183 / 0.193 | 269 | 0.361 / 0.472 | 0.51× |
| Random intercept | REML | 0.196 / 0.197 | 250 | 0.362 / 0.461 | 0.54× |
| Fixed categorical/weights/offsets | ML | 0.350 / 0.357 | 317 | 0.520 / 0.646 | 0.67× |
| Fixed categorical/weights/offsets | REML | 0.344 / 0.353 | 287 | 0.482 / 0.595 | 0.72× |
| Correlated intercept/slope | ML | 1.490 / 1.511 | 247 | 1.340 / 1.640 | 1.11× |
| Correlated intercept/slope | REML | 1.484 / 1.511 | 277 | 1.638 / 2.014 | 0.91× |
| Independent intercept/slope | ML | 1.166 / 1.169 | 277 | 0.845 / 1.176 | 1.38× |
| Independent intercept/slope | REML | 1.122 / 1.150 | 270 | 0.876 / 1.193 | 1.28× |
| InstEval crossed sparse | ML | 25.539 / 26.412 | 432 | 6.027 / 6.353 | 4.24× |
| InstEval crossed sparse | REML | 25.362 / 26.109 | 437 | 6.155 / 6.166 | 4.12× |

p99/p99.9 and throughput are not reported: five or six samples do not support
meaningful tail estimates, and this experiment measures isolated fits rather
than a service queue. lme4 peak RSS ranged from 306 to 539 MB; it is retained in
the raw report but is not directly compared because container/process accounting
differs.

## Correctness and compatibility

All ten dimension identities and cross-runtime fits passed. Across the corpus,
maximum absolute discrepancies were:

| Quantity | Maximum | Predeclared ceiling |
|---|---:|---:|
| Objective | `7.53e-7` | `1e-5` single-group / `1e-6` InstEval |
| Theta | `5.79e-5` | `2e-4` single-group / `1e-6` InstEval |
| Beta | `6.87e-8` | `1e-5` single-group / `1e-6` InstEval |
| Sigma | `2.34e-8` | `1e-5` single-group / `1e-7` InstEval |

All Kamino internal staged-objective, conditional-prediction, dense-Z absence,
time, and memory checks passed. All lme4 workers converged without recorded
warnings. `make check` passed 396 tests at 91.02% coverage before measurement.

## Decision

- Conclusion: accept E03 correctness/resource reporting and risk ownership.
- Comparative performance: no claim. The observed ratios are diagnostic only.
- InstEval within 2× lme4: unverified and not met by this observation; do not
  advertise it as supported performance.
- Regression policy: investigate >20% time or memory movement only after an
  exact-profile rerun; shared CI verifies report integrity and resource gates.
- Risks/response: [remaining-risk register](../REMAINING_RISKS.md), owned by
  Joshua Myers with dated reviews and closure evidence.
- Review: owner review on 2026-09-16. This is not independent statistical or
  release approval; beta/stable publication remains blocked on named reviewers.
