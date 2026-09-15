# Phase 2 general sparse evidence

Date: 2026-09-15
Scope: N03 ordinary nested/crossed random-intercept structures
Reference: lme4 2.0-6, source commit
`4aa26a91f9e676e9409f6cd8163ae92654ef1e7e`

## Structural and numerical evidence

The public adapter accepts explicit crossed terms such as
`(1 | plate) + (1 | sample)` and slash nesting such as `(1 | batch/cask)`.
Slash expansion, grouping-term ordering, factor levels, and sparse Z identity
match pinned lme4. Each observation stores one group index and one intercept
value per term; production code never constructs a dense observation-by-random-
coefficient indicator matrix.

Pastes and Penicillin contribute twelve fixed-theta cases across ML/REML and
interior, partial-zero, and all-zero covariance vectors. Maximum discrepancies
are:

| Quantity | Maximum absolute error |
|---|---:|
| Objective | `1.14e-13` |
| Fixed coefficients | `1.67e-13` |
| Spherical random modes | `1.46e-13` |

The same cases independently agree with Kamino's dense PLS evaluator. Four
optimized Pastes/Penicillin ML/REML fits have maximum objective discrepancy
`8.52e-9`, raw-theta discrepancy `1.04e-4`, beta discrepancy `4.59e-12`, and
prediction discrepancy `1.16e-5`. Kamino's Powell objective is no worse than
the pinned NLopt result within the declared `1e-7` objective tolerance. Known
groups and partially new nested/crossed combinations preserve per-term zero-
mode behavior and row-level new-group flags.

## InstEval scale gate

`benchmarks/general_sparse_v1.json` runs the actual lme4 InstEval crossed model
under ML and REML in isolated workers. Local macOS arm64/Python 3.12 results:

| Metric | ML | REML |
|---|---:|---:|
| Rows / fixed columns / random coefficients | 73,421 / 28 / 4,100 | same |
| Public fit seconds | 25.73 | 25.38 |
| Peak RSS, decimal MB | 336.61 | 356.78 |
| Objective absolute error vs lme4 | `6.12e-10` | `1.17e-10` |
| Theta maximum absolute error | `1.14e-7` | `1.51e-7` |
| Beta maximum absolute error | `5.79e-8` | `6.88e-8` |
| Accepted sparse-factor nonzeros | 1,036,622 | 1,036,622 |

The compact random design has 146,842 stored nonzeros. A dense Z would require
2,408,208,800 bytes and is absent. Both isolated cases pass the 1,500 MB and
180-second ceilings. Timings are observational outside dedicated hardware.

## Boundaries

The backend is SciPy SuperLU in symmetric mode, not CHOLMOD. Its structural
pattern and limits are cached, but SciPy repeats internal symbolic ordering per
numeric factor because it exposes no supported split symbolic/numeric API.
Single-group models retain the block fast path. Multi-factor random slopes,
categorical random terms, and nested/crossed prediction bundles remain
fail-closed. The bundle limitation belongs to the later Phase 2 full artifact-
recovery milestone and is not counted as N03 evidence.
