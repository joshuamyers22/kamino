# Phase 1 single-group resource evidence

Date: 2026-09-15

Scope: the complete public Phase 1 single-group subset at one million
observations and 10,000 observed groups, under ML and REML.

## Decision and workload

The versioned manifest is `benchmarks/single_group_v1.json`. It generates
deterministic, balanced, caller-data-free weighted/offset models for a random
intercept and a correlated numeric random intercept/slope. The resulting
dimensions are `(n, p, q, d) = (1,000,000, 1, 10,000, 1)` and
`(1,000,000, 2, 20,000, 3)` respectively.

Each case must complete through `kamino.lmer`, reproduce its staged prepared
fit and conditional training prediction, contain no dense `Z`, stay below
2,000 decimal MB peak RSS, and complete each fit before the 300-second worker
deadline. A dense `Z` would require 80 GB for the intercept case and 160 GB for
the slope case and is neither allocated nor used as a fallback.

The fit path now assembles theta-independent group and fixed-effect cross
products once into an immutable block workspace. Optimizer evaluations reuse
that workspace while retaining the row-wise residual calculation that passed
the existing dense and pinned-lme4 numerical tolerances. The workspace arrays
occupy 8.24 MB and 8.80 MB for the two cases; the compact specification arrays
occupy 48 MB and 64 MB. These figures exclude Python labels, dataframe state,
result arrays, interpreter/runtime libraries, and transient workspaces, all of
which are included in measured process peak RSS.

## Measurement

The committed runner starts each scenario/objective combination in a separate
process with all advertised BLAS thread controls set to one. Input generation
is excluded. It records the first public fit as cold, five following fits as
warm samples, five prepared fixed-theta warm samples, parse/encoding, block
assembly, fixed-theta evaluation, optimization, prediction, optimizer calls,
and process peak RSS. Independent blocks require no symbolic-analysis phase;
inference remains unavailable in Phase 1 and is recorded as such.

The final full local pass at revision `deaa271c71254742f48a4eb0dd66d456aeab4ef2`
ran on a 16 GB, eight-core Apple M1 Pro with macOS 15.1, CPython 3.12.14,
NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, and the locked environment.

| Structure | Kind | Cold fit | Warm median / p95 | Prepared fixed-theta median / p95 | Assembly | Optimization (evaluations) | Peak RSS |
|---|---|---:|---:|---:|---:|---:|---:|
| Random intercept | ML | 1.29 s | 1.31 / 1.37 s | 0.0167 / 0.0286 s | 0.0433 s | 0.469 s (25) | 880 MB |
| Random intercept | REML | 1.14 s | 1.32 / 1.37 s | 0.0181 / 0.0269 s | 0.0545 s | 0.573 s (25) | 908 MB |
| Correlated slope | ML | 12.95 s | 12.54 / 12.81 s | 0.0326 / 0.0334 s | 0.214 s | 11.67 s (266) | 1,257 MB |
| Correlated slope | REML | 12.35 s | 13.02 / 13.28 s | 0.0424 / 0.0612 s | 0.218 s | 12.36 s (285) | 1,230 MB |

All four correctness and resource checks passed. Staged and public objectives
and theta were identical; conditional predictions were byte-level numerically
identical to fitted values. Existing pinned-lme4 and independent dense tests ran
before the resource cases and passed.

## Interpretation and limits

This evidence establishes bounded completion for the advertised single-group
models, not a general sparse, nested, crossed, or inference claim. Shared CI
reruns the full manifest and retains its JSON report for 30 days. It enforces
correctness, completion, and the deliberately loose peak-memory ceiling; its
timings are not regression thresholds. A greater than 20% time or memory change
must be investigated only against repeated results from pinned dedicated
hardware.

No lme4 speed ratio is reported. Pinned lme4 uses NLopt BOBYQA while Kamino's
verified Phase 1 paths use SciPy bounded minimization and Powell. Calling those
end-to-end timings comparable would violate the plan's matched-optimizer
contract. Statistical fidelity remains established by the pinned Dyestuff,
Dyestuff2, Sleepstudy, and fixed-theta corpus; a future comparative performance
claim requires a controlled same-hardware, same-BLAS, matched-control run.
