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

The first full local pass at revision `e2dfe16cdca5c38588149ab6431f9e57f73d3c44`
ran on a 16 GB, eight-core Apple M1 Pro with macOS 15.1, CPython 3.12.14,
NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, and the locked environment. That
three-sample preliminary pass produced the following stable scale; the hosted
artifact uses the manifest's final one-cold/five-warm protocol.

| Structure | Kind | Public fit median / p95 | Prepared fixed-theta median / p95 | Assembly | Optimization (evaluations) | Peak RSS |
|---|---|---:|---:|---:|---:|---:|
| Random intercept | ML | 0.942 / 0.966 s | 0.0124 / 0.0146 s | 0.0339 s | 0.351 s (25) | 768 MB |
| Random intercept | REML | 0.940 / 0.943 s | 0.0150 / 0.0153 s | 0.0347 s | 0.379 s (25) | 813 MB |
| Correlated slope | ML | 9.18 / 9.24 s | 0.0306 / 0.0315 s | 0.177 s | 8.25 s (266) | 1,150 MB |
| Correlated slope | REML | 9.82 / 9.86 s | 0.0306 / 0.0311 s | 0.158 s | 8.91 s (285) | 1,148 MB |

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
