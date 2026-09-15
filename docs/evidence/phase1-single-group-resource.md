# Phase 1 single-group resource evidence

Date: 2026-09-15

Scope: the complete public Phase 1 single-group subset at one million
observations and 10,000 observed groups, under ML and REML.

## Decision and workload

The versioned manifest is `benchmarks/single_group_v1.json`. It generates
deterministic, balanced, caller-data-free weighted/offset models for a random
intercept, a four-column numeric/categorical fixed design with both formula and
argument offsets, a correlated numeric random intercept/slope, and independent
numeric intercept/slope terms sharing one group. The resulting dimensions are
`(n, p, q, d) = (1,000,000, 1, 10,000, 1)`,
`(1,000,000, 4, 10,000, 1)`, `(1,000,000, 2, 20,000, 3)`, and
`(1,000,000, 2, 20,000, 2)` respectively.

Each case must complete through `kamino.lmer`, reproduce its staged prepared
fit and conditional training prediction, contain no dense `Z`, stay below
2,000 decimal MB peak RSS, and complete each fit before the 300-second worker
deadline. A dense `Z` would require 80 GB for the intercept case and 160 GB for
the slope case and is neither allocated nor used as a fallback.

The fit path now assembles theta-independent group and fixed-effect cross
products once into an immutable block workspace. Optimizer evaluations reuse
that workspace while retaining the row-wise residual calculation that passed
the existing dense and pinned-lme4 numerical tolerances. The workspace arrays
occupy 8.24 MB, 8.48 MB, and 8.80 MB for the intercept, categorical fixed, and
both slope cases; the compact specification arrays occupy 48 MB, 72 MB, and
64 MB. These figures exclude Python labels, dataframe state,
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

The expanded full local pass used the working tree based on revision
`6099a1baff53952fa4c52d5ce70b36c437db68fa` and ran on a 16 GB, eight-core
Apple M1 Pro with macOS 15.1, CPython 3.12.14,
NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, and the locked environment.

| Structure | Kind | Cold fit | Warm median / p95 | Prepared fixed-theta median / p95 | Assembly | Optimization (evaluations) | Peak RSS |
|---|---|---:|---:|---:|---:|---:|---:|
| Random intercept | ML | 6.70 s | 6.63 / 6.71 s | 0.0485 / 0.0503 s | 0.103 s | 1.21 s (25) | 934 MB |
| Random intercept | REML | 6.79 s | 6.82 / 7.82 s | 0.0226 / 0.0258 s | 0.0546 s | 0.625 s (25) | 853 MB |
| Categorical fixed | ML | 5.30 s | 5.38 / 5.79 s | 0.0149 / 0.0178 s | 0.178 s | 0.400 s (26) | 1,161 MB |
| Categorical fixed | REML | 3.83 s | 3.71 / 3.78 s | 0.0144 / 0.0147 s | 0.148 s | 0.376 s (26) | 1,316 MB |
| Correlated slope | ML | 10.74 s | 10.85 / 10.95 s | 0.0323 / 0.0332 s | 0.173 s | 8.57 s (266) | 1,200 MB |
| Correlated slope | REML | 11.19 s | 11.31 / 11.42 s | 0.0308 / 0.0316 s | 0.169 s | 8.90 s (285) | 1,263 MB |
| Independent terms | ML | 8.57 s | 8.04 / 8.28 s | 0.0316 / 0.0328 s | 0.185 s | 5.71 s (179) | 1,302 MB |
| Independent terms | REML | 8.33 s | 8.40 / 8.44 s | 0.0318 / 0.0322 s | 0.169 s | 6.17 s (193) | 1,228 MB |

All eight correctness and resource checks passed. Staged and public objectives
and theta were identical; conditional predictions were byte-level numerically
identical to fitted values. Existing pinned-lme4 and independent dense tests ran
before the resource cases and passed.

[Hosted CI run 35008609955](https://github.com/joshuamyers22/kamino/actions/runs/35008609955)
repeated the then-current six-case covariance protocol at revision
`3aa689cbb9ee8ee8c91a6bac4435e015160b0079`. Every case passed and the retained
report recorded a maximum 647 MB peak RSS. Warm public-fit medians were 1.21/1.24
seconds for intercept ML/REML, 11.07/11.34 seconds for correlated-slope ML/REML,
and 7.77/7.71 seconds for independent-term ML/REML. The workflow now reads the
expanded eight-case manifest, including the categorical fixed design. Hosted
timings document completion behavior but are not a pinned performance baseline.

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
