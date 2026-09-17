# ADR 0015: E03 performance evidence and claim boundary

Status: accepted for E03 on 2026-09-16.

## Context

E03 requires reproducible time/memory reports and explicit ownership of remaining
risks. The plan also records an aspirational objective of running InstEval within
2× lme4. Kamino runs natively on macOS with NumPy/SciPy and Accelerate, while the
immutable lme4 oracle runs in a Linux ARM64 Docker VM with OpenBLAS. Kamino uses
SciPy bounded/Powell optimizers and lme4 uses NLopt BOBYQA. Those differences
make a speed ratio useful for diagnosis but invalid as a controlled comparative
performance claim.

## Decision

Separate three conclusions:

1. Correctness and resource gates are release evidence when the manifested cases,
   tolerances, ceilings, revisions, and artifact hashes pass.
2. Native Kamino and containerized lme4 timings on the same physical host are
   published only as observational measurements. They do not establish relative
   implementation performance.
3. The proposed InstEval-within-2× objective remains unverified. The observed
   E03 ratios are about 4.12–4.24× and must not be relabeled as a passing target.

Timing regressions above 20% trigger investigation only when rerun on the exact
hardware profile under comparable measurement conditions. Shared CI verifies
the committed reports and continues to enforce correctness/completion/resource
ceilings; it does not make timing-regression decisions.

## Consequences

E03 can close because it delivers reproducible raw reports, a hashed assessment,
and an owned risk register without converting an uncontrolled comparison into a
claim. A future matched runtime/BLAS/optimizer-policy experiment is required to
verify or reject the 2× objective. The general sparse SuperLU path remains a
measured optimization candidate, not a reason to weaken numerical tolerances or
replace the accepted correctness oracle.

Evidence is in
[`release-e03-performance-risk.md`](../evidence/release-e03-performance-risk.md)
and the machine-readable reports under [`benchmarks/results`](../../benchmarks/results).
