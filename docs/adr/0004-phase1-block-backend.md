# ADR 0004: Phase 1 block backend

Status: implemented for single-group random intercepts/slopes; general sparse extension in ADR 0006
Date: 2026-09-14

## Decision

Implement an owned, batched block-Cholesky path for Phase 1's one verified
independent grouping structure. Preserve the direct dense implementation only as
a small-model check. Do not advertise a general sparse backend in Phase 1.

SciPy 1.18.1 sparse LU is a useful feasibility comparator and exactly reproduced
the spike criterion to floating-point tolerance. It is not the lme4-like SPD
Cholesky contract: permutation control, determinant accounting, symbolic reuse,
fill behavior, boundary cases, and wheel behavior still need their own evidence.
Phase 2 must select an installable SPD backend or explicitly narrow its platform
and structure claims.

## Evidence and consequences

The executable spike is `tools/backend_spike.py`; reviewed feasibility
measurements are in `docs/evidence/phase0-backend-spike.md`. The production
single-group implementation is `src/kamino/block.py`, with fixed-theta and
Dyestuff2 boundary evidence in `docs/evidence/phase1-dyestuff2-block.md` and
correlated-slope evidence in `docs/evidence/phase1-sleepstudy.md`.
Independent same-group term evidence is in
`docs/evidence/phase1-independent-terms.md`.

The runtime implementation aggregates group cross-products with batched
`bincount` operations, factorizes one k-by-k block per group, and factorizes the
fixed-effect Schur complement. The verified public values are k=1 and k=2.
The production formula boundary now supplies one integer group index per
observation and has no dense `Z` member. Formulae evaluates only the fixed part
of the accepted formula. Small-model dense construction remains explicit in
independent tests. The manifested resource gate now passes every public
covariance structure under ML and REML at one million observations and 10,000
groups. The
fit assembles theta-independent cross-products once and reuses them across
optimizer evaluations; see `docs/evidence/phase1-single-group-resource.md`.
Multiple covariance terms sharing that group retain one block-diagonal relative
covariance factor. Their random-design cross-products remain in the same group
solve, so independence does not incorrectly split the likelihood calculation.
