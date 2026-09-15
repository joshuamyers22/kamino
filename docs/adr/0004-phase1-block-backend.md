# ADR 0004: Phase 1 block backend

Status: implemented for the random-intercept public slice; general sparse backend deferred
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
random-intercept implementation is `src/kamino/block.py`, with fixed-theta and
Dyestuff2 boundary evidence in `docs/evidence/phase1-dyestuff2-block.md`.

The runtime implementation aggregates group cross-products with batched
`bincount` operations and factorizes only the fixed-effect Schur complement.
The restricted Formulae boundary still materializes a dense indicator matrix
for validation, so end-to-end scale and memory claims remain pending until that
adapter is replaced with a sparse/encoded boundary and benchmarked.
