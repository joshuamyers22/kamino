# ADR 0006: General sparse backend

Status: accepted and implemented for ordinary nested/crossed random intercepts
Date: 2026-09-15

## Decision

Use the SuperLU implementation shipped in the supported SciPy wheels as
Kamino's first general sparse factorization backend. Assemble the weighted
random design and `C = I + Lambda' Z' W Z Lambda` as sparse matrices, retain the
theta-independent structural pattern, and factor numeric `C` with
`MMD_AT_PLUS_A`, pivot threshold zero, equilibration disabled, and symmetric
mode enabled. The backend owns sparse assembly, factorization, permutation,
solve, log determinant, fill reporting, and allocation/failure translation.

This is a sparse LU factorization of a symmetric positive-definite system, not
CHOLMOD and not an SPD-Cholesky compatibility claim. Kamino uses the absolute U
diagonal for `log|C|`; fixed-theta tests independently compare it with dense
Cholesky and pinned lme4. SciPy does not expose a supported split symbolic/
numeric SuperLU API, so the full structural pattern is cached and preflighted,
while SuperLU repeats its internal ordering analysis for each numeric factor.
That limitation is explicit rather than hidden behind an unsupported private
interface.

The proven independent single-group structures continue to use the Phase 1
block backend. Only coupled multi-factor random-intercept models route to the
general sparse backend. Random slopes across different grouping factors and
categorical random terms remain fail-closed until their own formula and oracle
corpora pass.

## Evidence and consequences

Pastes establishes parent-child coupling and lme4-compatible slash expansion.
Penicillin establishes crossed coupling. Twelve fixed-theta ML/REML cases cover
interior, partial-zero, and all-zero covariance vectors and agree with both
pinned lme4 and the independent dense PLS evaluator. Four optimized fits and
known/partially-new predictions pass their declared tolerances.

InstEval exercises 73,421 observations, 4,100 random coefficients, and 146,842
stored random-design nonzeros. The accepted factor contains 1,036,622 nonzeros;
the forbidden dense random design would require 2.41 GB. The local manifested
ML/REML run completes in 26 seconds per fit and peaks at 357 decimal MB. These
timings are machine-specific; shared CI enforces only the 180-second and 1,500
MB ceilings. See `docs/evidence/phase2-general-sparse.md`.

No new binary dependency is introduced: SciPy is already required and publishes
wheels for the supported Python/platform matrix. Dependency license and wheel
evidence are recorded in `docs/SOURCE_AND_LICENSE_INVENTORY.md`. A future
CHOLMOD backend requires a new ADR covering wheels, linking, redistribution,
notices, failure behavior, and parity; it is not implied by this decision.
