# Phase 1 compact formula-boundary evidence

Date: 2026-09-15

Scope: the accepted `response ~ 1 + (1 | group)` public formula only.

## Representation

The production formula path stores the random design in `RandomInterceptSpec`
as an immutable `int64` observation-to-group vector. Its length is `n`; the
number of random coefficients is the number of canonical observed levels. The
specification has no dense `Z` field.

The allowlisted parser establishes that the only grouped term is a random
intercept. Formulae 0.5.4 is then called with only `response ~ 1` and a
response-only frame. Consequently, the third-party formula backend cannot
construct an `n`-by-`q` grouped indicator matrix. Kamino derives indices from
the validated grouping labels and categorical level order.

Dense indicator construction remains confined to an explicitly named helper in
the small Dyestuff block-versus-PLS test. The independent marginal oracle accepts
the compact specification and assembles its small `n`-by-`n` covariance directly.
Neither path is used by the public fitter.

## Checks

- Categorical level order `("C", "A", "B")` produces group indices
  `(1, 2, 0)` and matching labeled random coefficients.
- A spy verifies Formulae receives `y ~ 1`, a one-column response frame, and no
  bar term.
- A 4,096-observation case with 4,096 distinct groups stores exactly 4,096
  `int64` group indices and exposes no dense random design.
- Compact arrays are copied, immutable, range checked, integer typed, and must
  cover every declared group.
- Dyestuff and Dyestuff2 ML/REML results, predictions, exact boundary behavior,
  and eight fixed-theta block comparisons remain unchanged.
- The complete local gate passes 105 tests with 95.98% branch-aware coverage,
  strict typing and linting, and source/wheel builds.

This closes dense random-indicator materialization for the current production
formula subset. It does not establish the planned million-row timing/peak-RSS
target, random-slope encoding, general sparse structures, or broader formula
semantics.
