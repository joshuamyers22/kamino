# Phase 1 compact formula-boundary evidence

Date: 2026-09-15

Scope: the accepted random-intercept formula and the numeric correlated-slope
formula `response ~ predictor + (1 + predictor | group)`.

## Representation

The production formula path stores the random design in `SingleGroupSpec` as an
immutable `int64` observation-to-group vector plus an n-by-k row-level random
design, where verified k is one or two. The number of random coefficients is
groups times k. The specification has no dense `Z` field.

The allowlisted parser establishes the grouped term before Formulae 0.5.4 is
called with only the fixed formula and without the grouping column. Consequently,
the third-party formula backend cannot construct an `n`-by-`q` grouped indicator
matrix. Kamino derives indices from the validated grouping labels and categorical
level order.

Dense indicator construction remains confined to an explicitly named helper in
the small Dyestuff block-versus-PLS test. The independent marginal oracle accepts
the compact specification and assembles its small `n`-by-`n` covariance directly.
Neither path is used by the public fitter.

## Checks

- Categorical level order `("C", "A", "B")` produces group indices
  `(1, 2, 0)` and matching labeled random coefficients.
- Spies verify Formulae receives `y ~ 1` for an intercept fit and
  `Reaction ~ Days` for Sleepstudy, with no grouping column or bar term.
- A 4,096-observation case with 4,096 distinct groups stores exactly 4,096
  `int64` group indices and exposes no dense random design.
- Compact arrays are copied, immutable, range checked, integer typed, and must
  cover every declared group.
- Dyestuff and Dyestuff2 ML/REML results, predictions, exact boundary behavior,
  and random-intercept fixed-theta comparisons remain unchanged.

This closes dense random-indicator materialization for the current production
formula subset. It does not establish the planned million-row timing/peak-RSS
target or broader formula semantics. Numeric random-slope encoding is covered by
the separate Sleepstudy evidence; general sparse structures remain outside this
claim.
