# ADR 0007: rank, estimability, and categorical random terms

Status: accepted for Phase 2 F02 on 2026-09-15.

## Context

lme4 drops rank-deficient fixed-effect columns with R's non-LAPACK `dqrdc2`
QR at tolerance `1e-7`. A generic SVD or LAPACK column-pivoted QR can retain a
different coefficient, which changes labels, contrasts, and new-data behavior.
Dropping the column without preserving the original coefficient space also
makes it impossible to distinguish estimable from non-estimable functions.

Categorical ordinary-bar terms expand to multiple row covariates per grouping
level. They must use the same compact block/sparse representation as numeric
slopes; a dense observation-by-random-coefficient indicator remains forbidden.
R's `contrasts=` argument applies to fixed effects, while random terms use the
factor's own contrast state. Kamino therefore needs separate explicit controls.

## Decision

- Port the `dqrdc2` column-moving policy: preserve source order and move a
  column right when its remaining norm is below `1e-7` times its original norm.
  Store the full names, retained/dropped indices, complete pivot, tolerance,
  and a normalized coefficient-aligned basis for `null(X_full)`.
- Fit only the retained full-rank matrix. Expose dropped full coefficients as
  unavailable (`NaN`), never as estimated zeros.
- Check every full-coordinate linear function against the stored null basis.
  Non-estimable functions return an explicit status with no estimate or SE.
  Non-estimable new-data rows return `estimable=False` and a `NaN` prediction.
- Keep `contrasts=` for fixed effects and add `random_contrasts=` for ordinary
  categorical random terms. Both accept the declared treatment/sum subset.
  Categorical `||` remains unsupported because its dummy-term splitting needs a
  separate reference contract.
- Encode categorical random rows with the owned serializable encoder and pass
  the resulting small `n × k` covariates to the existing block or sparse
  backend. Multiple grouping factors may include categorical slopes.
- Retain Powell for covariance vectors of at most three parameters, preserving
  the verified Phase 1/N03 path. Use bounded L-BFGS-B for larger vectors, where
  Powell demonstrated divergent searches on valid categorical covariance fits.

## Evidence and consequences

The pinned F02 corpus covers an exact numeric alias, an empty interaction cell,
a threshold-sensitive retained column, treatment/sum and mixed fixed/random
contrast states, and a categorical slope crossed with a random intercept.
Fixed-theta results also pass independent dense algebra. Raw Cholesky factors
are not treated as unique at singular fits; covariance matrices, modes, fitted
values, and predictions are the comparison targets.

Schema 1.2 bundles do not contain the full coefficient/null-space state or a
random categorical encoder. Saving either kind of F02 model therefore fails
closed until the separately gated artifact-recovery schema milestone.
