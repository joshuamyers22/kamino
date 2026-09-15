# Phase 2 F02 rank, estimability, and categorical evidence

Date: 2026-09-15

## Scope completed

- lme4 2.0-6 non-LAPACK QR column retention at tolerance `1e-7`.
- Full-to-retained coefficient identity, pivot/drop metadata, and normalized
  null-space basis.
- Explicit estimability for full-coordinate linear functions and prediction
  rows; dropped coefficients are unavailable rather than zero.
- Ordinary categorical random intercept/slopes with treatment or sum coding in
  the single-group block backend and across grouping factors in the sparse
  backend.
- Separate fixed `contrasts=` and `random_contrasts=` semantics, including the
  lmer fixed-sum/random-treatment case.
- Known/new random groups, reordered supported categories, unknown fixed-level
  rejection, and mathematically non-estimable new rows.

Categorical double-bar expansion, no-intercept random categorical terms, and
rank/categorical prediction bundles remain fail-closed.

## Independent evidence

The reviewed fixture is `oracle/fixtures/v1/f02_rank_categorical.json`, generated
by the immutable Linux ARM64 lme4 2.0-6 image recorded in
`oracle/manifest.json`.

| Evidence | Cases | Maximum absolute difference |
|---|---:|---:|
| Rank/drop contracts | 4 | Exact names, pivot, rank, and X hashes |
| Rank-deficient ML/REML fits | 2 | Objective `2.84e-14` locally |
| Categorical single-group fixed theta | 6 | Objective `1.14e-13` |
| Categorical single-group final fits | 6 | Objective `3.55e-7` |
| Single-group random covariance | 6 | `5.16e-5` |
| Single-group new-data prediction | 24 comparisons | `2.32e-5` |
| Crossed categorical fixed theta | 2 | Objective `2.85e-14` |
| Crossed categorical final fits | 2 | Objective `1.75e-6` |
| Crossed random covariance | 2 | `9.48e-5` |
| Crossed new-data prediction | 8 comparisons | `1.55e-5` |

Every fixed-theta categorical case also agrees with the independent dense PLS
route. The final-fit acceptance limits are `3e-6` for objective and `2e-4` for
random covariance and prediction. Singular covariance comparisons use the
implied covariance rather than non-unique raw factor coordinates.

## Local verification

`make check` passes formatting, lint, strict type checking, 283 tests with at
least 90% branch-aware coverage, and offline wheel/sdist builds. The clean-wheel
smoke test and hosted platform matrix are recorded separately when run.
