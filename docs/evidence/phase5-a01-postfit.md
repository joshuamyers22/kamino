# A01 postfit capability evidence

Status: complete locally on 2026-09-16; hosted verification is pending the
candidate commit.

Kamino now exposes `fit.postfit()` plus explicit `adapt_kamino()`,
`adapt_statsmodels_ols()`, and `adapt_statsmodels_mixedlm()` entry points. The
immutable basis contract carries full and retained coefficient identity,
covariance identity, inference method, contrast estimability, and exact formula
design reconstruction. Contract tests reject wrong result classes, nonformula
statsmodels fits, coefficient/covariance/null-space mismatches, and invalid
covariance/DF combinations.

The pinned A01 fixture uses a deterministic unbalanced 2×2 factorial with a
numeric covariate and random intercept. lme4 2.0-6 supplies the ML fit and
emmeans 2.0.2 supplies coefficient-aligned reference functions, marginal means,
standard errors, and pairwise adjustments. Equal, proportional, outer, cells,
and flat weighting all pass; none, Holm, Bonferroni, and Sidak families pass.
Maximum differences are `6.58e-11` for beta, `3.31e-10` for beta covariance,
zero for reference linear functions, `8.60e-12` for marginal means, `6.91e-10`
for standard errors, and `1.23e-9` for adjusted p-values, below the declared
`2e-8` tolerance.

Tests additionally cover explicit user weights, `at` values, formula and
argument offsets, `by` grouping, unavailable non-estimable coefficients,
rank-deficient statsmodels fits, one- and multi-DF Wald tests, confidence
intervals, robust-covariance/asymptotic pairing, and random-intercept/slope
variance decomposition including covariance cross terms. Unsupported Tukey
adjustment, malformed grids, empty denominators, missing source data, and
misaligned offsets fail before returning results.

The statsmodels external contract covers OLS, WLS, and MixedLM at the exact
optional version 0.14.6. OLS/WLS nonrobust tests match statsmodels residual-t
results; robust OLS/WLS deliberately switches to asymptotic inference; MixedLM
uses only its fixed-effect covariance block. Reference-grid predictions agree
with direct statsmodels formula predictions.

The marginaleffects feasibility spike found no public Python adapter contract
for a native Kamino result; its documented path accepts supported statsmodels
formula model objects. No deceptive wrapper was added. Direct Kamino support is
therefore an explicit deferred integration, not an A01 compatibility claim.

The reviewed Linux ARM64 oracle image is public at
`ghcr.io/joshuamyers22/kamino-oracle@sha256:be04b2666543d4981a9d5236a1258418b05b3f7e7ae47b67ed9591ebf42a2dd6`.
The fixture, generator, Dockerfile, image, source commits, and license boundary
are hash-pinned in `oracle/manifest.json`. A01 completes the capability spike
inside Kamino; a separately installable postfit distribution and its release
checklist remain a later Phase 5 gate.

Local release verification passed `make check` with 380 tests and 90.73% branch
coverage, the clean core-wheel smoke test without statsmodels installed, and a
fresh full oracle regeneration followed by source/output/container digest
verification. The optional external-adapter suite passed against statsmodels
0.14.6 in the frozen development environment.
