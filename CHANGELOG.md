# Changelog

## Unreleased

- Added the manifested one-million-observation/10,000-group ML/REML resource
  gate for both public single-group structures, with peak-RSS CI enforcement and
  reusable theta-independent block assembly.
- Added atomic, versioned prediction-only model bundles with non-executable
  payloads, strict integrity/resource validation, and installed-wheel round trips.
- Added end-to-end ML/REML fitting and prediction for one correlated numeric
  random intercept/slope, verified on Sleepstudy against a pinned lme4 fixture.
- Generalized the compact formula representation and block-Cholesky evaluator
  to `k` within-group random covariates while retaining O(nk + groups*k²)
  storage and no q-by-q factorization.
- Treat numerically singular vector-optimizer trial points as infeasible so a
  valid fit is not aborted by exploratory Powell evaluations.
- Replaced dense random-intercept indicator construction at the production
  formula boundary with a compact immutable observation-to-group encoding;
  Formulae now evaluates only the fixed-effects formula.
- Promoted the owned random-intercept block-Cholesky evaluator into the public
  fit path and added fixed-theta parity checks against dense PLS and independent
  marginal algebra.
- Added pinned Dyestuff2 ML/REML fixtures and public-API tests that accept the
  exact singular theta-zero boundary, zero random modes, and matching
  population/conditional predictions.
- Added the first public `lmer()` vertical slice for one random-intercept model,
  with ML/REML optimization, immutable labeled results, structured convergence
  outcomes, and explicit population/conditional prediction.
- Added pinned lme4 2.0-6 Dyestuff fit and prediction evidence, independent
  dense checks at fitted theta, new-group and offset semantics, and fixture
  provenance/licensing records.
- Extended clean-wheel smoke testing to perform a real public fit and prediction.
- Began Phase 0 with production scaffolding, compatibility records, and an
  independent fixed-parameter numerical walking skeleton.
- Pinned and built the lme4 2.0-6 R oracle; accepted eight weighted ML/REML
  fixed-theta fixtures and two optimized boundary fits covering offsets and
  covariance boundaries.
- Selected a restricted Formulae adapter and Phase 1 block-Cholesky backend from
  executable feasibility spikes; documented their unclaimed cases.
- Added strict local checks, clean-wheel smoke installation, and a production
  template-derived GitHub Actions runtime matrix.
- Added a six-case exact formula matrix corpus covering ordered treatment
  contrasts, fixed interactions, numeric `||`, nesting, and shared missing rows.
- Adopted the MIT License by owner decision and recorded the GitHub project URLs.
- Made the clean-environment wheel smoke test resolve runtime dependencies from
  the package index, avoiding a false assumption that CI caches contain registry
  metadata for every supported Python and operating system.
- Excluded the pinned Formulae adapter and R oracle image from routine Dependabot
  version bumps; both require explicit compatibility evidence before upgrades.
- Published the reviewed Linux ARM64 R/lme4 oracle to GHCR and recorded its
  immutable digest, platform, visibility, and publication time in the manifest.
