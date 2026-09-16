# Changelog

## Unreleased

- Added a versioned postfit linear-function contract with full/retained labels,
  null-space estimability, immutable covariance identity, and inseparable
  covariance/DF policy for asymptotic, residual-t, Satterthwaite, and
  Kenward–Roger inference.
- Added bounded reference grids, marginal means, pairwise contrasts, tidy
  coefficient results, and variance decomposition with explicit weighting,
  offset, adjustment, and unavailable-result behavior.
- Added explicit optional statsmodels 0.14.6 OLS/WLS and MixedLM adapters plus
  a pinned lme4 2.0-6/emmeans 2.0.2 unbalanced-factorial fixture and public
  immutable oracle image.
- Added bounded Kenward–Roger adjusted covariance and scaled F tests with ML-to-
  REML refit provenance, full pbkrtest intermediates, estimability/rank handling,
  explicit prior-weight/boundary refusal, and dense-memory ceilings.
- Added lme4-style named ML likelihood profiles for random SD/correlation,
  residual scale, and retained fixed effects, including nuisance reoptimization,
  optional variance/covariance presentation, adaptive traces, direct target
  evaluation, and explicit boundary/unbounded/nonmonotone status.
- Added pinned pbkrtest 0.5.5 and lme4 2.0-6 I03 fixtures plus a passing locked
  1,000-replicate KR type-I/profile-coverage assessment.
- Added full `(theta, sigma)` unprofiled-deviance Satterthwaite derivatives,
  reusable derivative diagnostics, estimable one-DF t tests, and rank-aware
  multi-DF F tests across block and general sparse fits.
- Added pinned lmerTest 3.1-3 derivative/test fixtures, explicit boundary and
  curvature refusal, and a passing locked 2,000-replicate one-/two-DF
  calibration assessment.
- Added explicit conditional/unconditional Gaussian simulation with weighted
  residuals and replicate/purpose-separated PCG64DXSM streams.
- Added exact response refitting and retained-fixed-effect parametric bootstrap
  across the block and coupled sparse backends, including bounded workers and
  exact serial/parallel replay.
- Added private resumable bootstrap ledgers with atomic canonical metadata,
  non-pickle response arrays, strict identity/integrity/resource validation,
  complete outcome accounting, and exact one-sided failure bounds.
- Added four stored-response refits against pinned lme4 and a locked 10,000-draw,
  1,000-refit statistical assessment; percentile/basic intervals remain
  descriptive pending separate nominal-coverage calibration.
- Added lme4-compatible non-LAPACK QR rank dropping with full/retained
  coefficient identity, null-space estimability checks, unavailable dropped
  coefficients, and explicit non-estimable prediction rows.
- Added treatment/sum categorical ordinary-bar terms through the block and
  crossed sparse backends, with separate fixed/random contrast controls and a
  pinned fixed-theta, ML/REML, covariance, and new-data corpus.
- Added bounded L-BFGS-B for covariance vectors above three parameters while
  retaining the verified Powell path for smaller vectors.
- Added a verified general sparse backend for coupled nested/crossed random-
  intercept terms while preserving the single-group block fast path.
- Added lme4-pinned Pastes and Penicillin term-order, fixed-theta, ML/REML fit,
  boundary, mode, and partial-new-group prediction evidence.
- Added an isolated InstEval ML/REML scale gate at 73,421 rows and 4,100 random
  coefficients with sparse-fill, peak-RSS, dense-Z absence, and oracle checks.
- Added configurable sparse preflight/fill limits and fail-closed nested/crossed
  bundle behavior pending the separately gated artifact-recovery schema.
- Added one shared model frame for response, fixed/random/group columns,
  weights, formula/argument offsets, Boolean subset selection, and explicit
  fail/omit missing-row policy with preserved row identities.
- Added multiple numeric/categorical fixed effects, treatment and sum contrasts,
  pairwise `*` expansion, and deterministic new-data encoding, verified through
  four weighted/offset ML/REML fits and predictions against pinned lme4.
- Versioned prediction bundles as schema 1.2 to preserve fixed-effect terms,
  categorical levels/contrasts, and formula-offset evaluation while retaining
  fail-closed schema-1.0/1.1 loading.
- Added independent numeric intercept and slope terms sharing one grouping
  factor through both `(1 + x || g)` and its explicit split, with pinned lme4
  ML/REML fits, predictions, fixed-theta algebra, and exact boundary evidence.
- Versioned prediction bundles as schema 1.1 to preserve covariance-term
  boundaries while retaining fail-closed schema-1.0 loading.
- Added the manifested one-million-observation/10,000-group ML/REML resource
  gate for all public single-group covariance structures, with peak-RSS CI
  enforcement and reusable theta-independent block assembly.
- Added atomic, versioned prediction-only model bundles with non-executable
  payloads, strict integrity/resource validation, and installed-wheel round trips.
- Added end-to-end ML/REML fitting and prediction for one correlated numeric
  random intercept/slope, verified on Sleepstudy against a pinned lme4 fixture.
- Generalized the compact formula representation and block-Cholesky evaluator
  to `k` within-group random covariates while retaining O(nk + groups*k²)
  storage and no q-by-q factorization.
- Treat numerically singular vector-optimizer trial points as infeasible so a
  valid fit is not aborted by exploratory covariance searches.
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
