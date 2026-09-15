# Compatibility Contract

Primary profile: lme4-2.0.6-unstructured-gaussian-v1
Legacy profile: lme4-1.1.37-shared-subset-v1

Kamino may claim compatibility only for rows marked passing in a released,
machine-readable report. Formula matrices, fixed-parameter algebra, optimized
fits, and user-facing behavior are separate dimensions.

Phase 0 covers direct float64 arrays and proves the weighted ML/REML objective
against independent dense algebra and eight fixed-theta lme4 cases. The pinned
cases include weights, offsets, correlated random slopes, and singular/zero
covariance factors. The maximum observed objective difference is `7.11e-15`.

A restricted Formulae adapter now exposes the first Phase 1 public slice. The
claim is deliberately narrower than the six-case formula feasibility corpus:

| Public behavior | Current claim |
|---|---|
| Formula | One random intercept (`y ~ 1 + (1 | g)`), or one numeric predictor used as both a fixed and correlated random slope (`y ~ x + (1 + x | g)`) |
| Fit | Dyestuff ML and REML objective, beta, theta, variance estimates, modes, fitted values, and residuals pass pinned lme4 tolerances |
| Boundary fit | Dyestuff2 ML/REML returns exact theta and random variance zero, matching pinned lme4 |
| Correlated slope | Sleepstudy ML/REML objective, beta, theta, covariance, modes, fitted values, and residuals pass declared pinned lme4 tolerances |
| Prediction | Dyestuff and Sleepstudy population/conditional means pass for training, known groups, and explicitly allowed new groups |
| Boundary | Zero between-group variance is returned as a valid boundary fit |
| Errors | Unsupported formulas, invalid frames, unbracketed optima, evaluation exhaustion, and unseen conditional groups fail explicitly |
| Backend | Owned single-group block evaluator with batched 1×1/2×2 Cholesky; no q-by-q random-effects factorization |
| Formula storage | One immutable integer group index plus k row covariates per observation; no dense random-effects indicator matrix |
| Persistence | Atomic versioned prediction-only bundle; strict non-executable load; exact new-data prediction round trip |
| Runtime | R-free wheel; Formulae 0.5.4, NumPy, pandas, and SciPy are runtime dependencies |

Prediction requires an explicit `mode`. Known groups use fitted conditional
modes; allowed new groups receive a zero random contribution and a row-level
flag. Training offsets are retained, while new data for an argument-offset fit
must provide a new offset vector.

The local Dyestuff maximum absolute differences are zero at printed precision
for the optimized objective, `3.55e-9` for theta, and `1.04e-7` for predicted
means. Acceptance thresholds are `1e-8`, `1e-6`, and `1e-5`, respectively.
Independent dense algebra is also checked at each fitted theta.

Dyestuff2 matches the pinned ML/REML objectives and exact theta-zero boundary;
its maximum prediction difference is `1.78e-15`. The block evaluator matches
the dense PLS and marginal oracles at eight fixed-theta ML/REML cases. Formulae
receives only the fixed-effects formula; Kamino owns the compact group encoding.
A 4,096-observation case with 4,096 distinct levels verifies linear random-design
storage. Sleepstudy adds ten fixed-theta correlated/singular cases and complete
ML/REML fit and prediction comparisons. Kamino's fitted objective is no worse
than the pinned lme4 result; maximum absolute differences are `2.58e-8` for the
objective, `3.49e-5` for raw theta, `0.0384` for a random-covariance element, and
`6.55e-4` for conditional predictions. These are evaluated against declared
case tolerances and the optimizer identity remains explicit. Prediction-only
bundles preserve the supported fitted state and exact new-data predictions while
omitting responses and training rows; they do not support refit, training
prediction, or inference after reload. No million-row runtime or peak-memory
claim is made yet. Multiple predictors or random terms, nested/crossed terms,
broader formula semantics, and all inference remain unclaimed. Positive weights
and argument offsets are accepted by the fitter but do not yet have final-fit
lme4 coverage outside the synthetic slope fixture. See `PROJECT_PLAN.md` section
2 for the feature-level contract.
