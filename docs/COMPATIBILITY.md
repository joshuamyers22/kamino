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
| Formula | Exactly one fixed intercept and one random intercept: `y ~ 1 + (1 | g)` |
| Fit | Dyestuff ML and REML objective, beta, theta, variance estimates, modes, fitted values, and residuals pass pinned lme4 tolerances |
| Boundary fit | Dyestuff2 ML/REML returns exact theta and random variance zero, matching pinned lme4 |
| Prediction | Dyestuff population and conditional means pass for training, known groups, and an explicitly allowed new group |
| Boundary | Zero between-group variance is returned as a valid boundary fit |
| Errors | Unsupported formulas, invalid frames, unbracketed optima, evaluation exhaustion, and unseen conditional groups fail explicitly |
| Backend | Owned random-intercept block evaluator; no q-by-q random-effects factorization |
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
the dense PLS and marginal oracles at eight fixed-theta ML/REML cases. The
formula adapter still materializes a dense validation matrix, so no large-scale
formula-path claim is made yet. Safe model bundles, fixed-effect predictors,
random slopes, multiple/nested/crossed terms, broader formula semantics, and all
inference remain unclaimed. Positive weights and argument offsets are accepted
by the fitter but do not yet have final-fit lme4 oracle coverage. See
`PROJECT_PLAN.md` section 2 for the feature-level contract.
