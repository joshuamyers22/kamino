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

A restricted Formulae adapter exposes the completed Phase 1 single-group slice,
the Phase 2 N03 general sparse slice, and F02 rank/categorical terms. The claim remains
narrower than general lme4 formula semantics:

| Public behavior | Current claim |
|---|---|
| Formula | Intercept plus additive numeric/categorical fixed effects, distinct pairwise `a * b`, and formula offsets; one ordinary numeric/categorical random intercept/slope, independent numeric terms with one shared group, or two-plus nested/crossed terms where categorical slopes are supported |
| Model frame | One subset-before-NA selection across response, fixed/random/group columns, weights, and both offset sources; explicit retained, omitted, and excluded row IDs |
| Fixed categories | Declared/observed level order with unused-level dropping and explicit treatment or sum coding; state is reused for prediction |
| Fixed rank | Pinned non-LAPACK QR drop policy at tolerance `1e-7`; full/retained names, pivot, dropped columns, and null-space basis are preserved |
| Estimability | Full-coordinate linear functions return estimate/SE only when estimable; non-estimable prediction rows are flagged and return `NaN` |
| Random categories | Ordinary bars support treatment/sum row encodings through explicit `random_contrasts=`; categorical `||` remains rejected |
| Fit | Dyestuff ML and REML objective, beta, theta, variance estimates, modes, fitted values, and residuals pass pinned lme4 tolerances |
| Boundary fit | Dyestuff2 ML/REML returns exact theta and random variance zero, matching pinned lme4 |
| Correlated slope | Sleepstudy ML/REML objective, beta, theta, covariance, modes, fitted values, and residuals pass declared pinned lme4 tolerances |
| Independent terms | Sleepstudy ML/REML double-bar and explicit split forms preserve a diagonal prior covariance while solving their nonzero joint cross-products; fits and predictions pass declared pinned lme4 tolerances |
| Nested/crossed terms | Pastes slash nesting and Penicillin crossing preserve lme4 term/level identity and coupled cross-products under ML and REML |
| Prediction | Numeric/categorical and rank-deficient population/conditional means pass for training, known groups, and explicitly allowed new groups; unknown fixed levels error |
| Simulation | Live fits support explicit unconditional new-random-effect simulation and conditional fixed-mode simulation with weighted residuals and replicate/purpose-separated PCG64DXSM streams |
| Response refit | A finite replacement response reuses the exact accepted rows, X/random design, weights, offsets, labels, estimation method, optimizer controls, and backend |
| Parametric bootstrap | Retained fixed effects support deterministic serial/parallel refits, complete outcome accounting, and descriptive percentile/basic intervals; nominal interval coverage is not claimed |
| Bootstrap ledger | Optional private atomic directory with canonical JSON and non-object response arrays; exact fit/source/plan/RNG/request identity is required to resume |
| Boundary | Zero between-group variance is returned as a valid boundary fit |
| Errors | Unsupported formulas, invalid frames, unbracketed optima, evaluation exhaustion, and unseen conditional groups fail explicitly |
| Backend | Owned single-group block evaluator with batched 1×1/2×2 Cholesky; coupled structures route to SciPy SuperLU with symmetric-mode controls, minimum-degree ordering, cached structural pattern, strict fill/resource limits, and verified determinant/solve behavior |
| Formula storage | One immutable integer group index plus k row covariates per term and observation; no dense random-effects indicator matrix |
| Persistence | Atomic versioned prediction-only bundle for Phase 1 models; nested/crossed, rank-deficient, and categorical-random saves fail closed pending the separately gated artifact-recovery schema |
| Resource scale | Manifested ML/REML public fits at 1M observations and 10k groups for all accepted covariance structures, below the declared 2,000 MB process-RSS ceiling |
| Crossed scale | InstEval ML/REML at 73,421 rows, 4,100 random coefficients, and 146,842 random-design nonzeros; no 2.41 GB dense Z allocation |
| Runtime | R-free wheel; Formulae 0.5.4, NumPy, pandas, SciPy, and threadpoolctl are runtime dependencies |

Prediction requires an explicit `mode`. Known groups use fitted conditional
modes; allowed new groups receive a zero random contribution and a row-level
flag. Training offsets are retained. Formula offsets are reevaluated from new
data, while an argument-offset fit always requires a new explicit offset vector.

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
prediction, or inference after reload. The expanded local million-row resource
gate peaks at 1,316 MB and reports timing as non-authoritative machine-specific
evidence; no lme4 speed ratio is claimed because the optimizer algorithms differ.
Independent Sleepstudy adds ten fixed-theta cases and two optimized fits; maximum
absolute differences are `4.72e-9` for objective, `1.32e-5` for theta, `0.0157`
for a covariance element, and `3.54e-4` for conditional prediction. Both formula
spellings fit identically, the covariance off-diagonal remains exactly zero, and
a weighted-offset synthetic case returns the exact zero slope boundary under ML
and REML. Four categorical fits (treatment/sum by ML/REML) add positive weights
and both offset sources: maximum errors are `1.67e-13` for objective, `3.91e-7`
for theta, `1.91e-8` for beta, and `1.87e-8` for prediction. The shared-frame
case matches lme4's subset-before-NA row order, X, weights, and total offset
exactly. Pastes and Penicillin add twelve fixed-theta cases with maximum
objective error `1.14e-13`, four final fits with maximum objective error
`8.52e-9`, and partial-new-group predictions within `1.16e-5`. InstEval adds
73,421-row ML/REML final fits with maximum objective error `6.12e-10`, theta
error `1.51e-7`, and beta error `6.88e-8`; the local resource run peaks at 357
MB. Categorical random slopes are now covered across grouping factors. F02 adds
four exact/threshold rank/drop
contracts, two rank-deficient fits, six categorical fixed-theta and six final
single-group fits, and two fixed-theta plus two final crossed categorical fits.
Maximum errors are `1.14e-13` at fixed theta, `1.75e-6` for optimized objective,
`9.48e-5` for random covariance, and `2.32e-5` for prediction. I01 adds four
stored-response ML/REML refits against pinned lme4 with maximum objective error
`5.68e-14`, plus a locked 10,000-draw moment and 1,000-refit assessment. All
1,000 refits are usable (93 valid singular fits), with a one-sided exact 95%
failure-rate upper bound of `0.002992`; serial and parallel results are exactly
identical. Numeric slopes across different grouping factors, categorical `||`,
transforms, no-intercept formulas, broader formula semantics, calibrated nominal
interval coverage, and Satterthwaite/KR/profile inference remain unclaimed. See
`PROJECT_PLAN.md` section 2 for the
feature-level contract.
