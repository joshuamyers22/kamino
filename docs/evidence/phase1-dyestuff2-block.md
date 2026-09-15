# Phase 1 Dyestuff2 block-backend evidence

Date: 2026-09-15

Reference: lme4 2.0-6, source commit
`4aa26a91f9e676e9409f6cd8163ae92654ef1e7e`

Model: `Yield ~ 1 + (1 | Batch)`

Data: lme4 generated `Dyestuff2`, 30 observations in six batches

## Boundary fits

| Quantity | Kamino ML | lme4 ML | Kamino REML | lme4 REML |
|---|---:|---:|---:|---:|
| Objective (`-2 logLik`) | 162.87303665382575 | 162.87303665382575 | 161.82827781228846 | 161.82827781228846 |
| Log likelihood | -81.43651832691287 | -81.43651832691287 | -80.91413890614423 | -80.91413890614423 |
| Fixed intercept | 5.6655999999999995 | 5.6656000000000013 | 5.6655999999999995 | 5.6656000000000013 |
| Relative theta | 0 | 0 | 0 | 0 |
| Residual sigma | 3.6532313513746524 | 3.653231351374652 | 3.7156842744757266 | 3.7156842744757266 |
| Random-intercept variance | 0 | 0 | 0 | 0 |
| Residual variance | 13.346099306666668 | 13.346099306666666 | 13.806309627586208 | 13.806309627586206 |

Both Kamino fits report successful convergence and a structured boundary state.
The accepted parameter is exactly `theta=0`, rather than a small positive
approximation, and every conditional random mode is exactly zero. The optimizer
diagnostic names `random-intercept-block-cholesky` as the execution backend.

The objective and theta errors are exactly zero. The maximum absolute prediction
error is `1.78e-15`. At the boundary, an independent fixed-model calculation
confirms the sample mean and the ML/REML residual-variance divisors; the dense
marginal-covariance oracle independently confirms the complete objective.

## Predictions

For both ML and REML, training, known-batch, population, and conditional means
are `5.6656` subject only to floating-point representation. Conditional and
population predictions are identical because the fitted random variance and all
conditional modes are zero. An explicitly allowed unseen batch also receives a
zero random contribution and a row-level new-group flag.

## Block evaluator

The public fitter now evaluates scalar theta with an owned batched
random-intercept block algorithm. It forms one scalar penalized block per group
and factorizes only the `p`-by-`p` fixed-effect Schur complement; it does not form
or factor a `q`-by-`q` random-effects normal matrix.

Eight fixed-theta ML/REML cases at theta values `0`, `0.25`, `0.75`, and `2`
match the original dense PLS and independent marginal oracles. A separate case
covers unsorted groups, positive nonuniform weights, and offsets. The formula
adapter still materializes a dense indicator matrix for validation, so large
formula-driven fits remain unclaimed until that boundary becomes sparse/encoded.

This evidence extends the random-intercept slice only. Random slopes, multiple
terms, safe model bundles, and inference remain outside the compatibility claim.

The complete local gate runs 96 tests with 95% branch-aware coverage, strict
typing and linting, oracle hash verification, and source/wheel builds.
