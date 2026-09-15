# Phase 1 Dyestuff public-API evidence

Date: 2026-09-15

Reference: lme4 2.0-6, source commit
`4aa26a91f9e676e9409f6cd8163ae92654ef1e7e`

Model: `Yield ~ 1 + (1 | Batch)`

Data: lme4 `Dyestuff`, 30 observations in six batches

The results below come from `kamino.lmer()` after regenerating the accepted
fixture with the immutable Linux ARM64 oracle in `oracle/manifest.json`.

## Fits

| Quantity | Kamino ML | lme4 ML | Kamino REML | lme4 REML |
|---|---:|---:|---:|---:|
| Objective (`-2 logLik`) | 327.3270598811351 | 327.3270598811351 | 319.65427684225745 | 319.65427684225745 |
| Log likelihood | -163.66352994056754 | -163.66352994056754 | -159.82713842112872 | -159.82713842112872 |
| Fixed intercept | 1527.4999999999998 | 1527.4999999999998 | 1527.5000000000007 | 1527.4999999999998 |
| Relative theta | 0.7525806926269722 | 0.7525806952731132 | 0.8483237916224261 | 0.8483237832570345 |
| Residual sigma | 49.51009997671826 | 49.510099950987836 | 49.51009989946356 | 49.510099965333865 |
| Random-intercept variance | 1388.333334297845 | 1388.3333426178185 | 1764.0500366615372 | 1764.0500065645801 |
| Residual variance | 2451.249999704637 | 2451.249997156806 | 2451.2499920548617 | 2451.2499985773525 |

The maximum absolute objective difference is zero at the printed precision.
The maximum theta difference is `8.37e-9`; the accepted theta tolerance is
`1e-6`. Independent dense marginal algebra also matches Kamino at each fitted
theta to `1e-10` for the objective, beta, and residual variance.

## Predictions

Conditional predictions for one new row in each known batch are:

| Batch | ML | REML |
|---|---:|---:|
| A | 1510.87177837556 | 1509.8931485572605 |
| B | 1527.8695160360985 | 1527.8912633653943 |
| C | 1554.4746706352023 | 1556.062225673778 |
| D | 1505.6985538701788 | 1504.4154614417414 |
| E | 1581.079825234306 | 1584.2331879821613 |
| F | 1485.0056558486535 | 1482.5047129796653 |

Population prediction is the fixed intercept (`1527.5`, subject to floating
roundoff) for both known and unseen batches. Conditional prediction rejects an
unseen batch by default; with `allow_new_groups=True`, its unknown random
contribution is zero, its mean is also `1527.5`, and the returned row is flagged
as a new group. Across all pinned prediction cases, the maximum absolute Kamino
versus lme4 difference is `2.44e-7`, below the `1e-5` acceptance threshold.

## Reproduction gates

`make check` runs strict linting, typing, 71 tests, 96% branch-aware coverage,
and artifact builds. `make wheel-smoke` installs the wheel off-tree and performs
a public fit and conditional known/new-group prediction. The oracle fixture and
generator hashes are verified by `tools/verify_oracle_output.py`.

This evidence supports only the compatibility boundary documented in
`docs/COMPATIBILITY.md`; it does not complete the full Phase 1 gate.
