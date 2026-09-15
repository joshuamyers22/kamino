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
| Objective (`-2 logLik`) | 327.3270598811351 | 327.3270598811351 | 319.6542768422575 | 319.65427684225745 |
| Log likelihood | -163.66352994056754 | -163.66352994056754 | -159.82713842112875 | -159.82713842112872 |
| Fixed intercept | 1527.5000000000011 | 1527.4999999999998 | 1527.5000000000002 | 1527.4999999999998 |
| Relative theta | 0.7525806974470957 | 0.7525806952731132 | 0.848323786799286 | 0.8483237832570345 |
| Residual sigma | 49.51009992984855 | 49.510099950987836 | 49.51009993744166 | 49.510099965333865 |
| Random-intercept variance | 1388.3333494532342 | 1388.3333426178185 | 1764.0500193088715 | 1764.0500065645801 |
| Residual variance | 2451.2499950635897 | 2451.249997156806 | 2451.2499958154604 | 2451.2499985773525 |

The maximum absolute objective difference is `5.69e-14`. The maximum theta
difference is `3.55e-9`; the accepted theta tolerance is `1e-6`. Independent
dense marginal algebra also matches Kamino at each fitted theta to `1e-10` for
the objective, beta, and residual variance.

## Predictions

Conditional predictions for one new row in each known batch are:

| Batch | ML | REML |
|---|---:|---:|
| A | 1510.8717783199743 | 1509.8931486008 |
| B | 1527.869516037334 | 1527.8912633644265 |
| C | 1554.4746707253762 | 1556.0622256031468 |
| D | 1505.6985537972992 | 1504.4154614988267 |
| E | 1581.0798254134181 | 1584.2331878418665 |
| F | 1485.0056557066002 | 1482.5047130909336 |

Population prediction is the fixed intercept (`1527.5`, subject to floating
roundoff) for both known and unseen batches. Conditional prediction rejects an
unseen batch by default; with `allow_new_groups=True`, its unknown random
contribution is zero, its mean is also `1527.5`, and the returned row is flagged
as a new group. Across all pinned prediction cases, the maximum absolute Kamino
versus lme4 difference is `1.04e-7`, below the `1e-5` acceptance threshold.

## Reproduction gates

`make check` runs strict linting, typing, 73 tests, 95% branch-aware coverage,
and artifact builds. `make wheel-smoke` installs the wheel off-tree and performs
a public fit and conditional known/new-group prediction. The oracle fixture and
generator hashes are verified by `tools/verify_oracle_output.py`.

This evidence supports only the compatibility boundary documented in
`docs/COMPATIBILITY.md`; it does not complete the full Phase 1 gate.
