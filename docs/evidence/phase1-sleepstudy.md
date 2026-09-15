# Phase 1 Sleepstudy correlated-slope evidence

Date: 2026-09-15

Reference: lme4 2.0-6, source commit
`4aa26a91f9e676e9409f6cd8163ae92654ef1e7e`

Model: `Reaction ~ 1 + Days + (1 + Days | Subject)`

Data: lme4 `sleepstudy`, 180 observations in 18 subjects

## Optimized fits

| Quantity | Kamino ML | lme4 ML | Kamino REML | lme4 REML |
|---|---:|---:|---:|---:|
| Objective (`-2 logLik`) | 1751.939344463198 | 1751.939344488990 | 1743.628271958491 | 1743.628271959952 |
| Fixed intercept | 251.405104848486 | 251.405104848485 | 251.405104848484 | 251.405104848486 |
| Fixed Days slope | 10.467285959596 | 10.467285959596 | 10.467285959596 | 10.467285959596 |
| Relative theta | (0.92922542, 0.01816571, 0.22264544) | (0.92919061, 0.01816575, 0.22264321) | (0.96673292, 0.01516905, 0.23090959) | (0.96674177, 0.01516906, 0.23090995) |
| Residual sigma | 25.59181560 | 25.59190704 | 25.59181574 | 25.59179572 |

The maximum optimized objective difference is `2.58e-8`; Kamino's value is
slightly lower in both fits. The maximum raw-theta difference is `3.49e-5`, the
maximum random-covariance element difference is `0.0384`, and the maximum
conditional fitted/prediction difference is `6.55e-4`. Acceptance ceilings are
`1e-6` for objective, `5e-5` for raw theta, `atol=1e-3, rtol=1e-4` for random
covariance, and `1e-3` for conditional prediction. Fixed effects agree within
`3e-12`.

The reference fit used NLopt 2.7.1 BOBYQA through lme4. Kamino records and
reports `scipy-powell`; it does not claim optimizer identity. The discrepancy is
consistent with different stopping points because the Kamino objective is lower,
while evaluation at the pinned lme4 theta reproduces its objective.

## Algebra and representation

Ten Sleepstudy fixed-theta cases cover ML/REML at zero covariance,
intercept-only, diagonal, correlated, and near-boundary factors. The maximum
objective error against pinned lme4 is `4.55e-13`. Beta, spherical modes,
conditional modes, log determinants, weighted residual sums, and penalized sums
also pass their fixed-theta gates. The independent dense marginal covariance
oracle agrees independently.

The block evaluator aggregates weighted random/random, random/fixed, and
random/response cross-products by group, factorizes 18 two-by-two penalized
blocks, and factorizes the two-by-two fixed-effect Schur complement. It never
forms or factorizes the 36-by-36 random-effects normal matrix. The compact
formula design stores 180 group indices plus a 180-by-2 random-covariate matrix;
it has no dense 180-by-36 `Z`.

Population and conditional predictions pass for training rows, existing
subjects at new Days values, and an explicitly allowed unseen subject. Missing,
nonnumeric, or nonfinite prediction covariates fail explicitly. A separate
weighted/offset synthetic optimized slope fixture returns an exact zero final
Cholesky diagonal and a structured boundary diagnostic under both ML and REML.

Sleepstudy input values and derived oracle outputs are repository-only
GPL-2.0-or-later evidence following lme4 package metadata. They are excluded
from the MIT wheel.
