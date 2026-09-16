# A02 cluster-robust inference evidence

Status: complete locally on 2026-09-16; hosted candidate run pending.

Kamino now exposes `fit.cluster_robust()` and the equivalent public
`cluster_robust()` factory. CR0, CR1, and CR2 share a labeled, immutable
analysis containing the GLS bread, fitted marginal working-target blocks,
inverse blocks, adjustment matrices, estimating matrices, marginal residuals,
and cluster score contributions. CR0/CR1 inference is asymptotic. CR2 binds its
covariance to contrast-specific Satterthwaite t tests and full-row-rank HTZ
joint F tests.

The implementation uses `y - offset - X beta` rather than conditional
residuals. Its working target is the fitted `I + Z Lambda Lambda' Z'` within
each independent cluster. Every random grouping factor must be nested in the
chosen cluster. Explicit prior weights, invalid crossed partitions, missing or
misaligned cluster identities, one-cluster inputs, singular hypotheses, and
resource-limit violations fail closed.

The reference fixture pins lme4 2.0-6 and clubSandwich 0.7.0 at source commit
`bc925c2c8f27cfb52ab82eab253f7f4b5253f57a`. It contains ML and REML fits with
16 schools nested in eight higher-level clusters plus the REML Sleepstudy
correlated random-intercept/slope fit. Across the three cases, all CR0/CR1/CR2
covariances, working targets, CR2 adjustments, estimating matrices, scores,
coefficient tests, and HTZ joint tests pass. Maximum absolute differences are
`7.89e-12` for covariance, `1.32e-11` for adjustments, `3.37e-6` for estimating
matrices, `1.01e-8` for test statistics, and `1.36e-10` for denominator DF.

The predeclared `statistical/a02_plan.json` assessment ran 2,000 REML fits with
12 independent clusters, nested school random intercepts, cluster intercept
and slope dependence, and heteroscedastic residuals. All 2,000 paired tests
were available. The one-DF CR2 Satterthwaite rejection rate was `0.0525`
(Wilson 95% interval `[0.04355, 0.06316]`); the two-DF HTZ rejection rate was
`0.0505` (`[0.04174, 0.06099]`). Both predeclared gates passed without redraws.

Contract tests also cover row-identified permutation invariance, subset
alignment, additive offsets, CR1 normalization, rank-deficient estimability,
nested sparse models, crossed-model rejection, prior-weight refusal, immutable
state, and allocation ceilings. ADR 0012 records the supported estimand and
restrictions.

The reviewed Linux ARM64 oracle is public at
`ghcr.io/joshuamyers22/kamino-oracle@sha256:17e45268be294316967064d0600a727463d738d7dfb0c68ec43769baebe8eb4d`.
