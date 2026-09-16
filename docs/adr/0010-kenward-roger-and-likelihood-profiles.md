# ADR 0010: Bounded Kenward–Roger and named likelihood profiles

Status: accepted, 2026-09-16.

## Context

I03 requires two different approximations with different estimation contracts.
Kenward–Roger (KR) is a REML covariance/test adjustment over a covariance model
that is linear in known observation-space component matrices. Likelihood
profiles use an ML baseline and constrain a reported SD, correlation, residual
scale, or fixed coefficient while reoptimizing every nuisance parameter.

Treating either method as another use of the I02 theta Hessian would be wrong.
KR also requires dense observation-space matrices in the pinned pbkrtest
construction, while fitting itself remains sparse or block structured.

## Decision

Kamino implements the pbkrtest 0.5.5 `vcovAdj` and `KRmodcomp` construction for
regular, unit-weight Gaussian fits. ML inputs are refitted to REML without
mutating the source fit. The public result exposes the unadjusted and adjusted
fixed-effect covariance, covariance-parameter information and covariance,
derivative matrices, scaling, numerator/denominator degrees of freedom, and
scaled and unscaled F tests. Hypotheses are checked in the full fixed-effect
coordinate space and redundant consistent rows are reduced by rank.

The dense KR path preflights observation count, covariance-component count, and
estimated workspace bytes. Prior weights, fitted covariance boundaries,
singular/ill-conditioned information, non-estimable hypotheses, and invalid
adjustments return explicit unavailable status or a resource error. No sparse-KR
claim is inferred from the scalable fitter.

Likelihood profiles always use a separate ML baseline when the source fit is
REML. Covariance targets use lme4 2.0-6 `.sigNN`/`.sigma` ordering and either
SD/correlation or optional variance/covariance presentation. Fixed-effect
targets remove the constrained design column, move its contribution to the
offset, and optimize the remaining covariance parameters. Covariance targets
use the unprofiled-scale ML deviance and optimize all other natural covariance
parameters. Every point retains the complete reported parameter vector,
objective difference, signed-root deviance, convergence state, and evaluation
count.

Adaptive traces have explicit bounds, point/evaluation ceilings, monotonicity
checks, baseline-improvement invalidation, boundary/unbounded endpoint status,
and monotone-cubic interval inversion. Callers may request explicit target
values for reproducible constrained evaluations.

## Consequences

The installed Python package remains R-free. Development evidence pins
pbkrtest 0.5.5 at source commit
`4ead4ff46e90831f4a2376cdcae0602045db0458` and lme4 2.0-6 at
`4aa26a91f9e676e9409f6cd8163ae92654ef1e7e`.

The calibration claim is limited to the locked, regular, unit-weight Gaussian
random-intercept regime. Weighted KR, KR at covariance boundaries, and nominal
variance-component profile coverage remain unclaimed. Prediction-only bundles
still cannot perform derivative inference or profiling because they omit the
training response and design.
