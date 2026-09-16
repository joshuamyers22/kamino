# ADR 0009: Satterthwaite inference uses full variance parameters

Status: accepted, 2026-09-15

## Decision

Kamino computes Satterthwaite tests from the complete variance-parameter vector
`eta = (theta, sigma)`. It numerically differentiates the unprofiled ML or REML
deviance, forms `Var(eta) = 2 H_D^-1`, and differentiates the fixed-effect
covariance with respect to the same coordinates. This is the construction used
by the pinned lmerTest 3.1-3 companion reference.

For a one-row contrast, Kamino uses the delta-method denominator and returns a
two-sided t test. Multi-row hypotheses use the positive eigenspace of the
hypothesis covariance and lmerTest's published component-DF aggregation to
return an F test. Redundant, consistent rows reduce to their numerical rank;
an inconsistent right-hand side is unavailable.

The derivative evaluator reuses the fitted block or general sparse workspace.
It does not construct a dense random-effects indicator matrix. Variance-
parameter count is capped at 32 by default, and derivative storage is bounded
by `O(d^2 + d p^2)` in addition to the selected fitting workspace.

## Failure policy

Inference is unavailable at a covariance boundary. Material negative curvature,
a singular or excessively ill-conditioned Hessian, non-finite/step-sensitive
derivatives, non-estimable contrasts, and nonpositive delta-method denominators
also return explicit unavailable statuses. Kamino does not substitute a large,
residual, or asymptotic denominator DF in these cases.

## Evidence

Dyestuff and Sleepstudy ML/REML Hessians, variance-parameter covariances,
fixed-covariance Jacobians, one-DF tests, and multi-DF tests are compared with
the pinned lmerTest source commit. A locked 2,000-replicate REML assessment
calibrates one- and two-DF null rejection in a declared regular random-intercept
regime. Dyestuff2 verifies the boundary refusal. Rank-deficient and redundant
hypotheses and the general sparse backend have separate tests.

This decision does not cover Kenward–Roger covariance adjustment, likelihood
profiles, variance-component tests, generalized mixed models, or inference from
prediction-only bundles.
