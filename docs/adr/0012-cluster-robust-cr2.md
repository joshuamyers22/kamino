# ADR 0012: Fitted-target CR2 for independent clusters

Status: accepted, 2026-09-16.

## Context

A mixed-model CR2 estimator cannot substitute the ordinary least-squares
`(I - H_cc)^(-1/2)` adjustment. It needs the fitted marginal working target,
the GLS weight blocks, and cross-cluster leverage induced by estimating fixed
effects. Cluster-robust tests must also keep the covariance estimator and its
small-sample degrees of freedom in one contract.

## Decision

Live Kamino results expose `cluster_robust()`. The default independent cluster
is the fitted outer grouping factor; callers may name another fitted grouping
factor or supply a row-identified pandas Series or accepted-row-order sequence.
Every random-effect level must occur in exactly one proposed cluster. Prior-
weight fits and non-nested/crossed cluster partitions are rejected before
matrix construction.

For cluster `c`, Kamino constructs the fitted relative marginal target
`T_c = I + Z_c (G / sigma^2) Z_c'`, its inverse `W_c`, and
`B = (sum X_c' W_c X_c)^(-1)`. Scores use the marginal residual
`y_c - o_c - X_c beta_hat`; fitted random effects are not subtracted. CR0 uses
the unadjusted scores and CR1 multiplies each estimating matrix by
`sqrt(J / (J - 1))`.

CR2 follows clubSandwich 0.7.0's inverse-variance mixed-model path. If `R_c` is
the upper Cholesky factor of `T_c`, then

```text
I_H,c = I - X_c B X_c' W_c
K_c   = R_c I_H,c T_c R_c'
A_c   = R_c' K_c^(-1/2) R_c
E_c   = X_c' W_c A_c
```

The covariance is `B (sum E_c e_c e_c' E_c') B`. Symmetric powers use the
pinned clubSandwich absolute eigenvalue cutoff, with materially negative
eigenvalues rejected. Cluster targets, weights, adjustments, estimating
matrices, scores, and bread are retained as immutable evidence.

One-row CR2 hypotheses use the clubSandwich coefficient-specific
Satterthwaite approximation. Full-row-rank multi-parameter hypotheses use the
actual HTZ approximation implemented by `Wald_test`, including its scale and
denominator degrees of freedom. CR0/CR1 tests are explicitly asymptotic
normal/chi-square. Redundant joint rows are rejected rather than silently
changing the stated hypothesis.

Observation-space work is bounded by observation, fixed-effect, largest-
cluster, total squared-cluster-size, byte, and conditioning limits. The public
runtime remains R-free. clubSandwich is a development-only GPL-3 oracle pinned
at commit `bc925c2c8f27cfb52ab82eab253f7f4b5253f57a`.

## Consequences

The release claim covers CR0, CR1, and CR2 for unit-weight Gaussian fits whose
random grouping factors are nested in user-declared independent clusters. A
cluster vector establishes a partition, not empirical independence; that
design assumption remains the user's responsibility. Custom working targets,
prior-weight robust inference, invalid crossed partitions, CR3, saddlepoint
tests, and other joint-test approximations remain separate gates.

The pinned oracle covers ML/REML higher-level clustering and a correlated
random-slope fit. A locked 2,000-replicate assessment covers one-DF
Satterthwaite and two-DF HTZ tests under a declared 12-cluster nested Gaussian
regime with cluster covariance and slope heterogeneity outside the fitted
working target.
