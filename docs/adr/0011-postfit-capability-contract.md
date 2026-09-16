# ADR 0011: Bounded postfit capability contract

Status: accepted, 2026-09-16.

## Context

A01 must prove that post-estimation can preserve coefficient labels,
estimability, covariance identity, and degrees-of-freedom semantics across
Kamino and two external model families. A broad duck-typed protocol would make
those invariants impossible to establish. A separate public distribution would
also be premature until the capability spike works against real external
objects.

## Decision

Kamino exposes a versioned `LinearFunctionBasis` and explicit factories for a
live Kamino fit, statsmodels OLS/WLS, and statsmodels MixedLM. Factories accept
only the concrete result classes they name and formula-fitted statsmodels
objects with retained design metadata. The contract preserves full coefficient
names, retained indices, a coefficient-aligned null-space basis, covariance
identity, inference identity, and design reconstruction state. Arrays are
copied and read-only.

Covariance and degrees of freedom are selected as one analysis contract:

- Kamino model covariance uses asymptotic normal/chi-square inference;
  Satterthwaite uses its contrast-specific DF state; Kenward–Roger uses the
  adjusted covariance and scaled F state.
- Nonrobust statsmodels OLS/WLS uses model covariance and residual t/F
  inference. A statsmodels robust covariance always uses asymptotic
  normal/chi-square inference, even if its wrapper requests `use_t=True`,
  because residual model DF are not inherited across covariance estimators.
- statsmodels MixedLM exposes its fixed block with model covariance and
  asymptotic normal/chi-square inference.

Reference grids support only direct fixed-predictor names, explicit `specs`,
`by`, `at`, offset behavior, and equal/proportional/outer/cells/flat/user
weights. Linear functions stay in full coefficient coordinates and are checked
for estimability before inference. Pairwise families support none, Holm,
Bonferroni, and Sidak. Tukey and multivariate-t adjustments fail closed pending
their own distribution and heterogeneous-DF design.

`tidy()` returns unavailable rows rather than fabricating dropped-coefficient
statistics. Performance output is source-bounded. Kamino's Nakagawa-style
summary averages over retained training rows and computes each random term as
the mean of `z_i' G z_i`, including covariance cross terms; weighted residual
variance is the mean of `sigma² / w_i`.

The official Python marginaleffects interface currently documents
statsmodels formula model objects as its supported path. Kamino will not forge
a statsmodels-shaped object or claim direct support. A native marginaleffects
adapter remains deferred until upstream exposes a public adapter contract or a
separately reviewed integration is accepted.

## Consequences

The capability spike remains in Kamino until an independent postfit
distribution has its own release checklist. statsmodels 0.14.6 is an exact
optional extra and development dependency, not a core runtime dependency.
The core wheel remains R-free and statsmodels-free.

Development evidence pins emmeans 2.0.2 at source commit
`fbaba0c2e222a7e17bf6c4db59f3575e43607f54` and lme4 2.0-6 at
`4aa26a91f9e676e9409f6cd8163ae92654ef1e7e`. The fixture verifies coefficient-
aligned linear functions rather than treating displayed estimates alone as
sufficient evidence.
