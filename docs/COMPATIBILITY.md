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

A Formulae adapter is selected for the Phase 1 subset. Six formula cases pass
matrix, row, and label checks, including explicit numeric `||` expansion and
nesting canonicalization. Public formula parsing, optimized fitting,
nested/crossed structures, small-sample inference, and production wheels remain
unclaimed. See `PROJECT_PLAN.md` section 2 for the feature-level contract.
