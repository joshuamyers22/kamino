# ADR 0001: Native PLS core

Status: accepted
Date: 2026-09-14

## Decision

Implement Kamino's Gaussian mixed-model likelihood with an owned, labeled PLS
core. Use Statsmodels as an external comparator and future adapter target.

## Rationale and consequences

The product objective is numerical fidelity to lme4's parameterization and
profiled objective. Delegating the fit to another model class would not establish
that contract. The core remains independent of dataframe, formula, optimizer,
and sparse-factorization vendors. This raises the verification burden, which the
three-oracle strategy addresses.
