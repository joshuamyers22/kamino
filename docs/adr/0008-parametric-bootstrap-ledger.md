# ADR 0008: deterministic parametric bootstrap and private refit ledger

Status: accepted, 2026-09-15

## Context

I01 requires unconditional and conditional Gaussian simulation, exact response
refitting, deterministic work allocation, visible failure accounting, and safe
resume after interruption. Prediction bundles intentionally omit training rows
and responses and therefore cannot support these operations.

Cross-language RNG identity is not a valid compatibility target. Refit parity
can instead be tested by giving Kamino and pinned lme4 the same stored response
arrays. Singular covariance estimates are valid outcomes and must not be
silently discarded or redrawn.

## Decision

- Use NumPy `PCG64DXSM` with a `SeedSequence` keyed by root seed, replicate ID,
  and a separate purpose identifier for random effects and residuals. Record the
  NumPy version and do not claim cross-version or R/Python draw identity.
- Unconditional simulation draws new random effects and weighted residuals.
  Conditional simulation fixes fitted random effects and draws only weighted
  residuals; the API names the mode explicitly.
- Refit the immutable accepted numerical design with only its response replaced.
  Formula parsing, rows, weights, offsets, contrasts, coefficient identity,
  estimation method, optimizer controls, and sparse limits remain unchanged.
- Make retained fixed effects the first public bootstrap statistic. Percentile
  and basic intervals use NumPy's linear quantile convention. They are
  descriptive until a separately locked outer coverage study passes.
- Bound parallel work explicitly. Each refit builds private workspace state;
  `threadpoolctl` limits native numerical libraries to one thread while the
  bounded Python worker pool is active. Replicate streams do not depend on work
  order.
- Store an optional private ledger as a mode-0700 directory containing an
  atomically replaced canonical JSON manifest and mode-0600, non-object `.npy`
  response arrays. Validate exact paths, schema, dimensions, finite values,
  hashes, fit fingerprint, package source hash, project-plan hash, RNG identity,
  and request identity before resume. Commit an empty manifest before the first
  refit and hold a crash-released operating-system lock so a second process
  cannot write the same ledger concurrently.
- Record every attempted replicate as success, warning, singularity, optimizer
  failure, numerical failure, or statistic failure. Singularity remains usable.
  Never redraw a failed replicate. Intervals fail closed when failures remain
  unless incomplete use is explicitly requested. Report the observed failure
  rate and a one-sided exact binomial upper bound.
- A ledger is sensitive, is not a self-contained model bundle, and requires the
  same live or reproducibly reconstructed fit. Prediction-only bundles remain
  response-free and inference-incapable.

## Consequences

Simulation and refitting work across the block and coupled sparse backends with
worker-count-invariant inputs and outputs. Interrupted work can resume without
repeating completed refits. The ledger may contain sensitive simulated
responses and must follow the caller's private-data retention policy.

This decision does not validate nominal percentile/basic interval coverage,
Satterthwaite/KR/profile inference, arbitrary user callbacks, or bootstrap from
a prediction-only bundle. Those claims remain separately gated.
