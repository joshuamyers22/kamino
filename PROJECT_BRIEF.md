# Project Brief

## Outcome

- Problem and affected users: Python users need a transparent Gaussian linear
  mixed-model fitter with a versioned, measurable subset of lme4 compatibility.
- Measurable success: the compatibility and release gates in `PROJECT_PLAN.md`
  pass for each advertised feature and platform.
- Initial journey: construct a labeled model, evaluate ML or REML by independent
  dense and PLS routes, and receive immutable estimates and diagnostics.
- Explicit non-goals: GLMMs, nonlinear/Bayesian models, general residual
  correlation, frequency/survey weights, and unqualified lme4 parity.

## Constraints and risks

- Runtime: CPython 3.11+; float64; R-free installed package.
- Reference: lme4 2.0-6 at CRAN source commit
  `4aa26a91f9e676e9409f6cd8163ae92654ef1e7e`; legacy corpus 1.1-37.
- Data: caller-owned arrays/dataframes may be sensitive and are never uploaded.
- Resource behavior: dimensions and allocations are validated before dense work.
- Highest risks: design-matrix drift, wrong likelihood constants, invalid
  uncertainty, false convergence, and unavailable numerical binaries.
- Recovery: input/model snapshots are required to refit; result writes will be
  atomic and versioned. Bootstrap work will use a resumable failure ledger.
- Owner: Josh Myers. Statistical reviewer and release maintainer must be assigned
  before beta.

## Core invariants

1. Weighted PLS and independently assembled marginal GLS agree at fixed theta.
2. ML/REML objective kind and likelihood constants are explicit.
3. Labels and row identities survive every boundary.
4. Boundary covariance is representable and distinct from numerical failure.
5. Unsupported structure or inference fails before expensive work.

## Acceptance evidence

| Requirement | Verification | Status |
|---|---|---|
| Phase 0 algebra | Dense/PLS equivalence and pinned lme4 tests | Pass locally |
| Formula semantics | Six-case pinned X/Z, row, and label corpus | Pass for Phase 0 |
| Oracle environment | Digest-pinned container, package snapshot, fixture hashes | Pass locally |
| Packaging | Frozen install, strict checks, clean wheel smoke | Pass locally |
| Backend choice | Phase 1 block ADR accepted; general sparse remains Phase 2 work | Pass for Phase 0 |
| Dyestuff public API | ML/REML final fits and explicit population/conditional prediction versus pinned lme4 | Pass locally for one random-intercept formula |
| Single-group block backend | Dense PLS/marginal parity and public Dyestuff/Dyestuff2/Sleepstudy execution | Pass locally |
| Dyestuff2 boundary API | ML/REML theta-zero fits and predictions versus pinned lme4 and closed form | Pass locally |
| Compact formula boundary | Immutable group index plus small per-row random covariates; no production dense random indicator | Pass locally for intercept and numeric-slope slices |
| Sleepstudy correlated slope | ML/REML fixed-theta, fitted results, covariance, modes, and prediction versus pinned lme4 | Pass locally |
| Sleepstudy independent terms | Numeric double-bar and explicit same-group split preserve zero prior covariance while matching pinned ML/REML fits and prediction | Pass locally |
| Safe prediction bundle | Atomic non-executable save/load with strict validation and Dyestuff/Dyestuff2/correlated-and-independent Sleepstudy prediction parity | Pass locally |
| Single-group resource benchmark | ML/REML random-intercept, correlated-slope, and independent-term public fits at 1M observations/10k groups | Expanded manifest passes locally; prior subset passes hosted CI |
| Phase 1 completion | Remaining alpha corpus | Pending |
