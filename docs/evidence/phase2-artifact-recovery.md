# Phase 2 stable-core artifact-recovery evidence

Date: 2026-09-16

## Scope completed

Prediction bundle schema 1.3 recovers every artifact required for explicit
new-data prediction in the advertised stable-core model scope:

- full fixed coefficient names, retained/drop indices, pivot, tolerances, and
  normalized coefficient null-space basis;
- fixed and random numeric/categorical encoders with ordered treatment/sum
  contrast state;
- covariance-term boundaries and exact ordered random coefficient labels;
- grouping source columns, canonical levels, and per-term effect offsets for
  nested and crossed structures.

The artifact remains prediction-only. No response, training row ID, training
group row, design row, fitted/residual vector, training offset, or refit state is
written.

## Executable evidence

Round trips compare loaded predictions directly with the same live fitted
object, so this gate is independent of optimizer variation after the fit:

| Recovery case | Coverage |
|---|---|
| Dyestuff/Dyestuff2 | ML/REML, interior and exact boundary, known/new groups |
| Sleepstudy | ML/REML correlated and independent numeric slopes |
| Fixed rank | Exact alias, estimable and non-estimable rows, population and conditional modes |
| Categorical random slope | Treatment and sum fixed/random encoders, known/new groups |
| Pastes | Slash-expanded nested terms and partially new combinations |
| Penicillin | Crossed random-intercept terms and partially new combinations |
| InstEval | ML/REML recovery at 73,421 rows and 4,100 random coefficients |
| Crossed categorical slope | ML/REML, categorical slope plus second grouping factor |

All loaded prediction values, `estimable` flags, and `new_group` flags are exact
against their live-result counterparts. The existing F02 and N03 tests retain
the independent pinned lme4 and dense-algebra fidelity evidence for those live
fits.

The rejection corpus covers unsupported/extra members, unsafe paths, duplicate
keys and members, object arrays, checksums, non-finite/wrong-shape arrays,
covariance inconsistency, invalid fixed-rank partitions and pivots, invalid
tolerances, unnormalized null-space state, malformed random encoders/terms,
resource limits, and atomic overwrite failure. Schemas 1.0, 1.1, and 1.2 are
also reconstructed from current test artifacts and loaded under their original
contracts.

## Verification

The focused bundle, F02, and general-sparse suite passes 98 tests locally. The
complete `make check` gate passes strict formatting, lint, type checking,
repository/E03 verification, 407 tests at 90.93% branch-aware coverage, and
offline wheel/sdist builds. The clean-wheel smoke installs into an isolated
environment and round-trips rank-deficient, categorical-random, and crossed
fits through the public API. The hosted CI identifier is recorded at handoff.
