# ADR 0016: stable-core prediction artifact recovery

Status: accepted for Phase 2 on 2026-09-16.

## Context

Prediction bundle schema 1.2 could reconstruct the Phase 1 fixed encoder but
not the additional state introduced by N03 and F02. A rank-deficient fit needs
its full coefficient coordinates, retained/drop map, and null-space basis to
reproduce estimability. A categorical random slope needs its independent random
contrast encoder. Nested and crossed fits need every grouping source, ordered
level map, coefficient block, and encoder so the flat random-effects vector
cannot be misinterpreted.

Reconstructing any of those artifacts from the formula would be ambiguous and
would make reload behavior depend on a future parser. Saving caller rows would
violate the established prediction-only privacy and capability boundary.

## Decision

- Schema 1.3 adds a checksum-protected, non-object float64
  `fixed_null_basis.npy` member and exact fixed-rank metadata: full names,
  retained/dropped indices, pivot, and rank/estimability tolerances.
- The manifest records the single-group random encoder and an ordered descriptor
  for every general sparse term. Each descriptor contains the canonical group
  name, source columns, group levels, coefficient labels, predictor identity,
  and owned numeric or categorical treatment/sum encoder.
- The existing random-effects vector remains term-major and group-major. Term
  descriptors and covariance-term sizes jointly define its offsets and the
  block-diagonal covariance layout; no random indicator matrix is stored.
- Loading validates exact fields and members, identifier/label uniqueness,
  encoder expansion, rank partitions, null-basis dimensions and normalized drop
  identity, term boundaries, flat effect length, ordered random labels, and the
  existing theta/covariance identity before constructing a model.
- Schemas 1.0–1.2 remain readable only under their original contracts. New
  saves always use schema 1.3.
- The loaded object remains `PredictionOnlyModel`. Training rows, response,
  offsets, design rows, residuals, modes in spherical coordinates, and refit
  state remain absent. Consequently training prediction, refit, bootstrap, and
  inference remain unavailable after reload.

## Consequences

The complete advertised stable-core model scope now supports safe population
and conditional new-data prediction after reload: full-rank and rank-deficient
fixed designs, treatment/sum categorical random slopes, independent same-group
terms, nested/crossed random intercepts, and a categorical slope crossed with
another grouping factor. New-group and non-estimable-row behavior is identical
to the live result.

Canonical group labels and fitted random effects remain potentially identifying
derived data. Schema 1.3 does not change the documented access, retention,
sharing, or deletion obligations.

## Evidence

Implementation and corruption tests are in `src/kamino/bundle.py` and
`tests/test_bundle.py`. End-to-end recovery tests are in
`tests/test_f02_rank_categorical.py` and
`tests/test_general_sparse_public_api.py`. Reviewed results are recorded in
`docs/evidence/phase2-artifact-recovery.md`.
