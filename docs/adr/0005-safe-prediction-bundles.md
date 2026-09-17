# ADR 0005: Safe prediction-only model bundles

Status: accepted and implemented; extended through schema 1.3 by ADR 0016
Date: 2026-09-15

## Decision

Kamino saves fitted public models as one atomic ZIP container with a canonical
JSON manifest and schema-defined float64 NumPy `.npy` members. Loading always uses
`allow_pickle=False`. The format is versioned independently of the package and
is fail-closed: exact member names, schema fields, dimensions, dtypes, finite
values, allocation limits, checksums, ordered labels, theta/covariance identity,
and optimizer metadata must validate before a model object is constructed.

The loaded object is `PredictionOnlyModel`, not a partially reconstructed
`LinearMixedModelResult`. It supports population and conditional prediction on
explicit new data. Training prediction, refitting, and inference are
unavailable because their required data are deliberately absent.

Saving creates a completed temporary sibling, flushes it, and publishes it
atomically. The default no-overwrite path uses an atomic hard link; replacement
requires `overwrite=True` and uses `os.replace`. Temporary files are removed on
failure. The manifest records the package version, installed-source hash,
project-plan hash, lme4 reference profile, backend, controls, diagnostics, model
content hash, and member hashes.

Schema 1.1 added covariance-term sizes so an independent same-group model cannot
reload as a correlated term. Schema 1.2 adds the owned fixed encoder—variable
kinds, ordered levels, treatment/sum coding, term expansion, and formula-offset
names—so categorical new-data prediction is deterministic. Schemas 1.0 and 1.1
remain readable for their original intercept/numeric fixed designs. Schema 1.3
adds full rank/null-space state and ordered categorical/nested/crossed random
term artifacts as specified by ADR 0016; schemas 1.0–1.2 remain readable under
their original contracts and new saves use 1.3.

## Privacy boundary

Prediction bundles exclude the training response, residuals, fitted values,
row identifiers, training design rows, training offsets, training group rows,
and spherical modes. Conditional prediction still requires fitted random effects
and ordered group labels, which can identify people or sites. Saving is always
explicit; bundles are never uploaded or logged by Kamino.

## Rejected alternatives

- Pickle and object arrays permit executable reconstruction and are rejected.
- Pretending a prediction artifact is a complete fitted result would expose
  unavailable operations and invite fabricated training state.
- A directory of independently replaced files has no single atomic publication
  point and complicates integrity checks.
- Checksums are corruption evidence, not publisher authentication; signatures
  remain a separate future feature.

## Evidence

Implementation is in `src/kamino/bundle.py`; public round-trip, corruption,
privacy, resource, and atomicity tests are in `tests/test_bundle.py`. Reviewed
evidence is recorded in `docs/evidence/phase1-safe-bundle.md`.
