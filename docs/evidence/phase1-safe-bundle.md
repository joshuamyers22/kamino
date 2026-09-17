# Phase 1 safe prediction-bundle evidence

Date: 2026-09-15

Scope: fitted results from the verified random-intercept, correlated numeric
random-intercept/slope, independent numeric intercept/slope, and categorical
fixed-effect public formulas.

## Round-trip contract

ML and REML fits for Dyestuff, Dyestuff2, correlated Sleepstudy, and independent
Sleepstudy are saved and loaded.
The loaded object preserves objective kind/value, likelihood, theta, beta,
coefficient covariance, residual variance, random covariance, conditional modes,
all coefficient/group labels, formula identity, optimizer/backend diagnostics,
fit controls, predictor encoding, and the new-data offset requirement. Retained
arrays are byte-exact and immutable. Population and conditional predictions for
known and explicitly allowed new levels are identical before and after reload.

Schema 1.1 records covariance-term sizes, and the independent model reloads with
`(1, 1)` rather than being silently widened to one correlated term. Schema 1.2
records numeric/categorical fixed variables, ordered levels, treatment/sum
coding, expanded terms, and formula-offset names. Downgraded schema-1.0 and
schema-1.1 numeric correlated models remain loadable; malformed term or encoder
maps fail closed. Schema 1.3 later extends this contract to stable-core recovery;
see `phase2-artifact-recovery.md`.

The clean installed-wheel smoke test performs slope and weighted categorical
fits, saves and loads them off-tree, and verifies prediction identity including
both offset sources. No R runtime is involved.

## Safety and privacy contract

At the Phase 1/schema-1.2 gate, the deterministic artifact contained exactly
`manifest.json` and five float64 `.npy` members: theta, beta, beta covariance,
random covariance, and conditional random effects. Current schema 1.3 adds a
sixth float64 null-space member without changing the privacy boundary. It
contains no response, fitted/residual vectors, spherical
modes, row IDs, training design/offset rows, or training grouping rows.

Before construction, the loader rejects unsupported schema versions, missing or
extra members, duplicate/unsafe paths, encrypted or unsupported compression,
oversized files/members/arrays, malformed or duplicate-key JSON, non-float64 and
object arrays, nonfinite values, checksum changes, shape/label disagreement,
non-PSD covariance, and theta/covariance disagreement. A model-level digest binds
metadata to member hashes. Pickle is disabled. Checksums are explicitly not an
authentication mechanism.

Saving defaults to no replacement. A temporary sibling is fully written and
flushed before atomic publication. A fault-injected replacement failure leaves
the previous artifact byte-for-byte unchanged and removes the temporary file.

Group labels, fixed factor levels, and conditional effects remain potentially
identifying derived data and are documented accordingly in
`docs/MODEL_BUNDLES.md`. Saving is explicit; Kamino performs no upload,
telemetry, or implicit persistence.
