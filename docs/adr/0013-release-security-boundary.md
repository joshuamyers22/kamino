# ADR 0013: Release security and artifact boundary

Status: accepted, 2026-09-16.

## Context

Kamino's wheel was runtime-only, but its source distribution still included the
development R oracle and repository operations material. The project also lacked
an executable SBOM/checksum boundary, tag release workflow, complete privacy
description, and incident procedures.

## Decision

The published sdist is reduced to package source plus `README`, changelog,
license, and build metadata. The wheel and sdist are inspected member-by-member,
including wheel RECORD hashes, identity/license metadata, path/symlink rules,
size ceilings, and absence of oracle/development content. The R oracle, GPL
fixtures, calibration reports, tests, tools, and repository governance remain in
Git but outside both Python distributions.

Every candidate build emits deterministic CycloneDX 1.5 documents for the core
and all published extras, SHA-256 checksums, and a source/lock/artifact manifest.
Tag releases must match package and dated changelog versions, rerun the release
gates, and receive GitHub build-provenance attestations before a GitHub release
is created. E02 does not enable PyPI publishing; that requires a separately
reviewed trusted-publishing decision.

GitHub security automation uses immutable action commits for full-history secret
scanning, dependency review, and CodeQL. GitHub-native dependency alerts,
security updates, secret scanning, push protection, and private vulnerability
reporting are enabled. Runtime remains network- and telemetry-free.

## Consequences

Development evidence no longer rides inside the sdist, which makes the MIT
distribution boundary match the documented oracle boundary. Users receive two
SBOM views because the optional statsmodels adapter materially changes the
dependency graph. Artifact provenance is verifiable without treating local
manifests as cryptographic attestations.

Prediction bundles and bootstrap ledgers remain explicitly sensitive despite
excluding raw training rows from bundles. A sole repository owner can override
hosted controls, so beta and stable releases remain blocked until accountable
review roles and E03 are complete.
