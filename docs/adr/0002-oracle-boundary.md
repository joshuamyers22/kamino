# ADR 0002: R oracle is a containerized development boundary

Status: accepted
Date: 2026-09-14

## Decision

Run pinned lme4 reference generation in a container under oracle/. Commit safe,
licensed fixture outputs and manifests; do not add R or rpy2 to the Kamino runtime.

## Consequences

Default Python tests stay fast and offline. Scheduled/release oracle jobs must
verify the image digest, R session, fixture inputs, generator source, and output
hashes. A missing container runtime makes the gate unavailable, never passing.
