# ADR 0014: PyPI distribution identity and trusted publication

Status: accepted by the owner, 2026-09-16.

## Context

Kamino's product, repository, and import package use the name `kamino`, but that
distribution name is already occupied on PyPI by an unrelated project. Moving
the import package would break the documented API and saved-model provenance.
Publishing with a long-lived token would also weaken the E02 release boundary.

## Decision

Publish the Python distribution as `kamino-lme` while retaining `kamino` as the
only import package. The build and release verifiers treat normalized archive
identity (`kamino_lme`) separately from import identity (`kamino`).

PyPI publication occurs only from an exact version tag after the complete
release job succeeds. A separate least-privilege job downloads the distributions
built by that job and publishes through the protected GitHub `pypi` environment
using PyPI trusted publishing and an ephemeral OIDC credential. No PyPI token is
stored in GitHub or the repository.

## Consequences

Users install with `pip install kamino-lme` and continue to write
`import kamino`. The distribution name is included in repository checks so it
cannot silently drift. The existing unrelated `kamino` PyPI project is neither
claimed nor depended upon. Published files and versions remain immutable; a bad
release is yanked and replaced by a new version rather than overwritten.
