# R Oracle

This directory generates versioned lme4 reference fixtures. The manifest records
the exact R package snapshot, source commits, system NLopt package, numerical
libraries, image identity, generator hash, and accepted output hashes.

The installed Kamino package does not depend on this environment. Do not commit
caller data, generated package caches, or unreviewed dataset copies here.

Build, run, and verify the oracle from the repository root:

```sh
make oracle
```

Generated evidence lands in the ignored `oracle/output/` directory. The target
fails if its hashes differ from the reviewed manifest and then runs the offline
lme4 fixture tests. A deliberate corpus update requires a new version or a
reviewed manifest-and-fixture change; never bless drift in place.

The local build disables nondeterministic BuildKit provenance attestations; the
source inputs remain recorded separately. The OCI identity is local until it is
published to an immutable registry, which is a remaining Phase 0 operational
gate. The Dockerfile itself uses a digest-pinned base, dated R package snapshot,
and immutable source commits.
