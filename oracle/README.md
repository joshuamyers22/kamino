# R Oracle

This directory generates versioned lme4 reference fixtures. The manifest records
the exact R package snapshot, source commits, system NLopt package, numerical
libraries, image identity, generator hash, and accepted output hashes.

The installed Kamino package does not depend on this environment. Do not commit
caller data, generated package caches, or unreviewed dataset copies here.
The tracked Dyestuff, Dyestuff2, Sleepstudy, Pastes, Penicillin, and InstEval
fixtures are reviewed exceptions, including the separate independent-term
Sleepstudy fixture:
their GPL-2.0-or-later boundary and source are recorded in
`fixtures/v1/README.md` and `THIRD_PARTY_NOTICES.md`, and they are excluded from
the Python wheel and sdist. The model-frame fixture is Kamino-owned MIT
synthetic data.

Build, run, and verify the oracle from the repository root:

```sh
make oracle
```

Generated evidence lands in the ignored `oracle/output/` directory. The target
fails if its hashes differ from the reviewed manifest and then runs the offline
lme4 fixture tests. A deliberate corpus update requires a new version or a
reviewed manifest-and-fixture change; never bless drift in place.

The local build disables nondeterministic BuildKit provenance attestations; the
source inputs remain recorded separately. The reviewed Linux ARM64 image is
publicly available at the immutable reference:

```text
ghcr.io/joshuamyers22/kamino-oracle@sha256:83e891ef07ea9dd45eee788fbe5d79ad0f44852dca49143c70c3d0c15ec9f14b
```

Use the digest, not the mutable discovery tag, for evidence generation. The
Dockerfile uses a digest-pinned base, dated R package snapshot, and immutable
source commits. Other architectures require a separately built and reviewed
oracle profile; this publication does not claim cross-architecture identity.
