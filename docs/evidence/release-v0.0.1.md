# Version 0.0.1 release evidence

Kamino LME 0.0.1 was published on 2026-09-16 from immutable tag
[`v0.0.1`](https://github.com/joshuamyers22/kamino/releases/tag/v0.0.1) at
commit `8f6d9144b206fed05276c1bf7ea0bd4c6f099e07`. The GitHub release is marked
pre-release and the `kamino-lme` distribution is public on
[PyPI](https://pypi.org/project/kamino-lme/0.0.1/).

Release workflow
[`35165406793`](https://github.com/joshuamyers22/kamino/actions/runs/35165406793)
passed exact tag/version/changelog verification, the quality and installed-wheel
gates, all locked statistical assessments, committed-oracle verification,
artifact/SBOM validation, GitHub build-provenance attestation, GitHub release,
and PyPI OIDC trusted publication.

The public wheel is 98,015 bytes with SHA-256
`1a30433087037392f8e82e3a4fc01c81ef942c0b62f69870ffae56f6120d6c4d`.
The public sdist is 90,517 bytes with SHA-256
`db3aa62f09d390d18b504f49b5f55fef7d3caeda3856ab06d1fbd442ce7ac9c0`.
Those hashes match the GitHub assets, PyPI metadata, and published
`SHA256SUMS`. All six retained GitHub assets pass `gh attestation verify`.

A clean Python 3.12 environment installed `kamino-lme==0.0.1` from the public
PyPI index with eight total packages. Distribution metadata and
`kamino.__version__` both reported `0.0.1`; a random-intercept ML fit converged
and population/new-group prediction returned finite values with the expected
new-group identity.

The release action initially exposed the ignored one-byte `dist/.gitignore` as
`default.gitignore`. It was removed from the GitHub release; it was never in the
wheel, sdist, checksum manifest, or PyPI upload. The follow-up workflow uses
explicit distribution/evidence patterns so future releases cannot include it.

This is a pre-alpha release. E03 and accountable independent statistical and
release reviewers remain beta/stable gates.
