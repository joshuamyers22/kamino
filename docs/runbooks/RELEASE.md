# Release runbook

Kamino releases are built from an exact reviewed tag. Local artifacts are never
uploaded. The owner-approved tag workflow publishes the tested distributions to
PyPI as `kamino-lme` through a protected `pypi` environment and OIDC trusted
publishing; the installed import remains `kamino`.

## Prepare

1. Assign a release maintainer and statistical reviewer; neither role is
   satisfied merely by the implementation author rerunning checks.
2. Resolve Critical/High findings and record owned residual risks.
3. Move user-visible entries from `Unreleased` to a dated
   `## [MAJOR.MINOR.PATCH] - YYYY-MM-DD` section, update `pyproject.toml`, and
   update compatibility/schema documentation when applicable.
4. From a clean frozen environment run `make check`, `make wheel-smoke`,
   `make statistical`, `make oracle-verify`, and `make supply-chain`.
5. Review both CycloneDX SBOMs, `SHA256SUMS`, the minimal sdist contents, the
   wheel metadata, dependency/security alerts, and the release-readiness
   checklist. Merge only after CI and Security workflows pass.

## Publish and verify

1. Create the exact `vMAJOR.MINOR.PATCH` tag on the reviewed commit and push the
   tag. The release workflow verifies tag/version/changelog identity and reruns
   all non-container release gates.
2. Confirm GitHub records build-provenance attestations for every distribution
   and supply-chain artifact and that the release is marked prerelease for
   `v0.*`.
3. Confirm the separate `pypi-publish` job deployed through the `pypi`
   environment, then verify the version and hashes at
   `https://pypi.org/project/kamino-lme/`.
4. Download the artifacts into an empty directory, verify `SHA256SUMS`, and run
   `gh attestation verify --repo joshuamyers22/kamino <artifact>` for each file.
5. Install `kamino-lme` from PyPI into a clean supported Python environment and
   repeat the
   documented public fit/prediction smoke journey.
6. Link the release, commit, hosted runs, SBOMs, oracle digest, and approval in
   the release-readiness record. Announce only the compatibility scope actually
   gated.

## Recovery

Stop publication if any identity, hash, attestation, oracle, statistical, or
installed-wheel check differs. A bad GitHub prerelease is marked affected and
removed from the recommended path; preserve hashes and incident evidence, then
forward-fix from a new commit and version. Never move or reuse a published tag.
If a future PyPI release is affected, yank it rather than deleting history and
publish a new version. Follow the numerical or security incident runbook when
correctness or confidentiality is implicated.
