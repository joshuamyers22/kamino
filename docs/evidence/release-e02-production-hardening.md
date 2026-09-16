# E02 production release-hardening evidence

Status: complete on 2026-09-16 at implementation commit
[`5a4af2e`](https://github.com/joshuamyers22/kamino/commit/5a4af2e493115dd3ba17a428ca64ab7d47ab0870).
The hosted [CI candidate run](https://github.com/joshuamyers22/kamino/actions/runs/35161218558)
and [security run](https://github.com/joshuamyers22/kamino/actions/runs/35161218574)
both passed.

The E02 review applied the threat-model, release-readiness, repository-standard,
and incident-response patterns from production-project-template commit
`6526db479fced81463d1a97e25ff3ceefbe9eee2`. Library-inapplicable service
controls—authentication, tenants, service SLOs, dashboards, backups, and
deployment health checks—are assigned to embedding applications rather than
marked as implemented.

The package boundary is now executable. The source distribution contains only
Kamino Python source, build metadata, README, changelog, and MIT license. The
wheel contains only the package and distribution metadata. The verifier checks
archive paths, duplicates, symlinks, special/generated data, size limits, license
and version identity, optional extras, and every wheel RECORD hash. It also
emits normalized core/all-extra CycloneDX 1.5 SBOMs, SHA-256 checksums, and a
source/lock/artifact manifest. Repeated local generation produced identical
artifact and SBOM hashes.

The source/license review found no third-party source in the Python package.
The clubSandwich-compatible implementation uses independently written Python
over published mathematical behavior; the GPL-3 R package and derived A02
fixture remain development-only and excluded from both distributions. The same
boundary applies to lme4, lmerTest, pbkrtest, and emmeans oracle material. This
is an engineering provenance assessment, not legal advice.

The completed threat model covers formulas, malformed bundles, ledger path and
replay behavior, allocations, numerical integrity, data exfiltration, secrets,
dependency/action compromise, release replay, license-boundary drift, and
privileged-maintainer risk. Privacy documentation records live-fit retention,
identifying bundle metadata/random effects, full simulated ledger responses,
diagnostic disclosure, deletion, and the absence of runtime telemetry/networking.

The release workflow verifies exact tag/package/changelog/commit identity,
repeats quality, statistical, committed-oracle, installed-wheel, and artifact
gates, creates GitHub attestations, and publishes only GitHub prerelease assets
for `v0.*`. PyPI publication remains deliberately disabled. Security automation
adds Gitleaks history scanning, dependency/license review, and CodeQL on pinned
action commits. GitHub-native alerts, automated security updates, scanning/push
protection, and private vulnerability reporting are part of the hosted E02 gate.
GitHub reports no open Dependabot or secret-scanning alerts. Its additional
non-provider-pattern and validity-check options remain unavailable/disabled;
full-history Gitleaks is the supplemental repository detector.

Stable release approval remains correctly blocked on E03, named accountable
reviewers, and exercising the checklist against an exact candidate tag.
