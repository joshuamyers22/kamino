# Kamino release-readiness checklist

Unchecked items block a stable release. E02 establishes this gate but does not
approve version 0.0.1 or PyPI publication.

## Product and correctness

- [x] Declared journeys and invariants have evidence-linked acceptance results.
- [x] Error, boundary, interruption, concurrency, and resource paths are tested.
- [x] Artifact schema compatibility and fail-closed older-schema loading are tested.
- [x] Unsupported methods and known calibration limits are documented.
- [ ] E03 performance evidence and remaining-risk ownership are approved.
- [ ] Accountable statistical and release reviewers are assigned.

## Build and supply chain

- [x] Frozen clean builds, wheels, sdists, and installed-wheel smoke tests run in CI.
- [x] Runtime dependencies, oracle sources, container base, and actions are locked.
- [x] Wheel/sdist allowlists exclude oracle, fixtures, tests, ledgers, and work data.
- [x] Core and all-extra CycloneDX SBOMs, checksums, and a source manifest are generated.
- [x] Tag releases receive GitHub build-provenance attestations.
- [x] Gitleaks, CodeQL, dependency review, Dependabot, and push protection are configured.
- [ ] The exact candidate tag and artifacts have completed the release workflow.

## Security and privacy

- [x] Threat model covers assets, actors, boundaries, abuse cases, and residual risks.
- [x] Runtime networking/telemetry is absent and explicit file outputs are documented.
- [x] Sensitive bundle, ledger, diagnostic, retention, and deletion behavior is documented.
- [x] Private vulnerability reporting and credential-rotation response are defined.
- [x] Authentication and tenant isolation are explicitly assigned to embedding applications.

## Operations and recovery

- [x] Support scope, severity, channels, and non-contractual triage targets are documented.
- [x] Release, security-incident, and numerical-incident runbooks define forward fix and escalation.
- [x] No service SLO, dashboard, backup, or disaster-recovery claim is made for this library.
- [ ] The release runbook has been exercised against the exact candidate tag.
- [ ] Post-release monitoring owner and review date are recorded.

## Approval

- Release/commit: not yet selected
- Evidence bundle: `docs/evidence/release-e02-production-hardening.md`
- Decision: blocked for stable publication pending E03, named reviewers, and an exact candidate run
- Approvers and date: pending
