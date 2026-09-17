# Kamino release-readiness checklist

Unchecked items block a stable release. Version 0.0.1 is an owner-approved
pre-alpha publication and does not satisfy the remaining beta/stable gates.

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
- [x] The exact `v0.0.1` tag and artifacts completed release workflow `35165406793`.

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
- [x] The release runbook was exercised against exact tag `v0.0.1`.
- [x] Post-release monitoring owner Joshua Myers will review alerts by 2026-09-23.

## Approval

- Release/commit: `v0.0.1` / `8f6d9144b206fed05276c1bf7ea0bd4c6f099e07`
- Evidence bundle: `docs/evidence/release-v0.0.1.md`
- Decision: pre-alpha publication complete; blocked for beta/stable publication pending E03 and named reviewers
- Approvers and date: Joshua Myers, owner, 2026-09-16; independent statistical/release approval pending for beta
