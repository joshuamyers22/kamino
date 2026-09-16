# Kamino threat model

Status: reviewed for E02 on 2026-09-16.

## Scope and ownership

- System/version: Kamino 0.0.1 pre-alpha library and its GitHub release path.
- Owner: Joshua Myers (`@joshuamyers22`). A second release/security reviewer
  remains required before beta.
- Review triggers: artifact/schema/parser changes, new I/O or network behavior,
  dependency or release-workflow changes, a security or numerical incident, or
  six months without review.
- In scope: Python runtime, formulas/data, prediction bundles, bootstrap
  ledgers, wheel/sdist, CI/release automation, and the development R oracle.
- Out of scope: security of caller applications, hostile Python code already
  executing in-process, package indexes, GitHub, and operating-system controls.

## Assets, actors, and boundaries

| Asset | Sensitivity | Integrity/availability need | Owner |
|---|---|---|---|
| Caller responses, predictors, row/group labels | Potentially confidential or identifying | Must not leave the process implicitly | Caller |
| Model bundles and bootstrap ledgers | Sensitive model/derived data | Integrity checked; explicit retention | Caller |
| Numerical estimates and inference | Scientific decision material | Exact method identity and fail-closed behavior | Project owner |
| Source, lock, fixtures, and release artifacts | Public, integrity-critical | Reviewable provenance and reproducibility | Release maintainer |
| GitHub and publishing identities | Privileged credentials | Least privilege, short lived | Repository owner |

Actors include ordinary callers, a caller supplying malformed data or artifacts,
a compromised dependency or CI action, a contributor with repository access,
and a maintainer making an erroneous release. Entry points are `lmer()` formula
and data arguments, prediction data, bundle loading, ledger paths, postfit
contrasts/clusters, package installation, pull requests, and release tags.

The runtime boundary is local and R-free. The R/GPL oracle is a development
container and is excluded from both wheel and sdist. GitHub Actions is the only
hosted build boundary; release jobs use short-lived GitHub identity rather than
stored publishing credentials.

## Abuse cases and controls

| Abuse case | Preconditions | Impact | Prevent/detect/respond controls | Evidence | Residual risk |
|---|---|---|---|---|---|
| Formula injection or unintended evaluation | Attacker controls a formula | Code execution or data disclosure | Restricted grammar and supported-term validation; no Python `eval`; unsupported calls/transforms fail closed | Formula contract tests; `src/kamino/formula.py` | Formulae remains a trusted runtime parser dependency |
| Malicious prediction bundle | Caller loads hostile ZIP/JSON/NumPy content | Traversal, deserialization, memory exhaustion, model substitution | No pickle/executable format; canonical manifest, member allowlist, hashes, schema/identity checks, compressed/uncompressed limits, no duplicate/path members | `tests/test_bundle.py`; ADR 0005 | Parsing consumes bounded local CPU and memory |
| Ledger path manipulation or concurrent replay | Attacker controls path or reuses partial work | Overwrite, disclosure, duplicated/incorrect bootstrap outcomes | Symlink refusal, owner-only permissions, exclusive writer lock, atomic writes, response hashes, seed/model identity, complete status ledger | `tests/test_i01_bootstrap.py`; ADR 0008 | Parent-directory security and backups are caller responsibilities |
| Resource-exhausting data, cluster, or hypothesis | Very large or adversarial dimensions | Process denial of service | Observation, parameter, sparse-fill, dense-matrix, bundle, ledger, bootstrap, profile, KR, and CR2 preflight ceilings | Resource and negative-path tests; `PRODUCTION_READINESS.md` | Ordinary fitting can still be compute intensive inside accepted ceilings |
| Non-finite or ill-conditioned numerical input | Crafted or pathological data | Crash or misleading result | Finite/type/rank/condition checks, structured unavailable results, convergence/boundary diagnostics, oracle and calibration gates | Test suite and method ADRs | Floating-point/BLAS defects and novel pathologies remain possible |
| Runtime data exfiltration | Sensitive caller data is fitted | Confidentiality loss | No runtime network calls or telemetry; explicit-only file writes; privacy documentation | `PRIVACY.md`; dependency/source review | In-process caller code and native dependencies have the process's privileges |
| Credential exposure in source/history/artifacts | Contributor commits a secret | Account compromise | GitHub push protection/secret scanning, full-history Gitleaks CI, artifact member allowlists, private reporting and rotation runbook | Security workflow; release artifact verifier | Provider detectors are not exhaustive |
| Dependency or CI action compromise | Registry or upstream account is compromised | Malicious installation/build/release | Frozen `uv.lock`, immutable action SHAs, Dependabot/dependency review, weekly CodeQL/Gitleaks, core/all-extra SBOMs, reviewed updates | `uv.lock`; workflows; SBOM gate | PyPI/GitHub and native wheel trust cannot be eliminated |
| Unauthorized or replayed release | Attacker or maintainer pushes a misleading tag | Compromised public artifact | Tag/version/changelog/commit checks, CI rebuild, checksums, GitHub artifact attestations, immutable source commit, no long-lived publisher token | Release workflow and runbook | Repository-admin compromise can bypass repository policy |
| GPL oracle material enters MIT artifacts | Build configuration drifts | License violation or unexpected runtime dependency | Minimal sdist/wheel allowlists, full oracle exclusion, license inventory, artifact tests, separate SBOMs | `tools/release_artifacts.py`; source/license inventory | License conclusions require human review and are not legal advice |
| Privileged insider weakens evidence | Maintainer alters implementation and fixture together | False statistical or security claim | CODEOWNERS, PR checklist, immutable reference identities, separate implementation/oracle evidence, hosted gates | `AGENTS.md`; `PROJECT_PLAN.md`; workflows | A sole owner can administratively override controls; second reviewer pending |
| Authentication/tenant-isolation bypass | Library is embedded in a service | Cross-user access | Not implemented by Kamino; embedding service must authenticate and isolate users and files | Scope statement | Misconfigured host applications remain outside scope |

## Decisions and remaining risks

- Accept local in-process execution and native numerical dependencies for the
  library model; Kamino is not a sandbox.
- Accept that live results and explicit artifacts contain sensitive derived
  state and cannot be made anonymous by renaming columns.
- Do not publish to PyPI in E02. The tag workflow produces an attested GitHub
  prerelease; trusted PyPI publishing requires a separately approved release.
- Block beta until an accountable statistical reviewer and release maintainer
  are assigned. Owner: Joshua Myers; review at the beta decision.
- Treat Critical/High correctness or confidentiality findings as release
  blockers and follow the applicable incident runbook.
