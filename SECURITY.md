# Security policy

## Supported versions

Kamino is pre-alpha. The latest published pre-1.0 release and current `main`
branch receive security fixes; older pre-1.0 releases may receive a forward fix
only.

| Version | Supported |
|---|---|
| 0.0.1 | Yes |
| Current `main` | Yes |
| Older commits or unreviewed forks | No |

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/joshuamyers22/kamino/security/advisories/new).
Do not open a public issue for suspected vulnerabilities and do not attach
private datasets, model bundles, bootstrap ledgers, credentials, or identifying
group labels. A minimal synthetic reproducer is preferred.

The owner targets acknowledgement within three business days and an initial
severity/containment decision within seven business days. These are triage
targets, not a contractual service-level agreement. Coordinated disclosure
timing depends on exploitability, affected releases, and whether downstream
users need time to update.

## Security boundary

Kamino is an in-process numerical library, not a sandbox or network service. It
does not authenticate users, make runtime network calls, or emit telemetry.
Callers control filesystem access and must not run untrusted Python code in the
same process. Formula strings, model data, prediction bundles, and bootstrap
ledger paths cross trust boundaries and are validated within the documented
scope.

See the [threat model](docs/THREAT_MODEL.md), [privacy policy](PRIVACY.md), and
[security incident runbook](docs/runbooks/SECURITY_INCIDENT.md).
