# Support policy

Kamino is pre-alpha statistical software. The supported surface is the exact
compatibility scope in `docs/COMPATIBILITY.md`, on CPython 3.11 through 3.14 for
the Linux, macOS, and Windows rows exercised by CI. Unsupported formulas and
inference regimes are expected to fail closed.

Use GitHub issues for reproducible bugs and documentation problems. Reports
must use synthetic data and include the Kamino version or commit, Python and OS,
formula structure, ML/REML choice, observed diagnostics, and expected behavior.
Use private vulnerability reporting for security problems or any reproducer
that cannot safely be made public.

Triage targets, rather than contractual service levels, are:

| Severity | Example | Initial target |
|---|---|---:|
| Critical | Exploitable artifact loading, credential or private-data exposure | 3 business days |
| High | Materially wrong estimate, test, likelihood, or prediction in declared scope | 5 business days |
| Medium | Fail-closed defect, severe resource regression, supported-platform failure | 10 business days |
| Low | Presentation, documentation, or unsupported-scope request | Best effort |

Security incidents follow `docs/runbooks/SECURITY_INCIDENT.md`; statistical
correctness incidents follow `docs/runbooks/NUMERICAL_INCIDENT.md`. Pre-1.0
fixes may require upgrading to the newest release, and schema compatibility is
limited to the versions explicitly documented by each artifact reader.

Support excludes model-selection advice, interpretation of a user's scientific
study, validation of the independent-cluster assumption, recovery of deleted
caller files, and guarantees for unlisted Python/platform/dependency versions.
