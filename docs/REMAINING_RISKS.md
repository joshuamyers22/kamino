# Remaining risk register

Status: owned E03 register as of 2026-09-16. The machine-readable source is
[`benchmarks/e03_risks_v1.json`](../benchmarks/e03_risks_v1.json).

| ID | Severity/status | Risk and control | Owner | Review |
|---|---|---|---|---|
| E03-R01 | High / open release blocker | Independent statistical and release reviewers are not assigned. Keep beta/stable publication blocked until named reviewers approve the exact candidate and checklist. | Joshua Myers (`@joshuamyers22`) | 2026-10-16 |
| E03-R02 | Medium / accepted with control | Kamino/lme4 timings use different OS/BLAS and optimizer stacks. Publish ratios as observational only; make no relative-performance or InstEval-2× claim. | Joshua Myers | 2026-12-16 |
| E03-R03 | Medium / accepted with control | Affinity, frequency, thermal state, and all background load are not isolated. Retain raw distributions and rerun on the same profile before acting on a >20% change. | Joshua Myers | 2026-12-16 |
| E03-R04 | Medium / open design decision | SuperLU is not an SPD Cholesky backend and does not expose separate symbolic timing. Retain fill/resource limits and resolve the sparse-Cholesky ADR only with correctness, wheel, memory, and license evidence. | Joshua Myers | 2026-12-16 |
| E03-R05 | Medium / accepted with control | Capacity evidence covers named representative workloads, not every accepted structure. Scope claims to the manifests and add a case before expanding them. | Joshua Myers | 2026-12-16 |
| E03-R06 | Medium / open product limit | Refit-capable bundles and nominal bootstrap interval coverage remain unsupported/unclaimed. Keep both capabilities fail-closed pending separate gates; schema 1.3 now covers stable-core prediction recovery only. | Joshua Myers | 2026-12-16 |

Critical/High correctness, confidentiality, or release-integrity findings block
the affected release. The owner reviews this register at the listed dates and
after a numerical incident, backend change, scope expansion, or release-candidate
freeze, whichever comes first.
