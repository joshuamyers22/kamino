# Production Readiness

Status: Phase 0 core evidence passes locally; phase exit and public release are
not approved.

| Requirement | Evidence | Status |
|---|---|---|
| F01 frame/X/Z identities | Six pinned cases: category, interaction, `||`, nesting, missing row | Pass for Phase 0 corpus |
| N01 weighted ML/REML | Dense oracle and pinned lme4 tests | Pass locally |
| N02 boundary behavior | Singular and zero-factor lme4 cases | Pass locally |
| E01 clean checks/build | `make check`, clean wheel smoke | Pass locally; CI pending |
| Oracle parity | Eight fixed-theta cases, max criterion error `7.11e-15` | Pass for stated corpus |
| Optimized walking skeleton | ML/REML boundary fit, max objective difference `3.18e-12` | Pass for Phase 0 |
| Formula decision | ADR 0003 and executable spike | Pass for Phase 1 subset |
| Backend decision | ADR 0004 and executable spike | Pass for Phase 1 block scope |
| Distribution license | Owner decision | Pending |

Open Phase 0 exit items are remote OS/Python matrix results, a published oracle
image identity, and an owner-selected distribution license. The walking skeleton
passes fixed-theta ML/REML within `3.56e-15` and optimized objective parity within
`3.18e-12`. The optimizer is intentionally experimental until Phase 1 adds
diagnostics and failure semantics.

No unavailable R, platform, statistical, or release check is counted as passing.
