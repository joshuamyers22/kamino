# Production Readiness

Status: Phase 0 evidence is complete. The Phase 1 Dyestuff/Dyestuff2 public-API
vertical slice passes locally through the block backend; the full Phase 1 and
package-release gates remain open.

| Requirement | Evidence | Status |
|---|---|---|
| F01 frame/X/Z identities | Six pinned cases: category, interaction, `||`, nesting, missing row | Pass for Phase 0 corpus |
| N01 weighted ML/REML | Dense oracle and pinned lme4 tests | Pass locally |
| N02 boundary behavior | Singular and zero-factor lme4 cases | Pass locally |
| E01 clean checks/build | `make check`, clean wheel smoke, [CI run 34914403645](https://github.com/joshuamyers22/kamino/actions/runs/34914403645) | Pass locally and remotely |
| Oracle parity | Eight fixed-theta cases, max criterion error `7.11e-15` | Pass for stated corpus |
| Optimized walking skeleton | ML/REML boundary fit, max objective difference `3.18e-12` | Pass for Phase 0 |
| Formula decision | ADR 0003 and executable spike | Pass for Phase 1 subset |
| Backend decision | ADR 0004 and executable spike | Pass for Phase 1 block scope |
| Distribution license | MIT in `LICENSE` and package metadata | Pass |
| Published oracle | Public Linux ARM64 GHCR image at manifest digest `sha256:83e891…f14b` | Pass |
| Dyestuff public fit | [ML/REML formula-to-result evidence](docs/evidence/phase1-dyestuff.md) against pinned lme4 plus independent dense algebra | Pass locally for one random-intercept formula |
| Dyestuff prediction | Population/conditional training, known-group, and new-group cases | Pass locally; maximum mean error `1.04e-7` |
| Random-intercept block backend | Eight fixed-theta dense PLS/marginal comparisons plus weighted-offset case | Pass locally; no q-by-q factorization |
| Dyestuff2 boundary fit | [ML/REML block-backend evidence](docs/evidence/phase1-dyestuff2-block.md) against pinned lme4 and closed-form invariants | Pass locally; exact theta/variance zero |
| Dyestuff2 prediction | Population/conditional training, known-group, and new-group cases | Pass locally; maximum mean error `1.78e-15` |
| Public optimizer failures | Boundary, bracketing, evaluation limit, and failure wrapping tests | Pass locally for scalar theta optimizer |
| Installed public API | Clean wheel fits and predicts off-tree with runtime dependencies | Pass locally |
| Phase 1 formula/artifact scope | Sparse/encoded formula boundary and safe result bundle | Pending |

The walking skeleton passes fixed-theta ML/REML within `3.56e-15` and optimized
objective parity within `3.18e-12`. The public scalar optimizer now exposes
structured success, boundary, bracketing, and evaluation-limit outcomes for the
one-random-intercept slice. Dyestuff2 establishes that a singular theta-zero fit
is a valid accepted result. The public oracle is platform-scoped to Linux ARM64
and referenced by digest in `oracle/manifest.json`.

No unavailable R, platform, statistical, or release check is counted as passing.
