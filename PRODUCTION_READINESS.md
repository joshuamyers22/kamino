# Production Readiness

Status: Phase 0 evidence is complete. The Phase 1 Dyestuff, Dyestuff2, and
Sleepstudy public-API slices pass locally through the compact formula boundary
and block backend; the full Phase 1 and package-release gates remain open.

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
| Single-group block backend | Eight synthetic and ten Sleepstudy fixed-theta slope cases plus random-intercept cases | Pass locally; no q-by-q factorization |
| Dyestuff2 boundary fit | [ML/REML block-backend evidence](docs/evidence/phase1-dyestuff2-block.md) against pinned lme4 and closed-form invariants | Pass locally; exact theta/variance zero |
| Dyestuff2 prediction | Population/conditional training, known-group, and new-group cases | Pass locally; maximum mean error `1.78e-15` |
| Compact formula boundary | [Encoded formula evidence](docs/evidence/phase1-compact-formula.md), including 4,096-level construction | Pass locally; no dense random indicator stored or requested from Formulae |
| Sleepstudy correlated slope | [ML/REML fit and prediction evidence](docs/evidence/phase1-sleepstudy.md) against pinned lme4 and dense algebra | Pass locally |
| Singular slope covariance | Synthetic ML/REML optimized fixture | Pass locally; exact zero slope diagonal returned |
| Public optimizer failures | Boundary, bracketing, evaluation limit, and failure wrapping tests | Pass locally for scalar and vector paths |
| Installed public API | Clean wheel fits and predicts off-tree with runtime dependencies | Pass locally |
| Phase 1 artifact scope | Safe result bundle | Pending |

The walking skeleton passes fixed-theta ML/REML within `3.56e-15` and optimized
objective parity within `3.18e-12`. The public optimizers expose structured
success, boundary, bracketing, and evaluation-limit outcomes. Dyestuff2 and the
synthetic slope fixture establish valid exact covariance boundaries. Sleepstudy
extends the public claim to one correlated numeric slope. The compact boundary
is verified structurally and for a 4,096-level case, but million-row performance
and peak RSS remain an open benchmark gate. The public oracle is platform-scoped
to Linux ARM64 and referenced by digest in `oracle/manifest.json`.

No unavailable R, platform, statistical, or release check is counted as passing.
