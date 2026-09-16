# Production Readiness

Status: Phase 0 and Phase 1 evidence is complete. Phase 2 N03/F02 and Phase 3
I01 add the general sparse backend, rank/estimability, categorical random terms,
and deterministic parametric bootstrap. Later artifact-recovery, calibrated
interval, inference-method, and package-release gates remain open.

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
| Independent same-group terms | [Numeric double-bar/explicit-term evidence](docs/evidence/phase1-independent-terms.md) against pinned lme4 and dense algebra | Pass locally; exact zero cross-covariance and boundary slope returned |
| Shared model frame/fixed expansion | [Categorical, row-selection, weight, and offset evidence](docs/evidence/phase1-model-frame.md) against pinned lme4 | Pass locally; treatment/sum ML/REML fits and predictions within declared tolerances |
| Public optimizer failures | Boundary, bracketing, evaluation limit, and failure wrapping tests | Pass locally for scalar and vector paths |
| Installed public API | Clean wheel fits and predicts off-tree with runtime dependencies | Pass locally |
| Safe prediction bundle | [Artifact evidence](docs/evidence/phase1-safe-bundle.md) and ADR 0005 | Pass locally for the verified Phase 1 formulas |
| Single-group resource benchmark | [Manifested million-row evidence](docs/evidence/phase1-single-group-resource.md) for random intercept, categorical fixed, correlated slope, and independent terms under ML/REML; [CI run 35017301412](https://github.com/joshuamyers22/kamino/actions/runs/35017301412) | Expanded eight-case gate passes locally at 1,316 MB and remotely at 987 MB |
| General sparse backend | [ADR 0006](docs/adr/0006-general-sparse-backend.md) and [N03 evidence](docs/evidence/phase2-general-sparse.md) | Pastes/Penicillin fixed-theta and ML/REML final fits pass locally |
| Nested/crossed structure | Pinned Pastes slash nesting and Penicillin crossing plus independent dense PLS | Pass locally; no missing parent-child/crossed coupling |
| Crossed resource scale | Manifested InstEval ML/REML at 73,421 rows, 4,100 random coefficients, and 146,842 stored Z nonzeros; [CI run 35025175111](https://github.com/joshuamyers22/kamino/actions/runs/35025175111) | Pass locally at 357 MB and remotely at 286 MB |
| N03 platform/wheel gate | Linux Python 3.11/3.14, macOS 3.12, Windows 3.12, clean wheel/sdist, and both resource manifests in CI run 35025175111 | Pass |
| Fixed rank/estimability | [ADR 0007](docs/adr/0007-rank-estimability-categorical-random.md) and [F02 evidence](docs/evidence/phase2-f02-rank-categorical.md) | Four pinned QR/drop contracts, ML/REML fits, linear functions, and non-estimable new rows pass locally and in [CI run 35032002878](https://github.com/joshuamyers22/kamino/actions/runs/35032002878) |
| Categorical random terms | Six fixed-theta and six final single-group fits plus two fixed-theta and two final crossed fits against pinned lme4 and dense PLS | Pass locally and in hosted CI for treatment/sum ordinary bars; categorical `||` remains rejected |
| F02 new-data corpus | Known/new groups, treatment/sum and mixed fixed/random contrasts, reordered categories, unknown fixed levels, and non-estimable aliases | Pass locally and in hosted CI; maximum prediction error `2.32e-5` |
| F02 platform/wheel gate | Linux Python 3.11/3.14, macOS 3.12, Windows 3.12, clean wheel/sdist, installed-package smoke, and resource manifests in CI run 35032002878 | Pass |
| F02 artifact boundary | Rank-deficient, categorical-random, and nested/crossed saves | Fail closed pending full Phase 2 artifact recovery, as specified |
| I01 simulation/refit | [ADR 0008](docs/adr/0008-parametric-bootstrap-ledger.md) and [I01 evidence](docs/evidence/phase3-i01-bootstrap.md) | Conditional/unconditional weighted draws and exact response refits pass block and sparse backend tests |
| I01 pinned refits | Four stored-response ML/REML × conditional/unconditional cases against lme4 2.0-6 | Pass; maximum objective error `5.68e-14`, maximum random-variance error `4.77e-5` |
| I01 locked assessment | 10,000 draws per mode, 1,000 refits, eight serial/parallel cases in `statistical/i01_report.json` | Pass locally and in [CI run 35038175252](https://github.com/joshuamyers22/kamino/actions/runs/35038175252); zero failed refits, 93 valid singular fits, exact worker identity, one-sided 95% failure upper bound `0.002992` |
| I01 private ledger | Atomic canonical manifest, non-object response arrays, integrity/identity/resource checks, interruption/resume, exclusive writer lock, and corruption rejection | Pass locally and across the hosted platform matrix; prediction bundles remain response-free and inference-incapable |
| I01 platform/wheel gate | Linux Python 3.11/3.14, macOS 3.12, Windows 3.12, clean wheel/sdist, installed-package smoke, locked assessment, and resource manifests in CI run 35038175252 | Pass |
| Bootstrap interval coverage | Percentile/basic descriptive summaries | Unclaimed pending a separately locked outer-calibration study |

The walking skeleton passes fixed-theta ML/REML within `3.56e-15` and optimized
objective parity within `3.18e-12`. The public optimizers expose structured
success, boundary, bracketing, and evaluation-limit outcomes. Dyestuff2 and the
synthetic slope fixtures establish valid exact covariance boundaries. Sleepstudy
extends the public claim to correlated and independent numeric slopes. The compact
boundary
is verified structurally and for a 4,096-level case. Safe bundles preserve exact
new-data prediction while omitting responses and training rows. The manifested
million-row/10,000-group resource cases pass under ML and REML for all public
covariance structures and the expanded fixed design without dense `Z`; timings
remain machine-specific rather than release promises. The coupled sparse path
adds Pastes, Penicillin, and InstEval while retaining the block fast path.
F02 adds the reference retained-column policy, coefficient-aligned null-space
state, explicit estimability, and ordinary categorical random slopes through
both backends. F02 and nested/crossed bundles deliberately remain unavailable
until the later artifact-recovery schema gate. I01 adds deterministic simulation,
response refitting, a bounded worker path, private resumable response ledgers,
and exact failure-rate bounds without claiming nominal interval coverage. The public oracle is
platform-scoped to Linux ARM64 and referenced by digest in
`oracle/manifest.json`.

No unavailable R, platform, statistical, or release check is counted as passing.
