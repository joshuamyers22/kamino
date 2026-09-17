# Production Readiness

Status: Phase 0 and Phase 1 evidence is complete. Phase 2 N03/F02, Phase 3 I01,
and Phase 4 I02/I03 add the general sparse backend, rank/estimability, categorical
random terms, deterministic parametric bootstrap, and calibrated Satterthwaite
and Kenward–Roger/profile inference. Phase 5 A01 adds the bounded postfit
capability spike, A02 adds validated cluster-robust inference, and E02/E03 add
release hardening plus reproducible performance/risk evidence. Stable-core
artifact recovery is complete; beta/stable reviewer gates remain open.

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
| Distribution identity | `kamino-lme` on PyPI installs the `kamino` import package; ADR 0014 | Version 0.0.1 published through trusted OIDC and verified from the public index |
| E02 threat/privacy boundary | [Threat model](docs/THREAT_MODEL.md), [privacy policy](PRIVACY.md), and ADR 0013 | Formula, artifact, ledger, resource, native dependency, retained-data, scanning, and response boundaries reviewed |
| E02 artifact boundary | `tools/release_artifacts.py` and minimal Hatch sdist inclusion | Wheel/sdist identity, path, link, size, license, optional-extra, RECORD, and development-oracle exclusion pass locally and in hosted CI |
| E02 SBOM/provenance | Core/all-extra CycloneDX 1.5, SHA-256 manifest, exact tag verification, GitHub attest-build-provenance workflow | Version 0.0.1 checksums and all six GitHub attestations verified after publication |
| E02 security automation | Pinned Gitleaks, dependency review, CodeQL, Dependabot, GitHub secret scanning/push protection, private vulnerability reporting | Configured; CodeQL and full-history secret scanning pass in the hosted E02 security run |
| E02 operations | [Release checklist](checklists/RELEASE_READINESS.md), support policy, and release/security/numerical runbooks | Complete for pre-alpha; stable approval remains blocked on named reviewers and exact candidate exercise |
| E03 performance/risk | [Repeated ten-case evidence](docs/evidence/release-e03-performance-risk.md), [ADR 0015](docs/adr/0015-e03-performance-claim-boundary.md), and [owned risks](docs/REMAINING_RISKS.md) | Pass locally at protocol commit `4db3174`; reports/hashes verify, all correctness/resource gates pass, comparative speed and InstEval 2× remain unclaimed |
| Published oracle | Public Linux ARM64 GHCR image at manifest digest `sha256:17e45268…eb4d` | Pass |
| Inference companion oracle | Pinned lmerTest 3.1-3, pbkrtest 0.5.5, and clubSandwich 0.7.0 outputs plus lme4 profiles | Pass locally; rebuilt image published by immutable GHCR digest |
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
| Safe prediction bundle | [Artifact evidence](docs/evidence/phase1-safe-bundle.md), [stable-core recovery evidence](docs/evidence/phase2-artifact-recovery.md), ADR 0005, and ADR 0016 | Schema 1.3 passes locally for the stable-core prediction scope; schemas 1.0–1.2 remain readable |
| Single-group resource benchmark | [Manifested million-row evidence](docs/evidence/phase1-single-group-resource.md) for random intercept, categorical fixed, correlated slope, and independent terms under ML/REML; [CI run 35017301412](https://github.com/joshuamyers22/kamino/actions/runs/35017301412) | Expanded eight-case gate passes locally at 1,316 MB and remotely at 987 MB |
| General sparse backend | [ADR 0006](docs/adr/0006-general-sparse-backend.md) and [N03 evidence](docs/evidence/phase2-general-sparse.md) | Pastes/Penicillin fixed-theta and ML/REML final fits pass locally |
| Nested/crossed structure | Pinned Pastes slash nesting and Penicillin crossing plus independent dense PLS | Pass locally; no missing parent-child/crossed coupling |
| Crossed resource scale | Manifested InstEval ML/REML at 73,421 rows, 4,100 random coefficients, and 146,842 stored Z nonzeros; [CI run 35025175111](https://github.com/joshuamyers22/kamino/actions/runs/35025175111) | Pass locally at 357 MB and remotely at 286 MB |
| N03 platform/wheel gate | Linux Python 3.11/3.14, macOS 3.12, Windows 3.12, clean wheel/sdist, and both resource manifests in CI run 35025175111 | Pass |
| Fixed rank/estimability | [ADR 0007](docs/adr/0007-rank-estimability-categorical-random.md) and [F02 evidence](docs/evidence/phase2-f02-rank-categorical.md) | Four pinned QR/drop contracts, ML/REML fits, linear functions, and non-estimable new rows pass locally and in [CI run 35032002878](https://github.com/joshuamyers22/kamino/actions/runs/35032002878) |
| Categorical random terms | Six fixed-theta and six final single-group fits plus two fixed-theta and two final crossed fits against pinned lme4 and dense PLS | Pass locally and in hosted CI for treatment/sum ordinary bars; categorical `||` remains rejected |
| F02 new-data corpus | Known/new groups, treatment/sum and mixed fixed/random contrasts, reordered categories, unknown fixed levels, and non-estimable aliases | Pass locally and in hosted CI; maximum prediction error `2.32e-5` |
| F02 platform/wheel gate | Linux Python 3.11/3.14, macOS 3.12, Windows 3.12, clean wheel/sdist, installed-package smoke, and resource manifests in CI run 35032002878 | Pass |
| Stable-core artifact recovery | Rank-deficient, categorical-random, nested/crossed, and crossed-categorical saves | Schema 1.3 preserves rank/null-space, encoder, term, prediction, estimability, and new-group identity; malformed state fails before construction |
| I01 simulation/refit | [ADR 0008](docs/adr/0008-parametric-bootstrap-ledger.md) and [I01 evidence](docs/evidence/phase3-i01-bootstrap.md) | Conditional/unconditional weighted draws and exact response refits pass block and sparse backend tests |
| I01 pinned refits | Four stored-response ML/REML × conditional/unconditional cases against lme4 2.0-6 | Pass; maximum objective error `5.68e-14`, maximum random-variance error `4.77e-5` |
| I01 locked assessment | 10,000 draws per mode, 1,000 refits, eight serial/parallel cases in `statistical/i01_report.json` | Pass locally and in [CI run 35038175252](https://github.com/joshuamyers22/kamino/actions/runs/35038175252); zero failed refits, 93 valid singular fits, exact worker identity, one-sided 95% failure upper bound `0.002992` |
| I02 Satterthwaite derivatives | [ADR 0009](docs/adr/0009-satterthwaite-full-variance-parameters.md) and [I02 evidence](docs/evidence/phase4-i02-satterthwaite.md) | Pass locally and in [CI run 35042478890](https://github.com/joshuamyers22/kamino/actions/runs/35042478890) for full `(theta, sigma)` Hessian/covariance/Jacobian and one-/multi-DF tests; boundary/unstable cases fail closed |
| I02 locked calibration | 2,000 predeclared REML simulations in `statistical/i02_report.json` | Pass locally and in hosted CI: 1,988 available, one-DF rejection `0.04980`, joint rejection `0.05181`, 12 explicit boundary-unavailable fits |
| I03 KR adjustment/tests | [ADR 0010](docs/adr/0010-kenward-roger-and-likelihood-profiles.md) and [I03 evidence](docs/evidence/phase4-i03-kr-profile.md) | Four Dyestuff/Sleepstudy ML/REML source cases match pinned pbkrtest covariance, information, derivatives, scaling, DF, and tests; weighted/boundary/over-limit cases fail closed locally and in [CI run 35120538919](https://github.com/joshuamyers22/kamino/actions/runs/35120538919) |
| I03 likelihood profiles | SD/correlation, residual-scale, and fixed-effect trajectories against pinned lme4 2.0-6 | ML baselines, constrained objectives, signed-root deviances, adaptive intervals, ordering, and REML-to-ML provenance pass locally and in hosted CI |
| I03 locked calibration | 1,000 predeclared regular Gaussian simulations in `statistical/i03_report.json` | Pass locally and in hosted CI: KR rejection `0.04213` on 997 available fits; profile rejection `0.04700`, coverage `0.953`, and no unavailable profiles |
| A01 capability contract | [ADR 0011](docs/adr/0011-postfit-capability-contract.md) and [A01 evidence](docs/evidence/phase5-a01-postfit.md) | Full/retained labels, null-space estimability, immutable covariance identity, and asymptotic/residual-t/Satterthwaite/KR pairing pass locally and in [CI run 35146918117](https://github.com/joshuamyers22/kamino/actions/runs/35146918117) |
| A01 reference grids | Pinned lme4 2.0-6/emmeans 2.0.2 unbalanced-factorial fixture | Five weighting modes and four pairwise adjustments pass; maximum marginal-mean error `8.60e-12`, SE error `6.91e-10`, adjusted-p error `1.23e-9` |
| A01 external adapters | Exact optional statsmodels 0.14.6 OLS/WLS and MixedLM contract tests | Formula design, rank deficiency, direct tests/predictions, covariance-block identity, and robust-covariance DF refusal pass locally |
| A01 marginaleffects feasibility | Official Python interface review | Direct Kamino adapter deferred: no public third-party native-result adapter contract was found; no unsupported wrapper is claimed |
| A01 distribution boundary | Capability layer in the Kamino wheel; statsmodels remains optional | Local spike complete; separate postfit distribution, hosted candidate evidence, and its independent release checklist remain open Phase 5 gates |
| A02 CR0/CR1/CR2 | [ADR 0012](docs/adr/0012-cluster-robust-cr2.md) and [A02 evidence](docs/evidence/phase5-a02-cluster-robust.md) | Pass locally and in hosted CI: three pinned clubSandwich cases match covariance, targets, adjustments, scores, Satterthwaite tests, and HTZ tests; prior weights and non-nested clusters fail closed |
| A02 locked calibration | 2,000 predeclared nested Gaussian simulations in `statistical/a02_report.json` | Pass locally and in hosted CI: 2,000 available, one-DF rejection `0.0525`, joint HTZ rejection `0.0505` |
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
both backends. Schema 1.3 completes prediction-only recovery for F02 and
nested/crossed models without adding training or refit state. I01 adds deterministic simulation,
response refitting, a bounded worker path, private resumable response ledgers,
and exact failure-rate bounds without claiming nominal interval coverage. The public oracle is
platform-scoped to Linux ARM64 and referenced by digest in
`oracle/manifest.json`.

No unavailable R, platform, statistical, or release check is counted as passing.
