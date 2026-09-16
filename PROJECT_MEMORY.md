# Kamino Project Memory

This is a bounded evidence index, not a task log or source of truth.

## Durable constraints

| Key | Constraint | Evidence | Verified |
|---|---|---|---|
| `reference-primary` | Compatibility targets lme4 2.0-6 ordinary Gaussian unstructured LMMs at an immutable CRAN mirror commit. | `PROJECT_PLAN.md` §2; `oracle/manifest.json` | 2026-09-14 |
| `runtime-r-free` | R is an oracle/build-time tool and never a Kamino runtime dependency. | `PROJECT_BRIEF.md`; `docs/adr/0002-oracle-boundary.md` | 2026-09-14 |
| `three-oracles` | Statistical changes require independent dense algebra, fixed-theta R outputs, and final-fit comparisons as applicable. | `PROJECT_PLAN.md` §10 | 2026-09-14 |
| `fail-closed` | Unsupported formulas, structures, and inference methods fail explicitly. | `PROJECT_PLAN.md` §§2, 8 | 2026-09-14 |

## Accepted decisions

| Key | Decision | Evidence | Verified |
|---|---|---|---|
| `native-core` | Kamino implements the requested lme4-style PLS core; Statsmodels is a comparator/adapter target. | `docs/adr/0001-native-core.md` | 2026-09-14 |
| `oracle-container` | The pinned R oracle is isolated from the Python runtime in a reproducible container. | `docs/adr/0002-oracle-boundary.md` | 2026-09-14 |
| `formula-backend` | Use a restricted Formulae adapter with owned row selection, offset extraction, and label canonicalization. | `docs/adr/0003-formula-backend.md` | 2026-09-14 |
| `phase1-block` | Implement batched independent-group block Cholesky; do not claim a general sparse backend. | `docs/adr/0004-phase1-block-backend.md` | 2026-09-14 |
| `distribution-license` | Distribute Kamino under the MIT License. | `LICENSE`; `pyproject.toml` | 2026-09-14 |
| `public-repository` | Host the source publicly at `joshuamyers22/kamino`. | GitHub repository; hosted CI evidence | 2026-09-14 |
| `oracle-publication` | Publish the reviewed Linux ARM64 oracle publicly by immutable GHCR digest. | `oracle/manifest.json`; GHCR package | 2026-09-14 |
| `phase1-first-slice` | Initially keep the public fit fail-closed to one fixed intercept and one random intercept until broader final-fit evidence passes. | `docs/COMPATIBILITY.md`; Dyestuff oracle tests | 2026-09-15 |
| `phase1-block-runtime` | Route supported public fits through the owned single-group block evaluator; retain dense PLS and marginal routes as independent small-model checks. | `src/kamino/block.py`; Dyestuff2 evidence | 2026-09-15 |
| `phase1-compact-formula` | Encode the production random design as one immutable group index plus small row covariates; Formulae receives only the fixed-effects formula, while explicit dense construction remains test-only. | `src/kamino/formula.py`; `docs/evidence/phase1-compact-formula.md` | 2026-09-15 |
| `phase1-correlated-slope` | Generalize the compact single-group block path to correlated numeric intercept/slopes and expose the verified Sleepstudy ML/REML fit and prediction slice. | `src/kamino/block.py`; `docs/evidence/phase1-sleepstudy.md` | 2026-09-15 |
| `phase1-safe-bundle` | Persist only prediction-required state in an atomic versioned ZIP with canonical JSON, non-object float64 arrays, strict limits and integrity validation; load as a capability-limited prediction model. | `docs/adr/0005-safe-prediction-bundles.md`; `docs/evidence/phase1-safe-bundle.md` | 2026-09-15 |
| `phase1-resource-benchmark` | Gate the complete public single-group ML/REML subset at 1M observations/10k groups with reusable theta-independent block assembly, no dense `Z`, and a shared-CI peak-RSS ceiling; keep timings non-authoritative outside pinned hardware. | `benchmarks/single_group_v1.json`; `docs/evidence/phase1-single-group-resource.md` | 2026-09-15 |
| `phase1-independent-terms` | Preserve independent covariance terms sharing one grouping factor while retaining their design cross-products in a joint block solve; expose equivalent numeric double-bar and explicit split formulas only for one numeric predictor. | `docs/evidence/phase1-independent-terms.md`; `oracle/manifest.json` | 2026-09-15 |
| `bundle-schema-1.1` | Persist covariance-term boundaries in prediction bundles; retain schema-1.0 loading as one correlated term. | `docs/adr/0005-safe-prediction-bundles.md`; `tests/test_bundle.py` | 2026-09-15 |
| `phase1-shared-frame` | Select response, fixed/random/group columns, weights, both offset sources, and subset through one frame; preserve retained/omitted/excluded row IDs. | `docs/evidence/phase1-model-frame.md`; `tests/test_model_frame_public_api.py` | 2026-09-15 |
| `phase1-fixed-expansion` | Support additive numeric/categorical fixed effects, pairwise `*`, and treatment/sum contrasts through an owned serializable encoder checked against Formulae and pinned lme4. | `docs/adr/0003-formula-backend.md`; `oracle/fixtures/v1/model_frame.json` | 2026-09-15 |
| `bundle-schema-1.2` | Persist the owned fixed encoder and formula-offset names; retain strict schema-1.0/1.1 reading for their original numeric design scope. | `docs/adr/0005-safe-prediction-bundles.md`; `tests/test_bundle.py` | 2026-09-15 |
| `phase2-general-sparse` | Route coupled nested/crossed random-intercept terms to SciPy SuperLU with a cached structural pattern, symmetric controls, minimum-degree ordering, explicit fill/resource limits, and no dense Z; retain the Phase 1 block route for proven independent structures. | `docs/adr/0006-general-sparse-backend.md`; `docs/evidence/phase2-general-sparse.md` | 2026-09-15 |
| `phase2-rank-estimability` | Reproduce the pinned non-LAPACK QR retained-column policy and preserve full/retained identity plus a coefficient-aligned null-space basis; non-estimable contrasts and rows return explicit unavailable results. | `docs/adr/0007-rank-estimability-categorical-random.md`; `docs/evidence/phase2-f02-rank-categorical.md` | 2026-09-15 |
| `phase2-categorical-random` | Encode treatment/sum categorical ordinary-bar terms as compact row covariates through the block and sparse backends; keep fixed and random contrast controls separate and categorical double-bar fail-closed. | `docs/adr/0007-rank-estimability-categorical-random.md`; `oracle/fixtures/v1/f02_rank_categorical.json` | 2026-09-15 |
| `phase3-i01-bootstrap` | Use replicate/purpose-separated PCG64DXSM streams, exact response replacement, bounded private refit workspaces, and an atomic private response ledger; retain every failure without redraw and count singular fits as valid outcomes. | `docs/adr/0008-parametric-bootstrap-ledger.md`; `statistical/i01_report.json`; `oracle/fixtures/v1/i01_bootstrap.json` | 2026-09-15 |
| `phase4-i02-satterthwaite` | Differentiate the unprofiled deviance and beta covariance in full `(theta, sigma)` coordinates; use `2 H_D^-1`, rank-aware tests, and fail closed at boundaries or unstable curvature. | `docs/adr/0009-satterthwaite-full-variance-parameters.md`; `statistical/i02_report.json` | 2026-09-15 |
| `phase4-i03-kr-profile` | Keep KR as a bounded dense REML component-matrix adjustment and profiles as separate named-target ML nuisance optimizations; neither silently substitutes another inference method. | `docs/adr/0010-kenward-roger-and-likelihood-profiles.md`; `statistical/i03_report.json` | 2026-09-16 |
| `phase5-a01-postfit` | Bind full coefficient identity, estimability, covariance, and DF in one versioned analysis contract; use explicit Kamino, statsmodels OLS/WLS, and statsmodels MixedLM adapters, and defer a direct marginaleffects integration without a public upstream adapter contract. | `docs/adr/0011-postfit-capability-contract.md`; `oracle/fixtures/v1/a01_postfit.json` | 2026-09-16 |
| `phase5-a02-cluster-robust` | Use fitted marginal targets and marginal residuals for CR0/CR1/CR2; require all random grouping factors nested in declared independent clusters, refuse prior weights, and bind CR2 covariance to Satterthwaite/HTZ inference. | `docs/adr/0012-cluster-robust-cr2.md`; `oracle/fixtures/v1/a02_cluster_robust.json`; `statistical/a02_report.json` | 2026-09-16 |
| `release-e02-hardening` | Keep the R/GPL oracle outside both Python artifacts; validate archive identity/content, emit core/all-extra SBOM/checksum manifests, attest tag builds, keep runtime network/telemetry-free, and treat bundles/ledgers as sensitive. | `docs/adr/0013-release-security-boundary.md`; `docs/THREAT_MODEL.md`; `tools/release_artifacts.py` | 2026-09-16 |

## Open decisions

| Key | Question | Evidence needed |
|---|---|---|
| `stable-formula-expansion` | Which transforms, no-intercept terms, and broader fixed/random-term semantics enter after the verified F02 categorical ordinary-bar subset? | Adversarial R X/Z and new-data corpus per proposed expansion |
| `sparse-cholesky` | Does a future CHOLMOD backend materially improve scaling enough to justify its narrower wheel/linking/license surface? | Matched parity, fill, memory, wheel, failure, and redistribution evidence against accepted SuperLU |
