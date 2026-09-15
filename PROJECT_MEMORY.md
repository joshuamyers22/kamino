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
| `phase1-first-slice` | Keep the first public fit fail-closed to one fixed intercept and one random intercept until broader final-fit evidence passes. | `docs/COMPATIBILITY.md`; Dyestuff oracle tests | 2026-09-15 |
| `phase1-block-runtime` | Route supported public fits through the owned random-intercept block evaluator; retain dense PLS and marginal routes as independent small-model checks. | `src/kamino/block.py`; Dyestuff2 evidence | 2026-09-15 |

## Open decisions

| Key | Question | Evidence needed |
|---|---|---|
| `sparse-backend` | Which general sparse factorization is supportable? | Parity, symbolic reuse, memory, wheels, and license review |
| `formula-expansion` | Which contrasts, `||`, nesting, and transforms enter the stable subset? | Adversarial R X/Z and new-data corpus |
