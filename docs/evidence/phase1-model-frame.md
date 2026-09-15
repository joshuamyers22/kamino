# Phase 1 shared model-frame and fixed-effect evidence

Date: 2026-09-15

Scope: the Phase 1 single-group public API with multiple numeric/categorical
fixed effects, treatment/sum contrasts, pairwise `*` expansion, positive prior
weights, formula and argument offsets, Boolean subsets, and explicit missing-row
handling.

## Implemented contract

Kamino now resolves every response, fixed predictor, random predictor, grouping
column, weight, and offset against one original-row boundary. Boolean subset
selection occurs before missing-row handling. `na_action="error"` identifies the
selected rows that contain missing values; `na_action="omit"` removes those rows.
The fitted result separately preserves retained, omitted, and subset-excluded row
IDs. Unused fixed and grouping levels are dropped without losing declared
categorical order. Pandas Series supplied for subset, weights, or offsets must
have the exact DataFrame index; reordered labels fail rather than being applied
positionally.

The fixed grammar is intentionally bounded to an intercept, additive identifiers,
distinct pairwise `a * b`, and `offset(name)`. Categorical identifiers default to
treatment coding and may request sum coding through `contrasts`. Kamino constructs
and serializes its own encoder state, then requires Formulae 0.5.4 to produce the
identical retained X before fitting. Formulae never receives the grouped term.
Categorical random slopes/double bars, different grouping factors, transforms,
and no-intercept formulas remain typed unsupported paths.

Prediction reevaluates the saved fixed encoder and formula offsets. An unseen
fixed factor level is an error. A fit that received an argument offset requires a
new explicit argument offset for every new-data prediction; it is added to all
formula offsets. This is safer than lme4's raw `predict.merMod` behavior, which
does not reevaluate an argument-only offset from `newdata`. The reviewed oracle
therefore records the lme4 new-data prediction plus the explicitly supplied
argument-offset vector, exactly matching Kamino's documented contract.

## Pinned lme4 evidence

`oracle/fixtures/v1/model_frame.json` is generated from Kamino-owned MIT literals
by the public digest-pinned lme4 2.0-6 ARM64 oracle. Its shared-frame case puts
missing values in response, fixed predictor, factor, group, weight, formula
offset, and argument offset positions, including one subset-excluded row. Kamino
matches lme4's nine retained row IDs, ordered four-column X, group values,
weights, and summed offsets exactly.

Four final fits cross treatment/sum coding with ML/REML for
`y ~ x + f + offset(o) + (1 | g)` using nonuniform positive weights and an
argument offset. Full estimates, covariance state, modes, fitted/residual values,
and population/conditional predictions for known and new groups are checked.
At every fitted theta, an independently assembled dense marginal-covariance
oracle agrees with the public objective, beta, and residual variance to `1e-10`.

| Quantity | Maximum absolute difference | Acceptance |
|---|---:|---:|
| Objective | `1.67e-13` | `1e-8` |
| Theta | `3.91e-7` | `1e-6` |
| Beta | `1.91e-8` | `1e-7` |
| Random covariance | `1.80e-7` | `1e-6` |
| Prediction | `1.87e-8` | `1e-6` |

Treatment and sum coding have the same ML objective and predictions. Their REML
criteria differ by the fixed-basis determinant term, matching lme4 rather than
incorrectly forcing contrast-invariant REML objective values. The existing
pinned interaction corpus also confirms the exact six-column `x * f` expansion.

## Persistence, wheel, and resources

Prediction bundle schema 1.2 records fixed variable kinds, levels, contrasts,
terms, and formula-offset names. A categorical sum-coded model round-trips with
exact predictions; strict schema-1.0 and schema-1.1 compatibility remains limited
to their original intercept/numeric encoders. The installed-wheel smoke path now
fits, predicts, saves, and reloads a weighted categorical model with both offset
sources.

The million-row/10,000-group manifest adds ML and REML categorical fixed-design
cases. All eight full cases pass without dense random-indicator materialization;
the new four-column case peaks at 1,316 MB, below the existing 2,000 MB ceiling.
See `docs/evidence/phase1-single-group-resource.md` for the measured protocol.
[Hosted CI run 35017301412](https://github.com/joshuamyers22/kamino/actions/runs/35017301412)
also passes the complete platform, wheel, quality, and eight-case resource matrix.
