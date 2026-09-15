# ADR 0003: Formula backend

Status: accepted for the Phase 1 subset
Date: 2026-09-14

## Decision

Use a narrow Formulae adapter for the Phase 1 formula subset. Keep Formulaic as
an evaluation-only candidate, not a runtime dependency: version 1.2.2 rejects
the grouped-term `|` operator without an owned parser extension.

Formulae 0.5.4 exactly reproduced the pinned R fixed matrices. Its random matrix
exactly matched after a label-derived permutation, but not positionally:
Formulae emitted term-major grouping columns while lme4 emitted group-major
blocks. Formulae also sorted unordered factors; materializing the R level order
as an ordered categorical preserved the expected levels. The adapter therefore
owns categorical levels and canonical labeled ordering and never treats vendor
column positions as the compatibility contract.

## Guardrails

The selected route uses one shared model frame, has a restricted default execution
policy, persists encoder state, and fails unsupported syntax before optimization.

Formulae's automatic complete-case behavior must not independently select rows;
Kamino constructs the shared frame first. Formulae also treats `offset(o)` as an
ordinary fixed column, so Kamino must extract and validate offsets before matrix
construction. New grouping levels remain errors. Raw double-bar syntax is
unsupported by Formulae; the Phase 0 adapter expansion `(1 + x || g)` to
`(1 | g) + (0 + x | g)` exactly matches the pinned R matrix for numeric `x`.
The six-case corpus also exactly matches treatment-coded fixed categories, a
fixed interaction, nesting, and shared missing-row selection after label
canonicalization. These are design-layer findings, not early solver or API
claims. Categorical random effects, custom contrasts, general transforms, and
new-data transforms remain unclaimed until their R corpora pass.

For the implemented single-group subset, Kamino's allowlisted formula parser
establishes the grouped term before calling Formulae. Formulae receives only the
fixed portion (`response ~ 1` or `response ~ predictor`) and no grouping column,
so it cannot construct a dense grouped matrix. Kamino derives the canonical
level map directly from the validated grouping column and stores one immutable
integer index plus one or two random-design values per row. This is a narrow
refinement of the accepted adapter, not evidence for broader owned formula
semantics.

The first numeric double-bar slice is now public. Both `(1 + x || g)` and its
explicit `(1 | g) + (0 + x | g)` expansion produce the same compact group-major
design and two distinct one-column covariance terms. Formulae still sees only
`response ~ x`. Categorical double-bar expansion and random terms with different
grouping factors remain rejected.

The completed Phase 1 adapter owns a deliberately small fixed grammar: additive
identifiers, distinct pairwise `*`, and `offset(identifier)`, always with an
intercept. It builds one model frame before any matrix work, applies Boolean
subset selection before its explicit `error`/`omit` missing policy, drops unused
levels in declared order, and passes only the retained fixed frame to Formulae.
An independently serializable encoder then reproduces and checks Formulae's X
using lme4-compatible treatment or sum coding. The encoder and formula-offset
names are persisted for new-data prediction. General calls/transforms,
no-intercept models, categorical random terms, and different grouping factors
still fail closed.

The executable probe is `tools/formula_spike.py` and its reviewed findings are
recorded in `docs/evidence/phase0-formula-spike.md`.
