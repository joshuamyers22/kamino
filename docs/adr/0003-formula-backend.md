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

For the implemented one-random-intercept subset, Kamino's allowlisted formula
parser establishes the grouped term before calling Formulae. Formulae receives
only `response ~ 1` and a response-only frame, so it cannot construct a dense
grouped matrix. Kamino derives the canonical level map directly from the already
validated grouping column and stores one immutable integer index per row. This
is a narrow refinement of the accepted adapter, not evidence for broader owned
formula semantics.

The executable probe is `tools/formula_spike.py` and its reviewed findings are
recorded in `docs/evidence/phase0-formula-spike.md`.
