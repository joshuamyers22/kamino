# Phase 0 formula backend evidence

Environment: Formulae 0.5.4, Formulaic 1.2.2, pandas 3.0.5, compared with six
synthetic lme4 2.0-6 formula-matrix fixtures.

| Check | Result |
|---|---|
| Formulae fixed matrix | Exact value and shape match |
| Raw lme4 formula containing `offset(o)` | Incorrect extra fixed column; adapter must extract offset first |
| Formulae random matrix, raw | Mismatch caused by column ordering |
| Formulae random matrix, label permutation `[0, 3, 1, 4, 2, 5]` | Exact value and shape match |
| Missing `x` | The one affected row was dropped; Kamino must choose rows first |
| Known grouping level at new data | Original random column count preserved |
| Unknown grouping level | Rejected with `ValueError` |
| Raw Formulae `||` | Rejected with `ParseError` |
| Adapter expansion of numeric `||` | Exact R X/Z match |
| Ordered treatment-coded fixed category | Exact R X and labels |
| Fixed category interaction | Exact R X and labels |
| Nested `g/h` after semantic label permutation | Exact R Z |
| Formulaic grouped `|` | Rejected with `FormulaSyntaxError` |

Decision: use a restricted Formulae adapter for the first formula subset, with
owned row selection, explicit ordered categories, label canonicalization, offset
extraction, syntax allowlisting, and persisted encoder state. The corpus verifies
matrix semantics for these six cases; it does not approve categorical random
effects, custom contrasts, general transforms, or prediction transforms. Re-run
with:

```sh
make formula-spike
```
