# Oracle fixture provenance

The JSON files in this directory are test evidence and are excluded from the
installed Kamino package.

| File | Input provenance | Modification | License boundary |
|---|---|---|---|
| `fixed_theta.json` | Kamino-owned synthetic literals | Selected reference outputs from pinned lme4 fits | Kamino MIT |
| `optimized.json` | Kamino-owned synthetic literals | Selected reference outputs from pinned lme4 fits | Kamino MIT |
| `formula_contract.json` | Kamino-owned synthetic literals | Selected formula matrices and labels from pinned lme4 | Kamino MIT |
| `dyestuff.json` | lme4 2.0-6 `Dyestuff`; Davies and Goldsmith (1972), section 6.4 | Converted to JSON and augmented with lme4 fit, diagnostic, and prediction outputs | GPL-2.0-or-later, following lme4 package metadata |

The exact lme4 source commit, R environment, generator hash, controls, and file
hashes are recorded in `oracle/manifest.json`. Regenerate with the immutable
container reference documented in `oracle/README.md`; do not hand-edit expected
statistics or bless unexplained drift.
