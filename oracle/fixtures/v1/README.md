# Oracle fixture provenance

The JSON files in this directory are test evidence and are excluded from the
installed Kamino package.

| File | Input provenance | Modification | License boundary |
|---|---|---|---|
| `fixed_theta.json` | Kamino-owned synthetic literals | Selected reference outputs from pinned lme4 fits | Kamino MIT |
| `optimized.json` | Kamino-owned synthetic literals | Selected reference outputs from pinned lme4 fits | Kamino MIT |
| `formula_contract.json` | Kamino-owned synthetic literals | Selected formula matrices and labels from pinned lme4 | Kamino MIT |
| `dyestuff.json` | lme4 2.0-6 `Dyestuff`; Davies and Goldsmith (1972), section 6.4 | Converted to JSON and augmented with lme4 fit, diagnostic, and prediction outputs | GPL-2.0-or-later, following lme4 package metadata |
| `dyestuff2.json` | lme4 2.0-6 generated `Dyestuff2`; Box and Tiao (1973), section 5.1.2 | Converted to JSON and augmented with lme4 boundary-fit, diagnostic, and prediction outputs | GPL-2.0-or-later, following lme4 package metadata |
| `sleepstudy.json` | lme4 2.0-6 `sleepstudy`; Belenky et al. (2003) | Converted to JSON and augmented with fixed-theta, fitted, diagnostic, covariance, and prediction outputs | GPL-2.0-or-later, following lme4 package metadata |
| `sleepstudy_independent.json` | lme4 2.0-6 `sleepstudy`; Belenky et al. (2003), plus a Kamino-owned synthetic boundary case | Converted to JSON and augmented with independent-term fixed-theta, fitted, diagnostic, covariance, boundary, and prediction outputs | GPL-2.0-or-later as one combined fixture containing lme4 dataset values |

The exact lme4 source commit, R environment, generator hash, controls, and file
hashes are recorded in `oracle/manifest.json`. Regenerate with the immutable
container reference documented in `oracle/README.md`; do not hand-edit expected
statistics or bless unexplained drift.
