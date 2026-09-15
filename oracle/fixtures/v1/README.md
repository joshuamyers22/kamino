# Oracle fixture provenance

The JSON files in this directory are test evidence and are excluded from the
Kamino wheel and sdist.

| File | Input provenance | Modification | License boundary |
|---|---|---|---|
| `fixed_theta.json` | Kamino-owned synthetic literals | Selected reference outputs from pinned lme4 fits | Kamino MIT |
| `optimized.json` | Kamino-owned synthetic literals | Selected reference outputs from pinned lme4 fits | Kamino MIT |
| `formula_contract.json` | Kamino-owned synthetic literals | Selected formula matrices and labels from pinned lme4 | Kamino MIT |
| `model_frame.json` | Kamino-owned synthetic literals | Shared-frame selection, treatment/sum categorical fits, weights, offsets, and predictions from pinned lme4 | Kamino MIT |
| `f02_rank_categorical.json` | Kamino-owned deterministic synthetic construction | QR/drop maps, rank-deficient fits, treatment/sum categorical random terms, crossed fixed-theta/final fits, and predictions from pinned lme4 | Kamino MIT |
| `dyestuff.json` | lme4 2.0-6 `Dyestuff`; Davies and Goldsmith (1972), section 6.4 | Converted to JSON and augmented with lme4 fit, diagnostic, and prediction outputs | GPL-2.0-or-later, following lme4 package metadata |
| `dyestuff2.json` | lme4 2.0-6 generated `Dyestuff2`; Box and Tiao (1973), section 5.1.2 | Converted to JSON and augmented with lme4 boundary-fit, diagnostic, and prediction outputs | GPL-2.0-or-later, following lme4 package metadata |
| `sleepstudy.json` | lme4 2.0-6 `sleepstudy`; Belenky et al. (2003) | Converted to JSON and augmented with fixed-theta, fitted, diagnostic, covariance, and prediction outputs | GPL-2.0-or-later, following lme4 package metadata |
| `sleepstudy_independent.json` | lme4 2.0-6 `sleepstudy`; Belenky et al. (2003), plus a Kamino-owned synthetic boundary case | Converted to JSON and augmented with independent-term fixed-theta, fitted, diagnostic, covariance, boundary, and prediction outputs | GPL-2.0-or-later as one combined fixture containing lme4 dataset values |
| `pastes_sparse.json` | lme4 2.0-6 `Pastes`; Davies and Goldsmith (1972), section 6.4 | Converted to JSON and augmented with nested fixed-theta, fitted, covariance, mode, and prediction outputs | GPL-2.0-or-later, following lme4 package metadata |
| `penicillin_sparse.json` | lme4 2.0-6 `Penicillin`; Davies and Goldsmith (1972), section 6.6 | Converted to JSON and augmented with crossed fixed-theta, fitted, covariance, mode, and prediction outputs | GPL-2.0-or-later, following lme4 package metadata |
| `insteval_sparse.json` | lme4 2.0-6 `InstEval`; Bates et al. (2015) | Selected columns converted to JSON and augmented with crossed ML/REML fit outputs and scale metadata | GPL-2.0-or-later, following lme4 package metadata |

The exact lme4 source commit, R environment, generator hash, controls, and file
hashes are recorded in `oracle/manifest.json`. Regenerate with the immutable
container reference documented in `oracle/README.md`; do not hand-edit expected
statistics or bless unexplained drift.
