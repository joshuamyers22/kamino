# Source and license inventory

This is the current boundary inventory, not legal advice. Exact transitive
versions are in `uv.lock` and `oracle/manifest.json`.

## Distributed Python package

| Component | Role | License evidence | Distribution |
|---|---|---|---|
| Kamino | Project source | MIT, selected by the owner on 2026-09-14 | Yes |
| NumPy 2.5.3 in the local Python 3.12 lock | Runtime linear algebra | SPDX metadata includes BSD-3-Clause and bundled notices | Dependency |
| Formulae 0.5.4 | Restricted runtime formula adapter | MIT package metadata | Dependency, pinned compatibility input |
| pandas 3.0.5 in the local Python 3.12 lock | Formula and dataframe boundary | BSD-3-Clause package metadata and bundled notices | Dependency |
| SciPy 1.17.1 in the local Python 3.12 lock | Scalar optimization | BSD-3-Clause package metadata and bundled notices | Dependency |
| Formulaic 1.2.2 | Rejected grouped-term candidate | MIT package metadata | Evaluation extra only |

Kamino copies no third-party source. Its MIT license is recorded in
`pyproject.toml` and the repository `LICENSE` file. Dependency and oracle
licenses remain separate and retain their own notices and conditions.

## Development-only R oracle

The R image and oracle fixtures are not part of the Python wheel. Most tracked
fixtures use Kamino-owned synthetic literals. `dyestuff.json` additionally
contains the lme4 `Dyestuff` values and derived fit/prediction outputs.

| Component | Version | License from package metadata |
|---|---:|---|
| lme4 | 2.0-6 | GPL (>= 2) |
| Matrix | 1.7-4 | GPL (>= 2), LICENCE file |
| reformulas | 0.4.4 | GPL-3 |
| nloptr | 2.2.1 | LGPL (>= 3) |
| minqa | 1.2.8 | GPL-2 |
| Rcpp | 1.1.2 | GPL (>= 2) |
| RcppEigen | 0.3.4.0.2 | GPL (>= 2), LICENSE file |
| jsonlite | 2.0.0 | MIT, LICENSE file |

The pinned lme4 2.0-6 package declares `GPL (>= 2)` and documents Dyestuff as
Davies and Goldsmith (1972), section 6.4. The JSON conversion and augmentation
are recorded in `oracle/fixtures/v1/README.md`; that file is treated as
GPL-2.0-or-later and explicitly excluded from the MIT wheel. See
`THIRD_PARTY_NOTICES.md`. Reference values remain traceable to the generator,
input, image, and source commit through `oracle/manifest.json`.
