# Source and license inventory

This is the Phase 0 boundary inventory, not legal advice. Exact transitive
versions are in `uv.lock` and `oracle/manifest.json`.

## Distributed Python package

| Component | Role | License evidence | Distribution |
|---|---|---|---|
| Kamino | Project source | Owner selection pending | Yes |
| NumPy 2.5.3 in the local lock | Runtime linear algebra | SPDX metadata includes BSD-3-Clause and bundled notices | Dependency |
| Formulae 0.5.4 | Selected future formula adapter | MIT package metadata | Phase 0 extra only |
| pandas 3.0.5 | Formulae dataframe boundary | BSD-3-Clause package metadata and bundled notices | Phase 0 extra only |
| Formulaic 1.2.2 | Rejected grouped-term candidate | MIT package metadata | Evaluation extra only |

Kamino copies no third-party source. The release remains blocked until the owner
selects Kamino's distribution license and the resulting compatibility review is
recorded in `pyproject.toml` and a `LICENSE` file.

## Development-only R oracle

The R image is not part of the Python wheel. The synthetic tracked fixture was
generated from Kamino-owned literals and contains no copied lme4 dataset.

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

Before any upstream dataset is committed, record its file-level provenance,
license, modification state, and redistribution permission. Reference values
alone must remain traceable to the generator, input, image, and source commit.
