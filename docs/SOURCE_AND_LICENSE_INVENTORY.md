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
| SciPy 1.17.1 on Python 3.11 and 1.18.1 on Python 3.12+ in the lock | Scalar optimization and SuperLU sparse factorization | BSD-3-Clause package metadata plus bundled OpenBLAS/LAPACK/SuperLU notices | Dependency/native wheels |
| Formulaic 1.2.2 | Rejected grouped-term candidate | MIT package metadata | Evaluation extra only |

Kamino copies no third-party source. Its MIT license is recorded in
`pyproject.toml` and the repository `LICENSE` file. Dependency and oracle
licenses remain separate and retain their own notices and conditions.

## Development-only R oracle

The R image and oracle fixtures are not part of the Python wheel. Most tracked
fixtures use Kamino-owned synthetic literals. `dyestuff.json` and
`dyestuff2.json`, `sleepstudy.json`, `sleepstudy_independent.json`,
`pastes_sparse.json`, `penicillin_sparse.json`, and `insteval_sparse.json`
additionally contain lme4 dataset values and derived fit/prediction outputs.
`model_frame.json` contains only Kamino-owned MIT synthetic literals and derived
lme4 outputs.

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

The pinned lme4 2.0-6 package declares `GPL (>= 2)`. It documents Dyestuff as
Davies and Goldsmith (1972), section 6.4, and generated Dyestuff2 as the boundary
example described by Box and Tiao (1973), section 5.1.2. It documents sleepstudy
as the Belenky et al. (2003) sleep-deprivation study, Pastes and Penicillin as
the Davies and Goldsmith production examples, and InstEval as the Bates et al.
(2015) evaluation corpus. The JSON conversions and augmentations are recorded
in `oracle/fixtures/v1/README.md`; all seven files are
treated as GPL-2.0-or-later and explicitly excluded from the MIT wheel and
sdist. See
`THIRD_PARTY_NOTICES.md`. Reference values remain traceable to the generator,
input, image, and source commit through `oracle/manifest.json`.
