# Reproducibility

Python dependencies are declared in `pyproject.toml` and locked in `uv.lock`.
Local and CI work uses frozen installs after the lock is generated. Numerical
evidence records Python, NumPy/SciPy, BLAS/LAPACK, OS, architecture, precision,
thread controls, model dimensions, ordered labels, and tolerances.

The primary R reference is lme4 2.0-6 at CRAN mirror commit
`4aa26a91f9e676e9409f6cd8163ae92654ef1e7e`. R is isolated under `oracle/` and
is never imported by the installed Python package. Oracle outputs require input,
generator, container, session, and output hashes. Regeneration creates a reviewed
corpus version and does not silently overwrite accepted expectations.

Default tests are offline and R-free. `make oracle` is an explicit integration
gate. Fixed seeds do not promise identical streams across library versions;
stored response arrays are used for cross-language refit comparisons.

The Phase 0 image uses digest-pinned `rocker/r-ver:4.5.2`, the dated R package
snapshot `2026-03-10`, Rcpp 1.1.2 at CRAN-mirror commit
`a76e1c9bab241d008940626641cf7ad48cbc7383`, and system NLopt
`2.7.1-5build2`. The complete observed session, image identity, generator hash,
and output hashes are in `oracle/manifest.json`. Its image identity is local
until a release registry publishes it.

Local Phase 0 evidence was produced on macOS arm64 with CPython 3.12.14 and
NumPy 2.5.3 using Accelerate. Other configured Python/OS rows remain pending
until CI runs; see `docs/evidence/phase0-local-environment.md`.
