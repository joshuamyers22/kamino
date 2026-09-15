# Phase 0 local environment

Observed 2026-09-14:

| Layer | Value |
|---|---|
| Host | macOS 15.1 arm64 |
| Python | CPython 3.12.14 |
| NumPy | 2.5.3 |
| Python BLAS/LAPACK | Apple Accelerate |
| R oracle | R 4.5.2, linux/arm64 |
| lme4 | 2.0-6, commit `4aa26a91f9e676e9409f6cd8163ae92654ef1e7e` |
| R BLAS/LAPACK | OpenBLAS pthread 0.3.26 |

The local Python gate and clean-wheel smoke test cover only macOS arm64 with
Python 3.12. The GitHub Actions matrix defines Python 3.11 through 3.14 and all
three major hosted operating systems, but those rows remain unverified until the
workflow runs in a configured remote repository.

Local acceptance results:

- 37 strict pytest cases pass with 96.81% branch-aware coverage.
- Ruff formatting/lint and strict Pyright pass without findings.
- The sdist and wheel build without network access; two consecutive builds were
  byte-identical.
- The wheel installs with dependencies into a clean temporary environment and
  imports successfully from outside the source tree.
- All six formula-contract cases pass exact binary matrix hashes and labels.
- Eight fixed-theta lme4 cases pass at `1e-12`; the ML/REML optimized walking
  skeleton passes at `1e-8`.
