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

The local Python gate and clean-wheel smoke test cover macOS arm64 with Python
3.12. Hosted [CI run 34914403645](https://github.com/joshuamyers22/kamino/actions/runs/34914403645)
passed at commit `d7df8cc44795e114fba04519ed7cb13fb1e23912` on 2026-09-14
(local date), covering Ubuntu/Python 3.11, Ubuntu/Python 3.14,
macOS/Python 3.12, Windows/Python 3.12, and the complete quality gate.

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

The first hosted run exposed an invalid assumption that a synchronized uv cache
would contain enough registry metadata for a second offline dependency solve.
The clean-wheel test now performs an ordinary dependency resolution in its fresh
environment; package construction itself remains an offline, locked operation.

The reviewed Linux ARM64 R oracle was published publicly on 2026-09-14 at
`ghcr.io/joshuamyers22/kamino-oracle@sha256:83e891ef07ea9dd45eee788fbe5d79ad0f44852dca49143c70c3d0c15ec9f14b`.
An anonymous manifest request resolved the same digest. The platform constraint
is explicit; no x86_64 oracle identity is inferred from this artifact.
