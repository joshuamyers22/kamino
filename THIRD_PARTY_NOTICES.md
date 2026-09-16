# Third-party notices

Kamino source code and the built Python package are licensed under the MIT
License. Third-party dependencies retain their own licenses; see
`docs/SOURCE_AND_LICENSE_INVENTORY.md`.

The repository's `oracle/fixtures/v1/dyestuff.json`, `dyestuff2.json`,
`sleepstudy.json`, `sleepstudy_independent.json`, `pastes_sparse.json`,
`penicillin_sparse.json`, `insteval_sparse.json`, `i01_bootstrap.json`,
`i02_satterthwaite.json`, and `i03_kr_profile.json` files
include values from
the `Dyestuff`, generated
`Dyestuff2`, `sleepstudy`, `Pastes`, `Penicillin`, and `InstEval` datasets
distributed with lme4 2.0-6. The lme4
package declares `GPL (>= 2)` in its package metadata. Treat those fixtures as
GPL-2.0-or-later, not MIT. Their provenance and modification details are recorded
in `oracle/fixtures/v1/README.md`. Oracle fixtures are not included in the Kamino
wheel and sdist.

The complete development-only `oracle/` directory, including its R generator,
container recipe, manifests, outputs, and fixtures, is excluded from both Python
distribution artifacts. It remains in the source repository for reproducible
compatibility review under the licenses described below.

The I02 fixture includes derived Satterthwaite derivative and test outputs from
lmerTest 3.1-3. The lmerTest package declares `GPL (>= 2)` in its package
metadata and is used only in the development oracle image, not by the MIT
Python package at runtime.

The I03 fixture includes derived Kenward–Roger outputs from pbkrtest 0.5.5.
pbkrtest declares `GPL (>= 2)` in its package metadata and is used only in the
development oracle image. I03 profile outputs are derived from lme4 2.0-6.

The A02 fixture includes derived cluster-robust covariance, adjustment, score,
Satterthwaite, and HTZ outputs from clubSandwich 0.7.0. clubSandwich declares
GPL-3 in its package metadata and is used only in the development oracle image.

The MIT-synthetic `a01_postfit.json` fixture contains derived fit and
post-estimation outputs from lme4 2.0-6 and emmeans 2.0.2. emmeans declares
`GPL (>= 2)` in its package metadata and is used only in the development oracle
image. The fixture is excluded from the Kamino wheel and sdist.
