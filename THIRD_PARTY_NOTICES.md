# Third-party notices

Kamino source code and the built Python package are licensed under the MIT
License. Third-party dependencies retain their own licenses; see
`docs/SOURCE_AND_LICENSE_INVENTORY.md`.

The repository's `oracle/fixtures/v1/dyestuff.json`, `dyestuff2.json`,
`sleepstudy.json`, `sleepstudy_independent.json`, `pastes_sparse.json`,
`penicillin_sparse.json`, and `insteval_sparse.json` files include values from
the `Dyestuff`, generated
`Dyestuff2`, `sleepstudy`, `Pastes`, `Penicillin`, and `InstEval` datasets
distributed with lme4 2.0-6. The lme4
package declares `GPL (>= 2)` in its package metadata. Treat those fixtures as
GPL-2.0-or-later, not MIT. Their provenance and modification details are recorded
in `oracle/fixtures/v1/README.md`. Oracle fixtures are not included in the Kamino
wheel and sdist.
