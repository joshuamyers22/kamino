# Contributing

Use Python 3.12 and uv. Run `make setup` once and `make check` before submitting
a change. Packaging, dependency, or workflow changes must also run
`make wheel-smoke` and `make supply-chain`. Statistical changes must identify the estimand/objective, preserve
labels, include a meaningful invariant or oracle comparison, and state whether
the pinned R oracle was run.

Do not update reference fixtures in the same change that alters the implementation
unless the reference version or fixture specification intentionally changed and
the compatibility diff is reviewed.

Use synthetic data in issues and tests. Never commit caller data, model bundles,
bootstrap ledgers, credentials, `.env` files, or raw diagnostic logs. Security
reports and sensitive reproductions belong in GitHub private vulnerability
reporting. Releases follow `docs/runbooks/RELEASE.md`; contributors must not
move tags or upload locally built artifacts.
