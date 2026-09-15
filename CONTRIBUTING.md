# Contributing

Use Python 3.12 and uv. Run `make setup` once and `make check` before submitting
a change. Statistical changes must identify the estimand/objective, preserve
labels, include a meaningful invariant or oracle comparison, and state whether
the pinned R oracle was run.

Do not update reference fixtures in the same change that alters the implementation
unless the reference version or fixture specification intentionally changed and
the compatibility diff is reviewed.
