# Kamino

Native Python Gaussian linear mixed models with a versioned, tested subset of
lme4 compatibility.

Status: Phase 1 pre-alpha. The public vertical slice fits a Gaussian model with
one fixed intercept and one random intercept, using ML or REML and the owned
block backend. The formula path stores random-intercept membership as one group
index per row and never constructs a dense random-effects indicator matrix.
Dyestuff regular fits and Dyestuff2 boundary fits/predictions are verified
against pinned lme4 2.0-6 outputs.

```python
from kamino import lmer

fit = lmer(
    "yield_value ~ 1 + (1 | batch)",
    {
        "yield_value": [1.0, 1.2, 0.8, 3.0, 3.1, 2.9],
        "batch": ["a", "a", "a", "b", "b", "b"],
    },
    reml=True,
)
conditional = fit.predict(mode="conditional")
population = fit.predict(
    {"batch": ["a", "unseen"]},
    mode="population",
)
```

The accepted formula is currently exactly `response ~ 1 + (1 | group)`.
Prediction mode is explicit; conditional prediction rejects unseen groups unless
`allow_new_groups=True`. Fixed effects, slopes, multiple random terms, general
sparse solving, model serialization, and inference remain unavailable. The
lower-level fixed-theta array API remains available for numerical development.

- [Production design and implementation plan](PROJECT_PLAN.md)
- [Compatibility contract](docs/COMPATIBILITY.md)
- [Dyestuff ML/REML and prediction evidence](docs/evidence/phase1-dyestuff.md)
- [Dyestuff2 block-boundary evidence](docs/evidence/phase1-dyestuff2-block.md)
- [Phase 0 production-readiness record](PRODUCTION_READINESS.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Pinned R oracle](oracle/README.md)
- [MIT license](LICENSE)
- [Original design, preserved for review history](docs/archive/lmerx-design.before-production-revision-2026-09-14.md)

The project and planned Python package are named `kamino` (formerly `lmerx`).
The plan applies the production-project-template baseline and defines statistical
fidelity, implementation milestones, and release evidence requirements.

For development, install the frozen environment with `make setup`, run the
offline gate with `make check`, and verify the built wheel with
`make wheel-smoke`. Docker is required only for `make oracle`.
