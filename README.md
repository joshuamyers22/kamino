# Kamino

Native Python Gaussian linear mixed models with a versioned, tested subset of
lme4 compatibility.

Status: Phase 1 pre-alpha. The public vertical slice fits a Gaussian model with
one grouping structure and either a random intercept, a correlated numeric
random intercept/slope, or independent numeric intercept and slope terms, using
ML or REML and the owned block backend. The formula
path stores group membership plus small row-level covariates and never constructs
a dense random-effects indicator matrix. Dyestuff, Dyestuff2, and Sleepstudy
fits and predictions are verified against pinned lme4 2.0-6 outputs.

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

fit.save("yield-model.kamino")

from kamino import load_model_bundle

saved_model = load_model_bundle("yield-model.kamino")
saved_prediction = saved_model.predict(
    {"batch": ["a", "unseen"]},
    mode="conditional",
    allow_new_groups=True,
)

# sleepstudy_data is a pandas DataFrame with these three columns.
slope_fit = lmer(
    "reaction ~ days + (1 + days | subject)",
    sleepstudy_data,
    reml=True,
)

independent_fit = lmer(
    "reaction ~ days + (1 + days || subject)",
    sleepstudy_data,
    reml=False,
)
```

Accepted formulas are `response ~ 1 + (1 | group)`,
`response ~ predictor + (1 + predictor | group)`, and the equivalent independent
forms `response ~ predictor + (1 + predictor || group)` or
`response ~ predictor + (1 | group) + (0 + predictor | group)` for one numeric
predictor.
Prediction mode is explicit; conditional prediction rejects unseen groups unless
`allow_new_groups=True`. Safe prediction-only model bundles are supported;
training rows and responses are deliberately not stored, so reloading does not
support refitting or training prediction. Random terms with different grouping
factors, categorical double-bar expansion, multiple predictors, general sparse
solving, refit bundles, and inference remain unavailable. The
lower-level fixed-theta array API remains available for numerical development.

- [Production design and implementation plan](PROJECT_PLAN.md)
- [Compatibility contract](docs/COMPATIBILITY.md)
- [Dyestuff ML/REML and prediction evidence](docs/evidence/phase1-dyestuff.md)
- [Dyestuff2 block-boundary evidence](docs/evidence/phase1-dyestuff2-block.md)
- [Sleepstudy correlated-slope evidence](docs/evidence/phase1-sleepstudy.md)
- [Sleepstudy independent-term evidence](docs/evidence/phase1-independent-terms.md)
- [Safe model bundles](docs/MODEL_BUNDLES.md)
- [Million-row single-group resource evidence](docs/evidence/phase1-single-group-resource.md)
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
