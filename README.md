# Kamino

Native Python Gaussian linear mixed models with a versioned, tested subset of
lme4 compatibility.

Status: Phase 4 pre-alpha; Phase 1, N03, F02, I01, and I02 are complete. The public
vertical slice fits a Gaussian model with one grouping
structure and either a random intercept, a correlated numeric
random intercept/slope, or independent numeric intercept and slope terms, using
ML or REML and the owned block backend. The formula path uses one shared model
frame, stores group membership plus small row-level covariates, and never
constructs a dense random-effects indicator matrix. Numeric and categorical
fixed effects, treatment/sum contrasts, one pairwise `*` expansion, weights,
additive offsets, subsets, and explicit missing-row handling are covered by
pinned lme4 2.0-6 outputs. Ordinary nested or crossed random-intercept terms use
the coupled sparse backend and are verified on Pastes, Penicillin, and InstEval.
Ordinary categorical random slopes and rank-deficient fixed effects have pinned
fit, prediction, and estimability evidence.

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

crossed_fit = lmer(
    "diameter ~ 1 + (1 | plate) + (1 | sample)",
    penicillin_data,
    reml=True,
)

nested_fit = lmer(
    "strength ~ 1 + (1 | batch/cask)",
    pastes_data,
    reml=False,
)

# Treatment coding is the default; use "sum" explicitly when required.
categorical_fit = lmer(
    "y ~ x * treatment + offset(exposure) + (1 | site)",
    model_data,
    weights=prior_weights,
    offset=argument_offset,
    contrasts={"treatment": "sum"},
    subset=analysis_rows,  # one Boolean per original row
    na_action="omit",  # or the fail-closed default, "error"
)

categorical_random_fit = lmer(
    "y ~ treatment + (1 + treatment | site)",
    model_data,
    contrasts={"treatment": "sum"},
    random_contrasts={"treatment": "sum"},
)

# Simulation mode is explicit; streams do not depend on worker order.
draws = fit.simulate(100, seed=20260915, mode="unconditional")

# The optional private ledger stores responses and outcomes for safe resume.
bootstrap = fit.parametric_bootstrap(
    1000,
    seed=20260916,
    workers=4,
    ledger_path="private-bootstrap-ledger",
)
failure_evidence = bootstrap.failure_accounting()
descriptive_interval = bootstrap.interval(method="percentile")

# Full (theta, sigma) derivatives; inference fails closed at boundaries.
satterthwaite = slope_fit.satterthwaite()
slope_test = satterthwaite.test([0.0, 1.0])
fixed_effects_test = satterthwaite.joint_test([[1.0, 0.0], [0.0, 1.0]])
```

The accepted fixed side contains an intercept, additive numeric/categorical
identifiers, distinct pairwise `a * b` expansion, and `offset(name)`. The random
side is one intercept, one correlated numeric/categorical intercept/slope, or
equivalent independent numeric intercept/slope terms sharing one group. A
random-slope predictor must also be a supported fixed effect. Multiple grouping
structures accept random intercepts and ordinary categorical slopes; nested slash
syntax expands with lme4-compatible term and level ordering. Prediction
mode is explicit; conditional prediction rejects unseen groups unless
`allow_new_groups=True`. Safe prediction-only model bundles are supported;
training rows and responses are deliberately not stored, so reloading does not
support refitting or training prediction. Bundles for nested/crossed,
rank-deficient, and categorical-random fits remain fail-closed until the Phase 2
artifact-recovery schema milestone. Live fitted results support deterministic
conditional/unconditional simulation, exact response refitting, and retained-
fixed-effect parametric bootstrap with a resumable private ledger. Prediction-
only bundles still cannot refit, bootstrap, or run derivative inference.
Satterthwaite one- and multi-DF tests are available from regular live fits and
were calibrated only in the declared Gaussian random-intercept regime;
boundaries and unstable derivatives return unavailable. Percentile/basic
bootstrap intervals remain descriptive, and KR/profile inference remains
unclaimed. Categorical double-bar expansion, numeric random slopes across
different grouping factors, transforms, and refit bundles remain unavailable. The
lower-level fixed-theta array API remains available for numerical development.

- [Production design and implementation plan](PROJECT_PLAN.md)
- [Compatibility contract](docs/COMPATIBILITY.md)
- [Dyestuff ML/REML and prediction evidence](docs/evidence/phase1-dyestuff.md)
- [Dyestuff2 block-boundary evidence](docs/evidence/phase1-dyestuff2-block.md)
- [Sleepstudy correlated-slope evidence](docs/evidence/phase1-sleepstudy.md)
- [Sleepstudy independent-term evidence](docs/evidence/phase1-independent-terms.md)
- [Shared model-frame and fixed-effect evidence](docs/evidence/phase1-model-frame.md)
- [Safe model bundles](docs/MODEL_BUNDLES.md)
- [Million-row single-group resource evidence](docs/evidence/phase1-single-group-resource.md)
- [General sparse nested/crossed evidence](docs/evidence/phase2-general-sparse.md)
- [Rank, estimability, and categorical random-term evidence](docs/evidence/phase2-f02-rank-categorical.md)
- [Parametric-bootstrap and refit-ledger evidence](docs/evidence/phase3-i01-bootstrap.md)
- [Satterthwaite derivative and calibration evidence](docs/evidence/phase4-i02-satterthwaite.md)
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
