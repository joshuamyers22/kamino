# Kamino

Native Python Gaussian linear mixed models with a versioned, tested subset of
lme4 compatibility.

Status: Phase 0 pre-alpha. The weighted fixed-covariance ML/REML numerical core
is implemented and verified; optimization and the public formula API are not.

```python
import numpy as np

from kamino import ModelSpec, ObjectiveKind, evaluate_fixed_theta

spec = ModelSpec.from_arrays(
    y=[1.0, 2.1, 2.9, 4.2],
    x=[[1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    z=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
)
result = evaluate_fixed_theta(spec, np.diag([0.8, 0.8]), kind=ObjectiveKind.REML)
print(result.objective, result.beta)
```

This is a fixed-theta evaluation API, not yet a fitted model. It accepts only
validated arrays. Formula parsing, covariance optimization, prediction, and
inference are deliberately unavailable until later compatibility gates pass.

- [Production design and implementation plan](PROJECT_PLAN.md)
- [Compatibility contract](docs/COMPATIBILITY.md)
- [Phase 0 production-readiness record](PRODUCTION_READINESS.md)
- [Pinned R oracle](oracle/README.md)
- [Original design, preserved for review history](docs/archive/lmerx-design.before-production-revision-2026-09-14.md)

The project and planned Python package are named `kamino` (formerly `lmerx`).
The plan applies the production-project-template baseline and defines statistical
fidelity, implementation milestones, and release evidence requirements.

For development, install the frozen environment with `make setup`, run the
offline gate with `make check`, and verify the built wheel with
`make wheel-smoke`. Docker is required only for `make oracle`.
