# I03 Kenward–Roger and likelihood-profile evidence

Status: complete locally on 2026-09-16; hosted CI evidence pending this commit.

Kamino exposes `fit.kenward_roger()` and `fit.profile()`. KR reproduces the
pinned pbkrtest 0.5.5 observation-space component construction, adjusted
fixed-effect covariance, covariance-parameter information/covariance,
derivative matrices, scaled F statistic, scaling, numerator/denominator degrees
of freedom, and p-value. An ML source fit is separately refitted with REML and
records that provenance.

The reviewed oracle contains Dyestuff and Sleepstudy from both ML and REML
source fits. Dyestuff recovers denominator DF 5. Sleepstudy recovers slope DF 17
and the two-DF joint-test denominator DF 16 with scaling `0.94117647`. Tests also
cover redundant and inconsistent hypotheses, fixed-effect estimability, the
Pastes general sparse fit, immutable outputs, covariance-boundary refusal,
prior-weight refusal, information conditioning, and dense-resource ceilings.

Profiles use the lme4 2.0-6 ML `devfun2` contract. The pinned corpus includes
Dyestuff random-intercept SD, residual scale, and fixed intercept, plus
Sleepstudy intercept SD, slope SD, correlation, residual scale, and fixed
slope. Tests compare ML baselines, direct nuisance-optimized signed-root
deviances at lme4-selected target values, adaptive 95% intervals, target order,
ML-from-REML refit provenance, and optional variance/covariance presentation.
Boundary truncation, unavailable/unbounded endpoints, optimizer failure,
nonmonotonicity, and a materially improved baseline are represented explicitly.

The locked plan is `statistical/i03_plan.json`. Its 1,000-replicate regular
Gaussian random-intercept assessment produced 997 available KR tests; three
fitted covariance boundaries were retained as unavailable and never redrawn.
KR rejected 42/997 (`0.04213`; Wilson 95% interval `0.03132–0.05645`). The
fixed-slope ML profile likelihood ratio was available in all 1,000 replicates,
rejecting 47 (`0.04700`; interval `0.03553–0.06194`) and yielding `0.953`
coverage. Every predeclared gate passed. The machine-readable result is
`statistical/i03_report.json`.

Local release verification passed `make check` with 366 tests and 90.44%
coverage, the clean installed-wheel smoke test, all I01/I02/I03 locked
statistical verifications, and oracle source/output/container digest
verification. Hosted CI evidence will replace the pending status after the
implementation commit runs on the public matrix.

KR is currently claimed only for regular, unit-weight Gaussian fits within the
accepted formula scope and configured dense limits. Profile calibration covers
the declared regular fixed-effect target, not covariance-boundary coverage.
