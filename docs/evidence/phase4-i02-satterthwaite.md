# I02 Satterthwaite evidence

Status: complete locally on 2026-09-15; hosted CI evidence pending this commit.

Kamino now exposes reusable full-variance-parameter derivative state from a live
fit through `fit.satterthwaite()`. The implementation differentiates the
unprofiled deviance in `(theta, sigma)`, uses `2 H_D^-1`, and differentiates the
fixed-effect covariance in the same coordinates. Both the compact block and
coupled general sparse workspaces are supported without dense random-indicator
materialization.

The pinned companion is lmerTest 3.1-3 at source commit
`35dc5885205d709cdc395b369b08ca2b7273cb78`, run with lme4 2.0-6. The reviewed
fixture contains Dyestuff and Sleepstudy under ML and REML. Tests compare the
full Hessian, variance-parameter covariance, beta-covariance Jacobians, one-DF
t tests, and one-/two-DF F tests. The implementation also verifies consistent
redundant-row rank reduction, inconsistent-right-hand-side refusal, full-space
estimability after fixed-rank dropping, immutable derivative arrays, and the
Pastes general sparse route.

The locked plan is `statistical/i02_plan.json`. Its 2,000-replicate REML
assessment produced 1,988 available fits. The one-DF null rejection rate was
`0.04980` (99/1,988; Wilson 95% interval `0.04107–0.06026`) and the joint two-DF
rate was `0.05181` (103/1,988; interval `0.04290–0.06245`). Twelve fits (`0.6%`)
landed at the fitted covariance boundary and were retained as unavailable; none
were redrawn. Every predeclared gate passed. The machine-readable result is
`statistical/i02_report.json`.

The claim is limited to regular Gaussian models in the accepted Kamino formula
scope. A fitted covariance boundary, unstable derivative, invalid curvature,
or non-estimable hypothesis returns no denominator DF or p-value. Kenward–Roger,
profiles, variance-component tests, and prediction-bundle inference remain out
of scope.
