# Numerical correctness incident runbook

1. Preserve the exact Kamino version, formula structure, ML/REML mode, platform,
   dependency/BLAS identity, diagnostics, and a licensed minimal reproducer.
   Keep private source data out of the repository.
2. Classify impact: objective/estimate, covariance or degrees of freedom,
   prediction, artifact identity, boundary/convergence status, or resource
   failure. Treat a materially wrong declared-scope result as High or Critical.
3. Determine the first affected version and model classes. Compare independent
   dense algebra, the pinned R reference, fixed-theta intermediates, and the
   final optimizer state; do not regenerate fixtures to match the defect.
4. Contain by documenting the affected regime, disabling a method if necessary,
   and yanking an affected published package when continued use is unsafe.
5. Forward-fix with a regression case, all applicable oracle/statistical gates,
   clean wheel and platform evidence, and review by the statistical owner.
6. Publish a new version and advisory describing which results require refitting
   or reinterpretation. Never move a tag or replace historical evidence.
7. Record systemic corrective actions, residual risk, owner, due date, and the
   evidence required for closure.
