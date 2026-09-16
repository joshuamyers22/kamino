# kamino — production design and implementation plan

Status: implementation specification; Phase 0/1 and Phase 2 N03 complete
Revised: 2026-09-15
Product: native Python Gaussian linear mixed models with a versioned, tested subset of lme4 fidelity
Engineering baseline: [production-project-template, commit 6526db479fced81463d1a97e25ff3ceefbe9eee2](https://github.com/joshuamyers22/production-project-template/tree/6526db479fced81463d1a97e25ff3ceefbe9eee2)

This document replaces the original sketch and its review addendum. All sections
below are normative requirements for future implementation unless explicitly
marked proposed, experimental, or deferred. There is no implemented fitter or
passing statistical suite established by this review. A production-quality plan
is not evidence that the resulting software is production ready.

The engineering template was inspected locally and its commit verified against
GitHub. Statistical requirements were checked against lme4 source, its methods
paper, and companion-package documentation. This review also found that the
hosted lme4 reference site identifies itself as 1.1-37.9000 while CRAN source has
advanced to 2.0-6. Source versions must therefore be explicit.

## 1. Project brief

### 1.1 Outcome and critical journeys

Researchers and application developers should be able to fit a documented
Gaussian mixed model in Python, inspect the same statistical quantities as a
pinned lme4 fit, and understand every material difference.

Critical journeys:

1. Fit a supported formula with ML or REML and obtain labeled coefficients,
   covariance estimates, random-effect conditional modes, likelihood, and diagnostics.
2. Predict for existing and new groups with explicit conditioning and uncertainty.
3. Obtain an estimable fixed-effect contrast or a supported interval/test, with
   the method, assumptions, reference distribution, and failure status attached.
4. Reproduce a fit or bootstrap from a versioned specification and input snapshot.
5. Install a built wheel in a clean supported environment without R.

Measurable success is the requirement scorecard in §14. Numerical agreement,
statistical calibration, and installability are separate gates.

### 1.2 Scope and release boundary

The first alpha covers one grouping structure, including multiple independent
terms sharing the same grouping partition. The first stable fitter adds nested
and crossed ordinary random-effects structures through a verified general solver.
Positive prior weights, offsets, missing-row policy, rank handling, safe result
storage, and diagnostics are foundational work.

The stable base inference contract is coefficient covariance, labeled statistics,
and validated parametric bootstrap. I02 separately gates the implemented
Satterthwaite tests; Kenward–Roger (KR), profile intervals, and robust cluster
inference retain their own release gates. The
long-term post-estimation scope remains estimated marginal means, marginal
effects, tidy output, variance diagnostics, and cluster-robust inference.

Non-goals for this roadmap: GLMMs, nonlinear mixed models, Bayesian estimation,
general residual correlation models, frequency/survey weights, automatic model
selection, and a claim to implement every current lme4 feature. lme4 2.0 structured
random-effect covariance wrappers are a separate extension; they are distinct
from residual correlation.

### 1.3 Constraints, ownership, and recovery

| Field | Project contract |
|---|---|
| Runtime | Python library; float64 numerical core; no R at runtime |
| Proposed support | CPython 3.11–3.14; Linux x86_64, macOS arm64, Windows x86_64; freeze verified dependency/platform combinations in Phase 0 |
| Development | Python 3.12 environment, uv lock, Ruff, strict Pyright, pytest/Hypothesis; versions resolved and locked during scaffolding |
| Initial scale | Small oracle problems; single-group fits through 1M observations/10k groups as a benchmark; crossed scale measured on InstEval |
| Data | Caller-owned research data may be sensitive; no automatic upload, telemetry collection, or raw-data retention |
| Persistence | Explicit save only; prediction-only bundle by default, optional private refit bundle |
| Resource policy | Preflight dimensions and allocations; configurable byte, evaluation, and worker limits; cancellable long operations |
| Availability | No hosted-service SLO; library errors and failed numerical methods return actionable structured evidence |
| Recovery | Atomic artifact publication; resumable bootstrap ledger; exact saved input/model needed to refit; no hidden recovery from caller data loss |
| Budget/deadline | Unspecified; phase gates determine progression, no promised calendar date |
| Accountable owner | Josh Myers for product/scope; assign statistical reviewer and release maintainer before beta |
| Support | Maintainer-owned issue triage, numerical incident reproduction, wheel yank/forward-fix procedure before stable release |

The five highest risks are a different design matrix, an incorrect objective,
invalid uncertainty, false convergence, and unsupported numerical binaries.
Threat controls and evidence for each appear below.

## 2. Reference contract and compatibility matrix

### 2.1 Pin both reference and semantics

Primary target: the ordinary unstructured Gaussian LMM subset of lme4 2.0-6,
[CRAN source commit 4aa26a91f9e676e9409f6cd8163ae92654ef1e7e](https://github.com/cran/lme4/tree/4aa26a91f9e676e9409f6cd8163ae92654ef1e7e).

Legacy regression target: the same shared subset in lme4 1.1-37,
[commit 79c411060a27b934ee4d541bdb983ea4db7269b2](https://github.com/cran/lme4/tree/79c411060a27b934ee4d541bdb983ea4db7269b2).
Keep separate expected outputs when behavior differs. Do not call the older
version current or use a moving branch as the numerical oracle.

The [2.0 release notes](https://github.com/cran/lme4/blob/4aa26a91f9e676e9409f6cd8163ae92654ef1e7e/inst/NEWS.Rd)
introduce structured covariance and configurable double-bar expansion and change
confidence-interval ordering. Model and parameter labels must drive comparison;
raw vector positions are insufficient. Pin the reformulas dependency and
double-bar option as well as lme4.

Phase 0 must build and commit an oracle manifest with exact R, Matrix,
reformulas, nloptr/NLopt, lmerTest, pbkrtest, emmeans, and clubSandwich versions;
source hashes; BLAS/LAPACK identity and thread settings; locale; contrasts;
NA policy; optimizer controls; container digest; and generator revision.
Source review alone does not establish that those versions install together.

Publish the claim as: “Numerically validated against lme4 [version] for the
features and tolerances in this compatibility report.” Separate four dimensions:
formula semantics, fixed-parameter algebra, optimized estimates, and user-facing
behavior. Python signatures and warning text need not imitate R.

### 2.2 Planned feature contract

“Required” below means a release requirement, not an implemented capability.
Unimplemented features raise a typed capability error before optimization.

| Feature | Alpha | Stable fitter | Required semantics or limitation |
|---|---|---|---|
| Gaussian ML/REML | Required | Required | REML default; objective kind always attached |
| Numeric fixed effects, intercept control, interactions | Required | Required | Match R matrix entries, names, order |
| Fixed categorical effects | Required | Required | Explicit levels and treatment/sum contrasts; ordered/poly contrasts gated separately |
| Numeric random intercepts/slopes with ordinary bar syntax | Required | Required | Full relative Cholesky parameterization |
| Multiple terms sharing one group | Required | Required | Preserve each term's covariance independence and combine columns only for solving |
| Numeric double-bar syntax | Required | Required | Pinned semantic splitting; matrix-valued expansions need their own fixtures |
| Categorical double-bar syntax | Reject initially | Deferred | Never imply independent dummy coefficients without an explicit tested expansion |
| Random categorical effects using ordinary bars | Deferred | Required for declared supported contrasts | R-equivalent term matrix and covariance dimensions |
| Nested and crossed grouping factors | Reject | Required | Correct expansion plus a general solver; no per-term independence shortcut |
| Structured wrappers such as cs(), ar1(), diag() | Reject | Deferred | Do not reinterpret lme4 2.0 syntax as an ordinary term |
| Positive prior precision weights | Required | Required | Unnormalized; residual variance is sigma² / weight |
| Zero, negative, nonfinite, frequency, survey weights | Reject | Reject | Strict-positive subset; R's handling of zero weights is not emulated |
| Offsets | Required | Required | Formula and argument offsets additive; explicit new-data requirements |
| Missing data and subset | Required | Required | Shared model frame; explicit row IDs, omit/fail policy; support exclude/restoration separately |
| Rank-deficient fixed effects | Strict rejection allowed | Required | Match retained-column policy and preserve full-design estimability |
| Stateful transforms | Allowlisted subset | Expand only with fixtures | Store training state; R poly() is not an arbitrary polynomial expansion |
| Singular covariance | Required | Required | Valid boundary fit when objective is defined; separate inference status |
| Fitted values and random effects | Required | Required | Include offsets; preserve term/group/coefficient identities |
| Population/conditional predictions | Required | Required | §7; unseen fixed levels always error |
| AIC/BIC/log-likelihood | Required | Required | §4.4 bookkeeping; no invented test DF |
| Satterthwaite/KR/profile/robust CR2 | I02, I03, and A02 complete for their declared regimes | Separately gated | No compatibility claim merely because fitting passes |

Defaults that intentionally differ from R must be documented in the same report:
restricted expression evaluation, explicit data argument, explicit prediction
mode, resource limits, stricter invalid-input policy, and refusal of unsupported
inference. No adapter may silently convert an unsupported feature.

## 3. Formula, sample, and parameter identity

### 3.1 One model frame

Parse to a real syntax tree with parentheses, precedence, quoting, calls, bars,
and double bars. Evaluate response, fixed terms, random terms, groups, weights,
offsets, and subset through one shared row-selection pipeline. Independent
drop-NA operations on X and Z are prohibited.

Record original positions, stable row IDs independent of dataframe index labels,
subset membership, omitted rows, factor levels, ordered-factor status, contrasts,
column names, term assignments, offsets, weights, and transform state. Validate
lengths before broadcasting and reject nonfinite retained values. Grouping
interactions use tuples internally so labels containing ":" cannot collide.

Mirror the pinned R model-frame order of subset, transform evaluation, NA
handling, and unused-level dropping for accepted expressions. This matters for
data-dependent transforms such as centering, scaling, polynomials, and splines.
Rebuilding those transforms on prediction data changes the model.

Evaluate formulae and a Formulaic extension against the same R design corpus.
Choose one after the spike; neither earns compatibility from its name. An owned
restricted grammar is acceptable if the supported subset is smaller and tested.
No arbitrary eval, imports, attribute access, or live caller scope in the default
grammar. User transforms require explicit registration and a serializable state
contract; execution remains a trusted-code opt-in.

The [lmer interface](https://lme4.github.io/lme4/reference/lmer.html) explains
weights, offsets, missingness, rank dropping, and traditional double-bar caveats.
Use pinned source/fixtures to resolve version differences.

### 3.2 Random-effects mapping

For term i with m_i grouping levels and k_i random-design columns, construct
Z_i with q_i = m_i k_i columns. Keep level-major/within-level coefficient order,
then match the reference term ordering and tie behavior through an explicit map.
Store Z in sparse form; never materialize a large indicator matrix.

Define one lower-triangular template T_i per ordinary covariance term:

$$
\Lambda_\theta =
\operatorname{blockdiag}_i(I_{m_i}\otimes T_i),\qquad
d=\dim(\theta)=\sum_i k_i(k_i+1)/2.
$$

Pack each lower triangle in the pinned R column-major order. For a two-column
term, T = [[theta_1, 0], [theta_2, theta_3]]. Diagonal entries have lower bound
zero; off-diagonal entries are unrestricted. Store a Lind-equivalent map,
lower bounds, term slices, level maps, and parameter labels.

Different terms are independent covariance blocks even if they share a group.
Combining their columns for a per-group solve must not introduce new covariance
parameters or remove cross-products in the penalized equations. Do not merge
duplicate or equivalent terms unless the pinned parser does so.

### 3.3 Rank and estimability

Let X_full have p_full columns and X the retained full-rank matrix of p columns.
Use a deterministic rank-revealing QR policy, retaining the full-to-reduced map,
pivot, rank threshold, and a basis N for null(X_full). The pinned implementation
uses a particular QR/drop policy; an SVD chosen independently need not drop the
same coefficient. Test exact aliases and threshold-sensitive cases separately.

A full-coordinate contrast l is estimable only if N' l is numerically zero.
Transform estimable contrasts into retained coordinates. Return an explicit
non-estimable status for the rest, including prediction rows. Dropped
coefficients may be represented as unavailable, never as estimated zeros with
zero standard error.

Require n > p for REML and a positive penalized residual sum of squares. Match
the pinned default group-size/nobs checks within the subset. Stricter rejections
and any expert overrides belong in the compatibility matrix. Do not discard
random-effect columns merely because their observed design is rank deficient:
the spherical penalty is what makes the random block solvable.

## 4. Mathematical specification

### 4.1 Model and weighted penalized least squares

All equations use the retained X and strictly positive weights. Let o be the
known offset, W = diag(w_i), and e = y - o:

$$
y=o+X\beta+Zb+\varepsilon,\quad
b=\Lambda_\theta u,\quad
u\sim N(0,\sigma^2 I_q),\quad
\varepsilon\sim N(0,\sigma^2 W^{-1}),
$$

with independent u and epsilon. The absolute random-effect covariance is
G = sigma² Lambda Lambda'. Define the relative marginal covariance

$$
H_\theta=W^{-1}+Z\Lambda_\theta\Lambda_\theta'Z',\qquad
\operatorname{Var}(y)=\sigma^2 H_\theta .
$$

This naming avoids multiplying sigma² twice when calculating coefficient
covariance. The model and PLS derivation follow
[Bates et al.](https://lme4.github.io/lme4/articles/lmer.pdf).
For fixed theta:

$$
r_\theta^2=\min_{u,\beta}
\left\|W^{1/2}(e-X\beta-Z\Lambda_\theta u)\right\|^2+\|u\|^2 .
$$

An implementation-ready elimination is:

$$
A=W^{1/2}Z\Lambda_\theta,\quad B=W^{1/2}X,\quad c=W^{1/2}e,
$$
$$
C=I_q+A'A,\quad D=A'B,\quad f=A'c,
$$
$$
S=B'B-D'C^{-1}D,\quad t=B'c-D'C^{-1}f,
$$
$$
\hat\beta=S^{-1}t,\quad \hat u=C^{-1}(f-D\hat\beta),\quad
\hat b=\Lambda_\theta\hat u.
$$

All inverse notation denotes factored solves. Evaluate r² from the weighted
residual norm plus the penalty, not subtraction of nearly equal quadratic
forms. Store each component separately. Obtain log|C| from the factor diagonal;
for R_X' R_X = S, log|S| = 2 sum log diag(R_X). Undo any sparse permutation
before exposing u or b. Empty fixed or random blocks have determinant one.

These relations also supply matrix-level identities against the independent
[PLS/GLS formulation](https://lme4.github.io/lme4/articles/PLSvGLS.pdf).
Normal-equation cancellation in S is a numerical risk. Detect it through
conditioning and backward residual checks; use a validated augmented QR route
or return a numerical failure. Adding a ridge, diagonal jitter, or clamping
negative pivots would change the target model.

### 4.2 Profiled objectives

Let nu = n for ML and nu = n-p for REML. Then

$$
d_{\rm ML}(\theta)
=\log|C|-\sum_i\log w_i+
n\left[1+\log\left(\frac{2\pi r_\theta^2}{n}\right)\right],
$$
$$
d_{\rm REML}(\theta)
=\log|C|-\sum_i\log w_i+\log|S|+
(n-p)\left[1+\log\left(\frac{2\pi r_\theta^2}{n-p}\right)\right].
$$

The weight determinant term must be retained even though it is constant during
one theta optimization. Omitting it corrupts absolute likelihoods and comparisons.
At a fixed theta, sigma_hat² = r²/nu. Nonpositive r², nonfinite objectives,
failed factorizations, or n <= p under REML produce typed failures.

For uncertainty in the variance parameters, retain the objective with sigma
unprofiled and beta eliminated:

$$
D(\theta,\sigma)=\log|C|-\sum_i\log w_i+
\mathbf{1}_{\rm REML}\log|S|+
\nu\log(2\pi\sigma^2)+r_\theta^2/\sigma^2.
$$

Implementations must agree with the pinned
[response likelihood code](https://github.com/cran/lme4/blob/4aa26a91f9e676e9409f6cd8163ae92654ef1e7e/src/respModule.cpp)
and fixed-theta reference outputs, not only with a fitted final scalar.

### 4.3 Independent dense oracle and fitted outputs

For small n, independently assemble H from term covariance contributions and
factor H directly. Compute GLS beta, Q = (e-X beta)'H^-1(e-X beta), log|H|,
and log|X'H^-1X|. Compare Q with r², log|H| with log|C|-log|W|, and the two
profiled criteria. The oracle must not call production Lambda assembly,
Schur-complement code, or objective helpers.

At the fitted theta:

$$
\widehat{\operatorname{Var}}(\hat\beta)=
\hat\sigma^2 S^{-1},\qquad
\widehat G_i=\hat\sigma^2 T_iT_i'.
$$

Report SD/correlation through G_i; theta entries are not correlations.
A zero marginal SD makes its correlation undefined. Use an unavailable value
and explicit status, not a fabricated zero or plus/minus one.

The conditional covariance of b given y at fixed beta and fitted covariance
parameters is sigma² Lambda C^-1 Lambda'. This is the target of ranef(condVar)
style output. It is not full uncertainty after estimating beta and covariance
parameters. Cross-term conditional covariance can be nonzero even when prior
terms are independent; selected diagonal blocks do not constitute the full
joint matrix. Validate extraction against [ranef](https://lme4.github.io/lme4/reference/ranef.html).

### 4.4 Likelihood bookkeeping and model comparison

Expose named objective_kind, objective, log_likelihood, sigma², p, d,
n_parameters, and df_residual_compat. For this unstructured Gaussian subset:

- log_likelihood = -objective/2 for the fitted ML or REML criterion;
- n_parameters = p + d + 1, including residual scale;
- AIC = -2 log_likelihood + 2 n_parameters;
- BIC = -2 log_likelihood + log(n) n_parameters;
- df_residual_compat = n - n_parameters, matching lme4 bookkeeping;
- n-p is the REML scale denominator, not the above residual DF;
- neither quantity is a universal denominator DF for fixed-effect tests.

Count all declared covariance parameters at a boundary, as the pinned reference
does; do not reduce AIC dimension because a variance estimate is zero. See
[pinned likelihood methods](https://github.com/cran/lme4/blob/4aa26a91f9e676e9409f6cd8163ae92654ef1e7e/R/lmer.R).

For comparisons, require identical retained observations/response, weights,
offsets, response transformation, and compatible model spaces. Compare different
fixed-effect models after fresh ML optimization. Merely evaluating an ML
criterion at a REML estimate is not an ML refit. REML comparisons require the
same fixed-effect design/basis convention, not just the same number of columns.
Check nesting in design/covariance spaces, not formula text. No generic AICc or
boundary-corrected LRT is included without a specific derivation.

## 5. Numerical backends and resource safety

### 5.1 Correct structural routing

The graph of the possible nonzeros of C determines whether a block solve is
valid. A single grouping partition produces independent per-level blocks after
all its term columns are combined. Nested factors generally have cross-products
between parent and child random effects:

For (1|g) + (1|g:h), the parent intercept column and each child intercept column
overlap on their observations. Reordering cannot remove their nonzero inner
products. A correct nested algorithm needs full top-level blocks, tree
elimination, or general sparse factorization. Top-level blocks may be large.

Routing contract:

1. Compute the structural graph using possible nonzeros over feasible theta,
   not observed numeric zeros at a particular iterate.
2. Accept the block backend only when connected components have proven
   independence and component-size/memory limits pass.
3. Route coupled models to the general sparse backend; reject unavailable
   capabilities before allocating factor workspaces.
4. Keep a small dense C backend for diagnostics; it is not an unbounded fallback.
5. Consider tree elimination only after equivalence and scaling evidence.

Zero theta is especially important: it can make C numerically equal to I without
changing the structural graph required at later iterates.

### 5.2 Backend contract and numerical discipline

A backend owns symbolic analysis, numerical factorization, solves, logdet,
permutation, workspace, and memory/capability diagnostics. Cache a symbolic
pattern for Z Lambda over all feasible theta, including structural zeros.
Invalidate it if the design, covariance structure, or index layout changes.

Require float64 and tested index widths. Do not form a dense n-by-n H outside
small-oracle/resource-gated operations. Preflight n, p, q, d, nnz, dense fixed
cross-products, expected factor fill, and selected covariance extraction cost.
A byte estimate is not an exact prediction of sparse fill: allocation failures
must still be caught and reported. Batch group solves with bounded scratch space.

Never share mutable objective/factor state between concurrent fits, optimizer
comparisons, profile points, or bootstrap workers. Reevaluate the accepted theta
before creating the immutable fit snapshot; a final finite-difference or rejected
trial must not leak its state into the result.

### 5.3 Dependency decision

Use NumPy/SciPy for the first verified block path. Evaluate CHOLMOD and alternative
sparse engines by fixed-theta parity, symbolic reuse, supported wheels, peak
memory, failure behavior, license obligations, and actual speed.

Do not promise a universal pure-Python dependency stack: NumPy, SciPy, NLopt,
and sparse backends may use native binaries even if kamino itself is a pure wheel.
Do not describe an optional extra as resolving linking/redistribution obligations.
Record exact dependency versions, build flags, linked modules, notices, and
wheel inspection evidence in an ADR before distributing that backend. Review
[SuiteSparse module declarations](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/Include/cholmod.h)
at the selected immutable revision.

Autodiff/JAX is experimental. Require explicit x64, supported primitives, and
independent derivative checks. Complex-step differentiation is valid only for
a compatible analytic implementation; ordinary conjugating Cholesky routines
cannot be assumed to support it. Do not infer a trustworthy Hessian from an
L-BFGS approximation.

## 6. Optimization, convergence, and singularity

### 6.1 Reference optimization

The reference path uses NLopt LN_BOBYQA through an owned adapter. The pinned
[lmerControl](https://github.com/cran/lme4/blob/4aa26a91f9e676e9409f6cd8163ae92654ef1e7e/R/lmerControl.R)
and [nloptwrap implementation](https://github.com/cran/lme4/blob/4aa26a91f9e676e9409f6cd8163ae92654ef1e7e/R/utilities.R)
define restart_edge = TRUE, boundary.tol = 1e-5, and wrapper defaults
xtol_abs = 1e-8, ftol_abs = 1e-8, maxeval = 100000.
The BOBYQA/Nelder–Mead two-stage sequence belongs to glmer; it is not lmer's
fallback rule.

Capture start values, bounds, NLopt version, all effective controls, initial
step behavior, restart/boundary evaluations, evaluation count, and termination
reason. Implement the pinned start heuristic or label an explicit-start
compatibility mode until default-start fixtures pass. Pin autoscale = FALSE
for the initial reference profile; any internal scaling must be invertible and
account for REML determinant changes.

A documented comparison optimizer is a diagnostic tool. Automatic retries, if
offered, are opt-in, bounded, and recorded with all outcomes. Choose among valid
candidates by the same objective and diagnostics. Never turn a failed first
attempt into an unexplained success.

### 6.2 Three independent statuses

Retain optimizer termination, numerical convergence, and inference availability
as different fields. A successful termination code is insufficient.

Diagnostics include objective/solve finiteness, backward residuals, active bounds,
finite-difference settings, projected gradient/KKT behavior, scaled-gradient
status, Hessian eigenvalues/conditioning, and optional cross-optimizer objective
spread. At a zero lower bound, the derivative into the feasible region can be
positive at a valid optimum; demanding an unconstrained zero gradient is wrong.

Preserve a reference-compatible diagnostic view and separately named enhanced
diagnostics. lme4 2.0 may skip derivative checks based on size/controls:
record not_computed with a reason, never infer that omitted checks passed.
Use step-size sensitivity and tighter refits to investigate apparent warnings;
the [convergence guide](https://lme4.github.io/lme4/reference/convergence.html)
describes the limitations of finite-difference diagnostics.

No epsilon floor on covariance diagonals, log-only parameterization, or
correlation clipping may eliminate the exact zero boundary from the reference
optimization domain.

### 6.3 Singularity compatibility versus spectral diagnostics

For the supported ordinary unstructured covariance, the pinned lme4 singularity
test checks whether any relative Cholesky diagonal is below the singularity
tolerance (default 1e-4; store the effective value). Match this as
singular_lme4. The exact implementation is in
[pinned covariance methods](https://github.com/cran/lme4/blob/4aa26a91f9e676e9409f6cd8163ae92654ef1e7e/R/covariance.R).

Also report per-term singular values/rotations of T_i, absolute/relative
covariance eigenvalues, condition estimates, and a separately named spectral
near-rank flag with its own tolerance. Do not substitute an eigenvalue rule and
call it identical to isSingular. Rotated rank loss in dimension >=3 need not
show an individually small variance or pairwise correlation near one.

Boundary covariance is a valid statistical outcome. Distinguish it from
nonidentifiability, failed optimization, and unsupported Hessian-based inference.
Where parameterization is nonunique, use G, fitted values, and objective
agreement as the main numerical evidence; do not demand an arbitrary theta
representation match.

## 7. Fit object, prediction, and reproducibility

### 7.1 Owned state and public outputs

ModelSpec owns labeled numerical arrays and immutable metadata: formula AST,
training encoder state, original/reduced coefficient maps, group/term maps,
weights, offsets, row mask, theta mapping, and compatibility profile.
FitResult owns estimates, objective, diagnostics, and inference capabilities.
Backend handles and mutable workspaces remain private and reconstructible.

Use separate names for fixed coefficients, random-effect modes, and
group-specific coefficients. lme4 fixef and coef are not interchangeable.
Preserve term identity when multiple terms share a group.

Keep NumPy arrays in the numerical domain. Use Polars for optional tabular
presentation/ingestion and pandas only behind required interoperability adapters.
Copy or make arrays read-only at ownership boundaries.

### 7.2 Prediction contract

Require an explicit mode in the Python API:

| Mode | Mean | Group behavior |
|---|---|---|
| population | o_new + X_new beta_hat | Ignore all fitted group effects; grouping columns unnecessary |
| conditional | o_new + X_new beta_hat + Z_new b_hat | Include selected/all fitted terms; known levels use matched effects |
| new-observation simulation | Draw response from a named conditioning scheme | Declare new/existing groups and simulated versus fixed random effects |

These map to R's population re.form = NA/~0 and conditional re.form = NULL
conventions; verify against [predict.merMod](https://lme4.github.io/lme4/reference/predict.merMod.html).
In a Gaussian identity-link model, integrating zero-mean random effects gives
the same mean as population prediction.

Unknown fixed-effect levels error. Unknown or missing grouping levels error
unless allow_new_groups is explicit; then only their unknown random contribution
is zero, with a row-level flag. Partial conditioning must validate each selected
term. Preserve input row order and masks, and distinguish missing predictor
values from missing group labels.

Store/reapply training contrasts and transform parameters. Formula offsets must
be evaluated for new data; an argument-only offset requires an explicit new
vector. Do not recycle the training offset. Prediction on a transformed response
remains on that modeled scale; inverse-transforming a mean generally does not
give the response-scale expectation.

For population mean SE, x' Cov(beta) x is a plug-in coefficient calculation.
Conditional-mean intervals and new-response intervals require distinct
covariance/conditioning calculations. New responses can need residual variance,
new-group variance, existing-group uncertainty, parameter uncertainty, and
cross-covariances. Do not manufacture conditional prediction intervals by adding
independent ranef SEs to fixed-effect SEs. Gate richer uncertainty behind an
explicit method and reference/simulation evidence.

### 7.3 Artifact schema and privacy

Use a versioned JSON manifest plus non-executable numeric/tabular payloads.
Record code/model/plan hashes, reference profile, all ordered labels, contrast
and transform state, controls, backend/build identity, diagnostics, and checksums.
Validate schema, dimensions, paths, size limits, and hashes on load; reject
pickle/object arrays and arbitrary code reconstruction.

Prediction-only artifacts need no training response. Full refit/bootstrap or
reference-grid recovery requires an explicitly saved private model frame or a
verified caller-supplied snapshot. Expose missing capabilities after reload.
Save atomically through a temporary sibling and explicit replace policy.
Checksums detect corruption; they do not authenticate a publisher.

Random-group labels and training metadata may identify people. Do not emit them
to logs by default. Training arrays are caller data, not fixture candidates.
Document deletion of saved bundles and RNG checkpoint retention.

## 8. Inference specifications

Every method returns estimate, SE or interval, statistic, null value, method ID,
conditioning, covariance estimator, reference distribution, numerator/denominator
DF where applicable, adjustment, and available/unsupported/failed status.
A request for one method must not silently substitute another.

### 8.1 Base summaries and estimable linear functions

Match the lme4 Gaussian summary's estimates, SEs, and coefficient/SE statistics.
Its default summary does not supply small-sample p-values. Offer explicitly
requested asymptotic normal tests/intervals labeled as approximations. Do not
relabel them as lmerTest output or use df_residual_compat as test DF.

For L beta = r, check estimability and row rank of L before inference.
Expose hypothesis rank and how redundant rows are handled. Type I/II/III ANOVA
requires separate, contrast-aware hypothesis construction; do not infer these
tables from a list of coefficient p-values.

### 8.2 Satterthwaite

Use the complete variance-parameter vector eta = (theta, sigma), or a documented
smooth equivalent with the correct Jacobian. Let H_D be the Hessian of the
unprofiled-scale objective D in §4.2 at the fit. For a regular interior optimum:

$$
\widehat{\operatorname{Var}}(\hat\eta)=2H_D^{-1},\quad
v=l'\operatorname{Cov}(\hat\beta;\eta)l,\quad
g=\nabla_\eta v,\quad
\nu_l=2v^2/(g'\operatorname{Var}(\hat\eta)g).
$$

The factor two follows from differentiating negative twice log-likelihood.
A profiled theta Hessian alone loses residual-scale uncertainty and its
cross-covariance. Differentiate coefficient covariance with sigma independent of
theta, not sigma reprofiled at each perturbed theta.

Validate the Hessian, Jacobians, variance-parameter covariance, one-DF contrast
SE/DF/p-values against the pinned
[lmerTest implementation](https://github.com/runehaubo/lmerTestR/blob/master/R/lmer.R)
and [method paper](https://www.jstatsoft.org/article/view/v082i13).
Archive immutable companion-package sources in the oracle manifest.

Multi-DF tests require the published joint-test construction; averaging
individual DFs is not valid. Declare pseudoinverse eigenvalue cutoffs and match
the reference in diagnostic tests, but do not bless unstable/boundary inference
because a generalized inverse returns a number. Nonpositive variance/DF,
material negative curvature, or unstable derivatives return an unavailable
approximation with evidence. OLS/WLS limiting cases must recover n-p for
ordinary estimable contrasts under REML.

### 8.3 Kenward–Roger

Implement the pbkrtest-supported Gaussian model class with covariance expressed
as a sum of known component matrices. Supply adjusted coefficient covariance,
the scaled F statistic, numerator/denominator DF, and p-value, not just a
modified standard error. Its [KR comparison contract](https://hojsgaard.github.io/pbkrtest/reference/kr-modcomp.html)
requires REML, a common covariance structure, and a valid fixed-effect
restriction.

If given an ML fit, make a separate REML refit and attach its provenance; do not
mutate the original fit. Correlated slopes require the covariance-component
mapping and derivatives to be checked, not a naive theta Hessian substitution.
Prior-weight, rank-deficient, near-singular, and large-n cases require explicit
support gates. Bound memory; do not hide a dense n-by-n KR computation behind a
scalable-fit claim.

### 8.4 Likelihood profiles

Provide lme4-style ML profiles on named SD/correlation, residual-scale, and
fixed-coefficient targets, with optional variance/covariance presentation.
Profile means fixing the reported target and reoptimizing nuisance parameters;
varying one raw theta coordinate is not a variance-component profile.

For a REML fit, retain a separate ML refit as the profile baseline, following
[lme4's devfun2 path](https://github.com/cran/lme4/blob/4aa26a91f9e676e9409f6cd8163ae92654ef1e7e/R/profile.R).
Store baseline estimates, objective differences, signed-root deviance,
interpolation/bracketing controls, and optimizer diagnostics. Match 2.0 labels
and output order separately from the legacy corpus.

A materially lower objective found during profiling invalidates the baseline:
improve/restart it or fail the profile. Represent unbounded endpoints,
boundary truncation, nonmonotone profiles, and unidentified correlations
explicitly. Nominal profile cutoffs are asymptotic; they do not automatically
provide boundary-calibrated coverage. There is no universal requirement that a
profile interval enclose a Wald interval.

### 8.5 Parametric bootstrap

Default unconditional simulation draws fresh independent z_b and z_e:

$$
y^*=o+X\hat\beta+Z\hat\Lambda\hat\sigma z_b+
\hat\sigma W^{-1/2}z_e .
$$

Refit the original specification and estimation method on each draw.
An explicit conditional mode instead fixes fitted random effects and resamples
residuals; it estimates a different sampling target. Preserve the distinction in
the API and evidence, following [bootMer](https://lme4.github.io/lme4/reference/bootMer.html).

Allocate deterministic streams by root seed, replicate ID, and purpose. Record
RNG/NumPy versions; fixed seeds do not imply identical R/Python or cross-version
draws. Compare R/Python refits on stored simulated response arrays, and validate
simulation moments separately. One versus multiple workers must use the same
replicate inputs and yield equivalent fitted outputs within numerical tolerance.

Use a bounded worker pool and BLAS threads, private fit workspaces, cancellation,
and a resumable ledger. Every replicate records success, warning, singularity,
optimizer failure, or statistic failure. A singular but valid fit is not
automatically a failed replicate. Default: report no interval/p-value if
unresolved failed replicates remain; users may inspect partial draws with an
explicit incomplete flag. Never silently redraw until enough successful fits.

Choose and document interval types (percentile/basic first), quantile convention,
replicate count, and Monte Carlo uncertainty. Bias-corrected/studentized
intervals require their own derivations and tests.

### 8.6 Likelihood-ratio tests

For ordinary nested fixed-effect hypotheses, use ML refits and the correctly
ranked restriction; label chi-square calibration as asymptotic. Random-covariance
null hypotheses can be on a boundary with unidentified nuisance correlations.
A universal half-chi-square mixture is not valid for arbitrary slope/covariance
tests; see [Self and Liang](https://doi.org/10.1080/01621459.1987.10478472).

For supported boundary hypotheses, prefer a parametric-bootstrap comparison
simulated under the null and refit both models per replicate. With B complete
replicates, use a declared Monte Carlo convention such as
(1 + count(T_b >= T_observed))/(B + 1), and record its finite-B resolution.
Check nestedness, matching data, and optimizer failures. Material negative LRT
statistics indicate fitting trouble; tiny negatives may be rounded only under
a predeclared numerical tolerance. Specialized mixture or restricted-LRT methods
remain separately named and gated.

## 9. Post-estimation scope and capability contracts

### 9.1 Product and interface boundary

Build kamino first in one repository. Keep postfit as a separately releasable
future distribution with no mandatory kamino dependency. Do not freeze a public
generic protocol until the kamino adapter and two external adapters pass the same
contract tests. Statsmodels OLS/WLS and MixedLM are useful candidates with
different rank/weight/covariance behavior.

Use small capabilities with explicit versions:

| Capability | Required information |
|---|---|
| Linear-function basis | Ordered coefficients/covariance, full-to-reduced mapping, null-space basis, design/transform state |
| Prediction | Explicit modes, new-data design/Jacobian, offsets, supported uncertainty and scales |
| Contrast inference | Contrast-specific one/multi-DF method; selected covariance must agree with the DF calculation |
| Working covariance | Cluster-aligned blocks/solves, model-based target, weight interpretation, parameter identities |
| Estimating equations | Defined normalization, bread, cluster scores, observation/cluster order, residual definition |
| Variance decomposition | Term-specific G and row-wise random designs, residual weights, declared averaging population |

No generic hasattr check can establish these statistical invariants. Contract
tests permute rows/parameters, omit levels, introduce aliases, and misalign
clusters. A custom covariance provider invalidates inherited model-based DF
unless the requested method explicitly supports that covariance estimator.

### 9.2 Estimated marginal means and marginal effects

Estimated marginal means are linear functions of a specified reference grid,
averaged with explicitly chosen weights. Support equal, proportional, outer,
cells, flat, and user-defined weights only when their denominators and empty-cell
rules match the pinned [emmeans reference](https://rvlenth.github.io/emmeans/reference/emmeans.html).
Declare nesting, at/by/covariate reduction, factor contrasts, offset reduction,
and estimability. Grids must respect nested factors and deterministic relations
between derived covariates; a Cartesian product can create impossible rows.

Carry the coefficient-aligned basis, covariance, non-estimability basis, and
contrast DF as required by the [extension contract](https://rvlenth.github.io/emmeans/reference/extending-emmeans.html).
Population versus conditional prediction, empirical versus grid averaging, and
response transformation are separate choices.

Define adjustment families across by-groups. Implement Holm/Bonferroni/Sidak
first; Tukey and multivariate-t need their appropriate family, DF, numerical
integration, and uncertainty rules. Heterogeneous contrast DFs do not define a
unique multivariate-t distribution. Unsupported combinations error.

Marginal effects may average observed rows or a user-selected grid. Slopes must
differentiate the complete stored transformation/design, including interactions;
central differences need step-size checks. Conditional slopes can include
random-slope contributions. Integrate with the existing
[marginaleffects project](https://github.com/vincentarelbundock/marginaleffects)
through a proven adapter when feasible; upstream contribution is a later
project action, not assumed permission to publish.

### 9.3 Cluster-robust inference

For independent clusters c and fixed working covariance V_c, define the GLS
bread B = (sum X_c' V_c^-1 X_c)^-1 and marginal residual
e_c = y_c - o_c - X_c beta_hat. A generic sandwich has

$$
s_c=X_c'V_c^{-1}A_c e_c,\qquad
\widehat{\operatorname{Var}}_{\rm CR}(\hat\beta)
=B\left(\sum_c s_cs_c'\right)B'.
$$

Define normalization and weight conventions explicitly. Conditional residuals
after subtracting fitted b are not interchangeable with marginal GLS residuals.
The familiar OLS expression A_c = (I-H_cc)^-1/2 is not a general mixed-model
CR2 algorithm. Implement the reference working-target adjustment and leverage
operators, retaining cluster-specific matrices and tests.

The [clubSandwich lmer adapter](https://github.com/jepusto/clubSandwich/blob/master/R/lmer.R)
currently rejects prior-weight fits and requires every random-effect grouping
factor to be nested in the chosen independent clustering variable. Honor those
restrictions in its compatibility profile; crossed models can invalidate the
proposed independent clusters. Weighted/crossed robust extensions need a
separate derivation and oracle.

Ship CR0/CR1 only after contract tests, then CR2 with contrast-specific
Satterthwaite inference. Multi-parameter tests must specify an actual supported
method such as HTZ and match [Wald_test](https://jepusto.github.io/clubSandwich/reference/Wald_test.html);
do not implement an undefined generic “AHR” test. CR3 is another independent gate.

### 9.4 Tidy output and variance diagnostics

Tidy output is presentation over validated capabilities. Do not populate
p_value, influence, leverage, or Cook's distance unless that method was
computed. Marginal/conditional residuals and scaled/weighted residuals need
separate definitions.

For random slopes, compute average random variance from
mean_i(z_i' G z_i), including covariance cross-terms; summing intercept variances
is wrong. Under weights, residual variance per row is sigma²/w_i.
For a declared averaging measure define V_fixed, V_random, V_residual and then

- marginal R² = V_fixed / (V_fixed + V_random + V_residual);
- conditional R² = (V_fixed + V_random) / the same denominator.

Name weighted/generalized versions separately unless validated against a pinned
reference. A single ICC need not describe all covariate values in a slope model:
report its definition and population or covariate dependence. Residual normality
or heteroscedasticity checks do not prove model validity, and ordinary independent
residual tests may be inappropriate. Generic AICc and influence claims remain
deferred.

## 10. Statistical validation plan

This section specializes the GitHub template's STATISTICAL_ANALYSIS_PLAN:
the estimands are likelihoods, model parameters, estimable contrasts, interval
coverage, and test size. It is a frequentist Gaussian implementation; Bayesian,
trading-cost, temporal-validation, and financial-data fields are inapplicable
unless a later example actually uses those settings.

### 10.1 Three complementary oracles

1. Independent small-n marginal-covariance algebra (§4.3).
2. Pinned R fixed-theta intermediate outputs, before comparing optimizers.
3. End-to-end optimized fits, predictions, and inference in each reference profile.

At each fixed theta retain X_full/X, Zt, weights, offsets, Lambda/theta mapping,
rank/drop metadata, theta bounds, log determinants, residual/penalty components,
beta, u, b, scale, and the ML/REML criterion. Compare at identity/interior,
nonzero-correlation, near-boundary, exact-boundary, and rescaled theta vectors.
Do not compare Cholesky entries across different fill permutations; compare
reconstructed operators, solves, and log determinants.

The generator uses version adapters for lFormula, mkLmerDevfun, getME, and new
covariance APIs. Evaluate the deviance closure at a theta before extracting
state and reset it before another extraction. A closure's mutable environment
is not a reusable immutable fixture.

Final-fit fixtures store -2*logLik(fit, REML=...) explicitly with objective kind;
do not use an ambiguous deviance() call. Request every companion inference
method, contrast, cluster variable, and weighting option explicitly. Unavailable
method results are typed expected outcomes, not JSON NaN disguised as success.

Store matrices in lossless binary/sparse coordinate form with shapes, index base,
and labels; JSON holds metadata and safe scalar values. Store input hashes,
redistribution rights, script hash, sessionInfo(), source revisions, warnings,
and expected capability failures. Regeneration creates a reviewed corpus
version; it must never rewrite golden outputs just to make a regression pass.

### 10.2 Corpus

| Case | Main evidence |
|---|---|
| Dyestuff/Dyestuff2 | Basic fit, exact/near-zero covariance, boundary diagnostics |
| sleepstudy | Correlated and independent slopes, ML/REML, contrasts, prediction |
| Pastes | Nested expansion and parent-child coupling |
| Penicillin | Small crossed structure |
| Oats/Machines | Split-plot/categorical contrasts; record source package and license |
| InstEval | Crossed scale, sparse fill, resource and performance behavior |
| Hand-checkable synthetic models | Positive weights, offsets, aliases, no intercept, zero fixed block, all-zero theta |
| Adversarial synthetic models | Unequal group sizes, three-plus-dimensional singular covariance, poorly scaled slopes, confounding and nonidentifiability |
| Formula/new-data cases | Unused/reordered levels, NA in different terms, nested transforms, unseen groups, impossible grids |
| Operational cases | Cancellation, factor failure, allocation failure, corrupt artifact, incomplete bootstrap |

Dataset redistribution requires provenance review before fixtures/data are
committed. Independently generated small examples allow work to begin without
assuming package datasets are freely redistributable.

### 10.3 Tolerances and conditioning

Use the per-element rule |actual-reference| <= atol + rtol*|reference|,
plus normalized residual/operator checks. Proposed starting ceilings below
apply to well-conditioned fixtures in native units; Phase 0 must measure them
on supported platforms before freezing the tolerance manifest.

| Quantity | atol | rtol | Additional gate |
|---|---:|---:|---|
| Encoded floating design values | 1e-12 | 1e-12 | Labels, masks, bounds, integer sparsity structure exact |
| Fixed-theta small-model criterion/logdet | 1e-8 | 1e-10 | Independent oracle and R |
| Fixed-theta solves and covariance | 1e-8 | 1e-8 | Normalized backward residual <= 1e-10 in regular cases |
| Final optimized criterion, small/medium models | 1e-6 | 1e-9 | Same objective convention; inspect large-model absolute errors separately |
| Final beta, sigma, G, fitted values | 1e-6 | 1e-5 | Identified regular models, aligned labels |
| Raw theta | 1e-5 | 1e-5 | Secondary check only where locally identifiable |
| Satterthwaite/KR SE, DF, statistic | 1e-5 | 1e-4 | Regular method-supported cases; covariance adjustment also checked |
| Profile endpoints | 1e-4 | 1e-3 | Check objective cutoff/bracketing, not just endpoint proximity |
| Deterministic p-values | 1e-8 | 1e-4 | Tail-relative/log-probability checks for very small p |
| Bootstrap summaries | No universal pointwise tolerance | — | Same draws for refit parity; otherwise Monte Carlo uncertainty |

Large objective constants can hide poor solutions under relative tolerance.
Additionally compare objective differences, fitted covariance, and optimizer
diagnostics; set case-specific ceilings before optimization tuning. Singular/
ill-conditioned cases use a separate declared contract: objective and identifiable
quantities must agree, undefined parameters are flagged, and no global tolerance
loosening is allowed. Parameter magnitudes alone cannot establish identification.

### 10.4 Valid invariants

- At Lambda = 0, reproduce weighted linear-model ML/REML, with the correct
  determinant constant and scale denominator.
- Multiplying all weights by c > 0 maps sigma² to c sigma² and theta to
  theta/sqrt(c); absolute G, fitted values, beta covariance, and optimized
  likelihood remain unchanged. Relative singularity threshold flags need not
  be invariant under this reparameterization.
- Replacing y by y-o and removing the offset gives the same estimates and
  criterion; original fitted values differ by o.
- Permuting rows or relabeling/reordering groups preserves equivalent results
  after alignment; compare tolerantly because floating-point reductions differ.
- Numeric (x||g) and its pinned semantic expansion produce the same model.
- A nonsingular fixed-basis change X -> X A leaves ML and fitted predictions
  invariant; REML criterion shifts by 2 log|det A|. Transform coefficients and
  covariance accordingly. Never assert raw REML invariance under arbitrary coding.
- Equivalent random-basis transformations preserve the fit only when the entire
  covariance family is transformed. Centering a slope generally changes an
  independent intercept/slope covariance restriction.
- Retained X satisfies weighted mixed-model normal-equation residual checks.
- Singular Lambda leaves C positive definite because of I; a factorization
  failure there indicates numerics or implementation trouble.
- Profile endpoints satisfy the declared likelihood cutoff where finite;
  no profile-versus-Wald ordering assertion is used.

### 10.5 Calibration, assessment separation, and failure accounting

Write a method-specific simulation manifest before tuning: data-generating
model, parameter grid, null/alternative, estimand, sample size, cluster count,
imbalance, covariance conditioning, interval/test method, seed bank, replicate
count, and pass thresholds.

Use separate development and locked assessment seeds. For each regular
advertised inference cell, propose 10,000 assessment datasets at nominal
alpha = 0.05 and a release envelope of [0.04, 0.06] for rejection rate and
[0.94, 0.96] for nominal 95% coverage. Report binomial Monte Carlo intervals;
apply simultaneous uncertainty control across the predeclared primary cells.
If uncertainty straddles a gate, the result is inconclusive; use a predeclared
larger fixed batch, not repeated favorable-seed selection. These are proposed
engineering accuracy targets, not a theorem about small-sample methods.

Few-group, imbalance, boundary, and near-nonidentification scenarios are mandatory
stress cells. Where a reference method itself is poorly calibrated, report both
parity and failure of calibration; restrict or label the supported regime.
Do not alter the algorithm and still claim exact reference-method equivalence.

Count every simulated dataset, including failed fits and unavailable inference.
Report coverage on completed intervals and failure rates separately, and bound
overall coverage treating missing intervals as both misses and hits. Stable
advertised regular regimes require failure rate <=0.5% and a satisfactory
overall bound. Failed bootstrap replicates remain visible under §8.5.
Nested bootstrap studies need a fixed compute budget and reported Monte Carlo
error from both simulation levels.

## 11. Architecture and production-template application

### 11.1 Chosen starting structure

Use the template's shared governance/delivery files and selectively adapt the
python-data-quant archetype. There is no existing Python-library archetype;
do not claim that a library generator exists. The project folder is
~/Projects/kamino, created on 2026-09-14 with this plan and its review archive.
The project and Python package are named kamino. Git initialization, application
scaffolding, and implementation remain future work.

The template's Statsmodels default is intentionally overridden for the core:
implementing a native lme4-compatible algorithm is the project objective.
Statsmodels remains a test comparator and future adapter target. Polars remains
a tabular boundary, while labeled float64 arrays represent the mathematics.
Record both choices as project ADRs; they require no new research into whether
to replace the requested fitter with another package.

Planned layout:

~~~text
kamino/
  .github/{CODEOWNERS,dependabot.yml,PULL_REQUEST_TEMPLATE.md,workflows/}
  AGENTS.md
  PROJECT_BRIEF.md
  PROJECT_MEMORY.md
  PRODUCTION_READINESS.md
  README.md
  CHANGELOG.md
  CONTRIBUTING.md
  SECURITY.md
  REPRODUCIBILITY.md
  LICENSE
  Makefile
  pyproject.toml
  uv.lock
  .python-version
  src/kamino/
    formula/             # syntax, encoders, shared frame, compatibility
    model/               # labeled ModelSpec and parameter maps
    core/                # PLS objective and owned numerical contracts
    backends/            # block and sparse adapters
    optimize/            # controls, reference NLopt, diagnostics
    inference/           # independent method implementations
    results/             # snapshots, prediction, presentation
    io/                  # validated artifact encoding/loading
    py.typed
  tests/{unit,contract,integration,oracle,statistical,artifact}/
  tools/{fixtures,bench,verify_release.py}
  oracle/
    Dockerfile
    renv.lock
    manifest.json
  docs/{methods,compatibility,adr,runbooks}/
  schemas/{model-spec,fit-result,fixture,telemetry-event}.schema.json
  checklists/RELEASE_READINESS.md
  templates/{ADR,STATISTICAL_ANALYSIS_PLAN,THREAT_MODEL,INCIDENT_REVIEW,
             AGENTIC_VERIFICATION_LOOP,WORK_NOTE,IMPROVEMENT_PLAN,
             TELEMETRY_REVIEW,PERFORMANCE_EXPERIMENT}.md
  notes/
~~~

Keep the Python and R locks authoritative for their own package managers.
Future postfit can move to its own repository with independent
versioning after the capability spike; do not create unused public package shells.

### 11.2 Template crosswalk

All links below identify the inspected immutable template revision.

| Template artifact | Application to kamino | Required evidence |
|---|---|---|
| [PROJECT_BRIEF](https://github.com/joshuamyers22/production-project-template/blob/6526db479fced81463d1a97e25ff3ceefbe9eee2/templates/PROJECT_BRIEF.md) | §1 users, journeys, constraints, owners | Completed brief and requirement IDs |
| [PRODUCTION_BLUEPRINT](https://github.com/joshuamyers22/production-project-template/blob/6526db479fced81463d1a97e25ff3ceefbe9eee2/docs/PRODUCTION_BLUEPRINT.md) | First vertical slice; pure model policy and owned solver/RNG/I/O boundaries | Architecture ADR and contract tests |
| [Repository standard](https://github.com/joshuamyers22/production-project-template/blob/6526db479fced81463d1a97e25ff3ceefbe9eee2/standards/PRODUCTION_REPOSITORY_STANDARD.md) | src layout, metadata, locks, build, CI, governance | Clean-clone and artifact evidence |
| [Python guide](https://github.com/joshuamyers22/production-project-template/blob/6526db479fced81463d1a97e25ff3ceefbe9eee2/docs/PYTHON_ENGINEERING_GUIDE.md) | Array ownership, shapes, typed APIs, numerical solves | Strict typing, shape/aliasing/error tests |
| [STATISTICAL_ANALYSIS_PLAN](https://github.com/joshuamyers22/production-project-template/blob/6526db479fced81463d1a97e25ff3ceefbe9eee2/templates/STATISTICAL_ANALYSIS_PLAN.md) | §10 and one completed plan per inference method | Estimand, sample, covariance, seeds, locked assessment, evidence hashes |
| [THREAT_MODEL](https://github.com/joshuamyers22/production-project-template/blob/6526db479fced81463d1a97e25ff3ceefbe9eee2/templates/THREAT_MODEL.md) | Formula execution, hostile artifacts, allocation growth, native binaries, private data | Negative-path and resource tests |
| [Adversarial review](https://github.com/joshuamyers22/production-project-template/blob/6526db479fced81463d1a97e25ff3ceefbe9eee2/templates/ADVERSARIAL_CODE_ARCHITECTURE_REVIEW.md) | Milestone review with defect severity and dispositions | Evidence-linked findings; unresolved blockers remain open |
| [Verification loop](https://github.com/joshuamyers22/production-project-template/blob/6526db479fced81463d1a97e25ff3ceefbe9eee2/templates/AGENTIC_VERIFICATION_LOOP.md) | Coherent changes, distinct evidence, resource/stop limits | Requirements and results, no transcripts or token-based correctness claims |
| [PERFORMANCE_EXPERIMENT](https://github.com/joshuamyers22/production-project-template/blob/6526db479fced81463d1a97e25ff3ceefbe9eee2/templates/PERFORMANCE_EXPERIMENT.md) | §12 correctness-first time/memory benchmarks | Baseline/candidate distributions and rollback criterion |
| [Release checklist](https://github.com/joshuamyers22/production-project-template/blob/6526db479fced81463d1a97e25ff3ceefbe9eee2/checklists/RELEASE_READINESS.md) | §14 requirement scorecard and signed/traceable release | Links to actual CI, oracle, calibration, wheel, license evidence |

Template tailoring: no production Docker image, service health endpoint, API
authentication, database backup, or runtime dependency export is needed for a
pure library. A digest-pinned R oracle container is required for the selected
reference workflow. No financial forecasting pipeline, Bayesian sampler,
automatic telemetry shipping, or time-aware split policy is inherited merely
because the starting archetype is named data-quant.

Keep AGENTS/PROJECT_MEMORY as bounded, dated evidence indexes. Import template
governance files without copying source_material books into the project.
Use explicit local diagnostic callbacks for fit timings/status, not automatic
network collection. Events omit response values, formulas containing sensitive
names, and group labels. Release-cycle aggregate issue/benchmark reviews can use
the template's telemetry/improvement records where there is actual evidence.

The owner selected the MIT License for Kamino on 2026-09-14. This does not
relicense or authorize redistribution of lme4 code or datasets; separately review
dependencies and any source translation. Independent implementation from
mathematics still requires accurate attribution.

## 12. Engineering verification and performance

### 12.1 Commands and CI

| Command | Meaning |
|---|---|
| make setup | Frozen development install; select explicit supported extras |
| make check | Ruff, strict Pyright, unit/contract/integration tests, risk-based coverage floor, docs checks, build |
| make oracle | Fixed-theta and fitted corpus checks in pinned reference environment |
| make statistical | Locked calibration plan and machine-readable report |
| make benchmark | Manifested correctness-gated time/memory measurements |
| make supply-chain | Vulnerability/license audit and SBOM verification |
| make release-check | All applicable gates plus clean wheel/sdist installation and evidence index |

Do not use an all-extras install as a substitute for separate backend support
matrices. Test base install, each supported extra, and incompatible/missing
backend cases. Include lower-bound and supported newer dependencies in dedicated
compatibility jobs; exact research reproduction uses the lock.

Default Python tests require neither R nor network. Scheduled and release jobs
run R oracles and simulations. A candidate-reference upgrade gets its own corpus
and reviewed diff, leaving the pinned release oracle intact. Floating upstream
checks may report drift but cannot silently redefine the contract.

CI must use least privileges, immutable action SHAs, explicit timeouts, safe
fork behavior, supported Python/OS/backend jobs, clean artifacts, and warning/
failure capture. Build wheel and sdist; build/install from the sdist in a clean
environment to detect missing files. Test the installed distribution outside
the source checkout, including import, fit, predict, and artifact reload.

Use a measured coverage floor for maintained numerical code and negative paths.
Coverage percentage alone is not acceptance evidence. Include targeted fault or
mutation checks capable of detecting a missing weight determinant, missing
cross-term coupling, double sigma scaling, parameter-order permutation, and
reuse of stale objective state.

### 12.2 Performance contract

Compare with pinned lme4 on the same hardware, BLAS thread count, model/sample,
optimizer controls, and precision. Record parse/encoding, matrix assembly,
symbolic analysis, fixed-theta evaluation, optimization, inference, and total
time separately, with n/p/q/d/nnz, evaluation counts, cold/warm behavior, median,
p95, and peak RSS.

The old 50ms/100ms and 10-second claims are unvalidated benchmark aspirations,
not release promises. Establish baselines in Phase 0 and publish actual results.
InstEval within 2x lme4 remains a proposed stable-scale objective; support may
not be claimed until both correctness and installation pass. Do not claim the
block solver beats lme4 on arbitrary nested models.

Pin dedicated benchmark hardware for timing regression decisions. Shared CI
checks completion/resource ceilings and retains timing reports without flaky
microbenchmark assertions. Proposed investigation threshold: >20% time or
memory regression beyond baseline variation; accept intentional tradeoffs only
with rationale and updated evidence. Optimize one measured stage at a time.

## 13. Dependency-ordered implementation roadmap

Each phase produces a usable evidence bundle and a template-based review.
Statistical correctness defects block the affected feature regardless of
schedule. The product owner, numerical implementer, statistical reviewer, and
release maintainer are accountable roles; assign names before their gates.

### Phase 0 — contract, oracle, and backend feasibility

Deliver the project brief, compatibility profiles, architecture/template
tailoring ADRs, source/license inventory, runtime matrix, pinned oracle container
and locks, dense oracle, hand-checkable examples, formula spike, and block/sparse
benchmarks. Build one small formula-to-result walking skeleton with ML and REML.

Gate: actual R environment builds; X/Z and fixed-theta algebra agree for regular,
weighted, offset, and boundary examples. Record formula/backend choices and
unresolved limitations. Unverified installers, licenses, or reference versions
remain open evidence, not guessed defaults.

Status 2026-09-14: complete. The local and hosted gates pass, the MIT license is
recorded, and the reviewed Linux ARM64 oracle is public at the immutable GHCR
digest recorded in `oracle/manifest.json`. Architecture expansion is future
profile work, not an implied property of the Phase 0 image.

### Phase 1 — first alpha: one verified grouping structure

Implement shared frame/encoders, theta assembly, block PLS, reference optimizer,
positive weights/offsets, strict rank checking, diagnostics, explicit prediction,
immutable results, safe bundle, built-wheel smoke tests, and rejection paths.
Use Dyestuff, Dyestuff2, sleepstudy, and synthetic cases.

Gate: all alpha compatibility rows pass design/fixed-theta/final-fit/prediction
tests; errors and boundary outcomes are structured; source and built artifacts
pass make check. No small-sample DF claim yet.

Status 2026-09-15: the Dyestuff vertical slice is complete through the public
API for one fixed intercept plus one random intercept. ML/REML estimates and
population/conditional predictions pass the pinned lme4 oracle and independent
dense checks; errors, new groups, argument offsets, immutable arrays, and the
installed-wheel path are exercised. This is partial Phase 1 evidence, not the
Phase 1 gate.

Status 2026-09-15, second slice: the owned random-intercept block evaluator is
now the public fit path. Dyestuff2 ML/REML fits accept the exact theta-zero
boundary and match pinned lme4 fits and predictions, the dense PLS route, the
independent marginal oracle, and closed-form fixed-model invariants.

Status 2026-09-15, third slice: the production formula boundary now encodes a
random intercept with one immutable group index per observation. Formulae sees
only the fixed-effects formula and cannot materialize the grouped indicator;
the compact specification has no dense `Z`. Explicit dense materialization is
limited to small-model PLS/oracle tests. The next open work is expanding the
formula/theta model and block backend to sleepstudy random slopes; that work is
recorded in the following slice.

Status 2026-09-15, fourth slice: the compact specification and block backend now
support one correlated numeric random intercept/slope term. Sleepstudy ML/REML
passes ten pinned fixed-theta cases, an independent dense marginal oracle,
optimized covariance/mode/result checks, and population/conditional prediction.
The public diagnostics identify SciPy Powell rather than implying the pinned R
NLopt optimizer ran. A separate synthetic optimized case returns the exact
singular slope boundary. Safe model bundles, resource benchmarks, and the
remaining alpha formula rows remain open.

Status 2026-09-15, fifth slice: safe prediction-only bundles now round-trip the
verified Dyestuff, Dyestuff2, and Sleepstudy ML/REML models and preserve exact
new-data predictions. The versioned atomic artifact contains canonical JSON and
non-object float64 arrays, validates paths, schema, dimensions, limits, hashes,
labels, covariance identity, controls, and diagnostics, and records package,
implementation, plan, backend, and reference-profile provenance. It omits the
training response and all training rows and reloads as an explicitly
capability-limited prediction model. Resource benchmarks and remaining alpha
formula rows remained open after that slice.

Status 2026-09-15, sixth slice: the complete public random-intercept and
correlated-slope subset now passes a manifested resource gate at one million
observations and 10,000 groups under ML and REML. Theta-independent cross-products
are assembled once per fit and reused by the optimizer; no dense `Z` or q-by-q
factorization is constructed. Local peak RSS was at most 1,257 MB, versus an
80/160 GB dense-`Z` estimate, and shared CI enforces completion/correctness plus
a 2,000 MB ceiling while retaining the machine-readable report. Timing remains
machine-specific and no lme4 speed ratio is claimed because the verified
optimizers differ. The remaining alpha formula/corpus rows remain open.

Status 2026-09-15, seventh slice: the compact covariance map now preserves
multiple independent random terms sharing the same grouping factor. Numeric
`(1 + x || g)` and the explicit `(1 | g) + (0 + x | g)` form produce identical
group-major designs and fits while retaining the nonzero within-group design
cross-products in one solve. Sleepstudy ML/REML passes ten pinned fixed-theta
cases, final fits, covariance/mode checks, and predictions; a weighted-offset
case returns the exact zero slope boundary. Prediction bundle schema 1.1 records
term boundaries and retains schema-1.0 reading. The expanded million-row/10,000-
group resource manifest passes ML and REML without dense `Z`. Categorical double
bars, terms
with different grouping factors, and the remaining alpha formula/corpus rows
remain open.

Status 2026-09-15, eighth slice: the Phase 1 shared model frame and fixed-effect
expansion are complete locally. Response, fixed/random/group columns, weights,
formula and argument offsets, and Boolean subset selection use one
subset-before-NA row pipeline with retained/omitted/excluded identities. Additive
numeric/categorical effects, pairwise `*`, treatment/sum contrasts, and saved
new-data encoding pass exact X/row contracts plus four pinned weighted/offset
lme4 ML/REML fits and predictions. Bundle schema 1.2 persists that design state.
The expanded eight-case million-row resource gate passes locally without dense
random indicators. Hosted CI run 35017301412 passes the complete platform,
installed-wheel, quality, and expanded resource matrix. This closes the declared
Phase 1 implementation gate; later package-release gates remain separate.

### Phase 2 — stable core: structures and robustness

Add the verified general sparse backend and ordinary nested/crossed models.
Add reference rank-dropping/estimability, categorical random terms for supported
contrasts, derivative diagnostics, resource limits, and full artifact recovery.
Exercise Pastes, Penicillin, InstEval, and rotated singular/ill-scaled examples.

Status 2026-09-15, first stable-core slice: N03 is complete for ordinary
nested/crossed random-intercept structures. Structural routing preserves the
single-group block fast path and sends coupled terms to a SciPy SuperLU backend
with a cached theta-independent pattern, symmetric controls, minimum-degree
ordering, determinant/solve checks, allocation/fill limits, and typed failures.
Pastes and Penicillin pass twelve fixed-theta ML/REML cases against pinned lme4
and independent dense PLS, four final fits, and partial-new-group predictions.
InstEval passes ML/REML at 73,421 rows and 4,100 random coefficients without the
2.41 GB dense Z; the local isolated run peaks at 357 MB. SciPy repeats internal
symbolic ordering because it exposes no supported split symbolic/numeric
SuperLU API; ADR 0006 records this explicit backend limitation. Hosted CI run
35025175111 passes Linux/Python 3.11 and 3.14, macOS/Python 3.12,
Windows/Python 3.12, clean wheels, and both resource manifests.
Multi-factor slopes and categorical random terms belong to F02; nested/crossed
prediction bundles remain fail-closed until full artifact recovery.

Status 2026-09-15, second stable-core slice: F02 is complete for the declared
formula subset. The owned adapter reproduces lme4's non-LAPACK QR column-moving
policy at tolerance `1e-7`, fits only retained columns, and preserves full names,
pivot/drop identity, and a normalized coefficient null-space basis. Linear
functions and new-data rows return an explicit non-estimable status rather than
inventing dropped estimates. Ordinary treatment/sum categorical random slopes
run through the compact block and coupled sparse backends, including a slope
crossed with another grouping factor; fixed and random contrast controls remain
separate to match lmer semantics. Six single-group and two crossed fixed-theta
cases agree with lme4 and dense PLS; eight final ML/REML fits and their known/new-
group predictions pass the reviewed F02 limits. Categorical double-bar and F02
bundle recovery remain fail-closed. ADR 0007 records the decisions.
Hosted CI run 35032002878 passes the quality and installed-wheel gates, the
Linux/Python 3.11 and 3.14, macOS/Python 3.12, and Windows/Python 3.12 matrix,
and the eight-case million-row resource benchmark.

Gate: every advertised stable structure has a real installable backend, oracle
evidence, resource behavior, and compatibility report. Numeric disagreement is
resolved at the earliest differing layer.

### Phase 3 — inference foundation and first stable fitter

Add unconditional/conditional simulation, reproducible bootstrap/refit ledger,
estimable linear contrasts, and explicit asymptotic summaries. Complete regular
calibration and failure-accounting gates, private artifact handling, runbooks,
supply-chain evidence, and reviewer sign-off for advertised methods.

Status 2026-09-15, first inference-foundation slice: I01 is complete for
deterministic Gaussian response simulation and retained-fixed-effect parametric
refitting from a live fitted result. Unconditional draws generate new random
effects and weighted residuals; conditional draws hold fitted random effects
fixed. PCG64DXSM streams are separated by root seed, replicate ID, and purpose,
and serial/parallel scheduling produces identical responses and statistics.
Every refit outcome remains visible, singular fits are valid completed draws,
and exact one-sided failure bounds accompany the observed rate. An optional
private ledger atomically stores canonical metadata and non-object response
arrays with fit, source, plan, RNG, request, integrity, and resource validation;
it requires the same live or reproducibly reconstructed fit and does not expand
prediction-bundle capabilities. Four stored-response ML/REML × conditional/
unconditional refits match pinned lme4. The locked assessment passes 10,000
draws per mode and 1,000 refits with zero failed refits, 93 valid singular fits,
a 0.002992 one-sided 95% failure-rate upper bound, and exact worker identity.
Percentile/basic intervals are descriptive; nominal coverage remains unclaimed
until a separately locked outer-calibration study passes. ADR 0008 records the
decision. Hosted CI run 35038175252 passes the locked statistical assessment,
quality and installed-wheel gates, Linux/Python 3.11 and 3.14, macOS/Python
3.12, Windows/Python 3.12, and the million-row resource benchmark.

Gate: §14 passes for the stable fitter scope. I02 Satterthwaite and I03
Kenward–Roger/profile inference have passed their separate evidence gates.

### Phase 4 — advanced lmer ecosystem fidelity

Deliver Satterthwaite one-DF then multi-DF tests, KR, likelihood profiles, and
supported model-comparison methods as separate increments. Each has a completed
statistical plan, exact companion reference, derivatives/intermediates,
boundary policy, calibration, memory ceiling, and documentation.

Gate: method-specific scorecards pass; output names never imply broader support
than the tested method/regime. Current and legacy oracle outputs stay separate.

Status 2026-09-16: I03 is complete for bounded Kenward–Roger inference on
regular unit-weight Gaussian fits and named ML likelihood profiles. KR exposes
the pbkrtest 0.5.5 adjusted covariance, component information/covariance,
derivative matrices, scaling, degrees of freedom, and scaled/unscaled F tests;
ML inputs receive a separate REML refit. The observation-space computation has
explicit count/byte ceilings and refuses weights, covariance boundaries, and
invalid information. Profiles use an ML baseline, lme4 2.0-6 `.sigNN` ordering,
SD/correlation or variance/covariance presentation, and nuisance reoptimization
for covariance, residual-scale, and retained fixed-effect targets. The pinned
Dyestuff/Sleepstudy corpus and locked 1,000-replicate calibration pass. KR
rejection is `0.04213` on 997 available fits; fixed-slope profile rejection is
`0.04700` with `0.953` coverage. Weighted/boundary KR and variance-component
profile coverage remain explicitly unclaimed. ADR 0010 records the decision.

### Phase 5 — postfit distribution

Prove small capability protocols with kamino and two external adapters. Implement
reference grids, contrasts, tidy/performance output, and a marginaleffects
adapter after its feasibility spike. CR0/CR1 precede independently validated CR2
and HTZ-style joint tests. Document adapter-specific limitations.

Gate: independent postfit installation, labeled permutation/estimability tests,
companion reference fixtures, method calibration, and its own release checklist.
Neither “generic” nor “equivalent” is an acceptance result.

Status 2026-09-16: A01 is complete as the in-repository capability spike. A
versioned immutable basis now binds full/retained labels, estimability,
covariance identity, and asymptotic/residual-t/Satterthwaite/KR inference.
Explicit adapters cover Kamino, statsmodels 0.14.6 OLS/WLS, and statsmodels
MixedLM; robust covariance cannot inherit model-based residual DF. Reference
grids implement the declared equal/proportional/outer/cells/flat/user weighting,
offset, `at`, `by`, estimability, and none/Holm/Bonferroni/Sidak contracts. A
pinned lme4 2.0-6/emmeans 2.0.2 fixture verifies coefficient-aligned linear
functions and results. The current marginaleffects Python API does not expose a
public native-result adapter contract, so direct Kamino integration is deferred
instead of simulated with a false statsmodels object. Independent distribution,
Tukey/multivariate-t, CR2/HTZ, and the separate release checklist remain open
Phase 5 gates.

Status 2026-09-16: A02 is complete for unit-weight Gaussian fits with every
random grouping factor nested in a declared independent cluster. CR0/CR1 use
the documented GLS score normalization; CR2 uses fitted marginal working
targets, marginal residuals, the clubSandwich inverse-variance adjustment,
contrast-specific Satterthwaite tests, and HTZ joint tests. ML/REML higher-level
clusters and correlated random slopes match pinned clubSandwich 0.7.0
covariance and intermediates. A locked 2,000-replicate assessment passes for
the declared 12-cluster nested regime. Prior weights, non-nested/crossed
partitions, custom targets, and CR3 remain unsupported.

### First ten reviewable changes

1. Template-derived library scaffold, completed brief, tailoring ADR, ownership,
   real lint/type/build gate, and initial requirement index.
2. Versioned compatibility/schema/tolerance manifest and fixture/data license inventory.
3. Reproducible primary/legacy R environments and input/output fixture generator.
4. Independent dense GLS oracle with hand-derived boundary/weight/offset cases.
5. Chosen formula adapter, common frame, labels/contrasts, and adversarial X/Z corpus.
6. Theta/Lambda mapping, fixed-theta PLS, and three-oracle comparisons.
7. Structural classifier and verified single-group block backend.
8. NLopt reference controls, accepted-state snapshot, convergence/singularity diagnostics.
9. Fit/predict/artifact vertical slice, transform recovery, row/new-group behavior.
10. Installed-wheel CI, benchmark manifest, capability rejection tests, and alpha report.

## 14. Acceptance scorecard and release procedure

Every requirement has a code/test/evidence link in PRODUCTION_READINESS.md.
“Pending” means the implementation has not been demonstrated by this design
review. No pre-scaffold checkbox is marked passed.

| ID | Requirement and evidence | Owner role | Release gate | Current |
|---|---|---|---|---|
| F01 | Exact accepted frame/X/Z/parameter maps, row masks and contrasts | Numerical implementer | Alpha | Public compact adapter passes numeric/categorical treatment/sum fixed effects, pairwise interaction, shared subset/NA rows, weights/offsets, Dyestuff/Dyestuff2, and correlated/independent numeric-slope Sleepstudy; unsupported stable-scope structures fail closed |
| N01 | Weighted ML/REML, fixed-theta dense/R/PLS agreement | Numerical implementer + statistical reviewer | Alpha | Phase 0 fixed-theta corpus passes |
| N02 | Bounds, exact singular fits, final-state and optimizer diagnostics | Numerical implementer | Alpha | Dyestuff2 plus correlated and independent synthetic slope boundaries and scalar/vector public diagnostics pass; wider covariance scope pending |
| P01 | Conditional/population prediction and safe artifact round trip | Numerical implementer | Alpha | Dyestuff/Dyestuff2, correlated/independent Sleepstudy, and categorical weighted/offset prediction pass; schema 1.2 round-trips the complete Phase 1 encoder state |
| E01 | Template baseline, locks, strict checks, clean wheel/sdist | Release maintainer | Alpha | Local and hosted matrix pass at `d7df8cc` |
| N03 | Nested/crossed solve, no missing coupling, resources and native wheels | Numerical implementer + release maintainer | Stable | Complete: Pastes/Penicillin oracle rows, InstEval ML/REML resource gate, clean native dependency wheels, and hosted platform/resource run 35025175111 pass |
| F02 | Rank dropping, estimability, categorical and new-data corpus | Statistical reviewer | Stable | Complete: four QR/drop contracts, coefficient null-space checks, treatment/sum and mixed-contrast categorical terms, six single-group plus two crossed final fits, and explicit new-data estimability pass pinned lme4/dense evidence locally and in hosted run 35032002878 |
| I01 | Complete bootstrap ledger, independent streams, calibration/failure bounds | Statistical reviewer | Stable | Complete: conditional/unconditional PCG64DXSM simulation, exact response refit, bounded worker-invariant bootstrap, atomic private ledger, four pinned-lme4 stored-response refits, and locked 10,000-draw/1,000-refit moment, bias, and failure assessment pass locally and in hosted run 35038175252; nominal interval coverage remains a separate gate |
| I02 | Satterthwaite full variance-parameter derivatives and calibrated tests | Statistical reviewer | Feature release | Complete: full `(theta, sigma)` unprofiled Hessian, `2 H_D^-1`, beta-covariance Jacobians, one-/multi-DF tests, pinned lmerTest 3.1-3 ML/REML intermediates, explicit boundary/curvature refusal, general sparse coverage, and a passing locked 2,000-replicate calibration |
| I03 | KR adjustment/scaling/DF and profile nuisance optimization | Statistical reviewer | Feature release | Complete: pinned pbkrtest 0.5.5 adjusted covariance/intermediates/scaled tests, ML-to-REML provenance, bounded dense failure policy, lme4 2.0-6 named ML nuisance profiles, explicit endpoints, and a passing locked 1,000-replicate KR/profile calibration |
| A01 | Postfit labels, covariance/DF consistency, external adapters | Adapter owner | Postfit release | Complete: versioned Kamino/statsmodels capability contracts, pinned emmeans/lme4 reference-grid evidence, estimability/label permutation checks, and explicit covariance/DF pairing; independent distribution remains a Phase 5 release gate |
| A02 | CR2 working target, independent clusters, valid joint inference | Statistical reviewer | CR2 release | Complete: fitted-target CR0/CR1/CR2, marginal scores, nesting and prior-weight refusal, Satterthwaite/HTZ tests, pinned clubSandwich intermediates, and a passing locked 2,000-replicate calibration |
| E02 | Threat model, privacy, license review, SBOM/provenance, support/runbooks | Release maintainer | Stable | Pending |
| E03 | Reproducible time/memory reports and owned remaining risks | Product owner | Stable | Pending |

Before stable publication:

1. Freeze the declared scope, package/schema/formula/oracle versions, dependency
   matrix, tolerance manifest, and method-specific statistical plans.
2. Run all applicable gates against the exact candidate commit and built artifacts.
   A skipped reference/calibration/platform check is unverified, not passing.
3. Publish an evidence index with numerical discrepancies, calibrated regimes,
   benchmark environment, failures, unsupported features, and owners/dates.
4. Complete the template release checklist and accountable statistical/release
   review. Resolve Critical/High correctness findings for shipped features.
5. Tag the reviewed commit; CI verifies metadata/tag agreement, builds artifacts,
   inspects bundled/linked dependencies, and creates checksums, provenance, and
   an SBOM. Use short-lived trusted publishing identity.
6. Verify the installed release. Preserve old corpus/artifacts for reproduction.
   Document upgrade/deprecation and result-schema migration rules.
7. For a numerical incident, identify affected versions/model classes, retain a
   minimal licensed reproducer, issue an advisory and yank/forward fix as needed.
   Reverting a dependency or code change requires rerunning the affected oracle
   and method gates; never rewrite historical evidence.

## 15. Review disposition and remaining decisions

### 15.1 Corrections integrated by this revision

| Prior problem | Resolution in the normative plan |
|---|---|
| Original algorithm contradicted later review addendum | Replaced with one consistent specification |
| All nested factors treated as tiny independent blocks | Actual graph routing and coupled/general solve (§5) |
| Weights/offsets optional and likelihood constants missing | Foundational weighted PLS and explicit ML/REML (§4) |
| lme4 behavior assumed timeless | Primary 2.0-6 and legacy 1.1-37 profiles (§2) |
| Incorrect lmer optimizer fallback claim | Pinned NLopt controls and bounded explicit comparisons (§6) |
| Spectral singularity rule proposed as identical to lme4 | Separate compatible Cholesky test and spectral evidence (§6.3) |
| Theta Hessian described as sufficient for Satterthwaite | Full variance parameters, residual scale, factor-two covariance (§8.2) |
| Profiles described as raw-theta scans | Named target constraints, ML baseline, nuisance reoptimization (§8.4) |
| Universal boundary LRT and mixed-model CR2 formulas | Hypothesis-specific calibration and reference restrictions (§8.6, §9.3) |
| Overbroad postfit protocol | Explicit estimability, identity, covariance and DF capabilities (§9) |
| Invalid test invariants and one tolerance for all cases | Weight/basis invariants, three oracles, conditioning-aware gates (§10) |
| Generic local production checklist | Immutable GitHub template, explicit tailoring, traceable phases (§11–14) |

### 15.2 Decisions to close with implementation evidence

| Decision | Resolution path | Deadline |
|---|---|---|
| Formula backend and accepted transform set | R matrix/NA/new-data spike; choose one | Phase 0 |
| Sparse engine and supported binaries | Build/benchmark ADR and wheel evidence | Before backend distribution |
| Project distribution license | MIT selected by owner and recorded in package/repository | Closed 2026-09-14 |
| Full Python/OS matrix and exact dependency versions | Clean installs and oracle environment solve | Phase 0 |
| Statistical reviewer and release maintainer | Product owner assigns accountable roles | Before beta |
| Final numerical and calibration thresholds | Freeze measured manifest before assessment/tuning | Before method validation |
| Package availability | Project/package name chosen: kamino; check registry availability before publication | Before alpha publication |
| Postfit API and repository | Two-external-adapter capability evidence | Phase 5 |

These decisions do not block completing this design. They do block claiming
implementation, numerical parity, or release readiness without their evidence.

Review provenance: the original 921-line document had SHA-256
112e0516ad73b7bc1dd9d2426018a2199c7d78c521771db3551bcc8a1caa6531.
It is preserved in the project review archive as
[original lmerx design](docs/archive/lmerx-design.before-production-revision-2026-09-14.md).
The project was renamed from lmerx to kamino on 2026-09-14; the archived original
retains its historical name and contents.
The review performed document/source inspection and mathematical consistency
checks. Document validation verified the backup hash, local link target, URL
syntax, section references, and balanced math/code fences. Small numerical
spot checks compared the specified PLS equations with independently assembled
dense GLS for ten ML/REML interior/boundary cases and two coupled nested cases.
The checks also verified the weight-rescaling and fixed-basis REML constants at
a normalized tolerance of 1e-10. These spot checks are not an R parity corpus.
R fitting, package builds, runtime parity, and simulation results remain
implementation work; no Rscript executable was available in the review shell.
