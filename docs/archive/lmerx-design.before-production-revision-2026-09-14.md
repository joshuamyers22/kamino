# An independent `lme4` for Python — design sketch

**Status:** draft / adversarially reviewed / **not approved to scaffold as written**
**Scope:** a native Python linear mixed-model fitter that reproduces `lme4::lmer` numerically, plus a companion suite covering `emmeans`, `marginaleffects`, `broom.mixed`, `performance`, and `clubSandwich`.

> **Review notice (2026-09-14):** Sections 13–21 are a production and
> adversarial review. Their corrections and release gates supersede conflicting
> statements in the original sketch. In particular, do not implement the v0.1
> nested-model shortcut, optimizer fallback, inference, or generic protocol as
> currently described without first closing the Critical/High findings.

---

## 1. Goals and non-goals

### Goals

- Fit `y ~ fixed + (slopes | group)` models by REML and ML, matching `lme4` to ~1e-6 on the deviance and ~1e-5 on variance components across a reference corpus.
- Support crossed and nested grouping factors, correlated and uncorrelated random slopes.
- No R dependency. No `rpy2`. Installable as a pure wheel wherever possible.
- Inference that is actually usable: Satterthwaite and Kenward-Roger denominator DF, profile-likelihood intervals, parametric bootstrap.
- A post-estimation layer that is **generic over model objects**, not welded to this fitter.

### Non-goals (at least for v1)

- GLMMs. PIRLS + Laplace + adaptive Gauss-Hermite is a separate and substantially larger project. Design the core so it can be added, but don't build it.
- Nonlinear mixed models (`nlmer`).
- `nlme`-style residual correlation structures (`corAR1`, `varPower`). Flag as a v2 fork — this is where `nlme` and `glmmTMB` beat `lme4` and where panel data actually lives.
- Bayesian estimation. `bambi`/`pymc` already fill that slot well.

---

## 2. Package split

Two distributions, not six.

| Distribution | Contents | Depends on core? |
|---|---|---|
| `lmerx` | formula parser, PLS solver, optimizer, fit object, DF methods, bootstrap | — |
| `postfit` | `emmeans`, `marginaleffects`, `broom.mixed`, `performance`, `clubSandwich` equivalents | **No** |

### Why `postfit` must not depend on `lmerx`

In R, `emmeans` and `clubSandwich` work across dozens of model classes because S3 dispatch lets third parties register methods. The Python equivalent is a `Protocol` plus an adapter registry. If we build these five as helpers on our own fit object, we get five packages that only work with our fitter — which is strictly worse than what already exists, because nobody will adopt a fitter to get marginal means.

Built against a protocol instead, `postfit` immediately works with `statsmodels`, `pyfixest`, `linearmodels`, `glum`, and `lmerx`. That is a package people install on its own merits, and it becomes the on-ramp for the fitter.

**Recommendation:** build `postfit` first, or at least concurrently. It is lower risk, independently valuable, and forces the core's public API to be honest.

```
lmerx/
  src/lmerx/
    formula/        # bar-syntax parser -> term AST
    core/           # Z, Lambda_theta, PLS, deviance, optimizer
    inference/      # vcov, Satterthwaite, KR, profile, bootMer
    results/        # LmerFit object, repr, summary
    datasets/       # sleepstudy, Penicillin, Pastes, Dyestuff, InstEval...
  tests/
    fixtures/       # frozen R golden files
postfit/
  src/postfit/
    protocol.py     # the model interface
    adapters/       # lmerx, statsmodels, pyfixest, linearmodels
    emmeans/
    slopes/         # marginaleffects
    tidy/           # broom.mixed
    perf/           # performance
    cluster/        # clubSandwich
```

---

## 3. The core algorithm

Follow Bates, Mächler, Bolker & Walker (2015, *JSS*) and the `lme4` "Computational Methods" vignette directly. Do not invent an alternative parameterization; matching `lme4` numerically is the entire point, and the profiled-deviance formulation is what makes it tractable.

### Model

$$
y = X\beta + Zb + \varepsilon, \qquad b \sim N(0, \Sigma_\theta), \qquad \varepsilon \sim N(0, \sigma^2 I_n)
$$

Reparameterize to spherical random effects $u$, with $b = \Lambda_\theta u$ and $\Sigma_\theta = \sigma^2 \Lambda_\theta \Lambda_\theta^\top$, so $u \sim N(0, \sigma^2 I_q)$. The **relative covariance factor** $\Lambda_\theta$ is block-diagonal: one block per random-effects term, each block itself block-diagonal over the levels of that term's grouping factor, repeating a single lower-triangular template $T_\theta$ of size $p_i \times p_i$ where $p_i$ is the number of columns in that term's model matrix.

$\theta$ is the vector of free lower-triangular entries across templates. Its length is $\sum_i p_i(p_i+1)/2$ — for `(1 + x | g)` that's 3, regardless of how many levels `g` has. **This is the whole trick:** the optimization is over a tiny parameter vector even when $q$ is in the tens of thousands.

### Penalized least squares

For fixed $\theta$, minimize

$$
r^2_\theta = \min_{u,\beta} \; \lVert y - X\beta - Z\Lambda_\theta u \rVert^2 + \lVert u \rVert^2
$$

via the sparse Cholesky factorization

$$
L_\theta L_\theta^\top = P\left(\Lambda_\theta^\top Z^\top Z \Lambda_\theta + I_q\right)P^\top
$$

with $P$ a fill-reducing permutation computed **once** from the sparsity pattern of $Z$, then reused for every $\theta$.

### Profiled criteria

$\beta$ and $\sigma^2$ drop out analytically:

$$
d(\theta \mid y) = \log \det(L_\theta)^2 + n\left[1 + \log\!\left(\frac{2\pi r^2_\theta}{n}\right)\right] \quad \text{(ML)}
$$

$$
d_R(\theta \mid y) = \log \det(L_\theta)^2 + \log \det(R_X)^2 + (n-p)\left[1 + \log\!\left(\frac{2\pi r^2_\theta}{n-p}\right)\right] \quad \text{(REML)}
$$

where $R_X$ is the fixed-effects factor from the augmented system. Optimize over $\theta$ alone, subject to box constraints (diagonal template entries $\geq 0$, off-diagonals free).

### Optimizer

`lme4` uses derivative-free BOBYQA from NLopt, with Nelder-Mead as a fallback and a second-stage restart. Two routes:

1. **Match `lme4`:** `nlopt` Python bindings (LGPL) or `Py-BOBYQA` (GPLv3 — licensing problem). Reproduces `lme4` convergence behaviour including its quirks.
2. **Improve on it:** make the deviance differentiable end-to-end and use L-BFGS-B. Analytic gradients of the profiled deviance exist and are in the vignette; alternatively autodiff the whole thing. This converges in far fewer function evaluations and hands you the $\theta$ Hessian for free, which you need anyway for Satterthwaite.

**Recommendation:** ship route 1 as the reference path for cross-validation against R, and route 2 as the default once golden tests pass. Keep both selectable — when a user reports a discrepancy with `lme4`, being able to flip to BOBYQA isolates optimizer differences from algebra differences immediately.

---

## 4. The sparse Cholesky problem

This is the single hard dependency decision and it deserves its own section.

The PLS step needs a Cholesky factorization that is (a) sparse, (b) re-factorizable with a **cached symbolic analysis** across hundreds of $\theta$ evaluations, and (c) exposes $\log\det$. In C that's `cholmod_analyze` once, `cholmod_factorize_p` per iteration.

### Options

| Option | Pros | Cons |
|---|---|---|
| `scikit-sparse` (CHOLMOD) | Exactly the right API; `cholesky_inplace` reuses symbolic factor; `logdet()` built in | CHOLMOD supernodal is GPL → forces GPL on the distribution; arm64 macOS wheels historically absent, needs `brew install suite-sparse` |
| `scipy.sparse.linalg.splu` | Ships with scipy | No symmetric exploitation, no clean symbolic reuse, wrong tool |
| Hand-rolled blockwise dense Cholesky | Pure NumPy, no deps, trivially differentiable, fast for the common case | Only exact when $\Lambda^\top Z^\top Z\Lambda + I$ is block-diagonal |
| `sksparse` optional, block-dense default | Best of both | Two code paths to test |

### The structural observation that makes this tractable

For a **single grouping factor** — and, with the right term ordering, for **nested** factors — $\Lambda_\theta^\top Z^\top Z \Lambda_\theta + I$ is exactly block-diagonal with one $p_i \times p_i$ block per level. No sparse library needed: batch the blocks into a 3-D array and call `np.linalg.cholesky` once over the batch. This is vectorized, allocation-free after warm-up, differentiable under JAX, and covers the overwhelming majority of real models (`sleepstudy`, every repeated-measures design, most panel structures).

**Crossed** factors (`(1|student) + (1|instructor)`) are where you genuinely need a fill-reducing sparse solve.

### Staged plan

- **v0.1** — dense block path only. Raise a clear `NotImplementedError` on crossed terms naming the limitation. Pure NumPy/SciPy, permissive license, works everywhere including your arm64 Macs.
- **v0.2** — CHOLMOD path behind an extra: `pip install lmerx[crossed]`. Import guarded, with an error message that tells the user the brew/apt incantation.
- **v0.3** — evaluate replacing CHOLMOD with a vendored LGPL-only supernodal path or a Rust `sprs`/`faer` extension to escape the GPL constraint, if adoption justifies it.

Licensing note worth deciding up front: if `lmerx` links CHOLMOD's supernodal module, the distribution is effectively GPL. Since `binspect` is going out as a permissive public package, a GPL sibling is an inconsistency you'd want to choose deliberately rather than stumble into. Making it an optional extra keeps the base package clean.

---

## 5. Formula layer

Parse R-style mixed-model formulas into a term AST. This is self-contained and testable in isolation, so build it first.

```
y ~ x + (1 | g)                    # random intercept
y ~ x + (x | g)                    # correlated intercept + slope
y ~ x + (x || g)                   # uncorrelated (expands to (1|g) + (0+x|g))
y ~ x + (1 | g/h)                  # nested: (1|g) + (1|g:h)
y ~ x + (1 | g) + (1 | h)          # crossed
y ~ x + (0 + x | g)                # slope only, no intercept
y ~ poly(x, 2) + (1 | g)           # transformations in fixed part
```

Fixed-effects side: delegate to **`formulaic`**. It's the modern successor to `patsy`, actively maintained, handles contrasts/transformations/stateful terms correctly, and is what `pyfixest` and `marginaleffects` already build on. Do not write a design-matrix builder.

Bar syntax is not something `formulaic` handles, so: split the RHS on top-level `|` groups yourself, hand each side to `formulaic` separately, and assemble $Z$ as the row-wise Khatri-Rao product of each term's model matrix with the indicator matrix of its grouping factor.

Term ordering matters for sparsity — `lme4` sorts random-effects terms by decreasing number of levels. Match that so factorizations are comparable.

---

## 6. Inference layer

The fit is the easy half. Inference is where `lme4` alone is insufficient and users end up installing `lmerTest` and `pbkrtest`. Build them in from the start.

| Method | Notes | Priority |
|---|---|---|
| Wald z | Free from $\widehat{\mathrm{Var}}(\hat\beta) = \sigma^2 (X^\top V^{-1} X)^{-1}$ | v0.1 |
| **Satterthwaite DF** | `lmerTest`'s default. Needs $\partial \mathrm{Var}(\hat\beta)/\partial\theta_j$ and the $\theta$ Hessian; $\nu = 2(\ell^\top V \ell)^2 / \mathrm{Var}(\ell^\top V \ell)$ by delta method. If the deviance is autodiffable, both derivatives come for free — a strong argument for route 2 in §3 | v0.1 |
| **Kenward-Roger** | Adjusted covariance $\Phi_A$ plus DF; needs second derivatives of $V$. `pbkrtest` is the reference. Better small-sample behaviour, more expensive | v0.2 |
| Profile likelihood CIs | $\zeta$ profiles over each $\theta_j$ and $\sigma$. The honest interval for variance components, which are bounded and skewed | v0.2 |
| Parametric bootstrap | `bootMer` equivalent. Simplest to implement correctly, most robust, embarrassingly parallel. Ship this early — it de-risks everything else | v0.1 |
| LRT with boundary correction | Testing a variance component at zero puts the null on the boundary; the null distribution is a $\bar\chi^2$ mixture, not $\chi^2_1$. `anova.merMod` gets this wrong by default and users are routinely misled. Get it right and say so | v0.2 |

Singular fits: detect when any $\theta$ template hits the boundary (variance → 0 or correlation → ±1) and surface it as a structured attribute on the fit object, not a warning string. `performance::check_singularity` then just reads it.

---

## 7. The model protocol

The contract everything in `postfit` is written against.

```python
from typing import Protocol, runtime_checkable
import numpy as np
import pandas as pd

@runtime_checkable
class FittedModel(Protocol):
    # --- identity -------------------------------------------------
    def coef(self) -> pd.Series: ...
    def vcov(self) -> np.ndarray: ...
    def nobs(self) -> int: ...

    # --- design ---------------------------------------------------
    def model_frame(self) -> pd.DataFrame: ...
    def model_matrix(self, data: pd.DataFrame | None = None) -> np.ndarray: ...
    def terms(self) -> "TermSpec": ...          # names, types, factor levels, contrasts

    # --- prediction -----------------------------------------------
    def predict(self, data: pd.DataFrame, *, re_form=None) -> np.ndarray: ...

    # --- inference ------------------------------------------------
    def df_residual(self, contrast: np.ndarray | None = None,
                    method: str = "satterthwaite") -> float: ...

    # --- optional, feature-detected via hasattr -------------------
    def bread(self) -> np.ndarray: ...          # clubSandwich
    def estfun(self) -> np.ndarray: ...         # clubSandwich
    def clusters(self) -> dict[str, np.ndarray]: ...
    def varcomp(self) -> pd.DataFrame: ...      # performance: ICC, R2
```

Adapters live in `postfit.adapters` and are registered by entry point, so a third-party fitter can add support without touching `postfit`. Required-vs-optional is enforced by capability checks with actionable errors: *"CR2 correction requires `bread()`; the `statsmodels.MixedLM` adapter does not implement it."*

---

## 8. Companion modules

### 8.1 `postfit.emmeans` — estimated marginal means

The conceptual core is the **reference grid**: a balanced factorial over the levels of every categorical predictor, with numeric predictors held at their means. Marginal means average predictions over the grid with specified weights — deliberately *not* over the empirical distribution of the data. (This is the distinction from `marginaleffects`, and it is worth documenting loudly because users conflate them constantly.)

- Reference grid construction, with `at=`, `by=`, and `cov_reduce=` controls.
- Averaging weights: `equal`, `proportional`, `cells`, `flat`.
- Contrast families: `pairwise`, `revpairwise`, `trt_vs_ctrl`, `poly`, `consec`, plus user-supplied contrast matrices.
- Multiplicity adjustment: Tukey, Sidak, Bonferroni, Holm, and the multivariate-*t* adjustment (needs `scipy.stats.multivariate_t` CDF — the slow one, so make it opt-in).
- Every estimate carries a DF from `df_residual(contrast=...)`, which is exactly why the protocol takes a contrast argument.

### 8.2 `postfit.slopes` — marginal effects

`marginaleffects`' three verbs, averaged over the observed data rather than a grid:

- `predictions()` — fitted values with SEs, at observed or counterfactual rows.
- `comparisons()` — contrasts of predictions under counterfactual manipulation (g-computation). Handles the `+1`, `factor level`, `IQR`, `SD` comparison types.
- `slopes()` — $\partial\hat y/\partial x$ by central difference, delta-method SEs from $\mathrm{Var}(\hat\beta)$ via the Jacobian.

AME / MEM / MER as aggregation modes over the same machinery. For mixed models, add the distinction that has no analogue in fixed-effects land: marginal effects **conditional on** the fitted random effects vs **population-level** (`re_form=None` vs `~0`). These differ and users need to choose explicitly.

A Python `marginaleffects` port already exists (Arel-Bundock). Two options: depend on it and contribute a mixed-model adapter upstream, or reimplement. **Depending is almost certainly right** — reimplementing a maintained package to get one adapter is wasted effort and splits the ecosystem. Revisit only if its model-interface assumptions turn out to be incompatible.

### 8.3 `postfit.tidy` — broom.mixed

Three functions, thin:

- `tidy(model, effects="fixed"|"ran_pars"|"ran_vals"|"ran_coefs", conf_int=False)` → DataFrame of `term, group, estimate, std_error, statistic, df, p_value, conf_low, conf_high`.
- `glance(model)` → one row: `nobs, sigma, loglik, aic, bic, deviance, df_residual, n_groups, converged, singular`.
- `augment(model, data=None)` → original data plus `fitted, resid, resid_pearson, hat, cooksd`, conditional and population-level fits separately.

Return **pandas** by default with a `polars` flag, or return whatever the caller passed in via the dataframe-interchange protocol. Given a Parquet-heavy data layer, polars support should not be an afterthought.

### 8.4 `postfit.perf` — performance

- **ICC**, adjusted and unadjusted/conditional, per grouping factor and total.
- **$R^2$** via Nakagawa, Johnson & Schielzeth: marginal (fixed only) and conditional (fixed + random). Requires the variance decomposition, so it needs `varcomp()`.
- `check_singularity()` — reads the boundary flag from §6.
- `check_collinearity()` — VIF on the fixed-effects design, with the correction for models containing interactions.
- `check_heteroscedasticity()`, `check_normality()` on both residual levels.
- `check_convergence()` — gradient norm and scaled Hessian eigenvalues at the optimum, the same diagnostic `lme4` warns on.
- `compare_performance()` — AIC/BIC/AICc table with the boundary-corrected LRT from §6.

### 8.5 `postfit.cluster` — clubSandwich

Cluster-robust variance with small-sample corrections. Genuinely the most independently valuable piece here, and the one most likely to be used by people who never touch the fitter.

$$
V^{CR} = \mathrm{bread} \cdot \left(\sum_g X_g^\top A_g \hat\varepsilon_g \hat\varepsilon_g^\top A_g^\top X_g\right) \cdot \mathrm{bread}
$$

- **CR0** — no correction, $A_g = I$.
- **CR1 / CR1S** — scalar DF multipliers.
- **CR2** — the Bell-McCaffrey bias-reduced adjustment, $A_g = (I - H_{gg})^{-1/2}$ computed under a working model. This is the one that matters with few clusters and the reason `clubSandwich` exists. For mixed models the working model is the fitted $V$, not identity — which is precisely why the protocol needs `bread()` rather than assuming OLS.
- **CR3** — jackknife-style.
- **Satterthwaite DF for CR2 Wald tests**, and the AHR (approximate Hotelling's $T^2$) test for multi-parameter hypotheses.

Correct behaviour with **few clusters** (< 40) is the acceptance criterion; that's the regime where naive CR0 over-rejects badly and where every published simulation lives.

---

## 9. Testing

### Golden files from R

One-time (well, versioned) R script generates fixtures; CI never runs R.

```r
# tools/generate_fixtures.R  — run manually, commit outputs
for (spec in model_corpus) {
  fit <- lmer(spec$formula, spec$data, REML = spec$reml)
  jsonlite::write_json(list(
    deviance   = deviance(fit, REML = spec$reml),
    theta      = getME(fit, "theta"),
    beta       = fixef(fit),
    sigma      = sigma(fit),
    vcov_beta  = as.matrix(vcov(fit)),
    ranef      = ranef(fit),
    logLik     = as.numeric(logLik(fit)),
    satt_df    = coef(summary(lmerTest::as_lmerModLmerTest(fit)))[, "df"],
    emm        = as.data.frame(emmeans::emmeans(fit, spec$emm_spec)),
    crve       = clubSandwich::vcovCR(fit, type = "CR2")
  ), path = spec$fixture, digits = 16)
}
```

### Corpus

| Dataset | Structure | Tests |
|---|---|---|
| `Dyestuff` | single factor, 6 levels | simplest possible case; `Dyestuff2` gives a **singular** fit — keep it, boundary handling is a first-class behaviour |
| `sleepstudy` | 18 subjects, correlated slope | the canonical `(1+Days|Subject)` |
| `Penicillin` | crossed, 24 × 6 | smallest crossed case |
| `Pastes` | nested, `(1|batch/cask)` | nesting expansion |
| `Oats`, `Machines` | classic split-plot | agreement with textbook ANOVA decompositions |
| `InstEval` | crossed, 73k obs, 2972 × 1128 | scale + performance benchmark |
| synthetic | known $\theta$, large $n$ | parameter recovery |

### Property tests (Hypothesis)

- As $\theta \to 0$, REML deviance → OLS deviance and $\hat\beta$ → OLS $\hat\beta$.
- Invariance of the fit to random-effects term ordering and to level relabeling.
- Permuting rows of the data leaves all estimates unchanged.
- `(1|g) + (0+x|g)` and `(x||g)` produce identical fits.
- Doubling all weights scales $\sigma^2$ predictably.
- Profile-likelihood CIs contain the point estimate and bracket the Wald interval in the expected direction for variance components.

Tolerances: `1e-8` on deviance, `1e-6` on $\theta$ and $\beta$, `1e-4` on Satterthwaite DF (optimizer path dependence bites here). Publish the tolerance table in the README — it is the package's actual claim.

---

## 10. Performance targets

Reference: `lme4` on the same machine, single-threaded.

| Model | Target |
|---|---|
| `sleepstudy` | < 50 ms |
| `Penicillin` | < 100 ms |
| `InstEval` | within 2× of `lme4` |
| 1M rows, `(1\|g)`, 10k groups | < 10 s |

The dense-block path should comfortably *beat* `lme4` on nested models, since `lme4` pays sparse-matrix overhead where a batched dense solve suffices. Say so in benchmarks — it's the most honest selling point available, and it's an argument nobody currently makes.

Benchmark harness in `tools/bench/`, results committed per release so regressions are visible in diffs.

---

## 11. Milestones

**M0 — formula layer.** Bar parser → term AST → $Z$ construction. Snapshot-tested against `getME(fit, "Zt")` sparsity patterns from R. No fitting at all. Fully testable in isolation, which makes it the right thing to build first.

**M1 — `postfit` skeleton + `cluster`.** Protocol, `statsmodels` and `pyfixest` adapters, CR0/CR1/CR2 with Satterthwaite DF. Ships independently of any of the above and is immediately useful. Validates the protocol against real models before the core exists to bias it.

**M2 — core fitter, dense-block path.** ML and REML, nested/single-grouping only, BOBYQA. Golden tests green on `Dyestuff`, `sleepstudy`, `Pastes`, `Oats`.

**M3 — inference.** Wald, Satterthwaite, parametric bootstrap. `lmerx` adapter for `postfit`. First release worth announcing.

**M4 — crossed effects.** CHOLMOD path behind an extra. `Penicillin`, `InstEval` green.

**M5 — `postfit` completion.** `emmeans`, `tidy`, `perf`. `slopes` via upstream `marginaleffects` adapter.

**M6 — KR, profile CIs, boundary-corrected LRT.** Feature parity with `lmerTest` + `pbkrtest`.

---

## 12. Open questions

1. **Autodiff or not.** A JAX-backed deviance gives gradients, the $\theta$ Hessian, and Satterthwaite derivatives essentially free, and makes the dense-block path trivially fast. Cost: a heavy optional dependency, and no sparse Cholesky in JAX — so the crossed path would stay NumPy-only and the two paths would diverge structurally. Alternative: derive analytic gradients from the vignette and stay in NumPy. Leaning toward analytic, with JAX as an experimental backend.

2. **Naming.** `lmerx` is a placeholder that leans on `lme4`'s recognition. Check PyPI and consider whether leaning on the R name helps discovery or invites confusion about compatibility guarantees.

3. **Weights and offsets.** `lme4` supports prior weights and offsets; they're cheap to add at the PLS stage but easy to get subtly wrong. Include in M2 or defer?

4. **Whether `postfit` should just be upstream contributions.** `marginaleffects` for Python exists; `pyfixest` has a growing post-estimation layer. The counterargument to building a new suite is that four of the five modules might be better as PRs. The one that clearly is *not* is `clubSandwich` — nothing in Python covers CR2/Bell-McCaffrey properly. Worth deciding before M1 rather than after.

5. **GLMM boundary.** Committing to a `lmerx.glmer` later constrains the core's internals now (PIRLS needs the same factorization machinery with a weights update between iterations). Cheap to accommodate structurally; expensive to retrofit. Decide the internal API shape at M2 even if the feature waits.

---

## 13. Adversarial and production review — 2026-09-14

### 13.1 Review metadata

- **Artifact reviewed:** `Downloads/lmerx-design.md` (388 lines before this addendum)
- **Review type:** design-only; no repository, implementation, branch, or commit exists
- **Production criteria:** [`plan_data/PRODUCTION_REPOSITORY_STANDARD.md`](plan_data/PRODUCTION_REPOSITORY_STANDARD.md)
- **Review format:** [`plan_data/ADVERSARIAL_CODE_ARCHITECTURE_REVIEW_TEMPLATE.md`](plan_data/ADVERSARIAL_CODE_ARCHITECTURE_REVIEW_TEMPLATE.md)
- **Production-plan structure:** [`PRODUCTION_PLAN.md`](PRODUCTION_PLAN.md), used as a structural example only; its domain-specific requirements are not inherited
- **Evidence date:** 2026-09-14
- **In scope:** mathematical design, formula semantics, numerical backends, inference, public protocol, test strategy, packaging, release, reproducibility, security, and operations appropriate to a library
- **Out of scope:** source-code quality and runtime verification, because there is no source repository to inspect

### 13.2 Executive verdict

- **Overall grade:** C (promising research design, not yet an executable production plan)
- **Release recommendation:** **block implementation of the advertised scope**; approve only a time-boxed feasibility phase
- **Highest risk:** the proposed dense-block solver is not valid for the full nested model class it claims to cover
- **Strongest property:** explicit numerical parity goals backed by R golden fixtures and intermediate-model test ideas
- **Recommended first improvement:** build a small algebra oracle and a model-structure classifier before choosing a sparse backend or public API

The project is viable if its first release is narrowed. It is not credible to
ship a new fitter, two difficult small-sample inference methods, five
post-estimation families, multiple third-party adapters, and crossed-effects
sparse infrastructure on one early roadmap. “Matches `lme4`” must become a
versioned compatibility contract, not a general aspiration.

### 13.3 Verification evidence

| Check | Result | Notes |
|---|---|---|
| Design file read | Pass | Entire pre-review document inspected |
| Production standard applied | Pass | Repository, CI, release, reproducibility, security, and completion gates added below |
| Mathematical claims | Desk review only | Compared with the `lme4` computational paper and current reference documentation |
| Formula ecosystem | Desk review only | Both `formulaic` and the existing mixed-formula package `formulae` identified for a build-vs-buy spike |
| Inference claims | Desk review only | Compared with `lmerTest`, `pbkrtest`, `emmeans`, and `clubSandwich` documentation |
| Dependency/licensing | Partial | Upstream license declarations checked; legal conclusions intentionally not made |
| Formatting/lint/type/tests/coverage/build | Not run | No implementation or repository exists; these are not passes |
| Performance | Not run | Existing millisecond targets have no hardware/software baseline |

## 14. Findings

Findings are ordered by severity and remediation value. “Location” names the
section in this design rather than a source-code line.

### [Critical] The nested-model dense-block claim is mathematically too broad

- **Location:** §4, “structural observation” and staged v0.1 plan
- **Principle:** correctness / architecture boundary
- **Evidence:** for `(1|g) + (1|g:h)`, columns belonging to `g` and `g:h` overlap on observations, so the corresponding blocks of $Z^T Z$ generally have nonzero cross-products. Reordering does not make those values disappear. A single grouping factor can be permuted into independent per-level blocks, but multiple nested terms require a tree/block-arrow solve, grouping by independent top-level clusters, or a general sparse factorization.
- **Failure mode:** the claimed batched $p_i\times p_i$ factorization produces the wrong objective or silently omits coupling; `Pastes` can fail before the advertised first fitter milestone.
- **Test protection:** the proposed final-fit golden files would detect some discrepancies late, but there is no fixed-$\theta$ matrix-level oracle to localize them.
- **Required change:** narrow the dense path initially to one random-effects grouping structure whose permuted normal matrix is proven block diagonal. Add a structural classifier that verifies the zero pattern, never one that infers support from formula labels such as “nested.” Prototype tree elimination separately.
- **Acceptance criteria:** for every accepted formula, compare the assembled penalized normal matrix, log determinant, $\hat\beta$, $\hat u$, and profiled criterion at at least five fixed $\theta$ vectors against both a dense marginal-covariance oracle and R fixtures. Unsupported structures fail before optimization with a typed capability error.
- **Estimated scope:** large

### [High] “Reproduces `lme4::lmer`” has an incomplete compatibility boundary

- **Location:** §§1, 3, 5, 9, and 12
- **Principle:** contract / naming / tests
- **Evidence:** current `lmer` behavior includes prior weights, offsets, missing-value handling, contrast control, rank-deficient fixed effects, factor-level ordering, subsets, and precise prediction rules. The sketch defers weights/offsets and does not specify the rest. `||` also only has the stated `lme4` behavior for numeric random-effect design columns. See [S2].
- **Failure mode:** users receive numerically different models from apparently identical formulas while the package still claims parity.
- **Required change:** publish a feature-by-feature compatibility matrix. For v0.1 either implement and test these semantics or say “compatible subset of `lmer`” and reject unsupported options. Pin each fixture corpus to an exact R, `lme4`, `Matrix`, contrast, and BLAS environment.
- **Acceptance criteria:** every public formula/argument is classified exact, intentionally different, experimental, or unsupported; unsupported behavior fails closed. Documentation never uses unqualified “reproduces `lme4`.”
- **Estimated scope:** large

### [High] The proposed generic `FittedModel` protocol cannot support its claims

- **Location:** §§2, 7, and 8
- **Principle:** interface segregation / dependency inversion
- **Evidence:** `emmeans` extensions need recovered fit data, a coefficient-aligned linear-function basis, covariance, a non-estimability basis, transformation metadata, and a contrast-specific DF function [S9]. CR2 for `lmerMod` also needs cluster-aligned estimating information, a working covariance target, and inverse-variance-weight semantics [S11]. `coef()`, `vcov()`, `model_matrix()`, `bread()`, and `estfun()` do not express these invariants.
- **Failure mode:** adapters appear structurally valid but return wrong contrasts, wrong row ordering, or invalid small-sample corrections.
- **Required change:** replace the single broad protocol with versioned capability protocols such as `LinearPredictorBasis`, `PredictionProvider`, `EstimatingEquationProvider`, `ClusterWorkingModel`, and `VarianceComponentsProvider`. Use labeled immutable arrays and explicit row/parameter identities. Keep pandas/polars at adapters, not in the numerical domain.
- **Acceptance criteria:** contract tests deliberately permute observation and coefficient order, include aliased coefficients, offsets, transformed terms, unseen levels, and non-estimable contrasts, and either reproduce the reference or return a typed unsupported result.
- **Estimated scope:** large

### [High] Inference is treated as an implementation detail instead of a separate statistical product

- **Location:** §6 and §8.5
- **Principle:** correctness / single responsibility
- **Evidence:** autodifferentiating an objective can provide derivatives, but it does not make covariance-parameter uncertainty, boundary behavior, or KR adjustments “free.” L-BFGS-B itself maintains a limited-memory approximation rather than returning the exact Hessian required here. `lmerTest` explicitly differentiates the coefficient covariance and uses the covariance of variance-parameter estimates [S6]. Boundary LRT distributions depend on the tested covariance structure; a universal $\bar\chi^2$ correction is not one algorithm [S8].
- **Failure mode:** anti-conservative p-values, undefined Hessians at the boundary, or confident output for a method outside its derivation.
- **Required change:** make each inference method an independently specified module with supported hypotheses, assumptions, reference algorithm, fallback policy, and simulation-calibrated error targets. For v0.1 ship covariance estimates and clearly labeled asymptotic statistics plus a reproducible parametric bootstrap; gate Satterthwaite, KR, and variance-component tests separately.
- **Acceptance criteria:** null simulations cover balanced/unbalanced, few-group, near-singular, rank-deficient, and boundary cases; empirical rejection and interval coverage meet predeclared tolerances. A failed/ill-conditioned approximation never silently falls back to a different method.
- **Estimated scope:** large

### [High] The milestone order maximizes breadth before proving the hard core

- **Location:** §§2 and 11
- **Principle:** risk management / cohesive releases
- **Evidence:** M1 attempts a generic protocol, multiple third-party adapters, and CR2 before the fitter establishes what information its own correct API must preserve. Full `postfit` is a separate ecosystem-scale project.
- **Failure mode:** substantial work lands on an unstable abstraction while the nested/sparse feasibility risk remains unresolved.
- **Required change:** begin with one vertical slice: formula → matrices → fixed-$\theta$ objective → optimizer → fitted object for a single grouping factor. Defer public generic adapters until this object and at least two external model adapters pass a capability spike. Treat `lmerx` and `postfit` as separately releasable products even if they temporarily share a workspace.
- **Acceptance criteria:** no `postfit` public API freeze before the capability spike; the first alpha fits and predicts a deliberately narrow model class end to end.
- **Estimated scope:** medium

### [High] Formula parsing is underestimated and duplicates an existing candidate

- **Location:** §5 and M0
- **Principle:** security / buy-versus-build / parsing correctness
- **Evidence:** splitting strings around top-level bars must still preserve `||`, nesting expansion, calls, quoting, operator precedence, evaluation environments, missing rows, factor contrasts, and state for new data. The maintained `formulae` package already builds common and group-specific matrices [S12]; `formulaic` provides extensible parsing, sparse output, reusable encoder state, and dataframe support [S13].
- **Failure mode:** subtly different $X/Z$, executable user expressions crossing a trust boundary, or a parser maintenance project larger than the fitter.
- **Required change:** time-box a build-vs-buy spike comparing `formulae`, a `formulaic` extension, and a minimal owned grammar. Define whether formulas are trusted code. If untrusted strings may enter a service, expose a restricted declarative grammar with an allowlisted transform registry and no arbitrary evaluation.
- **Acceptance criteria:** an adversarial corpus covers nested calls, quoted names, `|` inside expressions/strings, `||` with numeric and categorical predictors, slash expansion, missing values, reordered categories, and new data. The chosen route matches frozen R $X$, $Z^T$, column names, and row masks.
- **Estimated scope:** medium

### [High] Differential fixtures are necessary but insufficient and can silently age

- **Location:** §9
- **Principle:** tests / reproducibility
- **Evidence:** committed final outputs identify regressions against one oracle version, not mathematical correctness or parity with current supported `lme4`. Several proposed properties are not valid as written: “profile intervals bracket Wald intervals” is not a required property; “OLS deviance” is ambiguous between ML and REML; and one set of global tolerances cannot cover zero-boundary parameters and large criteria.
- **Failure mode:** the suite stays green while upstream semantics, fixture provenance, or a shared algebraic mistake invalidates results.
- **Required change:** use three oracles: (1) independent dense $V=ZGZ^T+R$ computations for small models, (2) fixed-$\theta$ intermediate fixtures from pinned R, and (3) final-fit differential tests. Regenerate in a pinned R container on a scheduled and release job; default Python CI can remain R-free.
- **Acceptance criteria:** each fixture has schema version, generation script hash, input-data hash/license, `sessionInfo()`, model controls, and expected row/column labels. Tolerances are per quantity with absolute/relative rules and explicit boundary logic.
- **Estimated scope:** medium

### [High] Sparse-backend licensing and distribution are decisions, not optional-install details

- **Location:** §4
- **Principle:** dependency boundary / release engineering
- **Evidence:** current CHOLMOD declares Cholesky as LGPL but Supernodal, MatrixOps, Modify, and GPU modules as GPL [S14]. `scikit-sparse` currently documents system SuiteSparse installation for pip/Homebrew/apt rather than promising a universal self-contained wheel [S15]. An optional extra does not by itself settle redistribution or linking obligations.
- **Failure mode:** missing wheels on a supported platform, accidentally GPL-coupled artifacts, or a published license that does not describe the shipped binary path.
- **Required change:** make a license-reviewed backend decision record. CI must inspect linked libraries for every wheel and test the exact installation instructions. Never state a legal conclusion based only on Python package metadata.
- **Acceptance criteria:** owner selects the project license; counsel or a qualified license review signs off before distributing a CHOLMOD-enabled artifact; the support matrix identifies which backends are installable on each OS/architecture.
- **Estimated scope:** medium

### [Medium] Optimizer behavior is described inaccurately and the autodiff path is underspecified

- **Location:** §3
- **Principle:** correctness / reproducibility
- **Evidence:** current `lmer` defaults to `nloptwrap`, whose default NLopt algorithm is BOBYQA [S3, S4]. The default two-optimizer BOBYQA/Nelder–Mead sequence applies to `glmer`, not `lmer`; `restart_edge` is not a general Nelder–Mead fallback. If JAX is evaluated, its default disables 64-bit values, which is incompatible with the stated tolerances unless x64 is explicitly required [S16].
- **Failure mode:** “reference” behavior differs before any algebra is exercised, and a nominally faster float32 backend fails parity.
- **Required change:** record exact optimizer algorithm, controls, start values, restart rule, scaling, convergence diagnostics, and derivative source. Treat optimizer agreement as a diagnostic matrix, not automatic fallback that hides failure.
- **Acceptance criteria:** reference runs reproduce pinned `lmerControl` settings; all successful fits expose optimizer, controls, evaluations, termination reason, gradient, Hessian status, boundary status, and alternative-optimizer comparison when requested.
- **Estimated scope:** medium

### [Medium] Diagnostics and prediction semantics are not specified tightly enough

- **Location:** §§6–8
- **Principle:** explicit state / API contract
- **Evidence:** `lme4::isSingular` tests whether covariance matrices lose rank; in dimension three or more this can occur without a small individual variance or a correlation near ±1 [S5]. Population-level versus conditional prediction also requires explicit behavior for new groups, missing random effects, offsets, and uncertainty.
- **Failure mode:** false “non-singular” flags, inconsistent post-estimation, or accidental conditional predictions in population summaries.
- **Required change:** use eigenvalue/rank diagnostics with a documented scale-aware tolerance and retain detailed rotations/eigenvalues. Replace `re_form=None` with an explicit prediction mode enum plus `allow_new_groups`, `include_parameter_uncertainty`, and response/linear-predictor scale.
- **Acceptance criteria:** prediction and singularity fixtures cover unseen groups, zero-variance dimensions hidden by covariance rotation, new factor levels, row preservation, and both conditional/population modes.
- **Estimated scope:** medium

### [Medium] Performance targets are not reproducible benchmarks

- **Location:** §10
- **Principle:** observability / tests
- **Evidence:** targets omit hardware, Python/R/package versions, BLAS vendor and thread counts, warm-up/JIT policy, data generation, peak memory, and correctness gate.
- **Failure mode:** optimizations are rewarded for changing semantics, and reported results cannot be compared across machines or releases.
- **Required change:** define a benchmark manifest and store JSON results with environment metadata. Separate parse, assembly, symbolic analysis, objective evaluation, optimization, and inference. Report median/p95 and peak RSS; include cold and warm JAX paths if applicable.
- **Acceptance criteria:** CI runs small regression ceilings; scheduled pinned hardware runs comparative benchmarks only after numerical checks pass.
- **Estimated scope:** small

### [Medium] Production engineering is absent from the original milestones

- **Location:** entire plan
- **Principle:** release / supply chain / governance
- **Evidence:** no repository layout, lockfile policy, CI matrix, coverage/type/API gates, trusted publishing, SBOM/provenance, security policy, compatibility versioning, deprecation policy, or rollback/yank procedure is planned.
- **Failure mode:** a numerically credible prototype is published as an unreproducible or unsafe package.
- **Required change:** adopt the local production repository standard with the library-specific tailoring in §18. Docker is optional for the shipped Python libraries but required for the pinned R fixture generator if that is the selected reproducibility mechanism.
- **Acceptance criteria:** all applicable items in §19 are green before a stable release.
- **Estimated scope:** medium

## 15. Adversarial challenge: how this plan fails in practice

| Attack or stress case | Likely present behavior | Required defense |
|---|---|---|
| Formula labeled “nested” but producing coupled $Z^T Z$ blocks | Wrong dense solve or late numerical mismatch | Inspect the actual sparsity graph and prove backend capability before fitting |
| Near-zero variance with a rotated zero eigen-dimension | Boundary check misses singularity | Eigen/rank diagnostic with structured evidence |
| Rank-deficient $X$ plus an `emmeans` contrast | Adapter returns a number for a non-estimable function | Preserve null-space basis and reject/mark non-estimable results |
| User formula contains arbitrary transform code | Code executes when a formula crosses a service boundary | Trusted-formula policy and restricted safe grammar for untrusted input |
| Huge-cardinality grouping factor or adversarial interaction | Memory/indices overflow or process exhaustion | Preflight $n,p,q,nnz$, integer-width and fill estimates; configurable resource ceilings |
| New prediction data has unknown fixed or random levels | Silent recoding or unintended population prediction | Stored encoder state and explicit new-level/new-group policy |
| CHOLMOD found but linked with unexpected modules | License and artifact mismatch | Link inspection, backend manifest, release gate |
| Bootstrap workers inherit BLAS threads and duplicate RNG streams | Oversubscription and irreproducible intervals | SeedSequence-style stream derivation, thread caps, failure ledger, resumable runs |
| Optimizer returns success on a flat/boundary solution | Invalid Hessian-based inference | Independent scaled-gradient/Hessian diagnostics and inference capability status |
| Frozen R fixtures were produced under a different contrast or package version | False regression failure or false parity claim | Complete fixture manifest and pinned regeneration image |
| CR2 adapter permutes cluster rows independently of scores | Plausible but wrong covariance | Labeled row identities and permutation contract tests |
| REML models with different fixed effects are compared by AIC/LRT | Invalid model selection claim | Comparison validator; refit with ML where required and document exceptions |

The most dangerous failure mode is not a crash. It is a plausible table of
coefficients, degrees of freedom, or marginal means computed from a subtly
different design matrix or row order. Every boundary must therefore preserve
labels and provenance, and unsupported capabilities must fail closed.

## 16. Corrected target architecture

### 16.1 Dependency map

```text
Public Python API / optional dataframe presenters
                  |
                  v
Fit, predict, bootstrap, profile use cases
                  |
                  v
ModelSpec + labeled arrays + PLS objective + inference policies
       |                  |                    |
       v                  v                    v
formula adapter      numeric backend      optimizer / executor / RNG ports
       |                  |                    |
       v                  v                    v
formulae/formulaic   NumPy/SciPy/CHOLMOD   NLopt/JAX/process backend
```

Rules:

- The mathematical core does not import pandas, polars, entry-point discovery,
  NLopt, CHOLMOD, or JAX.
- A `ModelSpec` owns row identity, coefficient identity, contrast/encoder state,
  offsets, weights, missing-row mask, grouping levels, $\Lambda$ index map, and
  compatibility metadata.
- Backends advertise capabilities from inspected matrix structure, precision,
  derivative support, and license/build identity.
- Result objects are immutable snapshots. They retain enough state for new-data
  matrices without retaining an unsafe live evaluation environment by default.
- Inference methods consume a fitted-state protocol and return a structured
  `available / unsupported / failed` status with diagnostics.
- Third-party postfit adapters translate to small capability protocols. Entry
  points register factories, not trusted arbitrary result objects.

### 16.2 Corrected numerical backend plan

1. **Dense oracle:** form $V=ZGZ^T+R$ for small $n$ and compute ML/REML by a
   deliberately independent route. This is slow and never the production path.
2. **Verified block path:** accept only structures for which a permutation into
   independent blocks is established from the actual sparsity graph. Initially,
   this should be one grouping structure, including its combined per-group
   random-effect columns.
3. **Nested/tree experiment:** prototype elimination by independent top-level
   clusters and compare against the dense oracle. Do not advertise until its
   supported pattern is formalized.
4. **General sparse path:** choose CHOLMOD simplicial/supernodal or another
   backend only after correctness, wheel, performance, and license spikes.
5. **Autodiff experiment:** require float64, verify first and second derivatives
   against complex-step/finite-difference checks away from boundaries, and
   compare behavior at boundaries. It does not become default merely because it
   is faster on `sleepstudy`.

### 16.3 Versioned compatibility contract

Track these versions independently:

- package/API version;
- fit-result serialization schema;
- formula/contrast semantics version;
- numerical-backend identity and version;
- R-oracle corpus version, including `lme4`/`Matrix` versions; and
- postfit capability-protocol version.

Never pickle fit objects as the stable interchange format. Define a
machine-readable manifest and a limited data format for coefficients,
covariances, model specification, diagnostics, and provenance; reconstructing
executable transform code must require an explicit trusted path.

## 17. Revised product scope and roadmap

No calendar estimate is credible until Phase 0 measures the hard dependencies.
Each phase ends at an evidence gate, not merely when code exists.

### Phase 0 — feasibility and contract freeze

- Write the compatibility matrix and narrow the first public claim.
- Create the repository under `~/Projects/<chosen-name>`; the reviewed document
  may remain in Downloads only as an input artifact.
- Decide a temporary monorepo workspace versus separate repositories for the two
  distributions; record ownership and independent release policy.
- Build the independent dense oracle and fixed-$\theta$ R fixture generator.
- Evaluate `formulae`, a `formulaic` extension, and an owned grammar on the same
  formula corpus.
- Prove or reject the block, nested/tree, general sparse, NLopt, and JAX paths
  with small prototypes.
- Audit PyPI names, project/trademark risk, dataset licenses, and backend licenses.

**Gate:** architecture decision records exist; the algebra oracle agrees with
pinned R at fixed parameters; one formula backend reproduces $X/Z$ for the
narrow corpus; the owner accepts the license and naming choices.

### Phase 1 — one production-shaped vertical fitter slice

- Support Gaussian LMMs with one verified grouping structure, ML and REML,
  numeric random slopes/intercepts, explicit contrasts, deterministic missing-row
  handling, and structured diagnostics.
- Implement reference NLopt BOBYQA controls and at least one comparison optimizer.
- Fit, predict, serialize diagnostics, and reload the safe result interchange.
- Test `Dyestuff`, `Dyestuff2`, and `sleepstudy`, plus generated adversarial cases.
- Establish lint, formatting, typing, unit/integration, coverage, build, and wheel
  smoke gates on the supported Python/platform matrix.

**Gate:** fixed-parameter and final-fit tolerances pass; unsupported formulas fail
before allocation/optimization; wheel installs in clean environments; no
small-sample DF claim is made.

### Phase 2 — robustness and inference foundation

- Add weights, offsets, rank-deficiency/estimability policy, prediction modes,
  new-level handling, and reproducible parametric bootstrap.
- Add scaled-gradient, Hessian/eigenvalue, and alternative-optimizer diagnostics.
- Run recovery, null-calibration, and coverage simulations; publish failures.
- Specify and implement Satterthwaite only after derivative and boundary gates.

**Gate:** the supported subset has a complete compatibility row for every
argument and behavior; bootstrap is reproducible across worker counts; inference
coverage targets pass.

### Phase 3 — structure expansion

- Add the proved nested/tree path or general sparse path.
- Add crossed effects only with installable supported backend artifacts.
- Introduce scale and fill-in benchmarks with preflight resource estimates.
- Add `Pastes`, `Penicillin`, and `InstEval` only after dataset redistribution and
  fixture licensing are recorded.

**Gate:** every structure routes by inspected capability; dense-oracle tests cover
small representatives; performance and memory targets have pinned baselines.

### Phase 4 — postfit capability spike

- Implement internal linear-function and estimating-equation capabilities.
- Build two external adapters before generalizing the interface; include one
  model with rank deficiency and one with weights/clusters.
- Decide whether marginal effects are contributed upstream to the existing
  Python `marginaleffects` project [S17].
- Ship CR0/CR1 first; gate CR2, Satterthwaite CR2 tests, and multi-parameter tests
  against reference fixtures and simulations.
- Treat `emmeans`, tidy summaries, and performance diagnostics as separate
  feature epics with explicit parity matrices, not one “completion” milestone.

**Gate:** permutation/estimability contract tests pass and each adapter publishes
its capability table. `postfit` may release independently only after this gate.

### Phase 5 — advanced inference and stable-release evidence

- Implement KR, profile intervals, and supported variance-component tests one at
  a time with method-specific validation.
- Run scheduled upstream-oracle checks and publish benchmark/compatibility reports.
- Complete security, provenance, SBOM, trusted publishing, documentation,
  deprecation, support, and incident/yank procedures.

**Gate:** the completion criteria in §20 are satisfied for a documented subset;
unimplemented `lme4`, `emmeans`, or `clubSandwich` features remain explicitly out
of scope rather than blocking an honest stable release.

## 18. Engineering and release plan from the production template

### 18.1 Repository baseline

Use the local production standard with library-specific omissions documented:

```text
<repository>/
├── .github/
│   ├── dependabot.yml
│   └── workflows/{ci,oracle,release}.yml
├── .gitignore
├── .python-version
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE                    # owner-selected, never inferred by the agent
├── Makefile
├── README.md
├── REPRODUCIBILITY.md
├── SECURITY.md
├── pyproject.toml
├── uv.lock
├── docs/
├── packages/                  # only if a two-distribution workspace is chosen
│   ├── lmerx/src/lmerx/
│   └── postfit/src/postfit/
├── tests/{unit,integration,oracle,statistical}/
└── tools/{fixtures,bench}/
```

A production Docker image is unnecessary for a pure library. A digest-pinned
development image is appropriate for reproducible R fixture generation. Commit
one authoritative Python lockfile, use frozen installs, and keep generated build
artifacts and environments out of Git.

### 18.2 Quality gates

- `make check` runs format check, lint, strict-enough typing, unit/integration
  tests, coverage, API compatibility checks, and package build.
- CI uses least-privilege permissions and actions pinned to commit SHAs.
- Test supported Python versions and OS/architecture combinations; backend
  extras get separate jobs rather than weakening the base matrix.
- Install and import each built wheel in a clean environment. Validate metadata,
  licenses, bundled/shared libraries, and absence of fixtures/secrets/caches.
- Statistical tests record seeds and environment. Slow simulation and R-oracle
  jobs are scheduled/release gates, clearly distinct from default unit tests.
- Treat warnings, optimizer failures, and bootstrap refit failures as structured
  test data. Never discard failed replicates without reporting the denominator.

### 18.3 Reproducibility and supply chain

- Fixture manifests include checksums and exact generator provenance.
- Release artifacts are built in CI from the reviewed tag, signed or accompanied
  by attestations, checksums, provenance, and an SBOM.
- Publish to PyPI with trusted publishing/short-lived identity; no long-lived
  upload token.
- Pin Git dependencies to immutable commits; prefer released dependencies.
- Dependabot/Renovate changes require the full compatibility and numerical suite.
- Document how to yank a bad wheel, disable a backend, and issue a corrected
  oracle corpus without silently rewriting old fixtures.

### 18.4 Documentation and governance

The README and compatibility report must state the supported model grammar,
inference methods, prediction semantics, exact/approximate differences from the
pinned `lme4` oracle, numerical tolerances, supported platforms, and known
failure modes. `REPRODUCIBILITY.md` covers the R oracle, dataset provenance,
threading/BLAS controls, RNG streams, benchmarks, and release recreation.

Adopt semantic versioning for the Python API plus explicit schema/oracle versions.
Define public/private API markers, a deprecation window, security-reporting path,
maintainer ownership, and the minimum upstream versions supported by each adapter.

## 19. Initial acceptance scorecard

| Area | Alpha gate | Stable gate | Current |
|---|---|---|---|
| Product claim | Narrow supported subset documented | Compatibility matrix complete and versioned | Red |
| Algebra | Dense and fixed-$\theta$ oracles agree | All supported structures covered | Red |
| Formula semantics | Chosen backend matches narrow R corpus | Adversarial grammar/new-data corpus passes | Red |
| Optimization | Pinned BOBYQA path and diagnostics | Cross-optimizer policy validated | Red |
| Inference | Asymptotic output labeled; bootstrap deterministic | Each advertised method calibrated | Red |
| Sparse support | Unsupported structures fail closed | Supported wheel/backend matrix passes | Red |
| Postfit protocols | Internal capability types only | Two external adapters pass contract suite | Red |
| Packaging | Clean wheel builds and imports | Reproducible signed/attested releases | Red |
| CI quality | Lint/type/test/build gates | Full OS/Python/backend matrix and release gates | Red |
| Documentation | Limitations and quickstart | API, methods, reproducibility, security, governance | Red |
| Performance | Manifested local baseline | Scheduled regression and memory reports | Red |

Red is expected for a pre-scaffold design. It is a work queue, not a judgment
that a check failed at runtime.

## 20. First ten pull requests

1. Scaffold the chosen repository layout, manifest, lock, license placeholder,
   contribution/security/reproducibility docs, and green empty quality gate.
2. Add the compatibility matrix and architecture/license decision-record template.
3. Add the pinned R fixture generator image, manifest schema, and three tiny
   hand-checkable datasets with redistribution records.
4. Implement labeled `ModelSpec` and the independent dense $V$ oracle.
5. Complete the formula-backend spike and commit the decision plus adversarial
   formula corpus; do not merge three production parsers.
6. Implement $\Lambda(\theta)$ assembly, fixed-$\theta$ PLS outputs, and
   intermediate oracle comparisons.
7. Implement the verified single-structure block backend with a fail-closed
   structural capability classifier.
8. Add the optimizer port, pinned NLopt BOBYQA adapter, diagnostics, and
   cross-optimizer comparison tests.
9. Add immutable fit results, explicit prediction modes, new-data encoder state,
   and safe serialization manifest.
10. Add clean-wheel CI, supported-platform smoke tests, benchmark manifest, and
    the first alpha compatibility report.

Do not start a public `postfit` package, CR2, KR, or a crossed-effects release in
these first ten changes. They depend on evidence the changes above are intended
to produce.

## 21. Decisions and completion criteria

### 21.1 Decisions required before implementation

1. What exact subset and reference versions does “`lmer` compatible” mean?
2. Which formula backend wins the measured spike, and are formulas trusted code?
3. Is the first release permissively licensed, and which numerical backends are
   legally and operationally compatible with that choice?
4. Is the two-distribution work held in one workspace or separate repositories,
   and who owns each release?
5. Which Python versions/platforms and BLAS configurations are supported?
6. What is the stable result-interchange schema, and what is explicitly unsafe
   to deserialize?
7. Which statistical error/coverage thresholds gate each inference method?
8. Which datasets and generated fixtures may be redistributed?

### 21.2 Definition of done for the first stable release

The first stable release is complete only when:

1. A clean clone installs in frozen mode and `make check` passes.
2. Built wheels install and smoke-test on every declared platform/Python pair.
3. Every accepted model passes matrix-level, fixed-$\theta$, final-fit, prediction,
   and serialization tests; every rejected structure fails before expensive work.
4. Numerical tolerances are per-output, boundary-aware, documented, and met by
   the release-oracle job.
5. Every advertised inference method passes its predeclared simulation and
   reference gates; failures remain visible in result metadata.
6. Formula, coefficient, observation, group, and cluster identities remain aligned
   through all adapter and post-estimation paths.
7. The package reports backend/build identity, optimizer diagnostics, singularity,
   convergence, and unsupported capabilities structurally.
8. Dependency locks, fixture/data provenance, benchmark environment, license
   review, SECURITY, REPRODUCIBILITY, and compatibility documentation are current.
9. CI uses least privilege and pinned actions; releases use immutable reviewed
   tags, trusted publishing, checksums, provenance, and an SBOM.
10. No credential, machine-specific path, generated cache, or unlicensed dataset
    is present; the worktree contains only intentional source artifacts.

### 21.3 Sources

Primary/upstream sources were preferred. Version-sensitive claims must be
rechecked when implementation starts.

- **[S1]** Bates, Mächler, Bolker, and Walker, [*Fitting Linear Mixed-Effects Models using lme4*](https://lme4.github.io/lme4/articles/lmer.pdf).
- **[S2]** `lme4`, [`lmer` reference](https://lme4.github.io/lme4/reference/lmer.html) — formula, weights, offsets, missing values, contrasts, rank deficiency, and `||` limitations.
- **[S3]** `lme4`, [`lmerControl` reference](https://lme4.github.io/lme4/reference/lmerControl.html) — current defaults, restart and convergence controls.
- **[S4]** `lme4`, [`nloptwrap` reference](https://lme4.github.io/lme4/reference/nloptwrap.html) — NLopt BOBYQA default and tolerances.
- **[S5]** `lme4`, [`isSingular` reference](https://lme4.github.io/lme4/reference/isSingular.html) — covariance-rank interpretation and higher-dimensional caveat.
- **[S6]** Kuznetsova, Brockhoff, and Christensen, [`lmerTest` JSS paper](https://www.jstatsoft.org/article/view/v082i13) — Satterthwaite implementation and derivative requirements.
- **[S7]** Halekoh and Højsgaard, [`pbkrtest` JSS paper](https://www.jstatsoft.org/article/view/v059i09) — KR and parametric-bootstrap model comparison.
- **[S8]** Self and Liang, [likelihood-ratio tests under nonstandard boundary conditions](https://doi.org/10.1080/01621459.1987.10478472).
- **[S9]** `emmeans`, [extension contract](https://rvlenth.github.io/emmeans/reference/extending-emmeans.html) and [reference-grid behavior](https://rvlenth.github.io/emmeans/reference/ref_grid.html).
- **[S10]** `emmeans`, [`emmeans` weights reference](https://rvlenth.github.io/emmeans/reference/emmeans.html) — equal, proportional, outer, cells, and flat weighting semantics.
- **[S11]** `clubSandwich`, [`vcovCR.lmerMod` reference](https://jepusto.github.io/clubSandwich/reference/vcovCR.lmerMod.html) — cluster, working target, inverse-variance, and estimating-function requirements.
- **[S12]** Bambinos, [`formulae` documentation](https://bambinos.github.io/formulae/api_reference.html) — existing mixed-effects formula parsing and design matrices.
- **[S13]** Formulaic, [upstream project and documented capabilities](https://github.com/matthewwardrop/formulaic).
- **[S14]** SuiteSparse, [CHOLMOD module license declarations](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/Include/cholmod.h).
- **[S15]** Scikit-Sparse, [installation documentation](https://pypi.org/project/scikit-sparse/).
- **[S16]** JAX, [default dtype and x64 policy](https://docs.jax.dev/en/latest/default_dtypes.html).
- **[S17]** `marginaleffects`, [upstream R/Python project](https://github.com/vincentarelbundock/marginaleffects).
