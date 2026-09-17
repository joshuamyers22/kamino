# E03 adversarial code and architecture review

## Review metadata

- Repository/remote: Kamino, `github.com/joshuamyers22/kamino`
- Branch and protocol commit: `main`, `4db31746c979a7b0924f3f46008c24df666663e3`
- Reviewer/date: Joshua Myers, 2026-09-16
- Scope: E03 benchmark manifests, runners, raw reports, assessment, claim policy,
  and risk ownership; generated fixture data and prior numerical implementation
  are evidence inputs rather than newly reviewed code.
- North star: PROJECT_PLAN §12.2 and acceptance item E03.
- Blocking threshold: unresolved Critical/High correctness, confidentiality, or
  release-integrity finding.
- Independence: owner self-review; not independent statistical/release approval.
- Ceiling: ten matched repeated fits, 300 seconds per worker, 1.2 GB/1.0 GB
  Kamino case ceilings; escalate rather than weaken tolerances.

## Executive verdict

- Overall grade: B+ for the bounded E03 scope.
- Recommendation: approve E03 with follow-up; continue to block beta/stable on
  independent reviewers.
- Highest risk: an observational cross-stack ratio could be mistaken for a fair
  performance comparison.
- Strongest property: raw samples, revisions, manifests, environment, fit
  discrepancies, decisions, and risk inputs are hash-linked and executable.
- First improvement: build a genuinely matched runtime/BLAS experiment before
  revisiting the InstEval 2× objective.

## Verification evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Formatting/lint | `uv run ruff check .` and `uv run ruff format --check .` | Pass | No suppressed E03 finding |
| Static typing | `uv run pyright` | Pass | Zero errors/warnings |
| Unit/integration | `uv run pytest --cov --cov-report=term-missing` | Pass | 396 tests, 91.02% |
| Build | `uv build --offline` | Pass | sdist and wheel built |
| E03 benchmark | `make benchmark-e03` | Pass | Ten matched cases; raw reports retained |
| Evidence integrity | `make benchmark-e03-verify` | Pass | Hashes, revisions, samples, gates, claim boundary, owners |
| Sanitizers/races | Not run | Unverified/not applicable | Python/R numerical library benchmark; no claim |

## Architecture map

The manifested generators/fixtures feed two delivery adapters: Python benchmark
workers call the public Kamino API and isolated staged internals; the R adapter
calls pinned lme4 in the immutable oracle container. The assessment layer reads
only raw JSON and declared tolerances. The verifier binds committed reports back
to manifests, hardware identity, oracle digest, revision, decisions, and risks.
No benchmark code is imported by the runtime package.

```text
manifests + fixtures -> isolated Kamino/lme4 workers -> raw JSON
raw JSON + risk/hardware records -> assessment -> offline verifier
```

## Findings

### Medium: unmatched stacks prohibit a relative-performance conclusion

- Location: `benchmarks/e03_*_v1.json`, `oracle/benchmark_e03.R`
- Evidence: native macOS/Accelerate/SciPy and Docker Linux/OpenBLAS/lme4 use
  different execution and optimizer stacks.
- Failure mode: a ratio is reported as implementation speedup/regression.
- Disposition: controlled by ADR 0015, report labels, verifier assertions, and
  the explicit `not-claimed` decision. Acceptance: a matched experiment must
  replace this baseline before any comparative claim. Status: open/controlled.

### Medium: the proposed InstEval 2× objective is not demonstrated

- Location: `benchmarks/results/e03_assessment_v1.json`
- Evidence: observational Kamino/lme4 median ratios are 4.24× ML and 4.12× REML.
- Failure mode: stable-scale language outruns measured performance.
- Disposition: the verifier requires `insteval_within_2x: unverified`; optimize
  only after profiling the repeated SuperLU factorization. Status: open.

### Medium: workstation noise limits small regression decisions

- Location: `benchmarks/hardware/apple_m1_pro_8c_16gb_2026_v1.json`
- Evidence: affinity, frequency, thermal state, and background load are not fully
  controlled.
- Failure mode: noise is mistaken for a code regression.
- Disposition: keep raw distributions, require exact-profile reruns, and use the
  20% threshold as an investigation trigger rather than an automatic failure.
  Status: accepted with control.

### Low: some phase timings cannot be cleanly separated

- Location: `tools/benchmark_general_sparse.py`, `oracle/benchmark_e03.R`
- Evidence: SuperLU symbolic analysis is internal to factorization; inference is
  outside the fit benchmark.
- Failure mode: phase totals are overinterpreted.
- Disposition: explicit status fields are retained and verified; no invented
  zero-time or inferred measurement is reported. Status: accepted.

## Clean-code and architecture disposition

Names and side effects are explicit; workers isolate peak RSS; subprocesses are
bounded; malformed/failed workers cannot produce a passing assessment; generated
environment state stays outside the runtime wheel; and the oracle dependency
points outward. The main duplication is intentional cross-language generation
used to test identity. The E03 scripts suppress no lint/type findings.

## Improvement plan

| Priority | Change | Owner | Verification | Status |
|---:|---|---|---|---|
| 1 | Design matched OS/BLAS/optimizer-policy comparison | Joshua Myers | New ADR plus repeated raw reports | Open |
| 2 | Profile InstEval factorization/evaluations before optimizing | Joshua Myers | Stage profile and unchanged oracle tolerances | Open |
| 3 | Assign independent statistical/release reviewers | Joshua Myers | Named approvals on exact candidate | Release blocker |

Review again on a sparse-backend change, >20% exact-profile movement, scope
expansion, or beta/stable candidate freeze. No Critical/High implementation
finding blocks E03 itself; the separate High governance risk continues to block
beta/stable publication.
