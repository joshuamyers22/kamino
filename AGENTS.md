# Agent Working Agreement

Human instructions in the current task take precedence. Before substantial work,
read `README.md`, `PROJECT_BRIEF.md`, and `PROJECT_MEMORY.md`, then verify relevant
claims against source, tests, ADRs, and runtime behavior.

## Statistical correctness

- Treat `PROJECT_PLAN.md` and accepted ADRs as the specification.
- Preserve row, coefficient, grouping, term, and covariance-parameter identity.
- Add an independent oracle or invariant before optimizing numerical code.
- Never weaken a tolerance, regenerate a fixture, suppress a diagnostic, or change
  an estimand merely to make a check pass.
- Mark skipped or unavailable oracle and simulation checks accurately.
- Keep unsupported formulas and inference methods fail-closed.

## Work and evidence

- Keep disposable output in ignored `.work/`; do not commit raw command logs.
- Record durable constraints and decisions in `PROJECT_MEMORY.md` with evidence.
- Use ADRs for consequential architecture, compatibility, dependency, or license
  choices. The project owner selects the distribution license.
- Run focused tests after coherent changes and `make check` before handoff.
- Do not store secrets, sensitive research data, formulas containing private names,
  hidden reasoning, or unrestricted logs in tracked files.
