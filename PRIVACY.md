# Privacy and data handling

Kamino processes caller-supplied model data locally. The library contains no
telemetry, analytics, update check, remote model service, or other runtime
network path. Installing dependencies and downloading releases are package-tool
operations outside Kamino's runtime.

## Data retained in memory

A live fitted result retains the accepted response, offsets, weights, fixed and
random design state, row identifiers, grouping labels, fitted random effects,
and numerical results needed for prediction, refitting, and inference. Python
memory is not cryptographically erased when objects are released. Processes
handling sensitive data should use ordinary host isolation, access controls,
encrypted swap where required, and bounded process lifetimes.

## Files written by explicit request

- Prediction bundles do not contain training rows or responses, but they do
  contain the formula and coefficient names, categorical and grouping levels,
  fitted random effects, covariance estimates, and provenance hashes. Those
  values can be identifying and bundles must be handled as sensitive model
  artifacts.
- A private parametric-bootstrap ledger contains complete simulated response
  arrays and fit outcomes. Kamino creates ledger directories and files with
  owner-only permissions where the operating system supports them, rejects
  symbolic-link ledger paths, and writes metadata atomically. The caller owns
  storage encryption, backup, sharing, retention, and secure deletion.
- Statistical reports and benchmark manifests contain aggregate synthetic or
  public-reference evidence. They must never be generated from private caller
  data for inclusion in the repository.

Kamino does not automatically delete user-selected output paths. Delete model
bundles and the entire bootstrap-ledger directory when their purpose and any
required audit retention period end. Remember that filesystem snapshots,
backups, and synchronized folders can retain copies after local deletion.

## Diagnostics and support

Exceptions and diagnostic objects may contain formulas, variable names, row
identifiers, grouping names, filesystem paths, dimensions, and optimizer
status. They are not uploaded automatically. Before filing an issue, replace
all sensitive names and values with a synthetic reproducer. Sensitive security
reports belong in GitHub's private vulnerability-reporting channel.

No claim is made that de-identified labels are anonymous. Group sizes, random
effects, rare categorical levels, and combinations of model metadata can permit
re-identification.
