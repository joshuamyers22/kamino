# Security incident runbook

1. Move discussion and reproductions to a private GitHub security advisory.
   Record absolute UTC timestamps, affected versions/artifacts, reporter contact,
   and evidence access restrictions. Do not copy private data into issues or
   project memory.
2. Triage exploitability and impact across formula parsing, bundle loading,
   ledger paths, native dependencies, CI, release artifacts, and credentials.
   Critical confidentiality or code-execution findings block releases.
3. Contain: disable the affected workflow or release, revoke/rotate exposed
   credentials, restrict compromised artifacts, and preserve immutable hashes.
   Do not destroy evidence needed to scope exposure.
4. Reproduce with a minimal synthetic case. Identify the first affected commit
   and every distributed version; inspect attestations and SBOMs for dependency
   incidents.
5. Correct with a regression test or scanner rule, run the complete affected
   gates, obtain accountable review, and publish a new version. Never rewrite a
   tag or silently replace an artifact.
6. Coordinate disclosure and downstream guidance. State affected regimes,
   workaround, fixed version, credential/data actions, and residual uncertainty.
7. Complete a blameless incident review with owned actions and verification.
   Close only after the fix and response controls are evidenced.
