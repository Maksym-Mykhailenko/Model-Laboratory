# Model Laboratory 1.18.0

Version 1.18.0 turns the release-readiness work into the active desktop release and closes the
remaining application-side reliability gaps found during the 1.17 review.

## Desktop safety

- Scientific sidecar calls now have finite operation-class timeouts. A timed-out sidecar is
  terminated and the next request starts a clean process.
- Files opened through operating-system associations remain in a native pending queue until the
  frontend has successfully opened and acknowledged them. Cancelling the unsaved-changes prompt
  preserves the request and exposes a **Pending file** retry action.
- Model names recovered from experiments and commit receipts are converted to portable YAML
  filenames before they reach a native save dialog.
- A later save removes stale, application-owned atomic-save temporary files for the same target;
  unrelated files and fresh recovery evidence are not touched.

## Windows release pipeline

- NSIS and MSI installer builds run version/tag consistency checks before packaging.
- Tagged releases require an imported Authenticode certificate and valid signatures; manual CI
  runs may still create clearly unsigned internal-test artifacts.
- Installer smoke automation covers clean installation, executable launch, `.mlab`, `.yaml`, and
  `.yml` open-command registration, and uninstall for both package formats. It automatically
  discovers an earlier published NSIS build when available and exercises the upgrade path.
- Every installer run emits a machine-readable pass/fail receipt with installer hashes; failed
  workflows retain that receipt as a diagnostic artifact, while successful receipts are shipped
  with the installers. Uninstall checks reject residual executables or application open commands.
- GitHub Actions dependencies are pinned to immutable commits, and draft releases are created with
  the runner's authenticated GitHub CLI.

## Release identity

Python, npm, Tauri, Rust, lockfiles, citation metadata, the desktop footer, and release tests use
`1.18.0`. The serialized scientific protocols remain compatible with 1.17.0: Model IR 3.0,
expression AST 1.2, `.mlab` 2.0, experiment state 6, Run 1.1, Artifact 1.0, and desktop sidecar
protocol 8.
