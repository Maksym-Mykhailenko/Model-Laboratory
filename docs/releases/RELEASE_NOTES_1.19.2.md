# Model Laboratory 1.19.2

Version 1.19.2 completes the Gaussian hierarchy flagship case and corrects release packaging.

## Frozen scientific case

The six-level Gaussian experiment has been rerun with 1.19.2 and frozen through the standard
canonical-review API. Its bundle contains a validated `metadata/authoring.json` and opens as
**Author-approved**. The scientific construction, precision profiles and numerical tolerances are
unchanged. The reference JSON identifies the new bundle, state, artifact and frozen review.

The complete case page, cited paper, YAML model, figure, reference JSON, frozen bundle and
checksum manifest are included in the tagged source and a separate release case archive.
Release verification rejects a draft bundle, stale version, missing asset or checksum mismatch.

Scientific source: Mykhailenko, M. (2026). *A scale-free measure of relaxation anisotropy in
precision-weighted variational inference*. SSRN. https://doi.org/10.2139/ssrn.5853487 (Section 4).

## Release corrections

- Windows installer filenames are normalised before hashing and upload. `SHA256SUMS.txt` uses
  the exact downloadable filenames and can be checked with `sha256sum -c SHA256SUMS.txt`.
- Contribution guidance and historical release notes no longer refer to the removed disclosure
  document.
- The security policy identifies the maintained 1.19.x release line.
- Development and CI use pytest 9.0.3, including the fix for CVE-2025-71176.
- The GTK3-compatible GLib dependency includes the upstream mutable-pointer fix for
  RUSTSEC-2024-0429. An optimised Linux iterator test gates this backport.
- Workflow tokens default to read-only repository access; release upload permission is scoped to
  a separate publication job.
- Each binary build emits an exact-environment CycloneDX SBOM and a complete case archive.
  The checksum manifest covers the final staged downloads.
- Interpreter validation and preflight reports have been regenerated with the current version.
  The preflight remains blocked where human review, model baselines and training hardware are absent.

## Compatibility

No persisted protocol versions change. Model IR remains 3.0, expression AST 1.2, `.mlab` 2.0,
experiment state 6, Run 1.1, Artifact 1.0 and sidecar protocol 8. Windows installers remain
unsigned, consistent with the existing distribution policy.
