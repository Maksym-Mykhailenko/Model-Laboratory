# Release-readiness implementation

This source package contains the release-critical work completed for version 1.18.0 after the
1.17.0 source review.

## Implemented

- Safe model lifecycle: visible dirty state; Save and Save As; Save/Discard/Cancel guards on New,
  Open, Example, associated-file open, committed-state open, and application close; and atomic
  same-directory native replacement after a flushed temporary write.
- Sidecar protocol 8: raw stdout framing across arbitrary chunks, cumulative response limits,
  strict positive request IDs, response-ID/status validation, malformed/empty/unsolicited frame
  rejection, oversized-input draining, bounded per-operation timeouts, and fail-closed sidecar restart.
- Lossless external-open queueing: associated files remain pending until the frontend successfully
  opens and acknowledges them; cancelling an unsaved-changes prompt exposes a retry action.
- Safer filesystem handling: generated YAML filenames are portable, and stale application-owned
  atomic-save temporary files are removed conservatively on a later save to the same target.
- First-run experience: a guided Define → Validate → Analyse introduction, reusable example
  gallery, eight safe-ID bundled examples, explicit blank/open paths, and no automatic computation
  before the user selects a starting action.
- Windows distribution: verified NSIS and MSI builds, mandatory Authenticode signing for version
  tags, SHA-256 manifests, clean-install/launch/association/uninstall smoke tests, pinned GitHub
  Actions dependencies, strict tag/version consistency, and draft GitHub releases.

## Verification completed in the implementation environment

- `451` Python tests passed.
- `9` frontend tests passed, plus JavaScript syntax validation.
- Python bytecode compilation, Tauri JSON parsing, workflow YAML parsing, and unique HTML-ID checks
  passed.
- Six scientific verification suites passed (`257` checks), and the 6,300-record interpreter
  corpus passed deep compiler/context replay.

The implementation environment did not include Rust or Windows, so native Rust compilation and
installer production are delegated to the included Windows workflow. Before publishing, run that
workflow with the Authenticode secrets configured, install both outputs on a clean Windows machine,
exercise Save/close recovery and file associations, verify both signatures, and compare the
downloaded files with `SHA256SUMS.txt`.
