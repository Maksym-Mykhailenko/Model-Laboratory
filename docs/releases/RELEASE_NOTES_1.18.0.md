# Model Laboratory 1.18.0

Model Laboratory is a local-first desktop environment for defining mathematical models, running
typed scientific analyses, and preserving the complete experiment as inspectable, reproducible
evidence. Version 1.18.0 is the first packaged Windows desktop release and completes the release-
readiness work begun in 1.17.

## Download

For most Windows users:

- **[Download the `.exe` installer](https://github.com/Maksym-Mykhailenko/Model-Laboratory/releases/download/v1.18.0/Model.Laboratory_1.18.0_x64-setup.exe)**
- [Download the MSI package](https://github.com/Maksym-Mykhailenko/Model-Laboratory/releases/download/v1.18.0/Model.Laboratory_1.18.0_x64_en-US.msi) if you specifically need MSI deployment
- [SHA-256 checksums](https://github.com/Maksym-Mykhailenko/Model-Laboratory/releases/download/v1.18.0/SHA256SUMS.txt)
- [Installer smoke-test receipt](https://github.com/Maksym-Mykhailenko/Model-Laboratory/releases/download/v1.18.0/INSTALLER_SMOKE_TEST.json)

> [!WARNING]
> These installers are not Authenticode-signed. Windows may display an **Unknown publisher** or
> SmartScreen warning. Verify the downloaded file against `SHA256SUMS.txt` before installation.

## Getting started

1. Install and open Model Laboratory.
2. Choose a bundled example, open a YAML model or `.mlab` experiment, or start with a blank model.
3. Review and validate the model before running an analysis.
4. Inspect the resulting typed artifacts, visualisations, provenance, and workload information.
5. Use the **Experiment** workspace to prepare and save a reproducible `.mlab` bundle.

Opening an experiment validates and inspects it; computation starts only after an explicit run or
reproduction action.

## What Model Laboratory provides

- A native Tauri desktop interface backed by a persistent local Python scientific engine.
- Typed model, run, artifact, view, provenance, and reproduction protocols.
- Content-addressed `.mlab` bundles containing model state, settings, artifacts, environment data,
  and integrity fingerprints.
- Exact, tolerance-based, and stochastic reproduction reports.
- Thirteen official scientific packs covering 34 object kinds and 45 capabilities.
- A bounded local authoring interpreter whose proposals pass through deterministic compilation,
  scientific validation, user review, and explicit acceptance.
- Workload estimation before execution and an explicit distinction between live and committed
  experiment state.

## What is new in 1.18.0

### Safer document lifecycle

- Visible unsaved state and Save/Discard/Cancel protection for replacement and close actions.
- Atomic native saves with conservative cleanup of application-owned stale temporary files.
- Lossless pending-file handling for documents opened through Windows file associations.
- Portable filenames for models recovered from experiments and commit receipts.

### Hardened desktop protocol

- Finite operation-class timeouts and fail-closed sidecar restart after a timeout.
- Strict request and response identity checks, bounded framing, malformed-frame rejection, and
  cumulative response limits.
- Oversized input is drained safely without desynchronising the next request.

### First-run experience

- A guided **Define → Validate → Analyse** introduction.
- Eight bundled examples with explicit blank-model and existing-file paths.
- No automatic scientific computation before the user chooses a starting action.

### Windows distribution

- Verified NSIS `.exe` and MSI installers produced from the same tested source.
- Automated clean install, launch, `.mlab`/`.yaml`/`.yml` association, and uninstall checks for
  both package formats.
- Version/tag consistency checks, SHA-256 manifests, pinned workflow dependencies, and a machine-
  readable installer receipt.

## Verification

The release workflow completed successfully with:

- 453 Python tests;
- 9 frontend contract tests;
- native Rust protocol and document-lifecycle tests;
- 257 reference, bundle, reproduction, expression, protocol, and official-pack checks;
- clean install, launch, file-association, and uninstall smoke tests for both Windows installers.

The serialized protocols remain compatible with 1.17.0: Model IR 3.0, expression AST 1.2,
`.mlab` 2.0, experiment state 6, Run 1.1, Artifact 1.0, and desktop sidecar protocol 8.

## Feedback

Report reproducible defects or usability problems through
[GitHub Issues](https://github.com/Maksym-Mykhailenko/Model-Laboratory/issues). Please follow
[`SECURITY.md`](https://github.com/Maksym-Mykhailenko/Model-Laboratory/blob/main/SECURITY.md)
for vulnerabilities rather than opening a public issue.

[Full changelog](https://github.com/Maksym-Mykhailenko/Model-Laboratory/compare/v1.17.0...v1.18.0)
