# Model Laboratory documentation

This directory contains the detailed technical and historical documentation for Model Laboratory.
The repository root README provides the short project introduction and development commands.

## Architecture

- [Desktop architecture](architecture/desktop.md) describes the Tauri host, Python sidecar,
  process lifecycle, IPC boundary, caching, and packaging model.
- [Compatibility and migration](compatibility/migration-notes.md) records the supported Model IR,
  expression AST, experiment-state, and `.mlab` migration paths.

## Scientific system

- [Official scientific packs](science/official-packs.md) is the authoritative catalogue of pack
  manifests, object kinds, capabilities, settings, artifacts, methods, and declared boundaries.

## AI authoring system

- [Protocol](ai/protocol.md) defines interpreter requests, outputs, clarification, compilation,
  acceptance, and provenance.
- [Boundary](ai/boundary.md) defines the local trust and execution boundary.
- [Training corpus](ai/training.md) documents corpus generation, review, and materialisation.
- [Baseline](ai/baseline.md) documents the untouched-model evaluation contract.
- [Fine-tuning](ai/fine-tuning.md) describes the QLoRA, candidate evaluation, registry, and
  promotion lifecycle.

## Verification and generated artifacts

- [Desktop and scientific verification](verification/desktop.md) summarises the verified source
  state and target-environment checks.
- [Generated-artifact policy](generated-artifacts.md) distinguishes current release inputs from
  companion historical artifacts.
- Machine-readable and generated verification results remain under `verification/`.

## Releases

- [Model Laboratory 1.17.0](releases/RELEASE_NOTES_1.17.0.md)
- Earlier release notes and the 1.16.1 interpreter update are retained in [releases](releases/).

## Project policy

- [Contributing](../CONTRIBUTING.md)
- [Security](../SECURITY.md)
- [Generative-AI development disclosure](../GENERATIVE_AI.md)
- [Licensing](../LICENSING.md)
