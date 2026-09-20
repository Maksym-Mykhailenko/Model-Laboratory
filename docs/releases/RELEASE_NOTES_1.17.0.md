# Model Laboratory 1.17.0

Version 1.17.0 establishes the openly licensed, repository-oriented source distribution of Model
Laboratory. It carries forward the validated scientific engine, experiment protocols, official
packs, and interpreter assets from the 1.16.1 implementation while giving the public release a
single consistent version and documentation structure.

## Release identity

- Python package, npm package, Tauri configuration, Rust package, lockfiles, desktop footer, and
  version-contract tests use `1.17.0`.
- Historical release notes, immutable corpus manifests, and versioned benchmark identities retain
  their original version references.
- Model IR 3.0, expression AST 1.2, `.mlab` 2.0, experiment state 6, Run 1.1, and Artifact 1.0 remain
  the current serialized protocol versions.

## Open-source distribution

- Original project material is distributed under Apache-2.0.
- `LICENSE`, `NOTICE`, and `LICENSING.md` define the project terms and scope.
- `THIRD_PARTY_LICENSES.md` records principal third-party components and distribution considerations.
- `GENERATIVE_AI.md` discloses the human-directed, AI-assisted development workflow and subsequent
  provenance practice.
- `CITATION.cff` supplies machine-readable software citation metadata.

## Documentation

- The root README is a concise project introduction, architecture summary, development guide, and
  documentation map.
- Detailed architecture, science, AI, compatibility, verification, and historical release material
  is organised under `docs/`.
- `CONTRIBUTING.md` defines the development, verification, generated-artifact, licensing, and
  generative-AI contribution practices.
- `SECURITY.md` records supported releases, reporting guidance, trust boundaries, and release
  handling.

## Generated-artifact composition

- The current 6,300-record corpus v1.5, QLoRA configuration v1.7, environment lock v1.2, 150-case
  benchmark v1.6, current reports, and referenced benchmark-construction inputs remain in the main
  source release.
- Tests now obtain generic training and clarification fixtures from current corpus v1.5.
- A dormant corpus-v1.2 migration helper was removed from the fully regenerative v1.5 generator.
- Unreferenced historical corpora, QLoRA configurations, environment locks, benchmark v1.1, and the
  corpus-v1.4 validation report are preserved in the companion historical-interpreter-artifacts
  archive with original relative paths and file-level SHA-256 values.
- `model-laboratory-v1.17.0-historical-interpreter-artifacts.zip` has archive-level SHA-256
  `c2acfa074ff2d9cd8f3e184374b7132b77665aa30f72346f2c194df2569ab2cf`.

## Verification

The release retains the established Python, frontend, reference, bundle, reproduction,
expression-AST, Model Graph/Run, official-pack, interpreter-asset, and corpus-replay verification
contracts. The final 1.17.0 results are recorded in `docs/verification/desktop.md` and the generated
reports under `verification/`.
