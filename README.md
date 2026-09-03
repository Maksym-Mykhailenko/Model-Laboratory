# Model Laboratory

Model Laboratory 1.17.0 is a native desktop environment for creating, analysing, preserving,
and reproducing mathematical experiments. It combines a Tauri 2 desktop host, a persistent Python
scientific sidecar, typed model and artifact protocols, content-addressed experiment bundles, and
a local authoring interpreter.

The project is intended for students, researchers, lecturers, and developers of research software.
It is local-first and licensed under Apache-2.0.

## Why Model Laboratory

Computational experiments often combine editable source, numerical settings, environment details,
plots, and derived results. Model Laboratory records these elements as one inspectable experiment
state. Its `.mlab` container binds models, run plans, artifacts, provenance, environment data, and
checksums so that opening, reviewing, and reproducing an experiment are separate explicit actions.

The local interpreter follows the same boundary. It proposes typed Model Transactions; the
deterministic compiler validates those proposals before the scientific engine can execute them.

## Highlights

- Canonical Model IR 3.0 and expression AST 1.2
- Typed scalar, vector, matrix, and extensible Model Graph objects
- Namespaced Run, Artifact, View, renderer, and comparator protocols
- Content-addressed `.mlab` 2.0 experiment bundles
- Exact, tolerance-based, and stochastic reproduction reports
- Immutable experiment commits with checksum-bound state restoration
- Local Qwen authoring with bounded context and explicit acceptance provenance
- Workload estimation and capability validation before execution
- Persistent Python sidecar with a native Tauri desktop interface
- Thirteen official scientific capability packs

## Scientific catalogue

The initial catalogue covers multidimensional mathematics and units; probability and stochastic
processes; graphs and networks; generative models and finite POMDPs; dynamical systems and control;
spatial fields and PDEs; geometry and meshes; mechanics and structures; statistical inference;
optimisation and inverse problems; electrical and electromagnetic systems; chemical and biological
systems; and machine learning and computational intelligence.

The exact object kinds, capability versions, assumptions, and method boundaries are documented in
[Official scientific packs](docs/science/official-packs.md).

## Reproducibility and trust boundary

Model Laboratory separates four document layers:

1. The **Model Graph** contains typed objects, relationships, assumptions, and asset references.
2. A **Run Record** binds one capability, its settings, workload, backend identity, and outputs.
3. A **Scientific Artifact** stores immutable renderer-independent results.
4. A **View** maps an artifact to a versioned renderer and display configuration.

Opening an experiment performs integrity and schema inspection. Scientific execution begins only
through an explicit run or reproduction action. Experiment files name required capabilities but do
not carry executable extension code.

## Current release state

Version 1.17.0 is the canonical source release for the initial architecture and thirteen-pack
roadmap. The verified catalogue contains 13 manifests, 34 object kinds, and 45 capabilities. The
release includes 444 Python regression tests, frontend contract tests, analytical reference checks,
bundle and reproduction checks, protocol checks, and official-pack verification.

The interpreter assets comprise a checksum-bound 150-case development benchmark and a
6,300-record compiler-replayed corpus. Its adapter lifecycle is at the review and environment-
capture phase, with a stratified 725-record human-review queue and versioned QLoRA configuration.

Generated artifacts required by the current code and tests remain in the main source release.
Earlier unreferenced interpreter artifacts are preserved in the companion historical archive; see
[Generated-artifact policy](docs/generated-artifacts.md).

## Development setup

Requirements:

- Python 3.11 or later
- Node.js 20 or later
- Rust and the Tauri platform prerequisites
- MSVC Rust host for a Windows native build
- Ollama only for optional local-interpreter execution

```bash
python -m pip install -r constraints-tested.txt
python -m pip install -r requirements-dev.txt
npm install
npm run desktop:dev
```

Run the principal checks:

```bash
python -m pytest -q
npm test
python verification/run_reference_verification.py
python verification/run_bundle_verification.py
python verification/run_reproduction_verification.py
python verification/run_expression_ast_verification.py
python verification/run_model_graph_protocol_verification.py
python verification/run_official_packs_verification.py
```

Build a Windows release:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
```

The build script verifies its prerequisites, packages the Python sidecar, generates an immutable
build-identity manifest, and invokes the Tauri release build.

## Documentation

Start with the [documentation index](docs/README.md). Principal references include:

- [Desktop architecture](docs/architecture/desktop.md)
- [Official scientific packs](docs/science/official-packs.md)
- [AI interpreter protocol](docs/ai/protocol.md)
- [AI interpreter boundary](docs/ai/boundary.md)
- [Training corpus](docs/ai/training.md)
- [Baseline evaluation](docs/ai/baseline.md)
- [Fine-tuning and candidate lifecycle](docs/ai/fine-tuning.md)
- [Compatibility and migration](docs/compatibility/migration-notes.md)
- [Verification state](docs/verification/desktop.md)
- [1.17.0 release notes](docs/releases/RELEASE_NOTES_1.17.0.md)

## Contributing, security, and citation

Contribution guidance is in [CONTRIBUTING.md](CONTRIBUTING.md). Security reports follow
[SECURITY.md](SECURITY.md). Citation metadata is provided in [CITATION.cff](CITATION.cff).

## Licence

Unless a file states otherwise, original Model Laboratory source code, tests, documentation,
schemas, examples, generated corpora, benchmark fixtures, and verification material are licensed
under the Apache License, Version 2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[LICENSING.md](LICENSING.md). Third-party components retain their respective terms as recorded in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) and their embedded notices.

Models, datasets, attachments, and `.mlab` experiments created by users retain their own terms.
