# Contributing to Model Laboratory

Contributions are welcome across the scientific engine, desktop application, protocols,
documentation, verification suites, and capability packs.

## Development environment

Use Python 3.11 or later and Node.js 20 or later. Native desktop work also requires Rust and the
Tauri prerequisites for the target platform.

```bash
python -m pip install -r constraints-tested.txt
python -m pip install -r requirements-dev.txt
npm install
```

Run the application with `npm run desktop:dev`.

## Before submitting a change

Run the checks relevant to the files changed. The complete source-level suite is:

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

Changes to canonical formats, numerical behavior, pack schemas, workload estimates, interpreter
contracts, or reproduction semantics should include focused regression tests and independent
reference evidence where applicable. Compatibility changes should also update the migration
documentation.

## Generated artifacts

Current corpus, benchmark, and verification artifacts are checksum-bound. Use the supplied builders
and validators, retain their manifests, and include all affected identity changes in the change
description. The repository policy is documented in
[docs/generated-artifacts.md](docs/generated-artifacts.md).

Do not commit downloaded model weights, local environments, platform build directories, generated
sidecar binaries, candidate adapters, environment receipts, or promotion ledgers.

## Contribution terms

Unless explicitly stated otherwise, contributions intentionally submitted for inclusion are
provided under Apache-2.0, consistent with section 5 of the project licence. Preserve third-party
copyright and attribution notices.

## Change description

A contribution should state:

- the problem and intended behavior;
- the affected protocol, schema, capability, or user workflow;
- compatibility and reproducibility effects;
- tests and verification performed; and
- any generated artifacts or documentation updated.
