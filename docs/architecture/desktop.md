# Model Laboratory 1.16.0 desktop architecture

Model Laboratory is a Tauri 2 desktop application. It is not a website or a packaged web
service. The WebView renders the workspace, Rust owns native dialogs and the process
boundary, and one persistent Python sidecar owns all scientific interpretation and
execution.

## Process boundary

```text
Tauri WebView
  ├── editor, controls, review, renderer state
  └── no scientific authority
          │ bounded Tauri commands
          ▼
Rust host
  ├── native open/save and double-click integration
  ├── persistent sidecar lifecycle
  ├── loopback-only Ollama bridge
  └── request/response byte limits
          │ newline-delimited JSON
          ▼
Python sidecar
  ├── strict model parser and Model Graph compiler
  ├── installed capability/comparator/renderer registries
  ├── Run → Artifact execution and reproduction
  ├── .mlab validation and migration
  └── deterministic Model Transaction compiler
```

Python, SymPy, SciPy, and a PyInstaller one-file payload initialise once. Requests are
serialized by the Rust-side mutex. If the protocol fails, the host terminates that process;
the next request starts a clean replacement. Deterministic legacy analysis responses use a
32 MiB, entry-bounded LRU.

## Scientific layers

The architecture deliberately avoids `model.functions[0]` as a universal product schema.

| Layer | Responsibility |
|---|---|
| Model Graph 3.0 | stable typed objects, relationships, expression definitions, assets, assumptions, ambiguities |
| Kind registry | strict schemas and local execution permission for namespaced object kinds |
| Capability registry | applicability, settings schema, workload, backend identity, outputs, compatible renderers |
| Run Record | frozen capability contract, settings, workload, backend/package identity, artifacts, provenance |
| Artifact registry | immutable schema-versioned scientific values and comparator contract |
| Renderer/View registry | artifact-to-view mapping and presentation-only state |
| Experiment/reproduction | generic ordered runs and artifact comparisons |

Unknown extension kinds are portable JSON, opaque, and non-executable. `.mlab` files contain
references to required local packs; they never contain executable plugin code.

## Desktop actions

The sidecar protocol is version 6. Principal actions are:

- `health`, `example_model`, and `inspect_model`;
- legacy convenience actions `analyse_model` and `run_sweep`;
- generic `run_capability`;
- `prepare_run_experiment`, `finalize_experiment`, `inspect_experiment`, and
  `reproduce_experiment`;
- `commit_experiment_state` and `inspect_committed_experiment` for the operational live/committed boundary;
- `read_content_chunk` for bounded binary artifact/asset access; and
- interpreter context, output processing, acceptance, status, pull, request, and cancel
  actions.

`inspect_experiment` always returns `execution_performed: false`. It validates checksums,
schemas, canonical reconstruction, workload, runs, artifacts, views, and environment data.
Only `reproduce_experiment` executes frozen runs.

## Binary content

Portable `.mlab` files are limited to 128 MiB. Base64 process requests have a 192 MiB
ceiling to accommodate that container. After inspection, embedded artifact/asset bytes are
held in a byte-bounded persistent content store. JSON exposes only an ephemeral handle,
member path, size, and digest; `read_content_chunk` returns at most one MiB. This keeps large
renderer payloads out of ordinary analysis JSON and establishes the handle contract needed
for later mesh, image, volume, and field renderers.

## Live versus committed execution state

Interactive state is deliberately not execution authority. The WebView may freely mutate source, parameters, numerical settings, and run plans, and these changes advance a monotonically increasing live input revision. Preparing an experiment creates a checksummed `ExperimentState` for exactly one revision.

An explicit **Commit prepared state** action sends that exact state plus its expected SHA-256 back through the Python boundary. The sidecar reconstructs and semantically validates the model, creates a `model-laboratory.committed-experiment/1.0` envelope, embeds the complete experiment state, binds the canonical Model IR, and computes a separate `commit_sha256`. Commit events may point to `parent_commit_sha256`, providing an auditable state-transition chain.

The desktop retains the committed envelope when the live model changes and marks the UI **Live · diverged**. Live edits never mutate the committed object; the embedded state is stored internally as canonical immutable JSON rather than a mutable nested mapping. A saved commit receipt can therefore be independently validated without running numerical analysis.

This is the required boundary for future physicalisation: external execution must be given a validated committed envelope and expected `commit_sha256`. It must not read current sliders, editor text, interpreter proposals, or other live controls. Publication is orthogonal: the existing `FrozenExperiment` author-review path decides whether an experiment is publishable, not whether it is the currently committed operational state.

## UI state correctness

The frontend maintains a monotonically increasing input revision. Parameter, source, and
numerical-setting changes clear every dependent legacy and generic result. An asynchronous
result commits only if its captured revision still matches. Resetting parameters is treated
as an input change. This prevents controls for one experiment state from coexisting with an
authoritative plot from another.

The **Capabilities** workspace is registry-driven. Scalar tabs are convenience views over
the built-in scalar pack; vector, matrix, optimisation, and future packs do not require new
fixed experiment fields.

## Packaging identity

The sidecar build bundles `models/`, all versioned interpreter prompt/schema assets, and an
immutable generated build-identity manifest. The
source/development and packaged paths therefore use the same `source_tree_sha256` contract
instead of hashing nonexistent files inside a one-file extraction directory.

Windows preflight rejects Python below 3.11, Node below 20, and non-MSVC Rust hosts before
the expensive install/test/build sequence begins.
