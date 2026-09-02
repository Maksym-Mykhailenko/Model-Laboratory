# Model Laboratory migration notes

## Version 1.16.0: learning pack and scientific-semantics corrections

Version 1.16.0 adds four strict Model Graph object kinds and four executable capabilities in the
Machine Learning and Computational Intelligence pack. Model IR remains 3.0, `.mlab` remains 2.0,
experiment state remains 6, Run remains 1.1 and Artifact remains 1.0, so historical documents do
not require a container migration.

Object-kind schemas remain at 1.0 so existing model sources retain their meaning. Packs whose
scientific implementations changed advance to pack 1.1, and each affected capability/artifact
contract advances to 1.1; mathematically unchanged capabilities remain at 1.0. HMM smoothing is
fully scaled and Viterbi treats structural zeros as impossible. Reducible Markov chains expose a
stationary family. Estimation artifacts explicitly distinguish identifiable and rank-deficient
problems. Tensor results preserve labels, physical dimensions and uncertainty. Degenerate
eigenspaces use their projector—not a platform-dependent display basis—for scientific identity.
Graph degree now has conventional multigraph semantics, with unique-neighbour counts separate.

Physical pack inputs with supported declared units are converted to canonical SI before numerical
work. Outputs record canonical and source units. Unsupported unit labels fail validation instead of
acting as presentation-only metadata. Reaction simulation no longer clips concentrations inside the
ODE vector field; only a small, documented post-integration projection is permitted.

Capability settings are now generically validated against their registered schemas before workload
estimation or execution. Defaults are frozen into the Run Record, and unknown/out-of-range settings
are rejected consistently. These are correctness hardenings, not migrations that silently
reinterpret archived artifacts: saved artifacts retain their exact historical payload and identity.
If an archived run requests a corrected 1.0 executor that is not installed, reproduction reports
`UNABLE TO REPRODUCE`; it never executes the 1.1 mathematics under the historical version label.

The deterministic Stage-6 corpus was regenerated from the unchanged seed because interpreter
contexts are checksum-bound to the installed official catalogue. Its 5,000-record size, splits,
review status, held-out exclusion policy and schema remain unchanged; only catalogue-derived
context/output/content identities changed.

## Version 1.15.0: statistics, optimisation, electrical and reaction/biological packs

Version 1.15.0 adds twelve strict Model Graph object kinds and sixteen executable capabilities
across four official packs: Statistical Inference and Data Modelling; Optimisation, Estimation
and Inverse Problems; Electrical, Electronic and Electromagnetic Systems; and Chemical,
Reaction and Biological Systems.

Model IR remains 3.0, `.mlab` remains 2.0, experiment state remains 6, Run remains 1.1 and
Artifact remains 1.0. Existing files therefore require no migration. New documents simply
declare the relevant namespaced kind and pack requirement. Older installations preserve those
objects as opaque and report a missing required pack rather than reinterpreting their contents.

Interpreter authoring selects each new pack and its declared dependency closure into a bounded
registry-derived context; the full source continues to remain in the deterministic engine. All
new scientific outputs use the existing numeric comparator, while solver iteration counts and
messages remain presentation-only and cannot change artifact identity.

## Version 1.14.0: dynamics, fields, geometry and mechanics packs

Version 1.14.0 adds ten strict Model Graph object kinds and fourteen executable capabilities
across four new official packs. Model IR remains 3.0, `.mlab` remains 2.0, experiment state
remains 6 and artifact/run schemas remain unchanged. The extension therefore requires no
reinterpretation of historical mathematics: 1.13 documents continue to compile with the same
kind versions and run contracts.

The interpreter no longer inserts every installed official schema into every scientific
prompt. It selects the requested/current pack and its declared dependency closure, while the
core-only contract remains byte-identical to the v1.12 frozen corpus. This keeps ODE, PDE,
geometry and mechanics authoring under the existing 32 KiB context-package ceiling as the
installed catalogue grows.

New artifacts use the existing numeric comparator and generic reproduction path. Their solver
methods, tolerances, sample coordinates, workload, implementation entry point and installed
backend package versions are frozen by the capability descriptor and Run Record. Renderer
views remain presentation-only and do not change artifact identity.

## Version 1.13.0: first official structured scientific packs

Version 1.13.0 adds eight locally registered object kinds and eleven capabilities across
multidimensional quantities, probability/Markov simulation, graphs/networks, and generative
inference/decision systems. It does not change Model IR 3.0 or `.mlab` 2.0: extension data was
already namespaced in the Model Graph, and generic run/artifact records already preserve new
results. Opening remains non-executing. A historical bundle is still compiled with its
historical registry, while current sources use the official installed registry.

Stochastic artifacts now select `org.modellab.comparator.stochastic`; this comparator first
checks exact sample-stream identity and otherwise applies only the statistical criteria frozen
inside the saved reference. Deterministic numerical status is not silently substituted for a
statistical claim. Reproduction-report schema 1.2 therefore adds the explicit
`STATISTICALLY REPRODUCED` status; `.mlab` format and experiment-state schemas are unchanged.

## Version 1.12.2: committed operational experiment state

Version 1.12.2 separates the mutable interactive/live model from an immutable operational commit. `ExperimentState` remains the checksummed scientific snapshot and `FrozenExperiment` remains the publication-approval object, but the desktop can now explicitly commit a prepared state into `model-laboratory.committed-experiment/1.0`. The commit embeds the exact state, binds source and canonical Model IR identities, carries its own `commit_sha256`, can reference a parent commit, and remains intact while live inputs diverge.

The sidecar protocol advances to 6 with `commit_experiment_state` and `inspect_committed_experiment`. The frontend exposes independent live/committed status and can save a commit receipt. Future physical execution is required to validate a committed receipt and expected commit identity rather than reading live UI state.

## Version 1.12.1: interpreter release-gate hardening

Version 1.12.1 advances the interpreter baseline/scorer, export receipt, and candidate registry
contracts. Existing 1.12.0 empirical baseline/training/export/evaluation evidence is not silently
upgraded: the untouched baseline must be rerun against benchmark v1.2 before a new ordinary
candidate can enter promotion. Registry v1.1 now represents experimental, trained, evaluated, and
approved lifecycle states, and final promotion independently revalidates the original non-
experimental training report.

## Version 1.12.0: complete local-interpreter candidate lifecycle

Version 1.12.0 adds bounded multi-turn clarification and adaptive production inference profiles,
without reducing the accepted instruction or source sizes. The current interpreter contracts are:

| Contract | Version |
|---|---:|
| application | 1.12.0 |
| desktop sidecar protocol | 5 |
| Model IR / Model Graph | 3.0 |
| expression AST | 1.2 |
| experiment state | 6 |
| `.mlab` container | 2.0 |
| interpreter context | 1.2 |
| interpreter output | 1.2 |
| interpreter compiler | 1.3 |
| interpreter proposal | 2.2 |
| interpreter acceptance | 1.3 |
| interpreter baseline contract/report | 1.3 |

Historical interpreter proposal/acceptance records remain readable. New provenance adds a
clarification-history hash/count and provider model role/registry entry identity. The new
Stage-6 v1.1 corpus and QLoRA v1.1 configuration are separate development artifacts; they do
not change `.mlab` mathematics.

Adapter export, held-out candidate evaluation, promotion, and rollback are now explicit,
checksum-bound phases. The desktop uses the frozen base when no registry exists, verifies a
selected approved candidate against the registry, and fails closed if the registry is invalid.
No candidate is silently selected and old frozen-base provenance is not rewritten.

## Version 1.9: scalar application to extensible scientific desktop

Version 1.9 completes the architectural migration that began when the prototype left
Streamlit. The native Tauri desktop remains the host, but scientific state is no longer
defined by fixed `evaluation`, `stationary_points`, and `parameter_sweep` slots.

At that milestone the versioned contracts were:

| Contract | Version |
|---|---:|
| application | 1.11.1 |
| desktop sidecar protocol | 5 |
| Model IR / Model Graph | 3.0 |
| expression AST | 1.2 |
| experiment state | 6 |
| `.mlab` container | 2.0 |
| interpreter context | 1.1 |
| interpreter output / compiler | 1.2 |
| interpreter proposal | 2.1 |
| interpreter acceptance | 1.2 |

The migration introduces:

- stable namespaced typed Model Graph objects, relationships, and structured assets;
- opaque preservation and non-execution of unavailable extension kinds;
- first-class vector and matrix functions;
- namespaced capability packs and typed immutable artifacts;
- generic Run Records, views, comparator routing, and capability sweeps;
- content-addressed artifact and asset namespaces plus bounded desktop chunk handles; and
- schema-aware Model Transactions from the local interpreter.

The existing scalar feature set is retained as the built-in Scalar Calculus pack and legacy
convenience UI. This is a semantic migration, not simply removal of an “exactly one scalar
function” validation guard.

## Version 1.10: Stage-6 corpus construction

Version 1.10 adds the deterministic interpreter training-corpus layer without changing the
scientific Model IR, `.mlab`, expression, or desktop sidecar contracts.  The corpus targets
interpreter output schema 1.2 and is deliberately separate from the Stage-5 held-out benchmark.

The new corpus contract is `model-laboratory-interpreter-training-corpus/1.0`.  Five thousand
synthetic examples are generated from 74 mathematical/authoring archetypes across 14 families.
Archetypes are permanently assigned to train or validation, proposal targets must compile through
the production interpreter/parser/validator, and every record carries both a full content hash and
a learning-content hash that excludes bookkeeping IDs.  The Stage-5 benchmark is used only as a
text/semantic deny-list.

A stratified 400-record review queue is frozen separately and remains explicitly
`pending-human-review`; v1.10 makes no claim that the original roadmap's manual expert-review
requirement has already been performed.  LoRA/QLoRA training remains Stage 7.

## Version 1.11: gated QLoRA candidate training

Version 1.11 does not change Model IR, expression AST, `.mlab`, experiment-state, or sidecar
protocol versions. It adds an offline development pipeline around the existing interpreter
contracts. The Stage-5 deterministic evaluation profile is versioned at 32,768 context tokens
and 4,096 output tokens to avoid reserving the 131,072-token production ceiling for each short
benchmark case.

The new `model-laboratory-interpreter-qlora-config/1.0` contract pins the upstream Qwen commit,
NF4/double-quantised load, all-linear rank-16 adapter, assistant-completion-only loss,
untruncated 16,384-token ceiling, optimiser, seed, and checkpoint policy. An exact downloaded
snapshot receipt, completed 400-item review queue, valid full untouched baseline, optional
CUDA stack, and suitable NVIDIA GPU are explicit gates. Adapters remain unpromoted artifacts;
there is no automatic change to the desktop's frozen baseline tag.


## Version 1.11.1: merged Stage-5 safety and strengthened Stage-7 gates

Version 1.11.1 keeps the useful 1.11 QLoRA pipeline and 32K/4K evaluation profile while restoring
the v1.10.1 Stage-5 failure boundary and atomic checkpoint/resume. Raw `ModelGraphError` values
from invalid interpreter proposals are converted into deterministic validation/proposal failures,
so one bad Qwen answer cannot terminate an 80-case campaign. Completed cases are checkpointed
atomically and can be resumed only when the benchmark, contract, scientific source identity, and
frozen runtime identity still match.

The Stage-7 gate no longer trusts a self-hashed aggregate report. It binds the exact Stage-5
benchmark and scientific implementation identity, verifies strict frozen Ollama runtime evidence,
replays every stored raw inference response through the scorer, and recomputes aggregate metrics.
The official QLoRA contract pins the immutable Stage-6 corpus content root plus exact train and
validation hashes, and normal preflight/training deep-replays all 5,000 records. Custom QLoRA
configuration is automatically experimental. Snapshot verification now requires an exact member
set plus receipt hashes and Qwen/tokenizer/index semantic checks, and optional ML dependencies are
version-checked and imported during preflight.

## Compatibility policy

`.mlab` format version is independent of application version. Version 1.9 loads formats
1.0–1.6 and current 2.0.

| Saved format | Historical identity | Migration rule |
|---|---|---|
| 1.0 | SymPy-backed expression representation | validate stored historical document; recompile source into current owned AST |
| 1.1 | Model IR 2.0 / AST 1.0 | project and verify historical canonical payload |
| 1.2 | Model IR 2.1 / compact decimal AST 1.1 | project and verify historical payload |
| 1.3 | frozen authoring review | retain review evidence |
| 1.4 | local interpreter lineage | retain acceptance 1.0 provenance |
| 1.5 | deterministic edit compiler | retain acceptance/compiler lineage |
| 1.6 | Model IR 2.2 / hyperbolic AST 1.2 | use hyperbolic language only for new function calls |
| 2.0 | Model Graph 3.0 and Run → Artifact | current native representation |

Historical mathematical meaning is version-aware. In particular, `sinh`, `cosh`, and
`tanh` were legal identifiers before they became reserved functions. When an old bundle is
reconstructed, its saved laboratory/AST lineage selects the old registry, so a parameter
named `sinh` remains a parameter. A genuine archived format-1.4 fixture exercises this path.

Migration never silently executes unavailable extension code. Unknown Model Graph kinds
remain visible and opaque. A saved Run requiring an unavailable capability produces
`UNABLE TO REPRODUCE` with the missing identity rather than `NOT REPRODUCED`.

## Canonical identity changes

Expression AST 1.2 and Model IR 3.0 are new schema versions because their accepted payloads
differ from earlier versions. Current real literals remain compact coefficient/exponent
values and are independent of Python's mutable decimal context.

Numerical diagnostic identity is now:

```text
code + severity + sorted structured details
```

`message` remains persisted presentation data but is excluded from scientific artifact
hashes and exact/numerical comparisons. This applies at the generic artifact layer, not only
to legacy scalar result fingerprints.

Run Records now record both the stable capability contract and local implementation/backend
identity. Result identity and environment/backend identity remain separately reported.

## Interpreter migration

The old whole-source protocol remains readable only as historical acceptance provenance.
Current inference receives a bounded registry-generated contract and emits a typed edit
program. Existing YAML is edited through `ruamel.yaml` round-trip nodes; comments, quotes,
collection styles, key order, and untouched indentation are preserved. The compiler no
longer performs `safe_load` → ordinary dictionary → `safe_dump` for an existing source.

Historical provider-provenance schemas remain readable, but from 1.12.3 ordinary native
authoring no longer trusts a mutable Ollama tag by name. The Rust bridge applies the bundled
frozen model lock before inference and refuses a mismatched manifest, extra/missing layer,
modified local blob, non-frozen GGUF, or inconsistent model size. The **Stage-5 evaluation
runner** independently performs the same local artifact checks and additionally binds its
report to the frozen upstream/model lock.

Version 1.12.4 upgrades current proposal/acceptance provenance to 2.3/1.4. New records carry
complete local artifact evidence and a checksum-linked clarification-turn chain. Historical
proposal/provider records remain inspectable, but their weaker tag/digest language cannot be
used to create a new current acceptance without regenerating the proposal.

QLoRA config/report, candidate report/policy, and the local interpreter model registry advance
to 1.2. Old local candidate registries do not silently retain production authority because they
lack the one-shot promotion-purpose/campaign/benchmark evidence. Re-evaluate a candidate under
the 1.12.4 promotion contract or roll back to the exactly verified frozen base.

Version 1.12.5 deliberately changes the not-yet-empirically-used candidate release contracts:
QLoRA config/report become 1.3, candidate evaluation report becomes 1.3, and the local interpreter
registry becomes 1.3. The shipped 80-case benchmark is development-only. Promotion evidence now
requires an external candidate-bound seal and completed consumption-ledger entry; 1.12.4 promotion
reports cannot be upgraded by inventing those missing facts. Because no candidate was promoted in
the supplied 1.12.4 state, the safe migration is to retain the frozen base, re-evaluate any future
candidate under 1.12.5, and create a fresh 1.3 registry entry.

Version 1.12.6 corrects the Stage-7 Python policy before empirical training. QLoRA config/report
advance to 1.4 and the training-environment policy advances to 1.2. Environment creation now
supports CPython `>=3.10,<3.14`; the immutable receipt still binds the exact selected executable,
patch version, installed payloads, platform, and CUDA stack. A 1.12.5 receipt is policy-bound to
the former exact-3.12.10 lock and is not silently reinterpreted. Create a new receipt under the
1.12.6 policy before training.

Version 1.12.7 adds an optional, local-only interpreter reference layer without changing the
frozen Stage-5 output schema or the reviewed Stage-6 corpus. No-attachment contexts remain
schema 1.2 and byte-compatible; attachment contexts use schema 1.3. New compiler, proposal, and
acceptance records use 1.4, 2.4, and 1.5 respectively and bind content-addressed manifests plus
the exact chunks disclosed to inference. Earlier proposal/acceptance records remain readable.
Reference files are ephemeral authoring inputs, not executable bundle members and not silently
reopened. Reattach them after a sidecar restart if a dialogue has not yet produced a proposal.

## Desktop/build migration

The Python sidecar is persistent, the bundled default model is explicit PyInstaller data,
and packaged builds carry a generated immutable build identity. Input changes invalidate
dependent results. Windows preflight validates Python, Node, and the Rust MSVC host before
building.

The Tauri document and sidecar request ceilings now accommodate the complete 128 MiB
portable bundle limit. Binary artifact/asset bytes are exposed after inspection through
bounded one-MiB chunks instead of ordinary analysis responses.
