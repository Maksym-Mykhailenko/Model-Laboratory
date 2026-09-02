# Version 1.17.0 desktop and scientific verification

Verification date: 2026-09-02 UTC

## Completed checks

| Layer | Result |
|---|---|
| Python regression suite | 444/444 passed |
| Frontend contract | 8/8 passed; `node --check` passed for `app.js` and `core.mjs` |
| Reference verification | 49/49 passed |
| Bundle verification | 37/37 passed |
| Reproduction verification | 51/51 passed |
| Expression AST | 33/33 passed |
| Model Graph / Run protocol | 33/33 passed |
| Official scientific packs | 54/54 passed; 13 manifests, 34 kinds, 45 capabilities |
| Interpreter benchmark assets | 150 cases validated; live frozen-Qwen run not performed |
| Training corpus | 6,300/6,300 deep compiler/context replay passed |
| Human review queue | 725/725 rows structurally bound; all decisions still pending |
| QLoRA preflight | `BLOCKED` truthfully on review, live baseline, environment receipt, pinned ML dependencies, and CUDA |

## Corrected boundaries

The desktop and compiler now reject unsupported scientific composition before it can be mistaken
for a working model. Independent schema availability does not imply an SDE solver, neural training,
a specialised CNN/GNN/RNN, a neural controller, structural optimisation, solver-coupled fitting,
or mesh/FEM PDE solving. Supported neighbouring requests—deterministic ODEs, fixed explicitly
weighted dense networks, uniform-grid PDEs, static truss analysis, and explicit nonlinear residual
fitting—remain available.

Committed experiment receipts now have a complete safe-open path. Opening a receipt validates its
immutable envelope and embedded experiment state, recompiles and checks the model identity, restores
parameters and numerical controls, and marks the restored live revision as matching the commit. It
does not execute reproduction or analysis.

## Runtime-dependent checks

This verification environment has no usable Rust toolchain, Ollama runtime with the frozen GGUF,
pinned optional training stack, frozen base-model snapshot, or CUDA GPU. Consequently it does not
claim a Rust/Tauri compile, packaged installer, live 150-case Qwen score, adapter training/export,
or candidate promotion. Run the native verification and build scripts on each target platform before
distributing an installer; run the live baseline, review workflow, environment freeze, and QLoRA
pipeline only on the authorised empirical workstation.
