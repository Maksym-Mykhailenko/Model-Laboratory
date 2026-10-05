# Version 1.19.0 desktop and scientific verification

Verification date: 2026-10-04 UTC

## Completed checks

| Layer | Result |
|---|---|
| Python regression suite | 460/460 passed |
| Frontend contract | 9/9 passed; `node --check` passed for `app.js` and `core.mjs` |
| Reference verification | 49/49 passed |
| Bundle verification | 37/37 passed |
| Reproduction verification | 51/51 passed |
| Expression AST | 33/33 passed |
| Model Graph / Run protocol | 33/33 passed |
| Official scientific packs | 57/57 passed; 14 manifests, 35 kinds, 46 capabilities |
| Interpreter benchmark assets | 150 cases validated; live frozen-Qwen run not performed |
| Training corpus | 6,300/6,300 deep compiler/context replay passed |
| Human review queue | 725/725 rows structurally bound; all decisions still pending |
| QLoRA preflight | `BLOCKED` truthfully on review, live baseline, environment receipt, pinned ML dependencies, and CUDA |

## Corrected boundaries

The desktop now supports bounded declarative composition of installed numerical primitives. The
new campaign verifies content-addressed recipe steps, safe derived expressions, symmetric
generalised eigenspectra, declarative rendering, and exact reproduction of the six-level Gaussian
flagship. Formula output shapes and cumulative stored values are bounded before allocation.

Composition does not imply an uninstalled solver. The compiler still rejects requests for SDEs,
neural training, specialised CNN/GNN/RNN architectures, neural controllers, structural
optimisation, solver-coupled fitting, or mesh/FEM PDE solving. Supported neighbouring requests—
deterministic ODEs, fixed explicitly weighted dense networks, uniform-grid PDEs, static truss
analysis, explicit nonlinear residual fitting, and the new recipe vocabulary—remain available.

Committed experiment receipts now have a complete safe-open path. Opening a receipt validates its
immutable envelope and embedded experiment state, recompiles and checks the model identity, restores
parameters and numerical controls, and marks the restored live revision as matching the commit. It
does not execute reproduction or analysis.

Sidecar requests have finite operation-class timeouts, and protocol failures terminate the process
before a later request starts a clean replacement. Associated documents stay queued until an exact
frontend acknowledgement; cancelling replacement preserves the pending file for retry. Atomic-save
cleanup matches only stale temporary names created for the same target.

## Runtime-dependent checks

This verification environment has no usable Rust toolchain, Ollama runtime with the frozen GGUF,
pinned optional training stack, frozen base-model snapshot, or CUDA GPU. Consequently it does not
claim a Rust/Tauri compile, packaged installer, live 150-case Qwen score, adapter training/export,
or candidate promotion. Run the native verification and build scripts on each target platform before
distributing an installer; run the live baseline, review workflow, environment freeze, and QLoRA
pipeline only on the authorised empirical workstation.
