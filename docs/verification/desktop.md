# Version 1.19.2 desktop and scientific verification

Verification date: 2026-10-06 UTC

## Completed checks

| Layer | Result |
|---|---|
| Python regression suite | 466/466 passed |
| Frontend contract | 9/9 passed; `node --check` passed for `app.js` and `core.mjs` |
| Reference verification | 49/49 passed |
| Bundle verification | 37/37 passed |
| Reproduction verification | 51/51 passed |
| Expression AST | 33/33 passed |
| Model Graph / Run protocol | 33/33 passed |
| Official scientific packs | 57/57 passed; 14 manifests, 35 kinds, 46 capabilities |
| Frozen flagship release case | `FROZEN`; current 1.19.2 reference; `EXACT REPRODUCTION`; all five asset checksums pass |
| GLib iterator backport | 3/3 release-optimised Rust regression tests passed with Rust 1.99.0 |
| SBOM | CycloneDX 1.6 schema and dependency references pass; 550 resolved source/build components |
| Published 1.19.1 checksum repair | Both installers independently downloaded and hashed; corrected manifest passes `sha256sum -c` |
| Interpreter benchmark assets | 150 cases validated; live frozen-Qwen run not performed |
| Training corpus | 6,300/6,300 deep compiler/context replay passed |
| Human review queue | 725/725 rows structurally bound; all decisions still pending |
| QLoRA preflight | `BLOCKED` truthfully on review, live baseline, environment receipt, pinned ML dependencies, and CUDA |

## Corrected boundaries

The Windows sidecar now retains distribution metadata for every scientific dependency recorded in
experiment environments and fails its packaging self-test if any version resolves as `not
installed`. Composed-analysis numerical reproduction compares the complete recipe semantics and
aligned numerical values while excluding only the value hashes derived from those same values.
Strict artifact identity still covers the hashes; platform-level eigensolver rounding can now
correctly produce `NUMERICALLY REPRODUCED` when it falls within the pre-frozen tolerances.

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

The Linux verification environment uses Python 3.12.14 and the tested scientific dependency pins.
The current release's source/build SBOM records that environment. Cargo resolves the patched
dependency graph with `--locked`, and the affected GLib iterator operations pass optimised Rust
tests. The new release asset pipeline stages the full case archive and exact download basenames;
its Python and PowerShell syntax and checksum handling have been checked locally.

The 1.19.2 Windows installers and new cross-platform CI have not been run in this local session.
The publication script waits for those checks and verifies all downloaded draft assets before
publishing. The existing 1.19.1 installers were downloaded and checked without changing their
contents. The environment has no frozen Qwen runtime, pinned optional training stack, frozen
base-model snapshot or CUDA GPU, so no live 150-case Qwen score, adapter training/export or
candidate promotion is claimed. QLoRA preflight remains `BLOCKED`, with separately bound,
current-version deep corpus validation reporting `PASS`.
