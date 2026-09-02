# Model Laboratory 1.12.1 release notes

Version 1.12.1 hardens the local interpreter training and promotion boundary. It does not add or
invent any empirical model result. The prior Stage-5 baseline report is deliberately invalid under
the new benchmark/scorer identity and must be rerun on the untouched frozen model.

## Security and release-gate hardening

### Training reports are no longer self-authorising

`validate_training_report()` now requires the frozen base-model directory and independently
reconstructs release-critical evidence instead of trusting a self-hashed JSON inventory. It checks
the exact frozen QLoRA config and seed, current Stage-7 implementation identity, PEFT LoRA adapter
configuration and every reported adapter byte, exact frozen base snapshot, Stage-6 manifest and
review queue, genuine review completion/counts, held-out benchmark exclusion, untouched baseline
validity, and the full trainer step/epoch contract. Minimal forged training reports are rejected.

### Experimental candidates cannot become production

The training lifecycle state is carried into export receipts and candidate evaluation reports.
Production export rejects `candidate_adapter_experimental`; evaluation policy contains an explicit
non-experimental training-state gate; and the final promotion boundary reopens the original
training evidence with `require_promotable=True` and requires exactly
`candidate_adapter_unpromoted`. Promotion also requires a matching previously evaluated registry
entry.

### Full candidate lifecycle registry

Registry schema 1.1 records `experimental`, `trained`, `evaluated`, and `approved` states. Training
and evaluation commands update the registry automatically. Evaluated records bind the training
report, export receipt, candidate evaluation report, candidate tag/ID, GGUF digest, observed Ollama
digest, and promotion eligibility. Only approved records may become active. The desktop status
surface now exposes the active candidate lifecycle plus its training-report and evaluation-report
SHA-256 identities.

## Held-out evaluation hardening

The benchmark advances to `verification/interpreter_baseline_v1.2.json`. Each of its eight
clarification cases now includes one held-out user answer and an expected proposal. The scorer runs
a second inference phase through the normal `clarification_history` path and reports separate
clarification-question, clarification-continuation, and end-to-end clarification accuracy.

A new `context_negotiation_first_action_accuracy` metric isolates the ability to request hidden
context correctly. Both it and clarification continuation are core no-regression promotion gates,
so improvements elsewhere cannot compensate for regressions in either behaviour.

## Compatibility and verification

The Stage-5 baseline protocol/report advance to 1.4, export receipt/plan to 1.1, candidate report
and promotion policy advance accordingly, and the interpreter model registry advances to 1.1.
The application version is 1.12.1 across Python, npm, Tauri, and Cargo metadata.

All 299 Python tests passed in bounded file groups, the seven frontend Node tests passed, Python
compile checks passed, and both frontend modules passed `node --check`. This environment has no
Rust toolchain, Ollama runtime, CUDA stack, or frozen model weights, so native compilation and live
baseline/training/export/evaluation remain target-host work rather than simulated evidence.
