# Model Laboratory 1.11.0 release notes

## Outcome

This release turns Stage 7 from an informal next step into an executable, provenance-bearing
QLoRA pipeline, while preserving the scientific gates established in Stages 5 and 6. It does
not claim that a model was trained in an environment where the required evidence and hardware
were absent.

## Stage audit

- Stage 5's benchmark, exact frozen-model verification, deterministic compiler/scorer, and raw
  evidence capture are implemented and validated. The included report is still
  `benchmark_validated_model_not_run`; all 80 cases must be run locally against the exact model.
- Stage 6's 5,000-record corpus passes full deterministic and deep compiler replay. Its 400
  selected records remain `pending-human-review`, so the human-review acceptance gate is not
  complete.
- Stage 7 now has strict QLoRA configuration, review and baseline gates, exact upstream snapshot
  receipts, CUDA/memory preflight, completion-only/no-truncation tokenisation, lazy dataset
  loading, candidate training, file fingerprints, and environment/provenance capture.

## Memory correction

The Stage-5 evaluator no longer reserves the production 131,072-token context for every short
benchmark case. Its frozen comparison profile is now 32,768 context tokens and 4,096 output
tokens, and the CLI prints progress and an estimated remaining time. This addresses the severe
RAM pressure observed with the earlier command.

Stage-7 corpus tokens are generated lazily per item. A length-only pass rejects an over-limit
record before model loading, but the full 5,000-record token table is never retained in memory.

## Verification

- Python: 279 passed.
- Frontend: 7 passed.
- Reference verification: 49/49.
- Bundle verification: 37/37.
- Reproduction verification: 51/51.
- Expression AST verification: 33/33.
- Model Graph/Run protocol verification: 33/33.
- Stage-6 deep compiler replay: 5,000/5,000.

Rust/Tauri execution, Ollama inference, and CUDA QLoRA training could not be run in the supplied
container. These are recorded as unavailable/blocked, not passed. See
`DESKTOP_VERIFICATION.md` and `AI_INTERPRETER_FINE_TUNING.md` for the exact workstation commands.
