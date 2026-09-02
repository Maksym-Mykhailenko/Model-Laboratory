# Model Laboratory 1.12.5

Version 1.12.5 closes the two remaining pre-empirical interpreter-pipeline gaps. It does not
include a trained adapter, private promotion benchmark, or fabricated empirical result.

## Development and promotion are separate protocols

- Candidate evaluation has no implicit benchmark or untouched-base-report default.
- The shipped 80-case set is explicitly public, repeatable development evidence and can never
  authorize promotion.
- A promotion benchmark is an external private artifact and is not included in the source archive.
- `seal_interpreter_promotion.py` binds the exact benchmark, its matching untouched-base report,
  current evaluator, campaign UUID, review provenance, export receipt, training-report identity,
  and quantized candidate GGUF.
- A checksum-bound consumption ledger uses cross-process locking. An exact reserved campaign may
  resume from its checkpoint; a completed benchmark or a benchmark/campaign reserved to another
  candidate cannot be reused.
- The report is atomically persisted before the ledger is marked complete; retrying the same
  finalization after interruption is idempotent.
- Candidate report 1.3 and registry 1.3 retain the seal, candidate freeze, reservation, benchmark,
  and evaluation-purpose identities. Python registration, final promotion, and the native Rust
  selector all validate the new registry contract.

## Immutable QLoRA environment

- QLoRA config 1.3 binds environment policy 1.1 and requires a pre-training receipt.
- The policy fixes CPython 3.12.10, `PYTHONHASHSEED=0`, direct ML versions, deterministic CUDA
  settings, minimum VRAM, and supported platform architecture.
- `freeze_interpreter_training_environment.py` streams every hash-declared installed wheel payload
  and records the complete transitive distribution inventory, wheel RECORD identities, Python
  executable, OS/platform, GPU, driver, CUDA, cuDNN, and deterministic process state.
- Preflight and training reject a missing receipt or any mismatch. Training report 1.3 binds the
  receipt and must prove that it predates training; later release validation independently reopens
  the receipt and checks the complete reported package inventory.
- The launcher sets the Python hash seed by re-executing before importing project or ML modules;
  setting it too late inside the running interpreter is no longer mistaken for determinism.

## Verification boundary

The source suite exercises the new environment-receipt and one-shot-ledger contracts. Live Ollama,
CUDA training/export, and native Tauri checks remain target-machine work and are not claimed here.
