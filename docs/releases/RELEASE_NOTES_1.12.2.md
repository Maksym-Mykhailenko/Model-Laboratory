# Model Laboratory 1.12.2 release notes

Version 1.12.2 introduces a first-class distinction between the mutable interactive/live model and a committed operational experiment state. This is an architectural prerequisite for eventual physicalisation.

## Committed experiment contract

- Added `model_lab.commit` and schema `model-laboratory.committed-experiment/1.0`.
- A commit embeds the complete checksummed `ExperimentState`, binds the model-source SHA-256 and canonical Model IR SHA-256, records a UTC commit timestamp and optional parent commit, and has its own `commit_sha256`.
- The embedded state is stored internally as canonical JSON text so nested mappings cannot mutate a frozen commit object.
- The envelope declares `live_state_authority: false` and `external_execution_requires_commit_sha256: true`.
- Committing is intentionally separate from publication approval. `FrozenExperiment` remains the author-review/publication contract.

## Desktop boundary

- Sidecar protocol advanced from 5 to 6.
- Added `commit_experiment_state`, which requires the exact prepared `state_sha256` and rejects stale state identity.
- Added `inspect_committed_experiment`, which validates a saved commit without executing numerical work.
- The frontend retains a committed state while live source, parameters, numerical settings, or run plans continue changing. It reports `Live · matches commit` or `Live · diverged`.
- Capability-plan changes now advance the same live input revision used by other experiment inputs.
- Commit receipts can be saved as JSON for later audit/physical-execution integration.

## Physicalisation rule

Future actuators, robots, instruments, or other external execution layers must consume a validated committed-state envelope and expected `commit_sha256`. Direct use of live editor/slider values is outside the execution contract.

## Verification

The new committed-state core, sidecar commands, immutable nested-state behavior, stale-state rejection, frontend revision relationship, and desktop controls are covered by regression tests. Existing interpreter release-gate hardening from 1.12.1 is unchanged.
