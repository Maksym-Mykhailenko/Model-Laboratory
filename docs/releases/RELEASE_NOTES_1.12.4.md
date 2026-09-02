# Model Laboratory 1.12.4

Version 1.12.4 closes the remaining non-empirical AI-interpreter release-contract gaps. It does
not include a trained adapter or claim baseline/candidate accuracy; those measurements still
belong to the subsequent empirical campaign.

## Exact local artifact identity

- The frozen Qwen lock is now `base_model_lock_v1.1.json` and pins the complete Ollama registry
  manifest, config, and every layer digest/media type/size.
- Rust production inference and Python baseline/candidate evaluation hash the exact local manifest,
  every referenced blob, and the exact GGUF bytes before use.
- Current provider records include `artifact_evidence`; tag names and digest prefixes are not
  accepted as current identity proof.
- Approved candidates are tied to their registry-entry hash, complete evaluated manifest digest,
  and exported GGUF digest.

## Complete clarification lineage

- Proposal 2.3 and acceptance 1.4 carry a checksum-linked clarification-turn chain.
- Each question-producing turn binds its conversation UUID, parent, context, raw output, question,
  and full provider/artifact evidence.
- A later turn must use the same verified artifact. Production profiles may change only through
  their recorded generation options; deterministic evaluator profiles remain valid across turns.
- Historical proposal/provider documents remain readable, but must be regenerated before they can
  become a new current acceptance.

## Reproducible QLoRA execution

- QLoRA config 1.2 binds `interpreter_training_environment_v1.0.json`.
- Training-critical packages are exact-version pinned and checked for both installation and import.
- CUDA, minimum VRAM, cuBLAS workspace configuration, deterministic PyTorch/Transformers settings,
  TF32/cuDNN behaviour, loader workers, seed, and full trainer-step contract are enforced.
- Training report 1.2 records the lock, observed Python/package inventory, hardware/backend facts,
  implementation identity, and all adapter artifacts; promotion revalidates the evidence.

## Promotion semantics

- Candidate report/policy and model registry are version 1.2.
- Development evaluations are repeatable but never promotable.
- A promotion evaluation requires an explicit campaign UUID, uses the sequestered default benchmark,
  and consumes that benchmark identity once.
- Eligibility requires no aggregate or family passed-case regression, explicit absolute floors,
  non-experimental training, and at least one additional correct case in a predeclared scientific
  benefit metric.

## Verification boundary

The Python suite and frontend unit suite cover the new identity, dialogue, deterministic-training,
candidate-policy, registry, bundle-provenance, and historical-read paths. Rust compilation, live
Ollama verification, CUDA training, and real candidate evaluation remain target-host checks and are
not inferred from fixtures.
