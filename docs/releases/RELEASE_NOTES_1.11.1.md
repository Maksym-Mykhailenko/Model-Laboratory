# Model Laboratory 1.11.1 release notes

Version 1.11.1 merges the strongest parts of 1.10.1 and 1.11.0. It retains the new Stage-7
QLoRA/review/download/provenance tooling and lower-memory 32K/4K Stage-5 evaluation profile,
while restoring the Stage-5 safety and resumability guarantees that regressed in 1.11.0.

## Restored from 1.10.1

- invalid Model Graph kinds and other model-construction errors are converted into ordinary
  interpreter proposal failures rather than raw campaign-crashing exceptions;
- per-case Stage-5 results are atomically checkpointed;
- rerunning the same campaign resumes only after checkpoint benchmark/contract/source/runtime
  identity verification;
- regression tests cover both failure containment and checkpoint resume.

## Retained from 1.11.0

- deterministic 32,768-context / 4,096-output Stage-5 profile with progress and ETA;
- QLoRA NF4/double-quantisation, all-linear rank-16 adapter configuration;
- human-review workflow and correction validation;
- pinned Hugging Face snapshot downloader/receipt;
- CUDA/memory/output-directory preflight;
- lazy completion-only training dataset and candidate provenance;
- explicit experimental/non-promotion boundary.

## Strengthened beyond both inputs

- Stage-7 baseline gate verifies the exact frozen 80-case benchmark and current scientific
  implementation identity, requires strict frozen Ollama runtime identity, checks raw-response
  hashes/JSON, replays all 80 recorded inference traces through the deterministic scorer, and
  recomputes aggregate metrics before accepting the baseline;
- QLoRA contract pins the Stage-6 corpus content root and exact train/validation SHA-256 values;
- normal preflight/training deep-replays all 5,000 corpus records; shallow corpus inspection is
  diagnostic-only and deliberately blocked;
- only the exact official QLoRA config hash can produce a normal candidate; custom configs are
  always experimental;
- frozen snapshot validation requires an exact member set, receipt hashes, locked tokenizer and
  three weight-shard hashes, and Qwen/generation/tokenizer/safetensors-index semantic checks;
- optional ML dependencies are version-checked and imported, not merely detected;
- optional Stage-7 dependencies are documented in third-party licensing.

No trained adapter is included. The 400 review-queue entries remain pending human review, and a
full empirical untouched-model Stage-5 report must still be produced on the target workstation
before a normal training candidate is permitted.
