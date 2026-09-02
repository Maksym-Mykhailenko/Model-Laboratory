# Generated-artifact policy

Model Laboratory keeps generated artifacts in the main source release when they are current runtime
inputs, current verification results, or construction inputs referenced by maintained scripts.
Historical generated artifacts with no current code or test dependency are distributed separately.

## Current source-release artifacts

The 1.17.0 source release retains:

- `training/interpreter_corpus_v1.5/`, the current 6,300-record corpus and review queue;
- `training/interpreter_qlora_v1.7.json`, the current candidate-training configuration;
- `training/interpreter_training_environment_v1.2.json`, the current environment lock;
- `verification/interpreter_baseline_v1.6.json`, the current 150-case benchmark;
- baseline v1.2, v1.3, and v1.5 construction inputs referenced by the maintained benchmark builders;
- current corpus validation, baseline, preflight, phase, pack, reference, bundle, reproduction,
  expression-AST, and Model Graph/Run verification reports.

## Dependency audit

Before the 1.17.0 split, tests referenced corpus v1.1 for two generic fine-tuning fixtures. Those
tests now exercise the same behavior against current corpus v1.5. A dormant migration helper was
the sole code reference to corpus v1.2 and was removed because the current corpus generator fully
regenerates all 6,300 records. The current code and tests therefore have no dependency on corpus
v1.0, v1.1, v1.2, or v1.4.

Legacy QLoRA configurations v1.0 through v1.6 and environment locks v1.0 and v1.1 are provenance
records rather than current pipeline inputs. Interpreter baseline v1.1 and the corpus-v1.4
validation report likewise have no maintained-script or test dependency.

## Companion archive

These historical files are preserved in
`model-laboratory-v1.17.0-historical-interpreter-artifacts.zip`, with their original project-relative
paths. The archive contains its own explanatory README and a SHA-256 manifest covering every
payload file. Its archive-level SHA-256 is
`c2acfa074ff2d9cd8f3e184374b7132b77665aa30f72346f2c194df2569ab2cf`.

The split changes repository composition only. It does not change the identities or contents of the
current corpus, benchmark, model locks, or verification evidence.
