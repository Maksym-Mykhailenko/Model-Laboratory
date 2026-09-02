# Interpreter Stage 7 and candidate release pipeline

Model Laboratory 1.17.0 implements the full fail-closed path from a reviewed Stage-6 corpus to a
locally selectable interpreter candidate. It does not bundle a trained adapter or invent the
empirical work: baseline inference, human review, CUDA training, export, and candidate evaluation
must be run on the authorised workstation.

## Gate state in this archive

| Gate | Implementation | Current evidence |
|---|---|---|
| untouched baseline | exact 150-case development runner, scorer, raw-evidence replay, frozen Ollama artifact lock | validation only; live run pending |
| corpus | 6,300 v1.5 records, including capability-boundary negatives and all 13 official packs with creation/edit/paired clarification turns and deep replay | built; 725 human decisions pending |
| QLoRA | pinned NF4/all-linear rank-16 configuration, pre-training full-environment receipt, deterministic CUDA policy, no truncation, CUDA and snapshot gates | implemented; not run |
| export | adapter report verification, base verification, PEFT merge, GGUF conversion, Q4_K_M quantisation, distinct Ollama tag and receipt | implemented; not run |
| candidate evaluation | explicit matching benchmark/base report, checkpoint/resume, raw-output replay, non-regression policy | implemented; not run |
| promotion/rollback | explicit report/registry hash confirmation, atomic registry, digest-checked desktop selection | implemented; no candidate promoted |

## Train a candidate

Use an isolated training environment. Normal training requires an NVIDIA GPU with at least
12 GiB VRAM; 16 GiB VRAM, 32 GiB system RAM, and substantial SSD space are recommended.

```powershell
py -3.12 -m venv .venv-training
.\.venv-training\Scripts\Activate.ps1
python -m pip install -r requirements-training.txt
python scripts\freeze_interpreter_training_environment.py
python scripts\run_interpreter_baseline.py
python scripts\run_interpreter_qlora.py
# Low-memory equivalent for the deep corpus gate:
python scripts\run_interpreter_qlora.py --corpus-workers 1
python scripts\download_interpreter_base_model.py D:\ModelLab\qwen3-base
python scripts\run_interpreter_qlora.py --train `
  --model-dir D:\ModelLab\qwen3-base `
  --output-dir D:\ModelLab\adapters\candidate-1
```

The QLoRA contract pins the exact upstream revision, tokenizer/weight snapshot, NF4 double
quantisation, all-linear LoRA rank 16/alpha 32/dropout 0.05, completion-only loss, maximum
16,384 tokens without truncation, batch/accumulation, optimiser, seed, and checkpoint policy.
It also binds `training/interpreter_training_environment_v1.2.json`: exact versions of PyTorch,
Transformers, Accelerate, PEFT, bitsandbytes, Hugging Face Hub, safetensors, and packaging;
CPython `>=3.10,<3.14` and `PYTHONHASHSEED=0`; CUDA and VRAM requirements;
`CUBLAS_WORKSPACE_CONFIG=:4096:8`;
deterministic PyTorch/Transformers execution; disabled TF32/cuDNN benchmarking; and zero data-loader
workers. The freeze command streams and hashes the Python executable and every installed wheel
payload, records the complete transitive distribution inventory, OS/platform, GPU, driver,
CUDA/cuDNN and deterministic process state, and writes
`training/interpreter_training_environment_receipt.json`. Any supported CPython patch release may
create the receipt. Preflight/training then require the current environment to match that exact
receipt, including the Python executable and patch version; the training report binds the receipt,
hardware/backend evidence, and adapter hashes.
Training output remains an unpromoted candidate plus a checksum-bound `training_report.json`. The
training command also records the candidate in registry lifecycle state `trained` or `experimental`;
that registry state is traceability only and does not make a candidate selectable.

## Export to a distinct Ollama candidate

Export accepts only the exact reported adapter and base snapshot, and only when independent
revalidation yields the non-experimental `candidate_adapter_unpromoted` state. An adapter marked
`candidate_adapter_experimental` is rejected before conversion or import work begins. Supply explicit files from a
reviewed llama.cpp checkout; their SHA-256 values are recorded. `--plan-only` verifies inputs
without loading weights.

```powershell
python scripts\export_interpreter_candidate.py `
  --adapter D:\ModelLab\adapters\candidate-1 `
  --base-model D:\ModelLab\qwen3-base `
  --output D:\ModelLab\exports\candidate-1 `
  --tag model-laboratory-interpreter:candidate-1 `
  --convert-hf-to-gguf D:\llama.cpp\convert_hf_to_gguf.py `
  --llama-quantize D:\llama.cpp\build\bin\Release\llama-quantize `
  --merge-device cuda --plan-only
```

Remove `--plan-only` only after reviewing the plan. CPU merge is supported but can use very
large system memory; prefer `--merge-device cuda` when suitable VRAM is available. Export never
overwrites the frozen tag and never promotes itself. It writes `export_receipt.json`, covering
the merged F16 GGUF, Q4_K_M GGUF, Modelfile, tools, commands, Ollama digest, and sizes.

## Evaluate the adapted model

Development evaluation is repeatable and deliberately non-promotable:

```powershell
python scripts\run_interpreter_candidate_evaluation.py `
  --adapter D:\ModelLab\adapters\candidate-1 `
  --base-model D:\ModelLab\qwen3-base `
  --export-receipt D:\ModelLab\exports\candidate-1\export_receipt.json `
  --benchmark verification\interpreter_baseline_v1.6.json `
  --baseline-report D:\ModelLab\evaluations\base-development.json `
  --output D:\ModelLab\evaluations\candidate-1-development.json `
  --purpose development
```

The shipped 150-case benchmark is development-only. For promotion, keep a different benchmark
outside the source tree, first run the untouched base against it, then seal it to the already
exported candidate:

```powershell
python scripts\run_interpreter_baseline.py `
  --benchmark D:\ModelLab\private\promotion-v1.json `
  --output D:\ModelLab\private\base-promotion-v1.json

python scripts\seal_interpreter_promotion.py `
  --benchmark D:\ModelLab\private\promotion-v1.json `
  --baseline-report D:\ModelLab\private\base-promotion-v1.json `
  --export-receipt D:\ModelLab\exports\candidate-1\export_receipt.json `
  --campaign-id REVIEWED_UUID `
  --reviewer "Reviewer identity" `
  --provenance "Private benchmark review record" `
  --output D:\ModelLab\private\promotion-v1.seal.json
```

Only then consume that campaign:

```powershell
python scripts\run_interpreter_candidate_evaluation.py `
  --adapter D:\ModelLab\adapters\candidate-1 `
  --base-model D:\ModelLab\qwen3-base `
  --export-receipt D:\ModelLab\exports\candidate-1\export_receipt.json `
  --benchmark D:\ModelLab\private\promotion-v1.json `
  --baseline-report D:\ModelLab\private\base-promotion-v1.json `
  --output D:\ModelLab\evaluations\candidate-1-promotion.json `
  --purpose promotion `
  --promotion-campaign-id REVIEWED_UUID `
  --promotion-seal D:\ModelLab\private\promotion-v1.seal.json
```

The runner verifies the adapter and export receipt, binds the complete installed local manifest
digest and exact exported GGUF bytes, and hashes every manifest blob,
runs all 150 held-out development cases with the frozen 32K/4K deterministic profile, prints progress/ETA,
and atomically checkpoints. Its validator replays every raw response through the current scorer.
A candidate is eligible only if passed-case counts do not regress for any aggregate or family,
the absolute schema/compiler/unsupported/scientific/clarification/context floors pass, and at
least one predeclared scientific-benefit metric gains a correct case. Clarification cases must
both ask the required question and correctly use a held-out answer; context negotiation has its own
first-action non-regression metric. A development report is never eligible. A promotion report
requires an explicit campaign UUID, external candidate-bound seal, and checksum-bound local ledger.
The ledger uses a cross-process file lock, permits exact checkpoint resume while reserved, and
rejects reuse after completion or by a different candidate. A completed evaluation is registered as lifecycle state
`evaluated`, including failed/non-promotable evaluations.

## Promote or roll back

First inspect the current registry and copy its exact checksum:

```powershell
python scripts\manage_interpreter_candidates.py status
```

Promotion requires both hashes to be supplied explicitly and rechecks the baseline, original
training report, frozen config/corpus/review/base-snapshot evidence, export artifacts, evaluation,
current registry revision, and installed Ollama digest. The final boundary requires the exact
`candidate_adapter_unpromoted` training state and a matching previously `evaluated` registry entry:

```powershell
python scripts\manage_interpreter_candidates.py promote `
  --adapter D:\ModelLab\adapters\candidate-1 `
  --base-model D:\ModelLab\qwen3-base `
  --candidate-report D:\ModelLab\evaluations\candidate-1.json `
  --export-receipt D:\ModelLab\exports\candidate-1\export_receipt.json `
  --benchmark D:\ModelLab\private\promotion-v1.json `
  --evaluation-baseline-report D:\ModelLab\private\base-promotion-v1.json `
  --promotion-seal D:\ModelLab\private\promotion-v1.seal.json `
  --confirm-report-sha256 REVIEWED_REPORT_SHA256 `
  --confirm-registry-sha256 CURRENT_REGISTRY_SHA256
```

Rollback is equally explicit and atomic:

```powershell
python scripts\manage_interpreter_candidates.py rollback `
  --target frozen_base `
  --confirm-registry-sha256 CURRENT_REGISTRY_SHA256
```

A rollback target may instead be any previously approved candidate entry ID. The desktop reads
the registry from its native application-data directory, fails closed on corruption, and accepts
an approved candidate only when its complete local manifest digest, exact exported GGUF digest,
verified size, and all referenced local blobs match the immutable entry.
Private candidates cannot be pulled by tag through the desktop.

## Empirical boundary

The scripts, contracts, and tests are implementation—not evidence that Qwen improved. Do not
publish an accuracy claim until the exact untouched and candidate campaigns have been run and
their reports reviewed. A candidate that merely improves JSON formatting but weakens any
scientific or safety family must not be promoted.
