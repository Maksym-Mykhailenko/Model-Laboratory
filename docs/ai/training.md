# Interpreter Stage 6 — training corpus v1.5

Stage 6 prepares deterministic supervised fine-tuning data. It does not train or ship a model.

## Frozen corpus

`training/interpreter_corpus_v1.5/` contains 6,300 synthetic, compiler-grounded records at seed
`20260826`: 5,039 train and 1,261 validation records. The 150-case v1.6 development benchmark is
used only as a text/semantic deny-list and remains excluded from training. Historical v1.1/v1.2
corpora and earlier benchmarks remain in the tree for provenance but are not current training inputs.

Every record binds its instruction, existing source, requested paths, context round, ordered
clarification history, strict output target, compilation evidence, and content/learning hashes.
Proposal targets pass the production parser, deterministic Model Transaction compiler, strict
model validator, expression AST, and Model Graph before admission. The 1,300 official-pack records
cover all 13 packs without exposing `org.modellab...` kind URIs in the instruction; they include
creation, edits of existing official objects, and paired clarification continuations. Official-pack
train/validation proposal semantics are disjoint rather than paraphrases of identical property
templates.

## Multi-turn records

Context negotiation is paired: a hidden exact path is requested in turn zero and disclosed for
the edit in turn one. Clarification is also paired: turn zero asks a genuine user question; turn
one includes a concrete user answer in `clarification_history` and proposes deterministic edits.
The second turn is not trained as a disconnected one-shot request.

## Human review gate

The stratified 725-record queue remains `pending-human-review`. Compiler validation is not human
or expert review. A production QLoRA candidate is blocked until a real reviewer decides each
item. Corrected targets are reconstructed with their exact dialogue/context and compiled again;
there is no bulk accept operation.

```bash
python scripts/review_interpreter_training_corpus.py
python scripts/review_interpreter_training_corpus.py --next
python scripts/review_interpreter_training_corpus.py \
  --id RECORD_ID --decision accepted --reviewer "Researcher name" \
  --notes "Checked instruction, target, and scientific semantics."
```

## Build and validation

```bash
python scripts/build_interpreter_training_corpus.py
python scripts/validate_interpreter_training_corpus.py
python scripts/validate_interpreter_training_corpus.py --deep
python scripts/validate_interpreter_training_corpus.py --deep --workers 1  # lower peak RAM
python scripts/materialize_interpreter_sft.py \
  training/interpreter_corpus_v1.5/train.jsonl \
  training/interpreter_corpus_v1.5/train.sft.jsonl
```

The compact JSONL is authoritative. SFT chat JSONL is derived from the same versioned production
system/user prompt contract. The current frozen identities are recorded in `manifest.json` and
`training/interpreter_qlora_v1.7.json`; changing any record or review decision changes them.
The QLoRA configuration also binds
`training/interpreter_training_environment_v1.2.json`, which fixes the direct training packages,
supported CPython range/hash seed, CUDA requirement, cuBLAS workspace setting, and deterministic
algorithm policy. Before empirical work, `freeze_interpreter_training_environment.py` creates the
required receipt containing every installed distribution and payload identity, the Python executable,
platform, GPU, driver, CUDA and cuDNN. CPython `>=3.10,<3.14` can create a receipt; training refuses
a missing receipt or any later difference from its exact recorded environment.

Stage 6 is technically constructed and reproducibly validated in this archive, but its human
review gate is deliberately not claimed as complete.
