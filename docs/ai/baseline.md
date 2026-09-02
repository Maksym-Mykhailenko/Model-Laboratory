# Untouched-Qwen interpreter baseline

Model Laboratory 1.17.0 retains the corrected Stage-5 evaluation contract for the local
authoring interpreter. Its purpose is to measure the **untouched, frozen** Qwen artifact before
any LoRA/QLoRA training begins. The held-out benchmark, scorer, prompt assets, compiler
identity, and decoding profile are versioned so the same campaign can later be rerun against
an adapter without moving the goalposts.

## Frozen base-model contract

`model_lab/baseline/base_model_lock_v1.1.json` pins:

- repository `Qwen/Qwen3-4B-Instruct-2507`;
- full upstream revision;
- Apache-2.0 licence identifier and copyright notice;
- `tokenizer.json` SHA-256;
- all three upstream safetensors shard SHA-256 values;
- Ollama tag `qwen3:4b-instruct-2507-q4_K_M`;
- Q4_K_M quantisation;
- the exact Ollama registry-manifest SHA-256;
- the config digest, media type, and byte size; and
- every layer digest, media type, and byte size, including the exact GGUF model blob,
  template, licence, and parameters.

Before a live campaign is accepted, the evaluator requires the complete tag digest to equal
the locked registry-manifest digest, hashes the exact local manifest bytes, rejects
missing/extra layers (including
adapter/system layers), hashes the **actual local GGUF and every referenced blob**, and checks
Qwen3 architecture, quantisation, and context length. A mutable tag alone is never accepted
as proof of the baseline artifact.

Production inference selects a bounded 32K/4K, 64K/8K, or 128K/8K profile from each request.
The fixed benchmark is much smaller, so evaluation uses a separate 32,768-token,
4,096-output deterministic profile (`seed=0`, `temperature=0.0`). Reserving 131,072 tokens
for every short case caused a multi-gigabyte KV-cache allocation and severe paging on ordinary
desktop hardware; the smaller profile remains frozen for both untouched and adapted campaigns.

## Shared production/evaluation protocol

The Rust production bridge and Python evaluator consume the same versioned assets under
`model_lab/baseline/`:

- `system_prompt_v1.2.txt`;
- `user_prompt_template_v1.0.txt`;
- `output_schema_v1.2.json`; and
- `few_shot_examples_v1.1.json`.

The output protocol has four actions:

- `propose_edits` — submit a bounded typed Model Transaction;
- `request_context` — request exact hidden paths from the existing local model;
- `needs_clarification` — ask the **user** a question because the request itself is
  scientifically under-specified; and
- `unable` — report that the requested work cannot be represented safely/supportedly.

`request_context` and `needs_clarification` are intentionally different. The former is a
bounded local retrieval mechanism. The latter starts a bounded user dialogue: the exact
question and explicit answer are returned in ordered `clarification_history` on the next turn.
It never creates or applies edits by itself.

## Repeatable development evaluation set

`verification/interpreter_baseline_v1.6.json` contains **150 cases across 28 mathematical,
official-pack, and protocol families**. It is `internal-curated`, not claimed to be independently expert-reviewed,
and is explicitly marked `training_exclusion: true`. From 1.12.5 it is explicitly the public,
repeatable **development** benchmark. It is never valid as a private promotion benchmark.

Promotion uses a separate external benchmark that is not shipped in the source archive. Its
untouched-base report is validated against that exact file, after which the benchmark is sealed to
an already-exported candidate and one UUID campaign. The generic report gate accepts an explicitly
supplied benchmark instead of silently substituting this development file.

The 13 official-pack families contribute creation coverage for all 34 official object kinds,
plus one existing-object edit and one held-out clarification-continuation case per pack. A benchmark
preflight requires every expected official object kind to be exposed by the schema context generated
from that exact case instruction.

Ten capability-boundary cases require an explicit `unable` response for SDEs, specialised or
trainable neural networks, neural controllers, structural optimisation, solver-coupled parameter
estimation, and mesh/FEM PDE solving. Nearby installed schemas are not treated as implicit composition.

Coverage includes:

- scalar and multivariate creation;
- edits of existing models;
- variables, parameters, constants, defaults, and domains;
- derived quantities and assumptions;
- constraints;
- vector and matrix functions;
- ambiguity recording and user clarification;
- units and other metadata;
- Model Graph objects and relationships;
- structured assets;
- unsupported/safety-boundary requests; and
- multiple large-model context-negotiation cases.

Every proposal gold model is compiled by the real parser/validator before the benchmark is
accepted. The loader rejects unknown benchmark fields, requires an explicit expected first
action for every case, and rejects exact or high-similarity overlap with the production
few-shot instructions. These cases must not be copied into Stage-6 generated or reviewed
training data.

## Corrected scoring

A response counts only after the strict output validator and deterministic Model Transaction
compiler accept it. The report separates:

- raw schema compliance;
- first-action accuracy;
- terminal-status accuracy;
- deterministic compiler acceptance for proposal cases;
- scientific semantic accuracy;
- stricter canonical Model IR identity;
- unsupported-request accuracy;
- clarification-question accuracy;
- clarification-continuation accuracy after the held-out user answer;
- end-to-end clarification accuracy;
- context-negotiation first-action accuracy; and
- family-level semantic accuracy.

Scientific comparison is **typed**, not a recursive “delete `source` keys” hash. Model Graph
relationship endpoints and extension data remain scientific structure. Scalar expressions and
relations are normalized symbolically so harmless algebraic rewrites such as `x+y` versus
`y+x`, `(x-a)**2` versus `(a-x)**2`, and expanded versus factored polynomials are not marked
wrong solely because of spelling. Canonical Model IR equality remains a separate stricter
metric.

A missing response never receives first-action credit. Transport/runtime/identity failures are
`BaselineInfrastructureError`s and invalidate the campaign instead of silently reducing model
accuracy.

## Audit evidence and provenance

Every live case preserves per round:

- exact context document and its checksum;
- exact rendered user-prompt checksum;
- raw model content;
- parsed output;
- validated output;
- context requests/actions;
- provider telemetry; and
- validation/compiler errors.

The aggregate report also records Model Laboratory/source-tree identity, evaluator and
interpreter-compiler source SHA-256 values, scorer version, prompt/schema/model-lock hashes,
benchmark hash, and verified runtime/model identity. This makes a published campaign
recomputable and makes scorer/compiler drift visible.

## Running Stage 5

Validate all fixed assets without inference:

```bash
python scripts/run_interpreter_baseline.py --validate-only
```

Run the empirical untouched-model campaign on a host where the exact frozen Ollama artifact
is installed:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
python scripts/run_interpreter_baseline.py --limit 3 \
  --output verification/interpreter_baseline_smoke.json
python scripts/run_interpreter_baseline.py
```

The smoke run verifies runtime identity and gives a realistic per-case timing without replacing
the authoritative report. The full runner prints case progress, elapsed time, and estimated
remaining time. After every completed case it atomically writes
`verification/interpreter_baseline_report.json.checkpoint.json`. If Python, Ollama, Windows, or
the evaluator stops, rerunning the same command verifies the benchmark/contract/source/runtime
identity and resumes the completed prefix instead of discarding hours of inference. Use
`--restart` only to deliberately discard that checkpoint. Invalid schema-valid model proposals
(including bad Model Graph kinds) are scored as model failures and do not abort the campaign;
transport/runtime failures still invalidate the campaign. The accepted report is written to
`verification/interpreter_baseline_report.json`.
`--validate-only` deliberately contains **no fabricated accuracy metrics** and records
`benchmark_validated_model_not_run`. A requested live run whose local runtime/artifact fails
identity or transport verification records `campaign_invalid_infrastructure` and exits
non-zero.

The supplied archive contains a successful asset-validation report, not a live empirical
baseline. Therefore Stage 5's deterministic harness is implemented, while the untouched-model
measurement remains a required local run. The benchmark is internal-curated and is not claimed
to have independent expert review.

## Fine-tuning comparison rule

Stage 6 must not use the held-out benchmark as training data. Stage 7 must compare the
untouched and adapted models with the same benchmark, scorer, prompt/schema contract, compiler,
and evaluation decoding profile. A fine-tune should be rejected if it improves JSON compliance
while degrading scientific semantic accuracy, unsupported-request accuracy, clarification
behaviour, or safe context negotiation.
