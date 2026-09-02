# Model Laboratory 1.12 interpreter protocol

This is the authoritative local-interpreter wire, dialogue, compiler, and provenance contract.
Qwen is an untrusted authoring assistant. It never executes mathematics and never writes source
without deterministic compilation, an exact review diff, and explicit user acceptance.

## Current versioned documents

| Document | Version |
|---|---:|
| interpreter context | 1.2 |
| raw interpreter output | 1.2 |
| deterministic edit compiler | 1.3 |
| proposal | 2.3 |
| accepted-proposal provenance | 1.4 |
| clarification-turn evidence | 1.0 |

Historical version triplets remain readable. Unknown fields, non-finite numbers, excessive
nested JSON, and values outside their byte/count limits are rejected.

## Exact Qwen output

Every response has exactly nine fields:

```json
{
  "schema": "model-laboratory-interpreter-output",
  "schema_version": "1.2",
  "action": "propose_edits",
  "context_sha256": "64 lowercase hexadecimal characters",
  "context_requests": [],
  "operations": [
    {"op": "set", "path": ["functions", "f"], "value": "sinh(x)"}
  ],
  "explanation": "Human-facing explanation",
  "warnings": [],
  "clarification_question": ""
}
```

| Action | Context requests | Operations | Clarification question | Explanation |
|---|---:|---:|---|---|
| `request_context` | 1–32 | 0 | empty | optional |
| `propose_edits` | 0 | 1–128 | empty | optional |
| `needs_clarification` | 0 | 0 | required | optional |
| `unable` | 0 | 0 | empty | required |

The model cannot return YAML, complete source, provider fields, commands, wildcards, JSON
Pointer, or arbitrary top-level data. A `remove` operation has a null value; `set` cannot.
Paths must be unique and non-overlapping.

## Model Transaction grammar

Allowed path shapes are:

```text
["name"]
["metadata", field]
[section, identifier]
[section, identifier, field]
```

Current sections are `variables`, `parameters`, `constants`, `derived_quantities`, `functions`,
`vector_functions`, `matrix_functions`, `constraints`, `assumptions`, `ambiguities`, `objects`,
`relationships`, and `assets`. The context generates their required/permitted fields from the
installed Pydantic and Model Graph registries. The expression function catalogue is generated
from `FUNCTION_ARITIES`; it currently contains `abs cos cosh exp log sin sinh sqrt tan tanh`.

## Bounded context and adaptive inference

The accepted instruction and source limits remain 64 KiB and 512 KiB. The full source remains
in Python. Qwen receives at most a 32 KiB deterministic context package and can request exact
hidden paths for up to six rounds. Existing content is editable only when the exact target or
its complete entry was visible; this includes the model name.

Production selects the smallest profile whose conservative byte-to-token estimate fits:

| Profile | `num_ctx` | `num_predict` |
|---|---:|---:|
| ordinary | 32,768 | 4,096 |
| medium | 65,536 | 8,192 |
| maximum | 131,072 | 8,192 |

All profiles retain `seed=0`, `temperature=0.7`, `top_p=0.8`, `top_k=20`, and `min_p=0.0`.
The maximum profile preserves the existing accepted-source contract; ordinary requests no
longer reserve its multi-gigabyte KV cache.

The held-out baseline and candidate campaigns deliberately use the separate deterministic
32,768/4,096 profile with temperature zero.

## Clarification dialogue

`needs_clarification` asks the user, not the source-context subsystem. The desktop retains the
exact original instruction and source, displays the exact question, and only continues after an
explicit answer. Each later context contains an ordered list of `{question, answer}` pairs.

Limits are eight turns, 8 KiB per question, 8 KiB per answer, and 16 KiB for canonical history.
Changing the instruction or model source invalidates the dialogue. Each question-producing turn
binds a conversation UUID, parent turn, exact context, exact raw output, question, and complete
provider/artifact evidence. The next turn must present the uninterrupted chain and the same
verified local model artifact. A proposal and its accepted provenance store this evidence chain,
the clarification-history SHA-256, and turn count, but not the private answer text.

## Deterministic compilation

For a proposal, the core:

1. parses and validates the complete local source;
2. recreates the exact context, including clarification history, and verifies its checksum;
3. validates and canonically orders typed operations;
4. rejects duplicate, overlapping, hidden, or invalid targets;
5. edits an existing round-trip YAML tree, preserving comments, quoting, collection style,
   key order, and untouched indentation;
6. recompiles through the safe parser, strict model schema, expression AST, and Model Graph;
7. records edit-program, compiler, source, Model IR, provider, dialogue, and proposal identities;
8. produces a stable current/proposed diff; and
9. requires acceptance of the exact displayed proposal SHA-256.

Operation and nested mapping order cannot alter the result identity.

## Provider identity and model selection

Current provider records contain exactly:

```text
provider, endpoint, runtime_version, model_tag, observed_model_digest,
identity_verification, expected_base_model, expected_quantization,
generation_options, model_role, registry_entry_sha256, artifact_evidence
```

`model_role` is `frozen_base`, `evaluation_candidate`, or `approved_candidate`. For the frozen
base, the native runtime verifies the bundled model lock, local manifest SHA-256, exact manifest
layer set, every referenced blob, the exact frozen GGUF SHA-256, and model size before inference.
Evaluation candidates are bound to an export receipt. Approved candidates are selected by a
checksum-bound local registry entry and must match its observed Ollama digest and size. An invalid
registry fails closed. A missing registry selects the frozen base, but still performs the full
frozen-artifact verification.

`artifact_evidence` records the verification mode, artifact-lock SHA-256, exact local manifest
SHA-256, exact GGUF SHA-256 and byte size, and the fact that every manifest blob was hashed.
Current proposals reject prefix-only or tag-only identities. Older provider records remain
readable only as historical provenance.

## Candidate lifecycle

Training never promotes itself. Export validates every reported adapter artifact and the frozen
base snapshot, merges PEFT weights, invokes explicit pinned-by-hash conversion/quantisation tools,
creates a distinct Ollama tag, and writes a receipt covering every output and command.

Candidate evaluation runs the complete held-out Stage-5 benchmark with the same scorer and
compiler. Validation replays each stored raw response; a self-hashed summary is insufficient.
Promotion requires no regression in passed-case counts for every aggregate scientific/safety
metric and every family, absolute schema/compiler/unsupported/scientific/clarification/context
floors, and at least one additional correct case in a predeclared scientific-benefit metric.
Repeatable development evaluations can never promote and have no implicit benchmark default. A
promotion evaluation requires an external private benchmark, matching untouched-base report,
candidate-bound seal created after export, explicit campaign UUID, and completed one-shot
consumption-ledger entry; reuse by another candidate is rejected. No promotion benchmark is shipped
with the application. Promotion also requires explicit confirmation of both the reviewed
candidate-report hash and current registry hash. Promotion and rollback use atomic replacement
and retain event history.

Inference never runs a capability, changes reference results, installs an extension, saves an
experiment, or publishes an `.mlab`.
