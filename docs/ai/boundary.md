# Local AI interpreter boundary

Model Laboratory 1.16.0 uses an optional local Ollama model as a formal-model authoring
assistant. It is not a numerical backend and does not receive execution authority.

The configured tag is `qwen3:4b-instruct-2507-q4_K_M`, but the tag is no longer trusted as
identity. Before inference, the Rust bridge loads the bundled frozen base-model lock, verifies
that `/api/tags` names the locked manifest identity, hashes the local manifest bytes, requires
exactly the locked layer set, verifies every referenced local blob by SHA-256, requires the exact
frozen GGUF model-layer SHA-256, and records the size derived from the exact verified GGUF bytes. A retagged or modified model therefore fails closed. The status response exposes the
resulting `artifact_evidence`; historical proposal-provider schemas remain readable, while new
proposals require the stronger evidence object.

## Runtime separation

```text
desktop interface
    │ explicit authoring request
    ▼
Rust bridge
    │ fixed loopback http://127.0.0.1:11434, proxy disabled
    ▼
Ollama exact frozen artifact
    │ strict typed response
    ▼
deterministic Python Model Transaction compiler
    │ ordinary model-validation pipeline
    ▼
reviewable source diff and hashes
    │ explicit user acceptance
    ▼
editor state
```

Ollama never receives file, network, plugin-installation, analysis, or experiment-execution
actions. The scientific sidecar never loads model weights. Only the native loopback bridge
talks to Ollama, and only one pull/inference operation can be active at a time. Cancellation
is explicit.

## Local reference attachments

The desktop picker accepts up to eight TXT, Markdown, YAML, JSON, CSV, TSV, or text-based PDF
references, each no larger than 16 MiB. The file path and raw bytes are never sent to Ollama.
The persistent Python sidecar performs strict UTF-8 decoding or deterministic page-ordered
`pypdf` text extraction, normalises line endings, hashes the source and extracted text, and
stores only bounded extracted text in memory. Scanned/image-only and encrypted PDFs fail closed.

The context builder publishes content-addressed manifests and selected 3 KiB UTF-8 chunks.
Qwen may request at most four omitted chunks by the reserved path
`["@attachment", attachment_id, zero_based_chunk_index]`; those paths are reference-only and
cannot be edit operations. Current proposals and acceptances retain the exact manifests and
hashes of every disclosed chunk, not the reference text itself, so review/reproduction evidence
is compact while remaining tamper-evident.

## Machine-generated contract

The Python sidecar generates the interpreter contract from the installed authoritative
registries:

- model sections and entry fields;
- namespaced Model Graph kind descriptors and strict schemas;
- installed capabilities and applicability;
- expression operators, constants, and `FUNCTION_ARITIES`; and
- edit/path/size limits.

The Rust prompt contains no independent mathematical function list. This prevents a prompt
from advertising an operator—such as `sinh`, `cosh`, or `tanh`—that the actual expression
compiler does not implement.

The fixed Stage-5 evaluator uses the same versioned system-prompt and structured-output-schema
files as the Rust bridge. Its few-shot examples demonstrate protocol behaviour only; the
authoritative scientific catalogue remains machine-generated from the current installation.

## Bounded context rather than full-source replacement

| Object | Limit |
|---|---:|
| instruction | 64 KiB |
| source retained and compiled locally | 512 KiB |
| one Qwen context package | 32 KiB |
| context requests per round | 32 |
| context rounds | 6 |
| edit operations | 128 |
| edit program | 128 KiB |
| inference output | 2 MiB native boundary |
| Ollama working context | adaptive 32,768 / 65,536 / 131,072 tokens |
| generated tokens | 4,096 / 8,192, paired with the selected context |
| clarification turns | 8, with a 16 KiB total history budget |

The full source and a complete replacement are never placed in the same working context.
For large models, Qwen receives checksummed inventories and exact visible paths. It can ask
for additional exact paths. Existing content cannot be changed unless that exact target or
complete containing entry was visible. This rule also applies to `name`; a long name reduced
to a size/hash record is not editable.

## Deterministic authority boundary

Qwen returns exactly one of:

- `request_context`;
- `propose_edits`; or
- `needs_clarification`; or
- `unable`.

Clarification is real dialogue, not a renamed context request. The desktop preserves the
original instruction/source, shows the exact question, records only an explicit user answer,
and resubmits the ordered bounded history. Source or instruction changes invalidate the
dialogue. Every clarification output creates a checksum-bound turn record containing the
exact context, raw output, question, full provider/artifact identity, parent turn, and
conversation UUID. The next turn must supply the complete lineage and the same verified local
artifact. The proposal and acceptance retain that lineage, its checksum, and turn count.

A proposal is a bounded set of typed `set`/`remove` paths, not YAML. The deterministic core:

1. recreates and verifies the context and instruction/source hashes;
2. validates every path and portable JSON value;
3. rejects duplicates, ancestor/descendant overlaps, hidden existing targets, and excessive
   depth/size;
4. applies independent operations in canonical path order;
5. uses a round-trip YAML representation for existing documents, preserving comments,
   quoting, collection style, key order, and untouched indentation;
6. validates the complete result with the safe YAML parser, strict Pydantic schema,
   expression AST, Model Graph kind registry, and mathematical validator;
7. records the context, edit-program, source, Model IR, compiler, provider, and proposal
   identities; and
8. requires the user to accept the exact reviewed proposal SHA-256.

No analysis runs while preparing, compiling, diffing, or accepting a model edit.

## Reproducible provenance

Current accepted edits use acceptance schema 1.4 and record:

- configured model tag and complete observed manifest digest;
- the artifact-lock, manifest, and model-blob SHA-256 values plus verified byte size;
- proof that every manifest blob was hashed locally;
- honest identity-verification mode;
- Ollama version and fixed generation options;
- instruction hash rather than the private instruction text;
- previous and accepted source hashes;
- accepted Model IR hash;
- edit-program/compiler identity;
- full checksum-linked clarification-turn evidence, when present; and
- proposal/acceptance hashes and time.

Provider identity also records whether inference used the frozen base or a checksum-bound
approved candidate and, for candidates, the immutable registry-entry hash.

The record is included in experiment state 6 and `.mlab` 2.0 author review. Acceptance 1.0
and 1.1 remain loadable as historical provenance; their older identity language is retained
as history, not promoted to a current verification claim.
