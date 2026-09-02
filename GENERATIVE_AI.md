# Generative-AI development disclosure

This document records the role of generative-AI tools in the development of Model Laboratory
through 2 September 2026. It concerns the development process itself. The local Qwen authoring
interpreter included in Model Laboratory is a product feature with its own protocol, provenance,
benchmark, review, and promotion controls.

## Development workflow

Model Laboratory has been developed through a human-directed, AI-assisted engineering workflow.
OpenAI ChatGPT and Codex have been used interactively for:

- elaborating technical requirements and acceptance criteria;
- implementing and refactoring source code;
- investigating defects and proposing corrections;
- designing and generating tests and verification programs;
- drafting and revising technical documentation; and
- inspecting and packaging development releases.

The project owner establishes the product direction, scientific scope, architectural requirements,
release priorities, and acceptance decisions. Development proceeds through explicit tasks and
iterative inspection. AI-assisted contributions are integrated with the surrounding codebase and
evaluated through the project's validation and verification controls.

## Verification and responsibility

The current engineering controls include:

- Python and frontend regression suites;
- independently specified analytical and exact numerical reference checks;
- bundle, reproduction, expression-AST, Model Graph, Run, and official-pack verification;
- deterministic compiler and schema validation;
- checksum-bound corpus, benchmark, build, and experiment identities;
- explicit workload and capability boundaries; and
- a separate human-review gate for interpreter-training records.

The project owner remains responsible for release decisions, scientific claims, licensing, and the
quality of distributed results. Passing automated checks is recorded as verification evidence for a
specific source state and does not transfer that responsibility to an AI system.

## Historical record

This file is the project-level disclosure for development completed through the date above.
Available conversation histories, supplied requirements, source revisions, release notes, test
results, and verification reports provide the retained historical record. Historical source lines
are not assigned to an interaction-by-interaction prompt ledger.

## Provenance practice for subsequent work

For substantive generative-AI assistance performed after this disclosure, the project records the
date, provider and model when available, prompt or task instruction, unedited response, affected
work, and the review or validation applied. Related commits or change records can reference the
corresponding interaction record. Any generative-AI assistance used to prepare a grant proposal is
recorded and disclosed according to the applicable funder's policy.

Deterministic code generation, schema compilation, numerical computation, static analysis, test
execution, and other conventional automation are identified separately from generative-AI use.
