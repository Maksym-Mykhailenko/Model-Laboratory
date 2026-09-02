# Model Laboratory 1.12.7

Version 1.12.7 makes local reference files a first-class, bounded input to the AI model
interpreter.

## Added

- Visible **Attach file** and per-file **Remove** controls in the desktop interpreter panel.
- Up to eight local TXT, Markdown, YAML, JSON, CSV, TSV, or text-based PDF references.
- Strict 16 MiB source, 8 MiB extracted-text, 32 MiB sidecar-store, and 2,000-page limits.
- Local UTF-8/newline normalisation and page-ordered `pypdf` extraction.
- Content-addressed manifests covering filename/media type, source and extracted-text hashes,
  extraction implementation/version, and deterministic chunking identity.
- Bounded 3 KiB context chunks and exact omitted-chunk negotiation through reserved reference
  paths that cannot be used as model edit paths.
- Proposal/compiler/acceptance provenance binding the manifests and hashes of every chunk Qwen
  actually saw.

## Compatibility and safety

- The Qwen response grammar remains 1.2, so the frozen Stage-5 benchmark and reviewed Stage-6
  targets are not rewritten merely to add references.
- Requests without attachments continue to use deterministic context schema 1.2.
- Attachment-aware contexts use 1.3; current compiler/proposal/acceptance schemas advance to
  1.4/2.4/1.5. Historical records remain inspectable.
- Paths and raw bytes never reach Ollama. The Python sidecar extracts locally and Qwen receives
  only the bounded context package.
- Encrypted, malformed, scanned/image-only, unsupported, invalid-UTF-8, excessive, stale, and
  tampered references fail closed. OCR and vision are deliberately outside this release.
- Changing active references clears any pending proposal or clarification dialogue. A proposal
  is still only applied after deterministic compilation, scientific validation, exact diff
  review, checksum-bound approval, and explicit acceptance.
