# Model Laboratory 1.12.3

## Frozen interpreter artifact identity

Ordinary desktop authoring no longer treats the configured Ollama tag as sufficient model identity. Before any frozen-base inference, the native Rust bridge now:

- validates the bundled `base_model_lock_v1.0.json` contract;
- canonicalises the digest reported by Ollama and requires the locked published manifest identity;
- locates and hashes the local Ollama manifest bytes and requires them to equal the API digest;
- requires exactly the locked manifest layer set, rejecting extra adapter/system layers;
- verifies every referenced local layer/config blob byte-for-byte by SHA-256;
- requires the model layer to equal the independently frozen GGUF SHA-256 `85e4a5b7…54b18b9`; and
- records the verified GGUF blob size from the exact bytes whose SHA-256 was checked.

The ready-status document exposes `frozen_base_evidence` containing the strong verification mode, base-model-lock SHA-256, local manifest SHA-256, exact model-blob digest, and verified blob size. Approved candidates retain their separate checksum-bound registry/export identity path.

## Native interpreter verification

Rust verification is now a release requirement rather than a manual footnote:

- `scripts/verify_native_interpreter.ps1` and `.sh` run `cargo check --locked`, native interpreter Rust tests, targeted Python/compiler contracts, and frontend tests;
- Windows and Unix release build scripts run the Rust format/check/test gates before packaging;
- the cross-platform GitHub workflow has a Windows `native-interpreter` job using the MSVC Rust toolchain; and
- `live_frozen_base_identity_smoke` is an ignored Rust integration test that can be explicitly run against a locally installed exact Ollama model (`-LiveOllama` in PowerShell or `MODEL_LAB_LIVE_OLLAMA_SMOKE=1` on Unix).

The current validation container does not contain `cargo`/`rustc` or Ollama, so the native compile/live gates cannot be executed here. Their absence is now detected and fails the native verifier instead of being silently skipped.

## Compatibility

The 1.12.2 live → prepared → committed → frozen/published state-authority boundary and the 1.12.1 training/evaluation/promotion hardening are unchanged.
