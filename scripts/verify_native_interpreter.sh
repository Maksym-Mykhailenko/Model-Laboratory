#!/usr/bin/env sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"

for command in python3 node npm cargo rustc; do
    command -v "$command" >/dev/null 2>&1 || { printf '%s\n' "$command is required for native interpreter verification." >&2; exit 1; }
done

cargo check --manifest-path src-tauri/Cargo.toml --locked
cargo test --manifest-path src-tauri/Cargo.toml --locked interpreter::tests
python3 -m pytest -q tests/test_desktop_interpreter_contract.py tests/test_interpreter.py tests/test_desktop_engine.py
npm run test:frontend

if [ "${MODEL_LAB_LIVE_OLLAMA_SMOKE:-0}" = "1" ]; then
    cargo test --manifest-path src-tauri/Cargo.toml --locked live_frozen_base_identity_smoke -- --ignored --nocapture
fi

printf '%s\n' "Native interpreter verification passed."
