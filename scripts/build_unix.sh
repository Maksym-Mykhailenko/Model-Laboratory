#!/usr/bin/env sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"

cargo check --manifest-path src-tauri/Cargo.toml --locked
cargo test --manifest-path src-tauri/Cargo.toml --locked interpreter::tests
python3 -m pip install --upgrade -r requirements-dev.txt -c constraints-tested.txt
npm ci
npm run test:frontend
python3 -m pytest -q
python3 verification/run_reference_verification.py
python3 verification/run_bundle_verification.py
python3 verification/run_reproduction_verification.py
python3 verification/run_expression_ast_verification.py
python3 verification/run_model_graph_protocol_verification.py
python3 verification/run_official_packs_verification.py
python3 scripts/build_sidecar.py
npx tauri build

printf '%s\n' "Installers are available under src-tauri/target/release/bundle."
