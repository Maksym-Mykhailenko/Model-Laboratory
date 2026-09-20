param(
    [switch]$LiveOllama
)

$ErrorActionPreference = "Stop"

function Assert-NativeSuccess([string]$Operation) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE."
    }
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

foreach ($Command in @("python", "node", "npm", "cargo", "rustc")) {
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "$Command is required for native interpreter verification."
    }
}

$RustDetails = (& rustc -vV | Out-String)
if ($LASTEXITCODE -ne 0) { throw "Could not inspect the Rust toolchain." }
$RustHostLine = ($RustDetails -split "`r?`n" | Where-Object { $_ -match '^host:\s+' } | Select-Object -First 1)
if (-not $RustHostLine) { throw "rustc -vV did not report a host toolchain." }
$RustHost = ($RustHostLine -replace '^host:\s+', '').Trim()
if ($IsWindows -and $RustHost -notmatch 'pc-windows-msvc$') {
    throw "Windows native verification requires an MSVC Rust host; found '$RustHost'."
}

Write-Host "Checking native interpreter Rust bridge..."
python scripts/build_sidecar.py
Assert-NativeSuccess "Scientific sidecar build"
cargo check --manifest-path src-tauri/Cargo.toml --locked
Assert-NativeSuccess "cargo check --locked"
cargo test --manifest-path src-tauri/Cargo.toml --locked interpreter::tests
Assert-NativeSuccess "Rust interpreter tests"
cargo test --manifest-path src-tauri/Cargo.toml --locked response_
Assert-NativeSuccess "Rust response protocol tests"

Write-Host "Checking Python/compiler and webview contracts..."
python -m pytest -q tests/test_desktop_interpreter_contract.py tests/test_interpreter.py tests/test_desktop_engine.py
Assert-NativeSuccess "Python native-interpreter contract tests"
npm run test:frontend
Assert-NativeSuccess "Frontend contract tests"

if ($LiveOllama) {
    Write-Host "Running live exact-frozen-base Ollama identity smoke test..."
    cargo test --manifest-path src-tauri/Cargo.toml --locked live_frozen_base_identity_smoke -- --ignored --nocapture
    Assert-NativeSuccess "Live frozen-base identity smoke test"
}

Write-Host "Native interpreter verification passed."
