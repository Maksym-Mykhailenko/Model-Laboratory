param(
    [switch]$LiveOllama
)

$ErrorActionPreference = "Stop"
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
cargo check --manifest-path src-tauri/Cargo.toml --locked
cargo test --manifest-path src-tauri/Cargo.toml --locked interpreter::tests

Write-Host "Checking Python/compiler and webview contracts..."
python -m pytest -q tests/test_desktop_interpreter_contract.py tests/test_interpreter.py tests/test_desktop_engine.py
npm run test:frontend

if ($LiveOllama) {
    Write-Host "Running live exact-frozen-base Ollama identity smoke test..."
    cargo test --manifest-path src-tauri/Cargo.toml --locked live_frozen_base_identity_smoke -- --ignored --nocapture
}

Write-Host "Native interpreter verification passed."
