param(
    [string]$CertificateThumbprint = "",
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [switch]$RequireSigned,
    [switch]$RunInstallerSmoke,
    [string]$PreviousNsisInstaller = ""
)

$ErrorActionPreference = "Stop"

function Assert-NativeSuccess([string]$Operation) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE."
    }
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.11 or newer is required."
}
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "Node.js 20 or newer is required for the Tauri build command."
}
if (-not (Get-Command rustc -ErrorAction SilentlyContinue)) {
    throw "Install the Rust MSVC toolchain from https://rustup.rs before building."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "Node.js 20 or newer is required for the Tauri build command."
}

$PythonVersionText = (& python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Could not determine the Python version. Python 3.11 or newer is required."
}
try {
    $PythonVersion = [Version]$PythonVersionText
} catch {
    throw "Python returned an invalid version string: $PythonVersionText"
}
if ($PythonVersion -lt [Version]"3.11.0") {
    throw "Python 3.11 or newer is required; found $PythonVersionText."
}

$NodeVersionText = (& node --version).Trim().TrimStart("v")
if ($LASTEXITCODE -ne 0) {
    throw "Could not determine the Node.js version. Node.js 20 or newer is required."
}
try {
    $NodeVersion = [Version]$NodeVersionText
} catch {
    throw "Node.js returned an invalid version string: $NodeVersionText"
}
if ($NodeVersion -lt [Version]"20.0.0") {
    throw "Node.js 20 or newer is required; found $NodeVersionText."
}

$RustDetails = (& rustc -vV | Out-String)
if ($LASTEXITCODE -ne 0) {
    throw "Could not inspect the Rust toolchain."
}
$RustHostLine = ($RustDetails -split "`r?`n" | Where-Object { $_ -match '^host:\s+' } | Select-Object -First 1)
if (-not $RustHostLine) {
    throw "rustc -vV did not report a host toolchain."
}
$RustHost = ($RustHostLine -replace '^host:\s+', '').Trim()
if ($RustHost -notmatch 'pc-windows-msvc$') {
    throw "The Tauri Windows build requires an MSVC Rust host; found '$RustHost'. Install an *-pc-windows-msvc toolchain with rustup."
}

Write-Host "Prerequisites: Python $PythonVersionText; Node.js $NodeVersionText; Rust host $RustHost"

python scripts/verify_release_version.py
Assert-NativeSuccess "Release metadata consistency verification"
python -m pip install --upgrade -r requirements-dev.txt -c constraints-tested.txt
Assert-NativeSuccess "Python dependency installation"
python scripts/build_sidecar.py
Assert-NativeSuccess "Scientific sidecar build"
cargo check --manifest-path src-tauri/Cargo.toml --locked
Assert-NativeSuccess "cargo check --locked"
cargo test --manifest-path src-tauri/Cargo.toml --locked interpreter::tests
Assert-NativeSuccess "Rust interpreter tests"
cargo test --manifest-path src-tauri/Cargo.toml --locked response_
Assert-NativeSuccess "Rust response protocol tests"
cargo test --manifest-path src-tauri/Cargo.toml --locked pending_document_
Assert-NativeSuccess "Rust pending-document tests"
cargo test --manifest-path src-tauri/Cargo.toml --locked document_temp_
Assert-NativeSuccess "Rust document-temporary-file tests"
npm ci
Assert-NativeSuccess "npm ci"
npm run test:frontend
Assert-NativeSuccess "Frontend contract tests"
python -m pytest -q
Assert-NativeSuccess "Python test suite"
python verification/run_reference_verification.py
Assert-NativeSuccess "Reference verification"
python verification/run_bundle_verification.py
Assert-NativeSuccess "Bundle verification"
python verification/run_reproduction_verification.py
Assert-NativeSuccess "Reproduction verification"
python verification/run_expression_ast_verification.py
Assert-NativeSuccess "Expression-AST verification"
python verification/run_model_graph_protocol_verification.py
Assert-NativeSuccess "Model-graph protocol verification"
python verification/run_official_packs_verification.py
Assert-NativeSuccess "Official-packs verification"

$TauriCli = Join-Path $ProjectRoot "node_modules\.bin\tauri.cmd"
if (-not (Test-Path $TauriCli -PathType Leaf)) {
    throw "The local Tauri CLI was not installed by npm ci."
}
$BuildArguments = @("build", "--bundles", "nsis,msi")
if ($CertificateThumbprint) {
    $SigningConfiguration = @{
        bundle = @{
            windows = @{
                certificateThumbprint = $CertificateThumbprint
                digestAlgorithm = "sha256"
                timestampUrl = $TimestampUrl
                tsp = $false
            }
        }
    } | ConvertTo-Json -Depth 5 -Compress
    $BuildArguments += @("--config", $SigningConfiguration)
    Write-Host "Authenticode signing enabled for certificate $CertificateThumbprint."
} elseif ($RequireSigned) {
    throw "-RequireSigned was supplied without -CertificateThumbprint."
} else {
    Write-Warning "Building unsigned installers. Windows may display an Unknown publisher or SmartScreen warning."
}

& $TauriCli @BuildArguments
Assert-NativeSuccess "Tauri Windows installer build"

$VerificationScript = Join-Path $PSScriptRoot "verify_windows_bundle.ps1"
if ($RequireSigned) {
    & $VerificationScript -RequireSigned
} else {
    & $VerificationScript
}

if ($RunInstallerSmoke) {
    $ReleaseVersion = (Get-Content (Join-Path $ProjectRoot "src-tauri\tauri.conf.json") -Raw | ConvertFrom-Json).version
    $SmokeParameters = @{
        ExpectedVersion = $ReleaseVersion
        EvidencePath = Join-Path $ProjectRoot "src-tauri\target\release\bundle\INSTALLER_SMOKE_TEST.json"
    }
    if ($PreviousNsisInstaller) {
        $SmokeParameters.PreviousNsisInstaller = $PreviousNsisInstaller
    }
    & (Join-Path $PSScriptRoot "smoke_test_windows_installers.ps1") @SmokeParameters
}

Write-Host "Installers are available under src-tauri\target\release\bundle."
