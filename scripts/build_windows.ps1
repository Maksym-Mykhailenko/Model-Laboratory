param(
    [string]$CertificateThumbprint = "",
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [switch]$RequireSigned,
    [switch]$RunInstallerSmoke,
    [string]$PreviousNsisInstaller = ""
)

$ErrorActionPreference = "Stop"

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
if ($LASTEXITCODE -ne 0) { throw "Release metadata consistency verification failed." }
cargo check --manifest-path src-tauri/Cargo.toml --locked
cargo test --manifest-path src-tauri/Cargo.toml --locked interpreter::tests
cargo test --manifest-path src-tauri/Cargo.toml --locked response_
cargo test --manifest-path src-tauri/Cargo.toml --locked pending_document_
cargo test --manifest-path src-tauri/Cargo.toml --locked document_temp_
python -m pip install --upgrade -r requirements-dev.txt -c constraints-tested.txt
npm ci
npm run test:frontend
python -m pytest -q
python verification/run_reference_verification.py
python verification/run_bundle_verification.py
python verification/run_reproduction_verification.py
python verification/run_expression_ast_verification.py
python verification/run_model_graph_protocol_verification.py
python verification/run_official_packs_verification.py
python scripts/build_sidecar.py

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
    Write-Warning "Building unsigned installers. Configure a certificate thumbprint for public distribution."
}

& $TauriCli @BuildArguments
if ($LASTEXITCODE -ne 0) {
    throw "The Tauri Windows installer build failed."
}

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
