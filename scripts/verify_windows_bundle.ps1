param(
    [string]$BundleRoot = "",
    [switch]$RequireSigned
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $BundleRoot) {
    $BundleRoot = Join-Path $ProjectRoot "src-tauri\target\release\bundle"
}
$BundleRoot = [System.IO.Path]::GetFullPath($BundleRoot)
if (-not (Test-Path $BundleRoot -PathType Container)) {
    throw "Windows bundle directory does not exist: $BundleRoot"
}

$NsisInstallers = @(Get-ChildItem (Join-Path $BundleRoot "nsis") -Filter "*.exe" -File -ErrorAction SilentlyContinue)
$MsiInstallers = @(Get-ChildItem (Join-Path $BundleRoot "msi") -Filter "*.msi" -File -ErrorAction SilentlyContinue)
if ($NsisInstallers.Count -lt 1) {
    throw "The build did not produce an NSIS .exe installer."
}
if ($MsiInstallers.Count -lt 1) {
    throw "The build did not produce an MSI installer."
}

$Installers = @($NsisInstallers + $MsiInstallers)
$ChecksumLines = foreach ($Installer in $Installers | Sort-Object FullName) {
    if ($Installer.Length -lt 1MB) {
        throw "Installer is implausibly small: $($Installer.FullName)"
    }
    $Signature = Get-AuthenticodeSignature -FilePath $Installer.FullName
    if ($RequireSigned -and $Signature.Status -ne "Valid") {
        throw "Installer signature is not valid ($($Signature.Status)): $($Installer.FullName)"
    }
    if (-not $RequireSigned -and $Signature.Status -ne "Valid") {
        Write-Warning "Unsigned installer: $($Installer.Name)"
    }
    $Hash = Get-FileHash -Algorithm SHA256 -Path $Installer.FullName
    "$($Hash.Hash.ToLowerInvariant())  $($Installer.Name)"
}

$ChecksumPath = Join-Path $BundleRoot "SHA256SUMS.txt"
$ChecksumLines | Set-Content -Path $ChecksumPath -Encoding ascii
Write-Host "Verified $($Installers.Count) Windows installers and wrote $ChecksumPath."
