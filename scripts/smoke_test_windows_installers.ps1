param(
    [string]$BundleRoot = "",
    [string]$PreviousNsisInstaller = "",
    [string]$ExpectedVersion = "1.18.0",
    [string]$EvidencePath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $BundleRoot) {
    $BundleRoot = Join-Path $ProjectRoot "src-tauri\target\release\bundle"
}
$BundleRoot = [IO.Path]::GetFullPath($BundleRoot)
if (-not $EvidencePath) {
    $EvidencePath = Join-Path $BundleRoot "INSTALLER_SMOKE_TEST.json"
}
$EvidencePath = [IO.Path]::GetFullPath($EvidencePath)
$NsisInstaller = Get-ChildItem (Join-Path $BundleRoot "nsis") -Filter "*.exe" -File | Select-Object -First 1
$MsiInstaller = Get-ChildItem (Join-Path $BundleRoot "msi") -Filter "*.msi" -File | Select-Object -First 1
if (-not $NsisInstaller -or -not $MsiInstaller) {
    throw "Both NSIS and MSI installers are required for smoke testing."
}

$Evidence = [ordered]@{
    schema = "model-laboratory-windows-installer-smoke"
    schema_version = "1.0"
    expected_version = $ExpectedVersion
    status = "RUNNING"
    started_at_utc = [DateTime]::UtcNow.ToString("o")
    completed_at_utc = $null
    runner = [ordered]@{
        os = [Environment]::OSVersion.VersionString
        architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    }
    installers = [ordered]@{
        nsis = $NsisInstaller.Name
        nsis_sha256 = (Get-FileHash -Algorithm SHA256 $NsisInstaller.FullName).Hash.ToLowerInvariant()
        msi = $MsiInstaller.Name
        msi_sha256 = (Get-FileHash -Algorithm SHA256 $MsiInstaller.FullName).Hash.ToLowerInvariant()
        previous_nsis = if ($PreviousNsisInstaller) { [IO.Path]::GetFileName($PreviousNsisInstaller) } else { $null }
        previous_nsis_sha256 = if ($PreviousNsisInstaller -and (Test-Path $PreviousNsisInstaller -PathType Leaf)) {
            (Get-FileHash -Algorithm SHA256 $PreviousNsisInstaller).Hash.ToLowerInvariant()
        } else { $null }
    }
    checks = [Collections.Generic.List[object]]::new()
    error = $null
}

function Add-SmokeCheck([string]$Name, [string]$Status, [string]$EvidenceText) {
    $Evidence.checks.Add([ordered]@{
        name = $Name
        status = $Status
        evidence = $EvidenceText
    })
}

function Write-SmokeEvidence {
    $Parent = Split-Path -Parent $EvidencePath
    if ($Parent) { New-Item -ItemType Directory -Path $Parent -Force | Out-Null }
    $Evidence | ConvertTo-Json -Depth 8 | Set-Content -Path $EvidencePath -Encoding utf8
}

function Invoke-CheckedProcess {
    param([string]$FilePath, [string[]]$ArgumentList, [int[]]$SuccessCodes = @(0))
    $Process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -PassThru -Wait
    if ($Process.ExitCode -notin $SuccessCodes) {
        throw "Process failed with exit code $($Process.ExitCode): $FilePath $($ArgumentList -join ' ')"
    }
}

function Get-ModelLaboratoryInstall {
    $Roots = @(
        "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*"
    )
    foreach ($Root in $Roots) {
        $Match = Get-ItemProperty $Root -ErrorAction SilentlyContinue |
            Where-Object { $_.DisplayName -eq "Model Laboratory" } |
            Select-Object -First 1
        if ($Match) { return $Match }
    }
    return $null
}

function Wait-ForInstall([bool]$Present) {
    for ($Attempt = 0; $Attempt -lt 30; $Attempt += 1) {
        $Record = Get-ModelLaboratoryInstall
        if ([bool]$Record -eq $Present) { return $Record }
        Start-Sleep -Milliseconds 500
    }
    if ($Present) { throw "Model Laboratory did not appear in the uninstall registry." }
    throw "Model Laboratory remained in the uninstall registry after uninstall."
}

function Resolve-AppExecutable($Record) {
    $Candidates = @()
    if ($Record.DisplayIcon) { $Candidates += ($Record.DisplayIcon -replace ',[0-9]+$', '').Trim('"') }
    if ($Record.InstallLocation) { $Candidates += Join-Path $Record.InstallLocation "Model Laboratory.exe" }
    $Candidates += Join-Path $env:LOCALAPPDATA "Model Laboratory\Model Laboratory.exe"
    $Candidates += Join-Path $env:ProgramFiles "Model Laboratory\Model Laboratory.exe"
    $Executable = $Candidates | Where-Object { $_ -and (Test-Path $_ -PathType Leaf) } | Select-Object -First 1
    if (-not $Executable) { throw "Could not locate the installed Model Laboratory executable." }
    return $Executable
}

function Assert-FileAssociation([string]$Extension, [string]$Executable) {
    $ExtensionKeyPath = "Registry::HKEY_CLASSES_ROOT\$Extension"
    if (-not (Test-Path $ExtensionKeyPath)) {
        throw "The $Extension file-association key is missing."
    }
    $ExtensionKey = Get-Item $ExtensionKeyPath
    $ProgIds = [Collections.Generic.List[string]]::new()
    $DefaultProgId = [string]$ExtensionKey.GetValue("")
    if ($DefaultProgId) { $ProgIds.Add($DefaultProgId) }
    $OpenWithPath = Join-Path $ExtensionKeyPath "OpenWithProgids"
    if (Test-Path $OpenWithPath) {
        (Get-Item $OpenWithPath).GetValueNames() | ForEach-Object {
            if ($_ -and -not $ProgIds.Contains($_)) { $ProgIds.Add($_) }
        }
    }
    $ExecutableName = [IO.Path]::GetFileName($Executable)
    foreach ($ProgId in $ProgIds) {
        $CommandPath = "Registry::HKEY_CLASSES_ROOT\$ProgId\shell\open\command"
        if (-not (Test-Path $CommandPath)) { continue }
        $Command = [string](Get-Item $CommandPath).GetValue("")
        if ($Command -and $Command.IndexOf($ExecutableName, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
            return $ProgId
        }
    }
    throw "$Extension is not registered with an open command for $ExecutableName."
}

function Assert-InstalledApplication([string]$PackageKind) {
    $Record = Wait-ForInstall $true
    if ($Record.DisplayVersion -ne $ExpectedVersion) {
        throw "$PackageKind installed version '$($Record.DisplayVersion)', expected '$ExpectedVersion'."
    }
    $Executable = Resolve-AppExecutable $Record
    $Associations = @(
        Assert-FileAssociation ".mlab" $Executable
        Assert-FileAssociation ".yaml" $Executable
        Assert-FileAssociation ".yml" $Executable
    )
    $App = Start-Process -FilePath $Executable -PassThru
    Start-Sleep -Seconds 8
    if ($App.HasExited -and $App.ExitCode -ne 0) {
        throw "$PackageKind application launch failed with exit code $($App.ExitCode)."
    }
    if (-not $App.HasExited) {
        Stop-Process -Id $App.Id -Force
        $App.WaitForExit()
    }
    return [ordered]@{
        record = $Record
        executable = $Executable
        associations = $Associations
    }
}

function Invoke-NsisUninstall($Record) {
    $Command = if ($Record.QuietUninstallString) { $Record.QuietUninstallString } else { $Record.UninstallString }
    if (-not $Command) { throw "The NSIS uninstall command is missing." }
    if ($Command -match '^"([^"]+)"\s*(.*)$') {
        $Executable = $Matches[1]
        $Arguments = $Matches[2]
    } else {
        $Parts = $Command -split '\s+', 2
        $Executable = $Parts[0]
        $Arguments = if ($Parts.Count -gt 1) { $Parts[1] } else { "" }
    }
    if ($Arguments -notmatch '(^|\s)/S($|\s)') { $Arguments = "$Arguments /S".Trim() }
    Invoke-CheckedProcess -FilePath $Executable -ArgumentList @($Arguments)
    Wait-ForInstall $false | Out-Null
}

function Assert-UninstalledApplication($InstallInfo) {
    Wait-ForInstall $false | Out-Null
    for ($Attempt = 0; $Attempt -lt 30 -and (Test-Path $InstallInfo.executable -PathType Leaf); $Attempt += 1) {
        Start-Sleep -Milliseconds 500
    }
    if (Test-Path $InstallInfo.executable -PathType Leaf) {
        throw "The application executable remained after uninstall: $($InstallInfo.executable)"
    }
    foreach ($ProgId in $InstallInfo.associations) {
        $CommandPath = "Registry::HKEY_CLASSES_ROOT\$ProgId\shell\open\command"
        if (-not (Test-Path $CommandPath)) { continue }
        $Command = [string](Get-Item $CommandPath).GetValue("")
        if ($Command -and $Command.IndexOf([IO.Path]::GetFileName($InstallInfo.executable), [StringComparison]::OrdinalIgnoreCase) -ge 0) {
            throw "The $ProgId open command remained registered after uninstall."
        }
    }
}

try {
    if ($PreviousNsisInstaller) {
        $PreviousNsisInstaller = [IO.Path]::GetFullPath($PreviousNsisInstaller)
        if (-not (Test-Path $PreviousNsisInstaller -PathType Leaf)) {
            throw "Previous NSIS installer does not exist: $PreviousNsisInstaller"
        }
        Invoke-CheckedProcess -FilePath $PreviousNsisInstaller -ArgumentList @("/S")
        Wait-ForInstall $true | Out-Null
        Invoke-CheckedProcess -FilePath $NsisInstaller.FullName -ArgumentList @("/S")
        $Upgrade = Assert-InstalledApplication "NSIS upgrade"
        Invoke-NsisUninstall $Upgrade.record
        Assert-UninstalledApplication $Upgrade
        Add-SmokeCheck "nsis-upgrade" "PASS" "Upgraded to $ExpectedVersion, launched the application, verified .mlab/.yaml/.yml associations, and uninstalled."
        Write-Host "NSIS upgrade smoke test passed."
    } else {
        Add-SmokeCheck "nsis-upgrade" "NOT_RUN" "No previous NSIS release asset was available."
    }

    Invoke-CheckedProcess -FilePath $NsisInstaller.FullName -ArgumentList @("/S")
    $Nsis = Assert-InstalledApplication "NSIS"
    Invoke-NsisUninstall $Nsis.record
    Assert-UninstalledApplication $Nsis
    Add-SmokeCheck "nsis-clean-install" "PASS" "Installed $ExpectedVersion, launched the application, verified .mlab/.yaml/.yml associations, and uninstalled."
    Write-Host "NSIS install, launch, association, and uninstall smoke test passed."

    Invoke-CheckedProcess -FilePath "msiexec.exe" -ArgumentList @("/i", "`"$($MsiInstaller.FullName)`"", "/qn", "/norestart") -SuccessCodes @(0, 3010)
    $Msi = Assert-InstalledApplication "MSI"
    Invoke-CheckedProcess -FilePath "msiexec.exe" -ArgumentList @("/x", "`"$($MsiInstaller.FullName)`"", "/qn", "/norestart") -SuccessCodes @(0, 3010)
    Assert-UninstalledApplication $Msi
    Add-SmokeCheck "msi-clean-install" "PASS" "Installed $ExpectedVersion, launched the application, verified .mlab/.yaml/.yml associations, and uninstalled."
    Write-Host "MSI install, launch, association, and uninstall smoke test passed."
    $Evidence.status = "PASS"
} catch {
    $Evidence.status = "FAIL"
    $Evidence.error = $_.Exception.Message
    throw
} finally {
    $Evidence.completed_at_utc = [DateTime]::UtcNow.ToString("o")
    Write-SmokeEvidence
    Write-Host "Installer smoke evidence written to $EvidencePath."
}
