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

function ConvertFrom-RegistryPathValue {
    param(
        [AllowNull()][object]$Value,
        [switch]$RemoveIconIndex
    )
    if ($null -eq $Value) { return $null }
    $Text = [Environment]::ExpandEnvironmentVariables(([string]$Value).Trim())
    if (-not $Text) { return $null }
    if ($RemoveIconIndex) {
        $Text = $Text -replace ',\s*-?\d+\s*$', ''
    }
    if ($Text -match '^"([^"]+)"') {
        return $Matches[1]
    }
    return $Text.Trim('"')
}

function Get-ShellAssociationExecutable([string]$Association) {
    try {
        if (-not ("ModelLaboratory.ShellAssociation" -as [type])) {
            Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace ModelLaboratory
{
    public static class ShellAssociation
    {
        private const uint ASSOCSTR_EXECUTABLE = 2;

        [DllImport("Shlwapi.dll", CharSet = CharSet.Unicode, ExactSpelling = true)]
        private static extern int AssocQueryStringW(
            uint flags,
            uint str,
            string association,
            string extra,
            StringBuilder output,
            ref uint outputLength);

        public static string GetExecutable(string association)
        {
            uint outputLength = 0;
            AssocQueryStringW(0, ASSOCSTR_EXECUTABLE, association, "open", null, ref outputLength);
            if (outputLength == 0)
            {
                return null;
            }

            var output = new StringBuilder(checked((int)outputLength));
            var result = AssocQueryStringW(
                0,
                ASSOCSTR_EXECUTABLE,
                association,
                "open",
                output,
                ref outputLength);
            return result == 0 ? output.ToString() : null;
        }
    }
}
'@
        }
        return [ModelLaboratory.ShellAssociation]::GetExecutable($Association)
    } catch {
        return $null
    }
}

function Test-EquivalentExecutablePath {
    param(
        [AllowNull()][string]$Expected,
        [AllowNull()][string]$Actual
    )
    if (-not $Expected -or -not $Actual) { return $false }
    try {
        $ExpectedPath = [IO.Path]::GetFullPath($Expected)
        $ActualPath = [IO.Path]::GetFullPath($Actual)
        return $ExpectedPath.Equals($ActualPath, [StringComparison]::OrdinalIgnoreCase)
    } catch {
        return $false
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

function Get-AppExecutableNames {
    $Names = [Collections.Generic.List[string]]::new()

    $ConfigPath = Join-Path $ProjectRoot "src-tauri\tauri.conf.json"
    if (Test-Path $ConfigPath -PathType Leaf) {
        $Config = Get-Content $ConfigPath -Raw | ConvertFrom-Json
        foreach ($ConfiguredName in @($Config.mainBinaryName, $Config.productName)) {
            if (-not $ConfiguredName) { continue }
            $ExecutableName = [string]$ConfiguredName
            if (-not $ExecutableName.EndsWith(".exe", [StringComparison]::OrdinalIgnoreCase)) {
                $ExecutableName += ".exe"
            }
            if (-not $Names.Contains($ExecutableName)) { $Names.Add($ExecutableName) | Out-Null }
        }
    }

    # Tauri's default WiX template keeps the Cargo binary filename even when productName differs.
    $ManifestPath = Join-Path $ProjectRoot "src-tauri\Cargo.toml"
    if (Test-Path $ManifestPath -PathType Leaf) {
        $InPackageSection = $false
        foreach ($Line in Get-Content $ManifestPath) {
            if ($Line -match '^\s*\[([^]]+)\]\s*$') {
                $InPackageSection = $Matches[1] -eq "package"
                continue
            }
            if ($InPackageSection -and $Line -match '^\s*name\s*=\s*"([^"]+)"') {
                $ExecutableName = "$($Matches[1]).exe"
                if (-not $Names.Contains($ExecutableName)) { $Names.Add($ExecutableName) | Out-Null }
                break
            }
        }
    }

    foreach ($FallbackName in @("Model Laboratory.exe", "model-laboratory.exe")) {
        if (-not $Names.Contains($FallbackName)) { $Names.Add($FallbackName) | Out-Null }
    }
    return $Names.ToArray()
}

function Add-ExecutableCandidate {
    param(
        [Collections.Generic.List[string]]$Candidates,
        [AllowNull()][string]$Path
    )
    if ($Path -and -not $Candidates.Contains($Path)) { $Candidates.Add($Path) | Out-Null }
}

function Resolve-AppExecutable($Record) {
    $Candidates = [Collections.Generic.List[string]]::new()
    $ExecutableNames = @(Get-AppExecutableNames)
    $DisplayIcon = ConvertFrom-RegistryPathValue -Value $Record.DisplayIcon -RemoveIconIndex
    if ($DisplayIcon -and [IO.Path]::GetExtension($DisplayIcon) -eq ".exe") {
        Add-ExecutableCandidate -Candidates $Candidates -Path $DisplayIcon
    }

    $InstallLocation = ConvertFrom-RegistryPathValue -Value $Record.InstallLocation
    $InstallRoots = [Collections.Generic.List[string]]::new()
    if ($InstallLocation) { $InstallRoots.Add($InstallLocation) | Out-Null }
    if ($env:LOCALAPPDATA) {
        $InstallRoots.Add((Join-Path $env:LOCALAPPDATA "Model Laboratory")) | Out-Null
        $InstallRoots.Add((Join-Path $env:LOCALAPPDATA "Programs\Model Laboratory")) | Out-Null
    }
    foreach ($ProgramFilesRoot in @($env:ProgramW6432, $env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if ($ProgramFilesRoot) {
            $CandidateRoot = Join-Path $ProgramFilesRoot "Model Laboratory"
            if (-not $InstallRoots.Contains($CandidateRoot)) { $InstallRoots.Add($CandidateRoot) | Out-Null }
        }
    }

    foreach ($InstallRoot in $InstallRoots) {
        foreach ($ExecutableName in $ExecutableNames) {
            Add-ExecutableCandidate -Candidates $Candidates -Path (
                Join-Path -Path $InstallRoot -ChildPath $ExecutableName
            )
        }
    }

    # If the binary is renamed later, constrain fallback discovery to the installer's own directory.
    if ($InstallLocation -and (Test-Path $InstallLocation -PathType Container)) {
        Get-ChildItem -LiteralPath $InstallLocation -Filter "*.exe" -File |
            Where-Object { $_.Name -notmatch '^(?i:unins|uninstall|model-lab-engine)' } |
            ForEach-Object {
                Add-ExecutableCandidate -Candidates $Candidates -Path $_.FullName
            }
    }

    $Executable = $Candidates | Where-Object { $_ -and (Test-Path $_ -PathType Leaf) } | Select-Object -First 1
    if (-not $Executable) {
        $CandidateSummary = if ($Candidates.Count) { $Candidates -join "; " } else { "<none>" }
        throw "Could not locate the installed Model Laboratory executable. InstallLocation='$InstallLocation'; DisplayIcon='$DisplayIcon'; candidates checked: $CandidateSummary"
    }
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
    if ($DefaultProgId) { $ProgIds.Add($DefaultProgId) | Out-Null }
    $OpenWithPath = Join-Path $ExtensionKeyPath "OpenWithProgids"
    if (Test-Path $OpenWithPath) {
        (Get-Item $OpenWithPath).GetValueNames() | ForEach-Object {
            if ($_ -and -not $ProgIds.Contains($_)) { $ProgIds.Add($_) | Out-Null }
        }
    }
    $ExecutableName = [IO.Path]::GetFileName($Executable)
    $ExpectedProgId = "Model Laboratory$Extension"
    $Diagnostics = [Collections.Generic.List[string]]::new()
    foreach ($ProgId in $ProgIds) {
        $CommandPath = "Registry::HKEY_CLASSES_ROOT\$ProgId\shell\open\command"
        if (-not (Test-Path $CommandPath)) {
            $Diagnostics.Add("$ProgId (open command key missing)") | Out-Null
            continue
        }
        $CommandKey = Get-Item $CommandPath
        $Command = [string]$CommandKey.GetValue("")
        if ($Command -and $Command.IndexOf($ExecutableName, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
            return $ProgId
        }

        # WiX emits advertised MSI ProgId/Extension/Verb rows. Windows Installer represents
        # their command with a Darwin descriptor instead of a literal executable command.
        $ResolvedExecutable = Get-ShellAssociationExecutable $ProgId
        if (Test-EquivalentExecutablePath -Expected $Executable -Actual $ResolvedExecutable) {
            return $ProgId
        }
        $AdvertisedDescriptor = $CommandKey.GetValue("command")
        $IsExpectedProgId = $ProgId.Equals($ExpectedProgId, [StringComparison]::OrdinalIgnoreCase)
        if ($IsExpectedProgId -and $null -ne $AdvertisedDescriptor) {
            return $ProgId
        }

        $ResolvedSummary = if ($ResolvedExecutable) { $ResolvedExecutable } else { "<unresolved>" }
        $AdvertisedSummary = if ($null -ne $AdvertisedDescriptor) { "present" } else { "absent" }
        $Diagnostics.Add(
            "$ProgId (default='$Command'; shell='$ResolvedSummary'; advertised descriptor=$AdvertisedSummary)"
        ) | Out-Null
    }
    $DiagnosticSummary = if ($Diagnostics.Count) { $Diagnostics -join "; " } else { "<no ProgIDs>" }
    throw "$Extension is not registered to open with $ExecutableName. Checked: $DiagnosticSummary"
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

function Remove-InstalledApplicationBestEffort {
    $Record = Get-ModelLaboratoryInstall
    if (-not $Record) { return }
    try {
        $UninstallCommand = [string]$Record.UninstallString
        if ([int]$Record.WindowsInstaller -eq 1 -or $UninstallCommand -match '(?i)msiexec(?:\.exe)?') {
            $ProductCode = [string]$Record.PSChildName
            if ($ProductCode -notmatch '^\{[0-9A-Fa-f-]+\}$' -and $UninstallCommand -match '\{[0-9A-Fa-f-]+\}') {
                $ProductCode = $Matches[0]
            }
            if ($ProductCode -notmatch '^\{[0-9A-Fa-f-]+\}$') {
                throw "Could not determine the MSI product code for failure cleanup."
            }
            Invoke-CheckedProcess -FilePath "msiexec.exe" -ArgumentList @("/x", $ProductCode, "/qn", "/norestart") -SuccessCodes @(0, 1605, 3010)
            Wait-ForInstall $false | Out-Null
        } else {
            Invoke-NsisUninstall $Record
        }
        Add-SmokeCheck "failure-cleanup" "PASS" "Removed the installed application after the smoke-test failure."
    } catch {
        Add-SmokeCheck "failure-cleanup" "FAIL" $_.Exception.Message
        Write-Warning "Installer failure cleanup did not complete: $($_.Exception.Message)"
    }
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
        $CommandKey = Get-Item $CommandPath
        $Command = [string]$CommandKey.GetValue("")
        $AdvertisedDescriptor = $CommandKey.GetValue("command")
        if (
            ($Command -and $Command.IndexOf([IO.Path]::GetFileName($InstallInfo.executable), [StringComparison]::OrdinalIgnoreCase) -ge 0) -or
            $null -ne $AdvertisedDescriptor
        ) {
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
    if ($Evidence.status -eq "FAIL") {
        Remove-InstalledApplicationBestEffort
    }
    $Evidence.completed_at_utc = [DateTime]::UtcNow.ToString("o")
    Write-SmokeEvidence
    Write-Host "Installer smoke evidence written to $EvidencePath."
}
