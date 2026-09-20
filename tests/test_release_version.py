from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_release_version", ROOT / "scripts" / "verify_release_version.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_release_metadata_and_notes_are_consistent() -> None:
    assert MODULE.verify("refs/tags/v1.18.0") == "1.18.0"


def test_release_tag_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="tag/version mismatch"):
        MODULE.verify("v9.9.9")


def test_release_workflows_pin_actions_and_require_signed_tags() -> None:
    workflows = list((ROOT / ".github" / "workflows").glob("*.yml"))
    uses = []
    for workflow in workflows:
        uses.extend(
            line.strip().removeprefix("- uses: ")
            for line in workflow.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("- uses: ")
        )
    assert uses
    assert all(re.fullmatch(r"[^@\s]+@[0-9a-f]{40}(?:\s+#.*)?", value) for value in uses)

    release = (ROOT / ".github" / "workflows" / "windows-release.yml").read_text(encoding="utf-8")
    assert 'throw "Tagged releases require WINDOWS_CERTIFICATE_BASE64."' in release
    assert "$Parameters.RequireSigned = $true" in release
    assert "RunInstallerSmoke = $true" in release
    assert "scripts/verify_release_version.py --expected-version" in release
    assert "Discover previous NSIS release for upgrade smoke test" in release
    assert "$Parameters.PreviousNsisInstaller = $env:MODEL_LAB_PREVIOUS_NSIS" in release
    assert "INSTALLER_SMOKE_TEST.json" in release

    smoke = (ROOT / "scripts" / "smoke_test_windows_installers.ps1").read_text(encoding="utf-8")
    assert 'schema = "model-laboratory-windows-installer-smoke"' in smoke
    assert '$Evidence.status = "PASS"' in smoke
    assert '$Evidence.status = "FAIL"' in smoke
    assert "nsis_sha256 = (Get-FileHash -Algorithm SHA256" in smoke
    assert "msi_sha256 = (Get-FileHash -Algorithm SHA256" in smoke
    assert "Assert-UninstalledApplication $Nsis" in smoke
    assert "Assert-UninstalledApplication $Msi" in smoke
    for extension in (".mlab", ".yaml", ".yml"):
        assert f'Assert-FileAssociation "{extension}" $Executable' in smoke


def test_windows_release_inputs_are_locked_and_smoke_paths_are_normalized() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".desktop-build/" in gitignore.splitlines()

    lock = tomllib.loads((ROOT / "src-tauri" / "Cargo.lock").read_text(encoding="utf-8"))
    packages = {(package["name"], package["version"]): package for package in lock["package"]}
    hyper = packages[("hyper", "1.11.1")]
    assert hyper["checksum"] == "27b501faa50e7a26c3d3560ca625132f4078a17771f4810baf70475ae48cbe43"
    assert ("tokio-macros", "2.7.2") in packages
    assert "tokio-macros" in packages[("tokio", "1.53.1")]["dependencies"]

    build = (ROOT / "scripts" / "build_windows.ps1").read_text(encoding="utf-8")
    assert 'Assert-NativeSuccess "cargo check --locked"' in build
    assert 'Assert-NativeSuccess "Python test suite"' in build
    assert 'Assert-NativeSuccess "Scientific sidecar build"' in build
    assert build.index("python scripts/build_sidecar.py") < build.index(
        "cargo check --manifest-path"
    )

    native_windows = (ROOT / "scripts" / "verify_native_interpreter.ps1").read_text(
        encoding="utf-8"
    )
    assert native_windows.index("python scripts/build_sidecar.py") < native_windows.index(
        "cargo check --manifest-path"
    )
    assert 'Assert-NativeSuccess "cargo check --locked"' in native_windows

    native_unix = (ROOT / "scripts" / "verify_native_interpreter.sh").read_text(
        encoding="utf-8"
    )
    assert native_unix.index("python3 scripts/build_sidecar.py") < native_unix.index(
        "cargo check --manifest-path"
    )

    smoke = (ROOT / "scripts" / "smoke_test_windows_installers.ps1").read_text(encoding="utf-8")
    assert "function ConvertFrom-RegistryPathValue" in smoke
    assert "function Get-AppExecutableNames" in smoke
    assert "ConvertFrom-RegistryPathValue -Value $Record.InstallLocation" in smoke
    assert '$ManifestPath = Join-Path $ProjectRoot "src-tauri\\Cargo.toml"' in smoke
    assert 'foreach ($FallbackName in @("Model Laboratory.exe", "model-laboratory.exe"))' in smoke
    assert 'Get-ChildItem -LiteralPath $InstallLocation -Filter "*.exe" -File' in smoke
    assert "candidates checked: $CandidateSummary" in smoke
    assert "function Get-ShellAssociationExecutable" in smoke
    assert 'DllImport("Shlwapi.dll"' in smoke
    assert "ASSOCSTR_EXECUTABLE" in smoke
    assert "function Test-EquivalentExecutablePath" in smoke
    assert 'Get-ShellAssociationExecutable $ProgId' in smoke
    assert '$ExpectedProgId = "Model Laboratory$Extension"' in smoke
    assert '$ProgIds.Add($ExpectedProgId) | Out-Null' in smoke
    assert '$AdvertisedDescriptor = $CommandKey.GetValue("command")' in smoke
    assert 'Extension default=\'$DefaultSummary\'' in smoke
    assert "advertised descriptor=$AdvertisedSummary" in smoke
    assert "Remove-InstalledApplicationBestEffort" in smoke

    workflow = (ROOT / ".github" / "workflows" / "cross-platform-verification.yml").read_text(
        encoding="utf-8"
    )
    assert "merge-multiple: true" not in workflow
    assert "--merge observations/*/*.json" in workflow
