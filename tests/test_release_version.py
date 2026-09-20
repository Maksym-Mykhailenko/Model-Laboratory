from __future__ import annotations

import importlib.util
from pathlib import Path
import re

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
