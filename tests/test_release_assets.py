from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from model_lab import __version__
from model_lab.bundle import create_run_mlab_bundle, load_mlab_bundle
from scripts.build_flagship_case import CASE_FILES
from scripts.prepare_release_assets import stage_assets
from scripts.repair_release_checksums import corrected_manifest
from scripts.verify_flagship_case import ROOT, verify_case


def test_flagship_release_is_frozen_current_and_reproduces() -> None:
    result = verify_case()
    assert result["authoring_status"] == "FROZEN"
    assert result["version"] == __version__
    assert result["reproduction"] in ("EXACT REPRODUCTION", "NUMERICALLY REPRODUCED")


def test_release_gate_rejects_a_valid_but_unfrozen_flagship(tmp_path: Path) -> None:
    for name in CASE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    bundle_path = tmp_path / CASE_FILES[1]
    frozen = load_mlab_bundle(bundle_path.read_bytes())
    bundle_path.write_bytes(create_run_mlab_bundle(state=frozen.state, model=frozen.model))
    with pytest.raises(ValueError, match="draft"):
        verify_case(tmp_path)


def _bundle_directory(root: Path) -> None:
    for directory, suffix in (("nsis", "x64-setup.exe"), ("msi", "x64_en-US.msi")):
        (root / directory).mkdir()
        (root / directory / f"Model Laboratory_{__version__}_{suffix}").write_bytes(directory.encode())
    (root / "SBOM.cdx.json").write_text(json.dumps({"bomFormat": "CycloneDX", "metadata": {"component": {"version": __version__}}}))
    (root / "INSTALLER_SMOKE_TEST.json").write_text(json.dumps({"status": "PASS", "expected_version": __version__}))


def test_staged_assets_use_download_names_and_have_a_complete_case(tmp_path: Path) -> None:
    _bundle_directory(tmp_path)
    staged = stage_assets(tmp_path, require_smoke=True)
    lines = (staged / "SHA256SUMS.txt").read_text().splitlines()
    covered = set()
    for line in lines:
        digest, name = line.split("  ", 1)
        assert " " not in name
        assert hashlib.sha256((staged / name).read_bytes()).hexdigest() == digest
        covered.add(name)
    assert covered == {p.name for p in staged.iterdir() if p.name != "SHA256SUMS.txt"}
    assert f"Model.Laboratory_{__version__}_x64-setup.exe" in covered
    assert f"Model.Laboratory_{__version__}_x64_en-US.msi" in covered
    archive = staged / f"Model.Laboratory_{__version__}_gaussian-hierarchy-case.zip"
    with zipfile.ZipFile(archive) as z:
        assert set(CASE_FILES) <= set(z.namelist())
        for line in z.read("examples/gaussian-hierarchy-dispersion-SHA256SUMS.txt").decode().splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256(z.read(name)).hexdigest() == digest


def test_staging_rejects_failed_smoke_and_colliding_names(tmp_path: Path) -> None:
    _bundle_directory(tmp_path)
    (tmp_path / "INSTALLER_SMOKE_TEST.json").write_text('{"status":"FAIL"}')
    with pytest.raises(ValueError, match="not PASS"):
        stage_assets(tmp_path, require_smoke=True)
    (tmp_path / "nsis" / f"Model.Laboratory_{__version__}_x64-setup.exe").write_bytes(b"collision")
    with pytest.raises(ValueError, match="collide"):
        stage_assets(tmp_path)


def test_published_manifest_repair_preserves_asset_hashes_and_checks_remote_digest() -> None:
    digest = hashlib.sha256(b"published installer").hexdigest()
    asset = {"name": "Model.Laboratory_1.19.1_x64-setup.exe", "digest": f"sha256:{digest}"}
    original = f"{digest}  Model Laboratory_1.19.1_x64-setup.exe\n"
    assert corrected_manifest(original, [asset]) == f"{digest}  {asset['name']}\n"
    with pytest.raises(ValueError, match="differs"):
        corrected_manifest(original, [{**asset, "digest": "sha256:" + "0" * 64}])
