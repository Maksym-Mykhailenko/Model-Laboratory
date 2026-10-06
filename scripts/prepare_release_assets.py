#!/usr/bin/env python3
"""Stage final GitHub asset basenames, the complete frozen case, SBOM and download checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_lab import __version__
from scripts.build_flagship_case import CASE_FILES
from scripts.verify_flagship_case import verify_case


def stage_assets(bundle_root: Path, *, require_smoke: bool = False) -> Path:
    verify_case()
    installers = sorted((bundle_root / "nsis").glob("*.exe")) + sorted((bundle_root / "msi").glob("*.msi"))
    if not any(p.suffix == ".exe" for p in installers) or not any(p.suffix == ".msi" for p in installers):
        raise ValueError("Both NSIS and MSI installers are required.")
    sbom_path = bundle_root / "SBOM.cdx.json"
    sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
    if sbom.get("bomFormat") != "CycloneDX" or sbom["metadata"]["component"]["version"] != __version__:
        raise ValueError("Release SBOM is absent or identifies a different release.")
    smoke_path = bundle_root / "INSTALLER_SMOKE_TEST.json"
    if require_smoke:
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        if smoke.get("status") != "PASS":
            raise ValueError("Installer smoke receipt is not PASS.")
        if smoke.get("expected_version") != __version__:
            raise ValueError("Installer smoke receipt identifies a different release.")
    filenames = [re.sub(r"[^A-Za-z0-9._-]", ".", p.name) for p in installers]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Normalised installer asset names collide.")
    if any(f"_{__version__}_" not in name for name in filenames):
        raise ValueError("Installer basename identifies a different release.")
    destination = bundle_root / "release-assets"
    destination.mkdir(parents=True, exist_ok=True)
    # Remove only the previous generated staging files; installers outside this folder remain intact.
    for old in destination.iterdir():
        if old.is_file():
            old.unlink()
        else:
            raise ValueError(f"Unexpected directory in release staging: {old}")
    for installer, name in zip(installers, filenames, strict=True):
        shutil.copyfile(installer, destination / name)
    shutil.copyfile(sbom_path, destination / sbom_path.name)
    if smoke_path.is_file():
        shutil.copyfile(smoke_path, destination / smoke_path.name)
    archive = destination / f"Model.Laboratory_{__version__}_gaussian-hierarchy-case.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in (*CASE_FILES, "examples/gaussian-hierarchy-dispersion-SHA256SUMS.txt", "CITATION.cff", "LICENSE"):
            z.write(ROOT / name, arcname=name)
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}"
             for p in sorted(destination.iterdir())]
    manifest = "\n".join(lines) + "\n"
    (destination / "SHA256SUMS.txt").write_text(manifest, encoding="ascii", newline="\n")
    (bundle_root / "SHA256SUMS.txt").write_text(manifest, encoding="ascii", newline="\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=ROOT / "src-tauri/target/release/bundle")
    parser.add_argument("--require-smoke", action="store_true")
    args = parser.parse_args()
    print(f"Staged release assets in {stage_assets(args.bundle_root, require_smoke=args.require_smoke)}")
