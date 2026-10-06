#!/usr/bin/env python3
"""Repair the previous checksum asset, wait for CI, then publish the verified patch release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_release_version import verify

REPOSITORY = "Maksym-Mykhailenko/Model-Laboratory"


def run(*arguments: str, capture: bool = False) -> str:
    completed = subprocess.run(arguments, cwd=ROOT, check=True, text=True,
                               stdout=subprocess.PIPE if capture else None)
    return completed.stdout.strip() if capture else ""


def wait_for_run(workflow: str, head: str, *, event: str, branch: str) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        entries = json.loads(run("gh", "run", "list", "--repo", REPOSITORY, "--workflow", workflow,
                                 "--commit", head, "--event", event, "--branch", branch,
                                 "--json", "databaseId,status,conclusion", "--limit", "10", capture=True))
        if entries:
            # The newest matching run is the one triggered by this source/tag push.
            run("gh", "run", "watch", str(entries[0]["databaseId"]), "--repo", REPOSITORY, "--exit-status")
            return
        time.sleep(5)
    raise RuntimeError(f"GitHub did not start {workflow} for {branch}. No release was published.")


def verify_downloads(directory: Path, version: str) -> None:
    required = {
        f"Model.Laboratory_{version}_x64-setup.exe",
        f"Model.Laboratory_{version}_x64_en-US.msi",
        f"Model.Laboratory_{version}_gaussian-hierarchy-case.zip",
        "SBOM.cdx.json", "INSTALLER_SMOKE_TEST.json", "SHA256SUMS.txt",
    }
    actual = {p.name for p in directory.iterdir() if p.is_file()}
    if not required <= actual:
        raise ValueError(f"Release is missing required downloads: {sorted(required - actual)}")
    covered = set()
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", 1)
        if Path(name).name != name or name in covered:
            raise ValueError("Invalid or duplicate release checksum filename.")
        digest = hashlib.sha256()
        with (directory / name).open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Release download checksum mismatch: {name}")
        covered.add(name)
    if required - {"SHA256SUMS.txt"} != covered:
        raise ValueError("Release checksum coverage differs from the required downloads.")
    smoke = json.loads((directory / "INSTALLER_SMOKE_TEST.json").read_text(encoding="utf-8-sig"))
    if smoke.get("status") != "PASS":
        raise ValueError("Release installer smoke evidence is not PASS.")
    if smoke.get("expected_version") != version:
        raise ValueError("Downloaded smoke receipt describes a different release.")
    sbom = json.loads((directory / "SBOM.cdx.json").read_text(encoding="utf-8"))
    if sbom["metadata"]["component"]["version"] != version:
        raise ValueError("Downloaded SBOM describes a different release.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="Push source/tag and publish only after successful CI and asset verification.")
    parser.add_argument("--repair-only", action="store_true", help="Replace only the v1.19.1 checksum asset.")
    args = parser.parse_args()
    if not (args.publish or args.repair_only):
        parser.error("Select --publish or --repair-only explicitly.")
    if not shutil.which("gh"):
        raise RuntimeError("GitHub CLI is required. Install it with: winget install --id GitHub.cli ; then run gh auth login")
    run("gh", "auth", "status")
    version = verify()
    tag = f"v{version}"
    if not args.repair_only:
        if run("git", "branch", "--show-current", capture=True) != "main":
            raise RuntimeError("Run publication from the main branch containing the reviewed patch.")
        if run("git", "status", "--porcelain", capture=True):
            raise RuntimeError("Commit the patch before publication; the working tree must be clean.")
        origin = run("git", "remote", "get-url", "origin", capture=True)
        if REPOSITORY.lower() not in origin.lower():
            raise RuntimeError("The origin remote does not identify the Model Laboratory repository.")
    repair_output = ROOT / ".desktop-build/release-repair/v1.19.1/SHA256SUMS.txt"
    run(sys.executable, str(ROOT / "scripts/repair_release_checksums.py"), "--tag", "v1.19.1",
        "--output", str(repair_output), "--upload")
    if args.repair_only:
        return 0
    head = run("git", "rev-parse", "HEAD", capture=True)
    run("git", "push", "origin", "main")
    wait_for_run("cross-platform-verification.yml", head, event="push", branch="main")
    remote_tag = run("git", "ls-remote", "origin", f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}", capture=True)
    if remote_tag:
        lines = remote_tag.splitlines()
        resolved = next((line.split()[0] for line in lines if line.endswith("^{}")), lines[0].split()[0])
        if resolved != head:
            raise RuntimeError(f"Existing {tag} identifies different source. It will not be moved.")
    else:
        local = subprocess.run(["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"], cwd=ROOT,
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if local.returncode == 0 and local.stdout.strip() != head:
            raise RuntimeError(f"Local {tag} identifies different source. It will not be moved.")
        if local.returncode != 0:
            run("git", "tag", "-a", tag, "-m", f"Model Laboratory {version}")
        run("git", "push", "origin", f"refs/tags/{tag}")
    wait_for_run("windows-release.yml", head, event="push", branch=tag)
    with tempfile.TemporaryDirectory(prefix="model-laboratory-release-check-") as temporary:
        directory = Path(temporary)
        run("gh", "release", "download", tag, "--repo", REPOSITORY, "--dir", str(directory))
        verify_downloads(directory, version)
    run("gh", "release", "edit", tag, "--repo", REPOSITORY, "--draft=false", "--latest")
    print(f"Published verified release: https://github.com/{REPOSITORY}/releases/tag/{tag}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Publication stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
