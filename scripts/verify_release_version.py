#!/usr/bin/env python3
"""Fail closed when release metadata or a version tag disagrees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$")


def _match(path: Path, pattern: str, label: str) -> str:
    text = path.read_text(encoding="utf-8")
    found = re.search(pattern, text, re.MULTILINE)
    if not found:
        raise ValueError(f"Could not find {label} in {path.relative_to(ROOT)}")
    return found.group(1)


def release_versions(root: Path = ROOT) -> dict[str, str]:
    package = json.loads((root / "package.json").read_text(encoding="utf-8"))
    package_lock = json.loads((root / "package-lock.json").read_text(encoding="utf-8"))
    tauri = json.loads((root / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8"))
    cargo = tomllib.loads((root / "src-tauri" / "Cargo.toml").read_text(encoding="utf-8"))
    cargo_lock = (root / "src-tauri" / "Cargo.lock").read_text(encoding="utf-8")
    locked = re.search(
        r'\[\[package\]\]\s+name = "model-laboratory"\s+version = "([^"]+)"',
        cargo_lock,
    )
    if not locked:
        raise ValueError("Could not find model-laboratory in src-tauri/Cargo.lock")
    return {
        "python": _match(root / "model_lab" / "__init__.py", r'^__version__ = "([^"]+)"$', "Python version"),
        "npm": str(package["version"]),
        "npm-lock-root": str(package_lock["version"]),
        "npm-lock-package": str(package_lock["packages"][""]["version"]),
        "tauri": str(tauri["version"]),
        "cargo": str(cargo["package"]["version"]),
        "cargo-lock": locked.group(1),
        "citation": _match(root / "CITATION.cff", r"^version:\s*([^\s]+)$", "citation version"),
        "desktop-footer": _match(root / "frontend" / "index.html", r"Desktop v([^<]+)", "desktop footer version"),
    }


def normalize_expected(value: str) -> str:
    value = value.strip()
    if value.startswith("refs/tags/"):
        value = value.removeprefix("refs/tags/")
    return value.removeprefix("v")


def verify(expected: str | None = None, root: Path = ROOT) -> str:
    versions = release_versions(root)
    unique = set(versions.values())
    if len(unique) != 1:
        details = ", ".join(f"{name}={version}" for name, version in versions.items())
        raise ValueError(f"Release versions disagree: {details}")
    version = unique.pop()
    if not SEMVER.fullmatch(version):
        raise ValueError(f"Release version is not supported SemVer: {version}")
    if expected is not None and normalize_expected(expected) != version:
        raise ValueError(f"Release tag/version mismatch: tag {expected!r}, metadata {version!r}")
    notes = root / "docs" / "releases" / f"RELEASE_NOTES_{version}.md"
    if not notes.is_file():
        raise ValueError(f"Release notes are missing: {notes.relative_to(root)}")
    return version


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-version", help="Version or refs/tags/vX.Y.Z tag that must match")
    args = parser.parse_args()
    try:
        version = verify(args.expected_version)
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"release-version verification failed: {error}", file=sys.stderr)
        return 1
    print(f"Release metadata is consistent: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
