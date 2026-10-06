#!/usr/bin/env python3
"""Correct a published checksum asset without rebuilding or replacing its installers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request

REPOSITORY = "Maksym-Mykhailenko/Model-Laboratory"


def corrected_manifest(original: str, assets: list[dict]) -> str:
    installers = {a["name"]: a for a in assets if a["name"].endswith((".exe", ".msi"))}
    lines = []
    seen = set()
    for line in original.splitlines():
        if not line.strip():
            continue
        digest, old_name = line.split("  ", 1)
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("Malformed SHA-256 in published manifest.")
        name = re.sub(r"[^A-Za-z0-9._-]", ".", old_name)
        if name not in installers or name in seen:
            raise ValueError(f"Checksum name does not identify a unique published installer: {old_name}")
        actual = installers[name].get("digest")
        if actual and actual.lower() != f"sha256:{digest.lower()}":
            raise ValueError(f"Published asset digest differs from manifest: {name}")
        lines.append(f"{digest.lower()}  {name}")
        seen.add(name)
    if seen != set(installers):
        raise ValueError("Published checksum manifest is incomplete.")
    return "\n".join(sorted(lines)) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="v1.19.1")
    parser.add_argument("--output", type=Path, default=Path("SHA256SUMS.txt"))
    parser.add_argument("--download-to", type=Path, help="Also download and independently hash the published installers.")
    parser.add_argument("--upload", action="store_true", help="Replace only SHA256SUMS.txt using authenticated GitHub CLI.")
    args = parser.parse_args()
    if args.output.name != "SHA256SUMS.txt":
        parser.error("The output basename must be SHA256SUMS.txt so upload replaces the correct asset.")
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", args.tag):
        parser.error("A version tag such as v1.19.1 is required.")
    request = urllib.request.Request(f"https://api.github.com/repos/{REPOSITORY}/releases/tags/{args.tag}",
                                     headers={"User-Agent": "Model-Laboratory-release-repair"})
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    if release.get("immutable"):
        raise ValueError("This release is immutable; its assets cannot be replaced.")
    assets = release["assets"]
    manifest_asset = next(a for a in assets if a["name"] == "SHA256SUMS.txt")
    with urllib.request.urlopen(manifest_asset["browser_download_url"], timeout=30) as response:
        original = response.read().decode("utf-8-sig")
    corrected = corrected_manifest(original, assets)
    if args.download_to:
        args.download_to.mkdir(parents=True, exist_ok=True)
        by_name = {a["name"]: a for a in assets}
        for line in corrected.splitlines():
            expected, name = line.split("  ", 1)
            path = args.download_to / name
            digest = hashlib.sha256()
            with urllib.request.urlopen(by_name[name]["browser_download_url"], timeout=60) as response, path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Installer download does not match recorded SHA-256: {name}")
            print(f"Verified downloaded {name}")
        (args.download_to / "SHA256SUMS.txt").write_text(corrected, encoding="ascii", newline="\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(corrected, encoding="ascii", newline="\n")
    print(f"Wrote corrected {args.output}")
    if args.upload:
        subprocess.run(["gh", "release", "upload", args.tag, str(args.output), "--clobber", "--repo", REPOSITORY], check=True)
        print(f"Replaced only the checksum asset on {args.tag}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
