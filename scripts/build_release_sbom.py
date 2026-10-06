#!/usr/bin/env python3
"""Generate a CycloneDX source/build inventory from the exact release environment and locks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from urllib.parse import quote
import uuid

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_lab import __version__


def python_inventory() -> tuple[list[dict], list[dict]]:
    """Include installed dependencies of the release requirements, excluding unused extras."""
    scopes: dict[str, str] = {}
    distributions = {}
    edges: dict[str, set[str]] = {}

    def visit(name: str, scope: str) -> None:
        name = canonicalize_name(name)
        if name in scopes and (scopes[name] == "required" or scope == "optional"):
            return
        distribution = metadata.distribution(name)
        scopes[name] = scope
        distributions[name] = distribution
        edges.setdefault(name, set())
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            dependency = canonicalize_name(requirement.name)
            installed = metadata.version(dependency)
            if requirement.specifier and installed not in requirement.specifier:
                raise ValueError(f"Installed {dependency} {installed} does not satisfy {raw}")
            edges[name].add(dependency)
            visit(dependency, scope)

    for filename, scope in (("requirements.txt", "required"), ("requirements-dev.txt", "optional")):
        for line in (ROOT / filename).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "-")):
                continue
            requirement = Requirement(line)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            installed = metadata.version(requirement.name)
            if installed not in requirement.specifier:
                raise ValueError(f"Installed {requirement.name} {installed} does not satisfy {line}")
            visit(requirement.name, scope)

    refs = {name: f"pkg:pypi/{name}@{distribution.version}" for name, distribution in distributions.items()}
    components = []
    for name, distribution in sorted(distributions.items()):
        component = {"type": "library", "bom-ref": refs[name], "name": name,
                     "version": distribution.version, "purl": refs[name], "scope": scopes[name]}
        licence = distribution.metadata.get("License-Expression")
        if licence:
            component["licenses"] = [{"expression": licence}]
        else:
            licence = distribution.metadata.get("License")
            if licence:
                component["licenses"] = [{"license": {"name": licence}}]
        components.append(component)
    dependencies = [{"ref": refs[name], "dependsOn": sorted(refs[dep] for dep in deps)}
                    for name, deps in sorted(edges.items())]
    return components, dependencies


def build_sbom(cargo_metadata: dict) -> dict:
    components, dependencies = python_inventory()
    root_ref = f"pkg:cargo/model-laboratory@{__version__}"
    application = {"type": "application", "bom-ref": root_ref, "name": "Model Laboratory",
                   "version": __version__, "licenses": [{"license": {"id": "Apache-2.0"}}]}
    cargo_lock = tomllib.loads((ROOT / "src-tauri/Cargo.lock").read_text(encoding="utf-8"))
    checksums = {(p["name"], p["version"]): p.get("checksum") for p in cargo_lock["package"]}
    cargo_refs = {p["id"]: f"pkg:cargo/{p['name']}@{p['version']}" for p in cargo_metadata["packages"]}
    for package in cargo_metadata["packages"]:
        if package["name"] == "model-laboratory":
            if package["version"] != __version__:
                raise ValueError("Cargo metadata identifies a different release.")
            continue
        ref = cargo_refs[package["id"]]
        component = {"type": "library", "bom-ref": ref, "name": package["name"],
                     "version": package["version"], "purl": ref,
                     "properties": [{"name": "model-laboratory:inventory", "value": "Cargo.lock; all targets"}]}
        if package.get("license"):
            component["licenses"] = [{"expression": package["license"]}]
        checksum = checksums.get((package["name"], package["version"]))
        if checksum:
            component["hashes"] = [{"alg": "SHA-256", "content": checksum}]
        if package["name"] == "glib" and package.get("source") is None:
            component["properties"].append({"name": "model-laboratory:source-patch",
                                           "value": "RUSTSEC-2024-0429; upstream PR 1343; src-tauri/vendor/glib/MODEL_LAB_PATCH.md"})
        components.append(component)
    for node in cargo_metadata["resolve"]["nodes"]:
        dependencies.append({"ref": cargo_refs[node["id"]],
                             "dependsOn": sorted({cargo_refs[d["pkg"]] for d in node["deps"]})})

    npm = json.loads((ROOT / "package-lock.json").read_text(encoding="utf-8"))
    if npm["version"] != __version__:
        raise ValueError("npm lock identifies a different release.")
    for path, package in npm["packages"].items():
        if not path:
            continue
        name = path.rsplit("node_modules/", 1)[-1]
        ref = f"pkg:npm/{quote(name, safe='/')}@{package['version']}"
        component = {"type": "library", "bom-ref": ref, "name": name,
                     "version": package["version"], "purl": ref, "scope": "optional",
                     "properties": [{"name": "model-laboratory:inventory", "value": "npm build tooling; all locked platforms"}]}
        if package.get("license"):
            component["licenses"] = [{"expression": package["license"]}]
        integrity = package.get("integrity", "")
        if integrity.startswith("sha512-"):
            import base64
            component["hashes"] = [{"alg": "SHA-512", "content": base64.b64decode(integrity[7:]).hex()}]
        components.append(component)

    for name in ("requirements.txt", "requirements-dev.txt", "constraints-tested.txt",
                 "package-lock.json", "src-tauri/Cargo.lock", "src-tauri/vendor/glib/src/variant_iter.rs",
                 "frontend/vendor/plotly.min.js"):
        component = {"type": "file", "bom-ref": f"file:{name}", "name": name,
                     "hashes": [{"alg": "SHA-256", "content": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}]}
        if name.endswith("plotly.min.js"):
            component["licenses"] = [{"license": {"id": "MIT"}}]
        components.append(component)
    # Combine the native root's dependency edge with Python, npm and bundled file inputs.
    root_dependencies = {c["bom-ref"] for c in components if not c["bom-ref"].startswith("pkg:cargo/")}
    retained = []
    for dependency in dependencies:
        if dependency["ref"] == root_ref:
            root_dependencies.update(dependency["dependsOn"])
        else:
            retained.append(dependency)
    retained.append({"ref": root_ref, "dependsOn": sorted(root_dependencies)})
    return {
        "bomFormat": "CycloneDX", "specVersion": "1.6", "version": 1,
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "metadata": {"timestamp": datetime.now(timezone.utc).isoformat(), "component": application,
                     "properties": [
                         {"name": "model-laboratory:scope", "value": "Resolved source/build inventory: installed Python runtime and build dependencies; all locked Cargo targets and npm build tools. Includes dependencies not shipped in the Windows binaries."},
                         {"name": "model-laboratory:python", "value": sys.version.split()[0]},
                     ]},
        "components": sorted(components, key=lambda item: item["bom-ref"]),
        "dependencies": sorted(retained, key=lambda item: item["ref"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cargo-metadata", type=Path, help="Reuse already captured locked Cargo metadata.")
    args = parser.parse_args()
    if args.cargo_metadata:
        cargo = json.loads(args.cargo_metadata.read_text(encoding="utf-8"))
    else:
        raw = subprocess.check_output(["cargo", "metadata", "--manifest-path", str(ROOT / "src-tauri/Cargo.toml"),
                                       "--locked", "--format-version", "1"], cwd=ROOT)
        cargo = json.loads(raw)
    result = build_sbom(cargo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}: {len(result['components'])} resolved components")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
