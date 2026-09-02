"""Build the scientific Python engine as a Tauri sidecar for the host platform."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_lab.build_identity import calculate_build_identity


BUILD_ROOT = ROOT / ".desktop-build" / "sidecar"
BINARIES = ROOT / "src-tauri" / "binaries"


def _rust_host() -> str:
    try:
        output = subprocess.check_output(["rustc", "-vV"], text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("Rust is required to identify the Tauri target triple.") from exc
    for line in output.splitlines():
        if line.startswith("host: "):
            return line.removeprefix("host: ").strip()
    raise SystemExit("rustc did not report a host target triple.")


def _suffix(target: str) -> str:
    return ".exe" if "windows" in target else ""


def build(target: str) -> Path:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    BINARIES.mkdir(parents=True, exist_ok=True)
    dist = BUILD_ROOT / "dist"
    work = BUILD_ROOT / "work"
    spec = BUILD_ROOT / "spec"
    build_identity = calculate_build_identity(ROOT)
    identity_path = BUILD_ROOT / "_build_identity.json"
    identity_path.write_text(
        json.dumps(build_identity, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        "model-lab-engine",
        "--distpath",
        str(dist),
        "--workpath",
        str(work),
        "--specpath",
        str(spec),
        "--paths",
        str(ROOT),
        "--collect-all",
        "plotly",
        "--add-data",
        f"{ROOT / 'models'}{os.pathsep}models",
        "--add-data",
        f"{ROOT / 'model_lab' / 'baseline'}{os.pathsep}model_lab/baseline",
        "--add-data",
        f"{identity_path}{os.pathsep}model_lab",
        str(ROOT / "desktop_engine.py"),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    built = dist / f"model-lab-engine{_suffix(target)}"
    if not built.is_file():
        raise SystemExit(f"PyInstaller did not create {built}.")
    destination = BINARIES / f"model-lab-engine-{target}{_suffix(target)}"
    shutil.copy2(built, destination)

    requests = "".join(
        json.dumps(request) + "\n"
        for request in (
            {"id": 1, "action": "health", "payload": {}},
            {"id": 2, "action": "example_model", "payload": {}},
            {"id": 3, "action": "health", "payload": {}},
        )
    )
    result = subprocess.run(
        [str(destination)],
        input=requests,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    if result.returncode != 0:
        raise SystemExit(
            "The built sidecar failed its protocol health check:\n" + result.stderr.strip()
        )
    responses = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if len(responses) != 3 or [item.get("id") for item in responses] != [1, 2, 3]:
        raise SystemExit("The built sidecar did not preserve its persistent protocol session.")
    health = responses[0]
    repeated_health = responses[2]
    if (
        not health.get("ok")
        or health.get("result", {}).get("engine") != "python-sidecar"
        or health.get("result", {}).get("process_mode") != "persistent"
        or not repeated_health.get("ok")
    ):
        raise SystemExit("The built sidecar did not pass its protocol health check.")
    packaged_identity = health["result"].get("build_identity", {})
    if packaged_identity.get("source_tree_sha256") != build_identity["source_tree_sha256"]:
        raise SystemExit("The packaged sidecar did not retain its source-tree identity.")
    example = responses[1]
    if (
        not example.get("ok")
        or example.get("result", {}).get("model", {}).get("name") != "Quadratic example"
    ):
        raise SystemExit("The packaged sidecar could not load its bundled example model.")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", help="Tauri/Rust target triple; defaults to rustc host")
    args = parser.parse_args()
    destination = build(args.target or _rust_host())
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
