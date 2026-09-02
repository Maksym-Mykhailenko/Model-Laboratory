"""Stable source/build identity across source and packaged desktop execution.

Source checkouts can hash the executable Python files directly.  A PyInstaller one-file
sidecar cannot: its source files are compiled into the executable and extracted into a
different runtime layout.  Release builds therefore embed the identity calculated from
the source checkout.  Packaged execution reads that embedded record rather than hashing
an incomplete temporary tree.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping


EMBEDDED_IDENTITY_RESOURCE = "_build_identity.json"


class BuildIdentityError(RuntimeError):
    """Raised when a complete source tree cannot be identified deterministically."""


def project_root() -> Path:
    """Return the source-project root for an ordinary Python installation."""
    return Path(__file__).resolve().parents[1]


def source_identity_files(root: Path) -> tuple[Path, ...]:
    """Return the complete ordered set of files defining the Python engine build."""
    model_package = root / "model_lab"
    required = (
        root / "desktop_engine.py",
        root / "requirements.txt",
        model_package / "__init__.py",
    )
    if not all(path.is_file() for path in required):
        missing = ", ".join(path.as_posix() for path in required if not path.is_file())
        raise BuildIdentityError(f"The executable source tree is incomplete: {missing}.")
    package_sources = tuple(
        path
        for path in sorted(model_package.glob("*.py"), key=lambda item: item.name)
        if path.name != EMBEDDED_IDENTITY_RESOURCE
    )
    return tuple(sorted((*required[:2], *package_sources), key=lambda item: item.as_posix()))


def calculate_source_tree_sha256(root: Path) -> str:
    """Hash the exact executable source/dependency inputs at ``root``."""
    digest = hashlib.sha256()
    for path in source_identity_files(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def calculate_build_id(root: Path) -> str:
    """Return an explicit build id or the source checkout's Git commit when available."""
    configured = os.environ.get("MODEL_LAB_BUILD_ID", "").strip()
    if configured:
        return configured
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return completed.stdout.strip() or "unavailable"


def calculate_build_identity(root: Path) -> dict[str, str]:
    """Create the record embedded in a packaged scientific sidecar."""
    return {
        "schema": "model-laboratory-build-identity-1",
        "source_tree_sha256": calculate_source_tree_sha256(root),
        "git_commit_or_build_id": calculate_build_id(root),
    }


def _embedded_build_identity() -> Mapping[str, Any]:
    try:
        resource = importlib.resources.files("model_lab").joinpath(
            EMBEDDED_IDENTITY_RESOURCE
        )
        value = json.loads(resource.read_text(encoding="utf-8"))
    except (FileNotFoundError, ModuleNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def source_tree_sha256() -> str:
    """Return one source identity in both a checkout and its packaged sidecar."""
    root = project_root()
    try:
        return calculate_source_tree_sha256(root)
    except BuildIdentityError:
        embedded = _embedded_build_identity().get("source_tree_sha256")
        if isinstance(embedded, str) and len(embedded) == 64:
            return embedded
        raise BuildIdentityError(
            "Packaged execution is missing its embedded source-tree identity."
        )


def git_commit_or_build_id() -> str:
    """Return a stable build id, falling back to the embedded release record."""
    configured = os.environ.get("MODEL_LAB_BUILD_ID", "").strip()
    if configured:
        return configured
    root = project_root()
    try:
        source_identity_files(root)
    except BuildIdentityError:
        embedded = _embedded_build_identity().get("git_commit_or_build_id")
        if isinstance(embedded, str) and embedded.strip():
            return embedded.strip()
        return "unavailable"
    return calculate_build_id(root)
