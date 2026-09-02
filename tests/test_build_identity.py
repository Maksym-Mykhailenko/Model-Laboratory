from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_lab import build_identity


ROOT = Path(__file__).resolve().parents[1]


def test_source_tree_identity_is_complete_and_deterministic() -> None:
    first = build_identity.calculate_source_tree_sha256(ROOT)
    second = build_identity.calculate_source_tree_sha256(ROOT)

    assert first == second
    assert len(first) == 64
    files = build_identity.source_identity_files(ROOT)
    assert ROOT / "desktop_engine.py" in files
    assert ROOT / "requirements.txt" in files
    assert ROOT / "model_lab" / "experiment.py" in files


def test_packaged_identity_fallback_uses_the_embedded_build_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = "a" * 64
    monkeypatch.setattr(build_identity, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        build_identity,
        "_embedded_build_identity",
        lambda: {
            "source_tree_sha256": expected,
            "git_commit_or_build_id": "release-test",
        },
    )

    assert build_identity.source_tree_sha256() == expected
    assert build_identity.git_commit_or_build_id() == "release-test"


def test_build_identity_record_is_json_portable() -> None:
    record = build_identity.calculate_build_identity(ROOT)

    restored = json.loads(json.dumps(record, sort_keys=True, allow_nan=False))
    assert restored["schema"] == "model-laboratory-build-identity-1"
    assert restored["source_tree_sha256"] == build_identity.calculate_source_tree_sha256(ROOT)
