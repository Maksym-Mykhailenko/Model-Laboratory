from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from model_lab.canonical import canonical_json_sha256
from model_lab.compatibility import (
    CompatibilityCampaignError,
    CompatibilityCaseOutcome,
    CompatibilityObservation,
    CompatibilityTarget,
    DEFAULT_COMPATIBILITY_TARGETS,
    merge_compatibility_observations,
)
from model_lab.reproduction import ReproductionStatus


def _observation(target: CompatibilityTarget, backend: str = "OpenBLAS") -> CompatibilityObservation:
    environment = (
        ("python", target.python_major_minor + ".9"),
        ("operating_system", "Darwin" if target.operating_system == "macOS" else target.operating_system),
        ("cpu_architecture", target.architecture),
        ("numpy_blas", backend),
    )
    report = {"target": target.key, "status": ReproductionStatus.EXACT.value}
    return CompatibilityObservation(
        target,
        environment,
        (
            CompatibilityCaseOutcome(
                "legacy-v1.1-expression-ast.mlab",
                "fixture-experiment",
                ReproductionStatus.EXACT,
                "compatible",
                canonical_json_sha256(report),
            ),
        ),
    )


def test_target_normalises_common_platform_architecture_names() -> None:
    assert CompatibilityTarget("Darwin", "aarch64", "3.12.4").key == "macOS-arm64-py3.12"
    assert CompatibilityTarget("Windows", "AMD64", "3.12").key == "Windows-x86_64-py3.12"


def test_observation_round_trip_preserves_real_evidence_checksum() -> None:
    observation = _observation(DEFAULT_COMPATIBILITY_TARGETS[1])
    restored = CompatibilityObservation.from_json(observation.to_json())

    assert restored == observation
    assert restored.observation_sha256 == observation.observation_sha256


def test_campaign_keeps_missing_platforms_visible() -> None:
    report = merge_compatibility_observations(
        [_observation(DEFAULT_COMPATIBILITY_TARGETS[1])]
    )

    assert not report.campaign_complete
    assert len(report.missing_targets) == 4
    assert "MISSING" in report.to_markdown()


def test_campaign_rejects_duplicate_runner_evidence() -> None:
    observation = _observation(DEFAULT_COMPATIBILITY_TARGETS[1])
    with pytest.raises(CompatibilityCampaignError, match="duplicate"):
        merge_compatibility_observations([observation, observation])


def test_complete_campaign_requires_successful_cases_and_backend_diversity() -> None:
    observations = [
        _observation(
            target,
            "Apple Accelerate" if target.operating_system == "macOS" else "OpenBLAS",
        )
        for target in DEFAULT_COMPATIBILITY_TARGETS
    ]
    report = merge_compatibility_observations(observations)

    assert report.campaign_complete
    assert report.backend_count == 2
    assert not report.missing_targets


def test_default_campaign_targets_match_ci_matrix() -> None:
    workflow_path = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "cross-platform-verification.yml"
    )
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    runner_platforms = {
        "ubuntu-latest": ("Linux", "x86_64"),
        "windows-latest": ("Windows", "x86_64"),
        "macos-14": ("macOS", "arm64"),
    }
    workflow_targets: set[CompatibilityTarget] = set()
    for entry in workflow["jobs"]["verify"]["strategy"]["matrix"]["include"]:
        operating_system, architecture = runner_platforms[entry["os"]]
        target = CompatibilityTarget(operating_system, architecture, entry["python"])
        assert entry["artifact"] == target.key.lower().replace("py3.", "py3")
        workflow_targets.add(target)

    assert workflow_targets == set(DEFAULT_COMPATIBILITY_TARGETS)
