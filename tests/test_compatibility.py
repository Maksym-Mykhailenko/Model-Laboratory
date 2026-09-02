from __future__ import annotations

import pytest

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
    assert len(report.missing_targets) == 3
    assert "MISSING" in report.to_markdown()


def test_campaign_rejects_duplicate_runner_evidence() -> None:
    observation = _observation(DEFAULT_COMPATIBILITY_TARGETS[1])
    with pytest.raises(CompatibilityCampaignError, match="duplicate"):
        merge_compatibility_observations([observation, observation])


def test_complete_campaign_requires_successful_cases_and_backend_diversity() -> None:
    observations = [
        _observation(target, "OpenBLAS" if index < 3 else "Apple Accelerate")
        for index, target in enumerate(DEFAULT_COMPATIBILITY_TARGETS)
    ]
    report = merge_compatibility_observations(observations)

    assert report.campaign_complete
    assert report.backend_count == 2
    assert not report.missing_targets
