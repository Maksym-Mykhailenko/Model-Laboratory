"""Structured evidence for the cross-machine reproducibility campaign."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping

from .canonical import canonical_json_sha256
from .reproduction import ReproductionStatus


COMPATIBILITY_OBSERVATION_SCHEMA = "model-laboratory-compatibility-observation"
COMPATIBILITY_OBSERVATION_SCHEMA_VERSION = "1.0"
COMPATIBILITY_CAMPAIGN_SCHEMA = "model-laboratory-compatibility-campaign"
COMPATIBILITY_CAMPAIGN_SCHEMA_VERSION = "1.0"


class CompatibilityCampaignError(ValueError):
    """Raised when cross-machine evidence is malformed or ambiguous."""


def _normalise_system(value: str) -> str:
    aliases = {"darwin": "macOS", "macos": "macOS", "windows": "Windows", "linux": "Linux"}
    return aliases.get(value.strip().lower(), value.strip())


def _normalise_architecture(value: str) -> str:
    aliases = {
        "amd64": "x86_64",
        "x86-64": "x86_64",
        "x64": "x86_64",
        "aarch64": "arm64",
    }
    return aliases.get(value.strip().lower(), value.strip().lower())


def _major_minor(value: str) -> str:
    parts = value.split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        raise CompatibilityCampaignError(f"Python version is malformed: {value!r}.")
    return ".".join(parts[:2])


def _backend_family(value: str) -> str:
    lowered = value.lower()
    for family, markers in (
        ("Apple Accelerate", ("accelerate", "veclib")),
        ("Intel MKL", ("mkl", "oneapi")),
        ("OpenBLAS", ("openblas",)),
        ("BLIS", ("blis",)),
    ):
        if any(marker in lowered for marker in markers):
            return family
    return value.strip() or "unavailable"


@dataclass(frozen=True, slots=True)
class CompatibilityTarget:
    operating_system: str
    architecture: str
    python_major_minor: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "operating_system", _normalise_system(self.operating_system))
        object.__setattr__(self, "architecture", _normalise_architecture(self.architecture))
        object.__setattr__(self, "python_major_minor", _major_minor(self.python_major_minor))
        if not self.operating_system or not self.architecture:
            raise CompatibilityCampaignError(
                "compatibility targets require an operating system and architecture."
            )

    @property
    def key(self) -> str:
        return f"{self.operating_system}-{self.architecture}-py{self.python_major_minor}"

    def to_dict(self) -> dict[str, str]:
        return {
            "operating_system": self.operating_system,
            "architecture": self.architecture,
            "python_major_minor": self.python_major_minor,
            "key": self.key,
        }


DEFAULT_COMPATIBILITY_TARGETS = (
    CompatibilityTarget("Linux", "x86_64", "3.11"),
    CompatibilityTarget("Linux", "x86_64", "3.12"),
    CompatibilityTarget("Windows", "x86_64", "3.12"),
    CompatibilityTarget("macOS", "arm64", "3.12"),
)


@dataclass(frozen=True, slots=True)
class CompatibilityCaseOutcome:
    name: str
    experiment_id: str
    status: ReproductionStatus
    environment_status: str
    report_sha256: str

    def __post_init__(self) -> None:
        if not self.name or not self.experiment_id:
            raise CompatibilityCampaignError("compatibility cases require a name and experiment id.")
        if len(self.report_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.report_sha256
        ):
            raise CompatibilityCampaignError("compatibility case report SHA-256 is invalid.")

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "environment_status": self.environment_status,
            "report_sha256": self.report_sha256,
        }


@dataclass(frozen=True, slots=True)
class CompatibilityObservation:
    """One real runner's outcomes and complete recorded environment."""

    target: CompatibilityTarget
    environment: tuple[tuple[str, str], ...]
    cases: tuple[CompatibilityCaseOutcome, ...]

    def __post_init__(self) -> None:
        names = [case.name for case in self.cases]
        if not self.cases or len(names) != len(set(names)):
            raise CompatibilityCampaignError(
                "an observation requires uniquely named compatibility cases."
            )
        environment_names = [name for name, _ in self.environment]
        if len(environment_names) != len(set(environment_names)):
            raise CompatibilityCampaignError("observation environment keys must be unique.")
        environment = dict(self.environment)
        detected = CompatibilityTarget(
            environment.get("operating_system", ""),
            environment.get("cpu_architecture", ""),
            environment.get("python", ""),
        )
        if detected != self.target:
            raise CompatibilityCampaignError(
                f"declared target {self.target.key} does not match recorded environment {detected.key}."
            )

    @property
    def blas_backend(self) -> str:
        return dict(self.environment).get("numpy_blas", "unavailable")

    @property
    def blas_backend_family(self) -> str:
        return _backend_family(self.blas_backend)

    @property
    def observation_sha256(self) -> str:
        return canonical_json_sha256(self.payload_dict())

    @property
    def successful(self) -> bool:
        return all(
            case.status in (
                ReproductionStatus.EXACT,
                ReproductionStatus.NUMERICAL,
                ReproductionStatus.STATISTICAL,
            )
            for case in self.cases
        )

    def payload_dict(self) -> dict[str, object]:
        return {
            "schema": COMPATIBILITY_OBSERVATION_SCHEMA,
            "schema_version": COMPATIBILITY_OBSERVATION_SCHEMA_VERSION,
            "target": self.target.to_dict(),
            "environment": [
                {"component": name, "version": version}
                for name, version in self.environment
            ],
            "cases": [case.to_dict() for case in self.cases],
        }

    def to_json(self) -> str:
        document = self.payload_dict()
        document["observation_sha256"] = self.observation_sha256
        return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "CompatibilityObservation":
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CompatibilityCampaignError("compatibility observation is not valid JSON.") from exc
        if not isinstance(document, dict):
            raise CompatibilityCampaignError("compatibility observation must be a JSON object.")
        supplied_sha256 = document.pop("observation_sha256", None)
        if supplied_sha256 != canonical_json_sha256(document):
            raise CompatibilityCampaignError("compatibility observation checksum is invalid.")
        try:
            target_raw = document["target"]
            target = CompatibilityTarget(
                target_raw["operating_system"],
                target_raw["architecture"],
                target_raw["python_major_minor"],
            )
            environment = tuple(
                (str(item["component"]), str(item["version"]))
                for item in document["environment"]
            )
            cases = tuple(
                CompatibilityCaseOutcome(
                    name=str(item["name"]),
                    experiment_id=str(item["experiment_id"]),
                    status=ReproductionStatus(item["status"]),
                    environment_status=str(item["environment_status"]),
                    report_sha256=str(item["report_sha256"]),
                )
                for item in document["cases"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CompatibilityCampaignError("compatibility observation is malformed.") from exc
        return cls(target, environment, cases)


@dataclass(frozen=True, slots=True)
class CompatibilityCampaignReport:
    """Merged matrix; missing runner evidence remains visible and never counts as success."""

    expected_targets: tuple[CompatibilityTarget, ...]
    observations: tuple[CompatibilityObservation, ...]

    def __post_init__(self) -> None:
        expected_keys = [target.key for target in self.expected_targets]
        observed_keys = [observation.target.key for observation in self.observations]
        if len(expected_keys) != len(set(expected_keys)):
            raise CompatibilityCampaignError("compatibility targets must be unique.")
        if len(observed_keys) != len(set(observed_keys)):
            raise CompatibilityCampaignError("duplicate observations for one target are not allowed.")
        unexpected = set(observed_keys) - set(expected_keys)
        if unexpected:
            raise CompatibilityCampaignError(
                "observations contain unexpected target(s): " + ", ".join(sorted(unexpected))
            )

    @property
    def missing_targets(self) -> tuple[CompatibilityTarget, ...]:
        observed = {item.target.key for item in self.observations}
        return tuple(target for target in self.expected_targets if target.key not in observed)

    @property
    def backend_count(self) -> int:
        return len({item.blas_backend_family for item in self.observations})

    @property
    def campaign_complete(self) -> bool:
        return (
            not self.missing_targets
            and all(item.successful for item in self.observations)
            and self.backend_count >= 2
        )

    def to_dict(self) -> dict[str, object]:
        observed = {item.target.key: item for item in self.observations}
        rows: list[dict[str, object]] = []
        for target in self.expected_targets:
            observation = observed.get(target.key)
            rows.append(
                {
                    "target": target.to_dict(),
                    "evidence": "MISSING" if observation is None else "RECORDED",
                    "observation_sha256": (
                        None if observation is None else observation.observation_sha256
                    ),
                    "blas_backend": None if observation is None else observation.blas_backend,
                    "blas_backend_family": (
                        None if observation is None else observation.blas_backend_family
                    ),
                    "cases": [] if observation is None else [case.to_dict() for case in observation.cases],
                }
            )
        return {
            "schema": COMPATIBILITY_CAMPAIGN_SCHEMA,
            "schema_version": COMPATIBILITY_CAMPAIGN_SCHEMA_VERSION,
            "complete": self.campaign_complete,
            "backend_count": self.backend_count,
            "missing_targets": [target.key for target in self.missing_targets],
            "matrix": rows,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def to_markdown(self) -> str:
        observed = {item.target.key: item for item in self.observations}
        lines = [
            "# Model Laboratory cross-platform compatibility matrix",
            "",
            f"Campaign complete: **{'yes' if self.campaign_complete else 'no'}**",
            "",
            "A row is recorded only from a real verification runner. Missing evidence is not inferred.",
            "",
            "| Target | Evidence | Exact | Numerical | Statistical | Failed/unable | BLAS backend |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for target in self.expected_targets:
            observation = observed.get(target.key)
            if observation is None:
                lines.append(f"| {target.key} | MISSING | — | — | — | — | — |")
                continue
            exact = sum(case.status is ReproductionStatus.EXACT for case in observation.cases)
            numerical = sum(
                case.status is ReproductionStatus.NUMERICAL for case in observation.cases
            )
            statistical = sum(
                case.status is ReproductionStatus.STATISTICAL for case in observation.cases
            )
            failed = len(observation.cases) - exact - numerical - statistical
            backend = observation.blas_backend.replace("|", "\\|")
            lines.append(
                f"| {target.key} | RECORDED | {exact} | {numerical} | {statistical} | {failed} | {backend} |"
            )
        lines.extend(
            [
                "",
                f"Distinct recorded BLAS backends: **{self.backend_count}** (two required for completion).",
                "",
            ]
        )
        return "\n".join(lines)


def merge_compatibility_observations(
    observations: Iterable[CompatibilityObservation],
    *,
    expected_targets: tuple[CompatibilityTarget, ...] = DEFAULT_COMPATIBILITY_TARGETS,
) -> CompatibilityCampaignReport:
    """Merge independently produced evidence without guessing missing outcomes."""
    return CompatibilityCampaignReport(expected_targets, tuple(observations))
