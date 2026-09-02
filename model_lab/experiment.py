"""Experiment-state capture and reproducibility verification for Model Laboratory.

An experiment state records the exact supplied model text, current parameter values,
selected visualisations, deterministic numerical settings, optional parameter-sweep
configuration, accepted local-interpreter provenance, software environment, strict result
fingerprints and numerical reference data.
The state is serialised as JSON and can be loaded later to reconstruct and verify the experiment.

The experiment state does not alter the model.  It records how the validated model was
examined at one point in time.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
from functools import lru_cache
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
import zlib
from typing import Any, Mapping

import numpy as np
import sympy as sp
from scipy.optimize import linear_sum_assignment

from . import __version__
from .analysis import ParameterSweepResult, StationaryPointAnalysisResult
from .build_identity import git_commit_or_build_id, source_tree_sha256
from .canonical import canonical_model_ir_sha256
from .capabilities import AnalysisCapability, AnalysisVisualisation, ModelVisualisation
from .critical_points import OneDimensionalCriticalPoint, TwoDimensionalCriticalPoint
from .evaluator import FunctionEvaluation, SurfaceEvaluation
from .issues import NumericalDiagnostic
from .interpreter import (
    InterpreterProposalError,
    validate_interpreter_acceptance_document,
)
from .model import ModelIR
from .parser import ModelParseError, parse_model_text
from .protocol import (
    ArtifactView,
    ProtocolError,
    RunOutcome,
    validate_artifact_document,
    validate_run_document,
    validate_view_document,
)
from .symbolic import analyse_one_variable_function, analyse_two_variable_function
from .validator import ModelValidationError, validate_legacy_model, validate_model
from .workload import WorkloadBudgetError, enforce_experiment_workload_budget


EXPERIMENT_FORMAT_VERSION = 6
_SUPPORTED_EXPERIMENT_FORMAT_VERSIONS = (1, 2, 3, 4, 5, 6)
_RECORDED_PACKAGES = (
    "python",
    "numpy",
    "scipy",
    "sympy",
    "pydantic",
    "PyYAML",
    "plotly",
)


class ExperimentStateError(ValueError):
    """Raised when an experiment-state document is malformed or fails checksum/semantic checks."""


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """Deterministic grid settings used for direct model evaluation."""

    points_1d: int = 1000
    points_per_axis_2d: int = 150

    def __post_init__(self) -> None:
        if not isinstance(self.points_1d, int) or not 2 <= self.points_1d <= 200_000:
            raise ExperimentStateError("points_1d must be an integer between 2 and 200000.")
        if not isinstance(self.points_per_axis_2d, int) or not 2 <= self.points_per_axis_2d <= 1000:
            raise ExperimentStateError(
                "points_per_axis_2d must be an integer between 2 and 1000."
            )


@dataclass(frozen=True, slots=True)
class StationaryPointSettings:
    """Deterministic search settings used by stationary-point analysis."""

    samples_1d: int = 2001
    seeds_per_axis_2d: int = 11
    root_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if not isinstance(self.samples_1d, int) or not 10 <= self.samples_1d <= 200_001:
            raise ExperimentStateError("samples_1d must be an integer between 10 and 200001.")
        if not isinstance(self.seeds_per_axis_2d, int) or not 3 <= self.seeds_per_axis_2d <= 101:
            raise ExperimentStateError(
                "seeds_per_axis_2d must be an integer between 3 and 101."
            )
        if not np.isfinite(self.root_tolerance) or self.root_tolerance <= 0:
            raise ExperimentStateError("root_tolerance must be a positive finite number.")


@dataclass(frozen=True, slots=True)
class SweepConfiguration:
    """Configuration of one completed parameter sweep."""

    parameter_name: str
    start: float
    end: float
    step_count: int
    selected_visualisation: str | None = None

    def __post_init__(self) -> None:
        if not self.parameter_name:
            raise ExperimentStateError("A sweep parameter name is required.")
        if isinstance(self.start, bool) or isinstance(self.end, bool):
            raise ExperimentStateError("Sweep bounds must be finite real numbers.")
        try:
            start = float(self.start)
            end = float(self.end)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ExperimentStateError("Sweep bounds must be finite real numbers.") from exc
        if not np.isfinite(start) or not np.isfinite(end):
            raise ExperimentStateError("Sweep bounds must be finite.")
        if start >= end:
            raise ExperimentStateError("Sweep start must be smaller than sweep end.")
        if not isinstance(self.step_count, int) or isinstance(self.step_count, bool) or not 2 <= self.step_count <= 201:
            raise ExperimentStateError("Sweep step_count must be an integer between 2 and 201.")
        # Canonicalise numerically equivalent integer/float inputs so serialise-load-serialise
        # preserves the exact experiment-state checksum.
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True, slots=True)
class NumericalReproductionSettings:
    """Explicit tolerances used when comparing saved and reproduced numerical values."""

    relative_tolerance: float = 1e-8
    absolute_tolerance: float = 1e-11

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.relative_tolerance)
            or self.relative_tolerance <= 0
            or self.relative_tolerance > 1e-4
        ):
            raise ExperimentStateError(
                "relative_tolerance must be a positive finite number no greater than 1e-4."
            )
        if (
            not np.isfinite(self.absolute_tolerance)
            or self.absolute_tolerance <= 0
            or self.absolute_tolerance > 1e-6
        ):
            raise ExperimentStateError(
                "absolute_tolerance must be a positive finite number no greater than 1e-6."
            )


@dataclass(frozen=True, slots=True)
class NumericalComparison:
    """Outcome of an explicit saved-value versus reproduced-value comparison."""

    name: str
    matches: bool
    comparison_kind: str
    max_absolute_deviation: float | None = None
    max_relative_deviation: float | None = None
    details: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReproducibilityCheck:
    """Comparison between a saved experiment and the currently reconstructed results.

    Result identity and execution-environment identity are intentionally independent.
    A strict result can reproduce exactly on a different supported environment; the
    environment difference remains explicit evidence in the report rather than changing
    the mathematical result comparison.
    """

    model_source_matches: bool
    laboratory_version_matches: bool
    environment_matches: bool
    result_matches: tuple[tuple[str, bool], ...]
    model_ir_matches: bool = True
    numerical_result_matches: tuple[tuple[str, bool], ...] = ()
    numerical_comparisons: tuple[NumericalComparison, ...] = ()
    legacy_tolerance_fingerprint_matches: tuple[tuple[str, bool], ...] = ()
    environment_differences: tuple[str, ...] = ()

    @property
    def all_results_match(self) -> bool:
        return bool(self.result_matches) and all(matches for _, matches in self.result_matches)

    @property
    def numerical_reproduction_available(self) -> bool:
        return bool(self.numerical_result_matches)

    @property
    def all_numerical_results_match(self) -> bool:
        return self.numerical_reproduction_available and all(
            matches for _, matches in self.numerical_result_matches
        )

    @property
    def all_results_match_numerically_or_strictly(self) -> bool:
        """Require numerical agreement where references exist and strict agreement otherwise."""
        if not self.result_matches:
            return False
        numerical = dict(self.numerical_result_matches)
        return all(
            numerical.get(name, strict_match)
            for name, strict_match in self.result_matches
        )

    @property
    def legacy_tolerance_fingerprint_available(self) -> bool:
        return bool(self.legacy_tolerance_fingerprint_matches)

    @property
    def exact_reproduction(self) -> bool:
        return self.model_source_matches and self.model_ir_matches and self.all_results_match

    @property
    def numerically_reproduced(self) -> bool:
        return (
            self.model_source_matches
            and self.model_ir_matches
            and self.numerical_reproduction_available
            and self.all_results_match_numerically_or_strictly
        )


@dataclass(frozen=True, slots=True)
class ExperimentState:
    """Complete reproducibility record for one laboratory experiment."""

    format_version: int
    laboratory_version: str
    created_at_utc: str
    model_source: str
    model_sha256: str
    parameter_values: tuple[tuple[str, float], ...]
    selected_model_visualisation: str
    evaluation_settings: EvaluationSettings
    stationary_settings: StationaryPointSettings
    sweep: SweepConfiguration | None
    result_fingerprints: tuple[tuple[str, str], ...]
    numerical_reproduction_settings: NumericalReproductionSettings = NumericalReproductionSettings()
    numerical_reference_data: tuple[tuple[str, dict[str, Any]], ...] = ()
    numerical_result_fingerprints: tuple[tuple[str, str], ...] = ()
    interpreter_acceptances: tuple[dict[str, Any], ...] = ()
    run_records: tuple[dict[str, Any], ...] = ()
    artifacts: tuple[dict[str, Any], ...] = ()
    views: tuple[dict[str, Any], ...] = ()
    environment: tuple[tuple[str, str], ...] = ()
    state_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format_version not in _SUPPORTED_EXPERIMENT_FORMAT_VERSIONS:
            raise ExperimentStateError(
                f"Unsupported experiment format version: {self.format_version}."
            )
        if not self.model_source.strip():
            raise ExperimentStateError("Experiment state contains an empty model source.")
        if _sha256_text(self.model_source) != self.model_sha256:
            raise ExperimentStateError("Experiment model-source hash does not match its contents.")
        names = [name for name, _ in self.parameter_values]
        if len(names) != len(set(names)):
            raise ExperimentStateError("Experiment parameter values contain duplicate names.")
        if any(not np.isfinite(value) for _, value in self.parameter_values):
            raise ExperimentStateError("Experiment parameter values must be finite.")
        fingerprint_names = [name for name, _ in self.result_fingerprints]
        if len(fingerprint_names) != len(set(fingerprint_names)):
            raise ExperimentStateError("Experiment result fingerprints contain duplicate names.")
        numerical_reference_names = [name for name, _ in self.numerical_reference_data]
        if len(numerical_reference_names) != len(set(numerical_reference_names)):
            raise ExperimentStateError(
                "Experiment numerical reference data contain duplicate names."
            )
        if any(not isinstance(reference, dict) for _, reference in self.numerical_reference_data):
            raise ExperimentStateError("Experiment numerical reference data must be JSON objects.")
        if self.format_version >= 2 and not self.numerical_reference_data and not self.artifacts:
            raise ExperimentStateError(
                "Experiment format version 2 or later requires numerical references or typed artifacts."
            )
        numerical_fingerprint_names = [name for name, _ in self.numerical_result_fingerprints]
        if len(numerical_fingerprint_names) != len(set(numerical_fingerprint_names)):
            raise ExperimentStateError(
                "Experiment numerical result fingerprints contain duplicate names."
            )
        if self.format_version < 3 and self.interpreter_acceptances:
            raise ExperimentStateError(
                "Interpreter acceptance provenance requires experiment format version 3."
            )
        if len(self.interpreter_acceptances) > 256:
            raise ExperimentStateError(
                "Experiment interpreter acceptance provenance exceeds 256 records."
            )
        if self.format_version == 3 and any(
            item.get("schema_version") != "1.0"
            for item in self.interpreter_acceptances
            if isinstance(item, Mapping)
        ):
            raise ExperimentStateError(
                "Experiment format version 3 supports only interpreter acceptance schema 1.0."
            )
        canonical_acceptances: list[dict[str, Any]] = []
        try:
            canonical_acceptances = [
                validate_interpreter_acceptance_document(item)
                for item in self.interpreter_acceptances
            ]
        except InterpreterProposalError as exc:
            raise ExperimentStateError(f"Invalid interpreter acceptance provenance: {exc}") from exc
        proposal_ids = [item["proposal_id"] for item in canonical_acceptances]
        acceptance_hashes = [item["acceptance_sha256"] for item in canonical_acceptances]
        if len(proposal_ids) != len(set(proposal_ids)) or len(acceptance_hashes) != len(
            set(acceptance_hashes)
        ):
            raise ExperimentStateError(
                "Experiment interpreter acceptance provenance contains duplicates."
            )
        for previous, current in zip(canonical_acceptances, canonical_acceptances[1:]):
            if (
                current["previous_model_source_sha256"]
                != previous["accepted_model_source_sha256"]
            ):
                raise ExperimentStateError(
                    "Experiment interpreter acceptance provenance is not a continuous source lineage."
                )
        if (
            canonical_acceptances
            and canonical_acceptances[-1]["accepted_model_source_sha256"]
            != self.model_sha256
        ):
            raise ExperimentStateError(
                "The last interpreter acceptance does not identify the embedded model source."
            )
        object.__setattr__(self, "interpreter_acceptances", tuple(canonical_acceptances))
        if self.format_version < 6 and (self.run_records or self.artifacts or self.views):
            raise ExperimentStateError(
                "Typed runs, artifacts, and views require experiment-state format version 6."
            )
        try:
            canonical_runs = tuple(validate_run_document(item) for item in self.run_records)
            canonical_artifacts = tuple(
                validate_artifact_document(item) for item in self.artifacts
            )
            canonical_views = tuple(validate_view_document(item) for item in self.views)
        except ProtocolError as exc:
            raise ExperimentStateError(f"Invalid Run -> Artifact state: {exc}") from exc
        artifact_ids = [str(item["artifact_id"]) for item in canonical_artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ExperimentStateError("Experiment contains duplicate scientific artifacts.")
        known_artifacts = set(artifact_ids)
        for run in canonical_runs:
            if not set(run["artifact_ids"]).issubset(known_artifacts):
                raise ExperimentStateError("A run record refers to an unavailable artifact.")
        for view in canonical_views:
            if not set(view["artifact_ids"]).issubset(known_artifacts):
                raise ExperimentStateError("An artifact view refers to an unavailable artifact.")
        object.__setattr__(self, "run_records", canonical_runs)
        object.__setattr__(self, "artifacts", canonical_artifacts)
        object.__setattr__(self, "views", canonical_views)
        environment_names = [name for name, _ in self.environment]
        if len(environment_names) != len(set(environment_names)):
            raise ExperimentStateError("Experiment environment contains duplicate components.")

    @property
    def fingerprint_map(self) -> dict[str, str]:
        return dict(self.result_fingerprints)

    @property
    def parameter_map(self) -> dict[str, float]:
        return dict(self.parameter_values)

    @property
    def environment_map(self) -> dict[str, str]:
        return dict(self.environment)

    def payload_dict(self) -> dict[str, Any]:
        """Return the canonical serialisable payload, excluding the outer checksum."""
        payload = {
            "format_version": self.format_version,
            "laboratory_version": self.laboratory_version,
            "created_at_utc": self.created_at_utc,
            "model_source": self.model_source,
            "model_sha256": self.model_sha256,
            "parameter_values": [
                {"name": name, "value": value} for name, value in self.parameter_values
            ],
            "selected_model_visualisation": self.selected_model_visualisation,
            "evaluation_settings": {
                "points_1d": self.evaluation_settings.points_1d,
                "points_per_axis_2d": self.evaluation_settings.points_per_axis_2d,
            },
            "stationary_settings": {
                "samples_1d": self.stationary_settings.samples_1d,
                "seeds_per_axis_2d": self.stationary_settings.seeds_per_axis_2d,
                "root_tolerance": self.stationary_settings.root_tolerance,
            },
            "sweep": None
            if self.sweep is None
            else {
                "parameter_name": self.sweep.parameter_name,
                "start": self.sweep.start,
                "end": self.sweep.end,
                "step_count": self.sweep.step_count,
                "selected_visualisation": self.sweep.selected_visualisation,
            },
            "result_fingerprints": [
                {"name": name, "sha256": digest}
                for name, digest in self.result_fingerprints
            ],
            "numerical_reproduction_settings": {
                "relative_tolerance": self.numerical_reproduction_settings.relative_tolerance,
                "absolute_tolerance": self.numerical_reproduction_settings.absolute_tolerance,
            },
            "numerical_reference_data": [
                {"name": name, "reference": reference}
                for name, reference in self.numerical_reference_data
            ],
            "numerical_result_fingerprints": [
                {"name": name, "sha256": digest}
                for name, digest in self.numerical_result_fingerprints
            ],
            "environment": [
                {"component": component, "version": version}
                for component, version in self.environment
            ],
        }
        if self.format_version >= 3:
            payload["interpreter_acceptances"] = list(self.interpreter_acceptances)
        if self.format_version >= 6:
            payload["run_records"] = list(self.run_records)
            payload["artifacts"] = list(self.artifacts)
            payload["views"] = list(self.views)
        return payload

    def with_checksum(self) -> "ExperimentState":
        """Return a copy carrying a SHA-256 corruption-detection checksum."""
        digest = _sha256_json(self.payload_dict())
        return ExperimentState(
            format_version=self.format_version,
            laboratory_version=self.laboratory_version,
            created_at_utc=self.created_at_utc,
            model_source=self.model_source,
            model_sha256=self.model_sha256,
            parameter_values=self.parameter_values,
            selected_model_visualisation=self.selected_model_visualisation,
            evaluation_settings=self.evaluation_settings,
            stationary_settings=self.stationary_settings,
            sweep=self.sweep,
            result_fingerprints=self.result_fingerprints,
            numerical_reproduction_settings=self.numerical_reproduction_settings,
            numerical_reference_data=self.numerical_reference_data,
            numerical_result_fingerprints=self.numerical_result_fingerprints,
            interpreter_acceptances=self.interpreter_acceptances,
            run_records=self.run_records,
            artifacts=self.artifacts,
            views=self.views,
            environment=self.environment,
            state_sha256=digest,
        )

    def with_integrity_hash(self) -> "ExperimentState":
        """Backward-compatible alias for :meth:`with_checksum`."""
        return self.with_checksum()

    def to_json(self, *, indent: int = 2) -> str:
        state = self.with_checksum()
        document = state.payload_dict()
        document["state_sha256"] = state.state_sha256
        return json.dumps(document, indent=indent, sort_keys=True, ensure_ascii=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "ExperimentState":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExperimentStateError(f"Experiment state is not valid JSON: {exc.msg}.") from exc
        if not isinstance(raw, dict):
            raise ExperimentStateError("Experiment state must be a JSON object.")

        required = {
            "format_version",
            "laboratory_version",
            "created_at_utc",
            "model_source",
            "model_sha256",
            "parameter_values",
            "selected_model_visualisation",
            "evaluation_settings",
            "stationary_settings",
            "sweep",
            "result_fingerprints",
            "environment",
            "state_sha256",
        }
        optional = {
            "numerical_reproduction_settings",
            "numerical_reference_data",
            "numerical_result_fingerprints",
        }
        try:
            document_format_version = int(raw.get("format_version"))
        except (TypeError, ValueError) as exc:
            raise ExperimentStateError("Experiment format_version must be an integer.") from exc
        if document_format_version >= 3:
            required.add("interpreter_acceptances")
        if document_format_version >= 6:
            required.update({"run_records", "artifacts", "views"})
        allowed = required | optional
        extra = set(raw) - allowed
        missing = required - set(raw)
        if extra:
            raise ExperimentStateError(
                "Experiment state contains unknown field(s): " + ", ".join(sorted(extra)) + "."
            )
        if missing:
            raise ExperimentStateError(
                "Experiment state is missing field(s): " + ", ".join(sorted(missing)) + "."
            )

        supplied_digest = raw.get("state_sha256")
        payload = {key: value for key, value in raw.items() if key != "state_sha256"}
        computed_digest = _sha256_json(payload)
        if not isinstance(supplied_digest, str) or supplied_digest != computed_digest:
            raise ExperimentStateError("Experiment state checksum does not match its contents.")

        try:
            evaluation_raw = _require_mapping(raw["evaluation_settings"], "evaluation_settings")
            stationary_raw = _require_mapping(raw["stationary_settings"], "stationary_settings")
            sweep_raw = raw["sweep"]
            sweep = None
            if sweep_raw is not None:
                sweep_map = _require_mapping(sweep_raw, "sweep")
                sweep = SweepConfiguration(
                    parameter_name=str(sweep_map["parameter_name"]),
                    start=float(sweep_map["start"]),
                    end=float(sweep_map["end"]),
                    step_count=int(sweep_map["step_count"]),
                    selected_visualisation=(
                        None
                        if sweep_map.get("selected_visualisation") is None
                        else str(sweep_map["selected_visualisation"])
                    ),
                )

            parameter_values = tuple(
                (str(item["name"]), float(item["value"]))
                for item in _require_list(raw["parameter_values"], "parameter_values")
                if isinstance(item, dict)
            )
            if len(parameter_values) != len(_require_list(raw["parameter_values"], "parameter_values")):
                raise ExperimentStateError("Each parameter_values entry must be an object.")

            result_fingerprints = tuple(
                (str(item["name"]), str(item["sha256"]))
                for item in _require_list(raw["result_fingerprints"], "result_fingerprints")
                if isinstance(item, dict)
            )
            if len(result_fingerprints) != len(_require_list(raw["result_fingerprints"], "result_fingerprints")):
                raise ExperimentStateError("Each result_fingerprints entry must be an object.")


            numerical_settings_raw = raw.get("numerical_reproduction_settings")
            if numerical_settings_raw is None:
                numerical_settings = NumericalReproductionSettings()
            else:
                numerical_settings_map = _require_mapping(
                    numerical_settings_raw, "numerical_reproduction_settings"
                )
                numerical_settings = NumericalReproductionSettings(
                    relative_tolerance=float(numerical_settings_map["relative_tolerance"]),
                    absolute_tolerance=float(numerical_settings_map["absolute_tolerance"]),
                )

            numerical_references_raw = raw.get("numerical_reference_data", [])
            numerical_reference_data = tuple(
                (str(item["name"]), dict(_require_mapping(item["reference"], "numerical_reference_data.reference")))
                for item in _require_list(
                    numerical_references_raw, "numerical_reference_data"
                )
                if isinstance(item, dict) and "reference" in item
            )
            if len(numerical_reference_data) != len(
                _require_list(numerical_references_raw, "numerical_reference_data")
            ):
                raise ExperimentStateError(
                    "Each numerical_reference_data entry must contain a name and reference object."
                )

            numerical_fingerprints_raw = raw.get("numerical_result_fingerprints", [])
            numerical_result_fingerprints = tuple(
                (str(item["name"]), str(item["sha256"]))
                for item in _require_list(
                    numerical_fingerprints_raw, "numerical_result_fingerprints"
                )
                if isinstance(item, dict)
            )
            if len(numerical_result_fingerprints) != len(
                _require_list(numerical_fingerprints_raw, "numerical_result_fingerprints")
            ):
                raise ExperimentStateError(
                    "Each numerical_result_fingerprints entry must be an object."
                )

            interpreter_acceptances_raw = raw.get("interpreter_acceptances", [])
            interpreter_acceptances = tuple(
                dict(item)
                for item in _require_list(
                    interpreter_acceptances_raw, "interpreter_acceptances"
                )
                if isinstance(item, dict)
            )
            if len(interpreter_acceptances) != len(
                _require_list(interpreter_acceptances_raw, "interpreter_acceptances")
            ):
                raise ExperimentStateError(
                    "Each interpreter_acceptances entry must be an object."
                )

            run_records = tuple(
                dict(item)
                for item in _require_list(raw.get("run_records", []), "run_records")
                if isinstance(item, dict)
            )
            artifacts = tuple(
                dict(item)
                for item in _require_list(raw.get("artifacts", []), "artifacts")
                if isinstance(item, dict)
            )
            views = tuple(
                dict(item)
                for item in _require_list(raw.get("views", []), "views")
                if isinstance(item, dict)
            )
            if len(run_records) != len(raw.get("run_records", [])):
                raise ExperimentStateError("Each run_records entry must be an object.")
            if len(artifacts) != len(raw.get("artifacts", [])):
                raise ExperimentStateError("Each artifacts entry must be an object.")
            if len(views) != len(raw.get("views", [])):
                raise ExperimentStateError("Each views entry must be an object.")

            environment = tuple(
                (str(item["component"]), str(item["version"]))
                for item in _require_list(raw["environment"], "environment")
                if isinstance(item, dict)
            )
            if len(environment) != len(_require_list(raw["environment"], "environment")):
                raise ExperimentStateError("Each environment entry must be an object.")

            state = cls(
                format_version=int(raw["format_version"]),
                laboratory_version=str(raw["laboratory_version"]),
                created_at_utc=str(raw["created_at_utc"]),
                model_source=str(raw["model_source"]),
                model_sha256=str(raw["model_sha256"]),
                parameter_values=parameter_values,
                selected_model_visualisation=str(raw["selected_model_visualisation"]),
                evaluation_settings=EvaluationSettings(
                    points_1d=int(evaluation_raw["points_1d"]),
                    points_per_axis_2d=int(evaluation_raw["points_per_axis_2d"]),
                ),
                stationary_settings=StationaryPointSettings(
                    samples_1d=int(stationary_raw["samples_1d"]),
                    seeds_per_axis_2d=int(stationary_raw["seeds_per_axis_2d"]),
                    root_tolerance=float(stationary_raw["root_tolerance"]),
                ),
                sweep=sweep,
                result_fingerprints=result_fingerprints,
                numerical_reproduction_settings=numerical_settings,
                numerical_reference_data=numerical_reference_data,
                numerical_result_fingerprints=numerical_result_fingerprints,
                interpreter_acceptances=interpreter_acceptances,
                run_records=run_records,
                artifacts=artifacts,
                views=views,
                environment=environment,
                state_sha256=supplied_digest,
            )
        except (KeyError, TypeError, ValueError, ExperimentStateError) as exc:
            if isinstance(exc, ExperimentStateError):
                raise
            raise ExperimentStateError(f"Malformed experiment state: {exc}.") from exc

        return state


def _require_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentStateError(f"Experiment field '{field_name}' must be an object.")
    return value


def _require_list(value: object, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ExperimentStateError(f"Experiment field '{field_name}' must be a list.")
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _numpy_backend_summary(kind: str) -> str:
    config = getattr(np.__config__, "CONFIG", {})
    dependencies = config.get("Build Dependencies", {}) if isinstance(config, dict) else {}
    details = dependencies.get(kind, {}) if isinstance(dependencies, dict) else {}
    if not isinstance(details, dict) or not details:
        return "unavailable"
    keys = ("name", "version", "openblas configuration", "found")
    parts = [f"{key}={details[key]}" for key in keys if key in details]
    return "; ".join(parts) if parts else "unavailable"


@lru_cache(maxsize=1)
def current_environment() -> tuple[tuple[str, str], ...]:
    """Return software, platform and numerical-backend metadata for reproducibility."""
    values: list[tuple[str, str]] = [
        ("python", platform.python_version()),
        ("python_implementation", platform.python_implementation()),
        ("python_compiler", platform.python_compiler()),
        ("operating_system", platform.system()),
        ("os_release", platform.release()),
        ("os_version", platform.version()),
        ("cpu_architecture", platform.machine()),
        ("processor", platform.processor() or "unavailable"),
        ("machine_byteorder", sys.byteorder),
        ("float_radix", str(sys.float_info.radix)),
        ("float_mantissa_bits", str(sys.float_info.mant_dig)),
        ("float_max_exponent", str(sys.float_info.max_exp)),
        ("python_hash_algorithm", sys.hash_info.algorithm),
        ("source_tree_sha256", source_tree_sha256()),
        ("git_commit_or_build_id", git_commit_or_build_id()),
        ("numpy_blas", _numpy_backend_summary("blas")),
        ("numpy_lapack", _numpy_backend_summary("lapack")),
        (
            "numpy_default_bit_generator",
            type(np.random.default_rng(0).bit_generator).__name__,
        ),
    ]
    for package in _RECORDED_PACKAGES:
        if package == "python":
            continue
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        values.append((package, version))
    return tuple(values)


def fingerprint_model_ir(model: ModelIR) -> str:
    """Return the canonical semantic fingerprint of the validated Model IR."""
    return canonical_model_ir_sha256(model)


def compile_experiment_model(state: ExperimentState) -> ModelIR:
    """Parse the embedded source with the expression registry of its saved release.

    Before 1.8, ``sinh``, ``cosh`` and ``tanh`` were legal identifiers rather than
    function names.  A historical experiment must therefore retain that grammar even
    after the current application adds those functions.
    """
    if _sha256_text(state.model_source) != state.model_sha256:
        raise ExperimentStateError("Experiment model-source hash does not match its contents.")
    try:
        specification = parse_model_text(state.model_source)
        release = tuple(int(item) for item in re.findall(r"\d+", state.laboratory_version)[:2])
        legacy_registry = len(release) >= 2 and release < (1, 8)
        return (
            validate_legacy_model(specification)
            if legacy_registry
            else validate_model(specification)
        )
    except (ModelParseError, ModelValidationError) as exc:
        raise ExperimentStateError(f"Embedded model cannot be reconstructed: {exc}") from exc


def validate_experiment_state_for_model(
    state: ExperimentState,
    model: ModelIR,
    *,
    enforce_workload_budget: bool = True,
    reconstructed_embedded_model: ModelIR | None = None,
) -> None:
    """Verify saved controls/settings against the reconstructed model and workload policy."""
    from .capabilities import detect_capabilities
    from .registry import capability_registry

    embedded_model = reconstructed_embedded_model or compile_experiment_model(state)
    if fingerprint_model_ir(embedded_model) != fingerprint_model_ir(model):
        raise ExperimentStateError(
            "The supplied Model IR does not match the model embedded in the experiment state."
        )

    expected_parameters = tuple(parameter.name for parameter in model.parameters)
    saved_parameters = tuple(name for name, _ in state.parameter_values)
    if saved_parameters != expected_parameters:
        raise ExperimentStateError(
            "Experiment parameter state does not match the reconstructed model declaration order."
        )
    for name, value in state.parameter_values:
        parameter = model.parameter(name)
        if not parameter.domain.contains(value):
            raise ExperimentStateError(
                f"Saved parameter '{name}' = {value} lies outside the reconstructed model domain."
            )

    if state.format_version >= 6 and state.artifacts:
        model_hash = fingerprint_model_ir(model)
        if any(item.get("model_ir_sha256") != model_hash for item in state.artifacts):
            raise ExperimentStateError(
                "A typed scientific artifact belongs to a different canonical Model IR."
            )
        if any(item.get("model_ir_sha256") != model_hash for item in state.run_records):
            raise ExperimentStateError("A run record belongs to a different canonical Model IR.")
        if enforce_workload_budget and sum(
            int(item.get("workload_units", 0)) for item in state.run_records
        ) > 10_000_000:
            raise ExperimentStateError(
                "Saved typed runs exceed the default total workload-inspection limit."
            )
        return

    report = detect_capabilities(model)
    try:
        selected = ModelVisualisation(state.selected_model_visualisation)
    except ValueError as exc:
        raise ExperimentStateError(
            f"Unknown saved model visualisation: {state.selected_model_visualisation}."
        ) from exc
    if selected not in report.visualisations:
        raise ExperimentStateError(
            f"Saved model visualisation '{selected.value}' is not applicable to the reconstructed model."
        )

    if state.sweep is not None:
        try:
            parameter = model.parameter(state.sweep.parameter_name)
        except KeyError as exc:
            raise ExperimentStateError(
                f"Saved sweep parameter '{state.sweep.parameter_name}' is not declared by the model."
            ) from exc
        if not parameter.domain.contains(state.sweep.start) or not parameter.domain.contains(state.sweep.end):
            raise ExperimentStateError("Saved sweep bounds lie outside the reconstructed parameter domain.")
        if state.sweep.selected_visualisation is not None:
            try:
                selected_sweep = AnalysisVisualisation(state.sweep.selected_visualisation)
            except ValueError as exc:
                raise ExperimentStateError(
                    f"Unknown saved sweep visualisation: {state.sweep.selected_visualisation}."
                ) from exc
            compatible = capability_registry.compatible_analysis_visualisations(
                AnalysisCapability.PARAMETER_SWEEP,
                model,
            )
            if selected_sweep not in compatible:
                raise ExperimentStateError(
                    f"Saved sweep visualisation '{selected_sweep.value}' is not applicable to the reconstructed model."
                )


    if enforce_workload_budget:
        try:
            enforce_experiment_workload_budget(
                model,
                state.evaluation_settings,
                state.stationary_settings,
                state.sweep,
            )
        except WorkloadBudgetError as exc:
            raise ExperimentStateError(
                f"Saved experiment exceeds the computational budget: {exc}"
            ) from exc


def _normalised_array(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64).copy()
    value[value == 0.0] = 0.0  # normalise negative zero
    value[np.isnan(value)] = np.nan  # normalise NaN payloads
    return np.asarray(value, dtype="<f8")


def _array_fingerprint(array: np.ndarray) -> str:
    value = _normalised_array(array)
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes({"shape": value.shape, "dtype": "float64-le"}))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _diagnostic_payload(diagnostic: NumericalDiagnostic) -> dict[str, object]:
    """Return the complete persisted diagnostic, including presentation text."""
    return {
        "code": diagnostic.code,
        "severity": diagnostic.severity.value,
        "message": diagnostic.message,
        "details": [list(item) for item in diagnostic.details],
    }


def diagnostic_record_payload(diagnostic: NumericalDiagnostic) -> dict[str, object]:
    """Return a complete canonical record for persistence and presentation."""
    return _diagnostic_payload(diagnostic)




def _diagnostic_scientific_payload(diagnostic: NumericalDiagnostic) -> dict[str, object]:
    """Return the canonical scientific identity of a numerical diagnostic.

    Human-facing message text is deliberately excluded.  Details are sorted by the
    ``NumericalDiagnostic`` invariant, making mapping-order differences irrelevant.
    """
    return {
        "code": diagnostic.code,
        "severity": diagnostic.severity.value,
        "details": [list(item) for item in diagnostic.details],
    }


def _saved_diagnostic_scientific_payload(value: object) -> dict[str, object] | None:
    """Canonicalise one saved full diagnostic record for scientific comparison."""
    if not isinstance(value, Mapping):
        return None
    code = value.get("code")
    severity = value.get("severity")
    details = value.get("details", [])
    if not isinstance(code, str) or not isinstance(severity, str) or not isinstance(details, list):
        return None
    canonical_details: list[tuple[str, str]] = []
    for item in details:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        key, detail_value = item
        if not isinstance(key, str) or not isinstance(detail_value, str):
            return None
        canonical_details.append((key, detail_value))
    canonical_details.sort()
    return {
        "code": code,
        "severity": severity,
        "details": [list(item) for item in canonical_details],
    }

def fingerprint_evaluation(result: FunctionEvaluation | SurfaceEvaluation) -> str:
    """Return a strict deterministic fingerprint for a direct numerical evaluation."""
    if isinstance(result, FunctionEvaluation):
        payload = {
            "type": "FunctionEvaluation",
            "x_name": result.x_name,
            "y_name": result.y_name,
            "x": _array_fingerprint(result.x),
            "y": _array_fingerprint(result.y),
            "status": result.numerical_status.value,
            "diagnostics": [
                _diagnostic_scientific_payload(item) for item in result.diagnostics
            ],
        }
    else:
        payload = {
            "type": "SurfaceEvaluation",
            "x_name": result.x_name,
            "y_name": result.y_name,
            "z_name": result.z_name,
            "x": _array_fingerprint(result.x),
            "y": _array_fingerprint(result.y),
            "z": _array_fingerprint(result.z),
            "status": result.numerical_status.value,
            "diagnostics": [
                _diagnostic_scientific_payload(item) for item in result.diagnostics
            ],
        }
    return _sha256_json(payload)


def _float_token(value: float) -> str:
    value = 0.0 if value == 0.0 else float(value)
    return value.hex()


def _point_payload(point: OneDimensionalCriticalPoint | TwoDimensionalCriticalPoint) -> dict[str, object]:
    if isinstance(point, OneDimensionalCriticalPoint):
        return {
            "type": "1d",
            "x": _float_token(point.x),
            "value": _float_token(point.value),
            "second_derivative": _float_token(point.second_derivative),
            "classification": point.classification,
        }
    return {
        "type": "2d",
        "x": _float_token(point.x),
        "y": _float_token(point.y),
        "value": _float_token(point.value),
        "eigenvalues": [_float_token(value) for value in point.eigenvalues],
        "classification": point.classification,
    }


def fingerprint_stationary_result(result: StationaryPointAnalysisResult) -> str:
    points = sorted(
        (_point_payload(point) for point in result.points),
        key=lambda item: _canonical_json_bytes(item),
    )
    payload = {
        "parameter_values": [(name, _float_token(value)) for name, value in result.parameter_values],
        "points": points,
        "status": result.numerical_status.value,
        "diagnostics": [
            _diagnostic_scientific_payload(item) for item in result.diagnostics
        ],
        "statistics": list(result.statistics.values),
    }
    return _sha256_json(payload)


def fingerprint_parameter_sweep(result: ParameterSweepResult) -> str:
    payload = {
        "parameter_name": result.parameter_name,
        "function_name": result.function_name,
        "variable_names": list(result.variable_names),
        "start": _float_token(result.start),
        "end": _float_token(result.end),
        "step_count": result.step_count,
        "fixed_parameters": [
            (name, _float_token(value)) for name, value in result.fixed_parameters
        ],
        "steps": [
            {
                "parameter_value": _float_token(step.parameter_value),
                "points": sorted(
                    (_point_payload(point) for point in step.points),
                    key=lambda item: _canonical_json_bytes(item),
                ),
                "error": step.error,
                "status": step.numerical_status.value,
                "diagnostics": [
                    _diagnostic_scientific_payload(item) for item in step.diagnostics
                ],
            }
            for step in result.steps
        ],
    }
    return _sha256_json(payload)


def fingerprint_symbolic_analysis(model: ModelIR) -> str | None:
    """Fingerprint all symbolic derivatives applicable to the current scalar model."""
    if len(model.variables) == 1 and len(model.functions) == 1:
        result = analyse_one_variable_function(model)
        payload = {
            "type": "one_variable",
            "function": result.function_name,
            "variable": result.variable_name,
            "first_derivative": sp.srepr(result.first_derivative),
            "second_derivative": sp.srepr(result.second_derivative),
        }
    elif len(model.variables) == 2 and len(model.functions) == 1:
        result = analyse_two_variable_function(model)
        payload = {
            "type": "two_variable",
            "function": result.function_name,
            "variables": list(result.variable_names),
            "gradient": [sp.srepr(item) for item in result.gradient],
            "hessian": [[sp.srepr(item) for item in row] for row in result.hessian.tolist()],
        }
    else:
        return None
    return _sha256_json(payload)




# ---- Legacy format-v1 tolerance-normalised fingerprints ------------------------------


def _quantise_scalar(value: float, settings: NumericalReproductionSettings) -> float:
    value = float(value)
    if not np.isfinite(value):
        return value
    magnitude = abs(value)
    if magnitude <= settings.absolute_tolerance:
        return 0.0
    exponent = math.floor(math.log10(magnitude))
    scale = 10.0 ** exponent
    quantum = max(settings.absolute_tolerance, settings.relative_tolerance * scale)
    return float(np.rint(value / quantum) * quantum)


def _quantised_array(
    array: np.ndarray, settings: NumericalReproductionSettings
) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64).copy()
    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size:
        magnitudes = np.abs(finite_values)
        near_zero = magnitudes <= settings.absolute_tolerance
        quantised = finite_values.copy()
        quantised[near_zero] = 0.0
        active = ~near_zero
        if np.any(active):
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                exponents = np.floor(np.log10(magnitudes[active]))
                scales = np.power(10.0, exponents)
                quanta = np.maximum(
                    settings.absolute_tolerance,
                    settings.relative_tolerance * scales,
                )
                quantised[active] = np.rint(finite_values[active] / quanta) * quanta
        values[finite] = quantised
    values[values == 0.0] = 0.0
    values[np.isnan(values)] = np.nan
    return np.asarray(values, dtype="<f8")


def _numerical_array_fingerprint(
    array: np.ndarray, settings: NumericalReproductionSettings
) -> str:
    value = _quantised_array(array, settings)
    digest = hashlib.sha256()
    digest.update(
        _canonical_json_bytes(
            {
                "shape": value.shape,
                "dtype": "float64-le",
                "rtol": settings.relative_tolerance,
                "atol": settings.absolute_tolerance,
            }
        )
    )
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def numerical_fingerprint_evaluation(
    result: FunctionEvaluation | SurfaceEvaluation,
    settings: NumericalReproductionSettings,
) -> str:
    """Legacy tolerance-normalised fingerprint retained for format-v1 compatibility."""
    if isinstance(result, FunctionEvaluation):
        payload = {
            "type": "FunctionEvaluation",
            "x_name": result.x_name,
            "y_name": result.y_name,
            "x": _numerical_array_fingerprint(result.x, settings),
            "y": _numerical_array_fingerprint(result.y, settings),
            "status": result.numerical_status.value,
            "diagnostics": [item.code for item in result.diagnostics],
        }
    else:
        payload = {
            "type": "SurfaceEvaluation",
            "x_name": result.x_name,
            "y_name": result.y_name,
            "z_name": result.z_name,
            "x": _numerical_array_fingerprint(result.x, settings),
            "y": _numerical_array_fingerprint(result.y, settings),
            "z": _numerical_array_fingerprint(result.z, settings),
            "status": result.numerical_status.value,
            "diagnostics": [item.code for item in result.diagnostics],
        }
    return _sha256_json(payload)


def _numerical_point_payload(
    point: OneDimensionalCriticalPoint | TwoDimensionalCriticalPoint,
    settings: NumericalReproductionSettings,
) -> dict[str, object]:
    if isinstance(point, OneDimensionalCriticalPoint):
        return {
            "type": "1d",
            "x": _quantise_scalar(point.x, settings),
            "value": _quantise_scalar(point.value, settings),
            "second_derivative": _quantise_scalar(point.second_derivative, settings),
            "classification": point.classification,
        }
    return {
        "type": "2d",
        "x": _quantise_scalar(point.x, settings),
        "y": _quantise_scalar(point.y, settings),
        "value": _quantise_scalar(point.value, settings),
        "eigenvalues": [_quantise_scalar(value, settings) for value in point.eigenvalues],
        "classification": point.classification,
    }


def numerical_fingerprint_stationary_result(
    result: StationaryPointAnalysisResult,
    settings: NumericalReproductionSettings,
) -> str:
    points = sorted(
        (_numerical_point_payload(point, settings) for point in result.points),
        key=lambda item: _canonical_json_bytes(item),
    )
    payload = {
        "parameter_values": [
            (name, _quantise_scalar(value, settings)) for name, value in result.parameter_values
        ],
        "points": points,
        "status": result.numerical_status.value,
        "diagnostics": [item.code for item in result.diagnostics],
    }
    return _sha256_json(payload)


def numerical_fingerprint_parameter_sweep(
    result: ParameterSweepResult,
    settings: NumericalReproductionSettings,
) -> str:
    payload = {
        "parameter_name": result.parameter_name,
        "function_name": result.function_name,
        "variable_names": list(result.variable_names),
        "start": _quantise_scalar(result.start, settings),
        "end": _quantise_scalar(result.end, settings),
        "step_count": result.step_count,
        "fixed_parameters": [
            (name, _quantise_scalar(value, settings)) for name, value in result.fixed_parameters
        ],
        "steps": [
            {
                "parameter_value": _quantise_scalar(step.parameter_value, settings),
                "points": sorted(
                    (_numerical_point_payload(point, settings) for point in step.points),
                    key=lambda item: _canonical_json_bytes(item),
                ),
                "status": step.numerical_status.value,
                "diagnostics": [item.code for item in step.diagnostics],
            }
            for step in result.steps
        ],
    }
    return _sha256_json(payload)



_MAX_REFERENCE_ARRAY_BYTES = 128 * 1024 * 1024
_MAX_FULL_REFERENCE_ARRAY_BYTES = 8 * 1024 * 1024
_REFERENCE_ARRAY_CHUNK_BYTES = 1024 * 1024
_MAX_CANONICAL_ARRAY_SAMPLES = 4096


def _encode_full_array_reference(value: np.ndarray) -> dict[str, object]:
    """Encode a bounded full float64 array, chunking larger inline payloads."""
    raw = value.tobytes(order="C")
    base = {
        "shape": list(value.shape),
        "dtype": "float64-le",
    }
    if len(raw) <= _REFERENCE_ARRAY_CHUNK_BYTES:
        compressed = zlib.compress(raw, level=9)
        return {
            **base,
            "encoding": "zlib+base64",
            "data": base64.b64encode(compressed).decode("ascii"),
        }

    chunks: list[dict[str, object]] = []
    for offset in range(0, len(raw), _REFERENCE_ARRAY_CHUNK_BYTES):
        chunk = raw[offset : offset + _REFERENCE_ARRAY_CHUNK_BYTES]
        chunks.append(
            {
                "raw_size": len(chunk),
                "data": base64.b64encode(zlib.compress(chunk, level=9)).decode("ascii"),
            }
        )
    return {
        **base,
        "encoding": "zlib+base64-chunks",
        "chunk_raw_bytes": _REFERENCE_ARRAY_CHUNK_BYTES,
        "chunks": chunks,
    }


def _canonical_sample_indices(element_count: int) -> list[int]:
    sample_count = min(element_count, _MAX_CANONICAL_ARRAY_SAMPLES)
    if sample_count == element_count:
        return list(range(element_count))
    denominator = sample_count - 1
    return [
        (index * (element_count - 1)) // denominator
        for index in range(sample_count)
    ]


def _array_reference_summary(value: np.ndarray) -> dict[str, object]:
    flat = value.reshape(-1)
    finite = flat[np.isfinite(flat)]
    return {
        "element_count": int(flat.size),
        "finite_count": int(finite.size),
        "nan_count": int(np.count_nonzero(np.isnan(flat))),
        "positive_infinity_count": int(np.count_nonzero(np.isposinf(flat))),
        "negative_infinity_count": int(np.count_nonzero(np.isneginf(flat))),
        "finite_minimum": None if not finite.size else float(np.min(finite)),
        "finite_maximum": None if not finite.size else float(np.max(finite)),
        "finite_mean": None if not finite.size else float(np.mean(finite)),
    }


def _encode_array_reference(array: np.ndarray) -> dict[str, object]:
    """Serialise an array according to the bounded result-type storage policy.

    Small arrays are stored completely.  Larger full references are split into bounded
    independently compressed chunks.  Arrays above the full-reference threshold use a
    deterministic canonical sample plus numerical summaries; the strict result
    fingerprint continues to cover the complete array.
    """
    value = _normalised_array(array)
    raw_bytes = value.size * value.dtype.itemsize
    if raw_bytes > _MAX_REFERENCE_ARRAY_BYTES:
        raise ExperimentStateError(
            "Numerical reference array is too large to embed in an experiment state."
        )
    if raw_bytes <= _MAX_FULL_REFERENCE_ARRAY_BYTES:
        return _encode_full_array_reference(value)

    indices = _canonical_sample_indices(int(value.size))
    samples = np.asarray(value.reshape(-1)[indices], dtype="<f8")
    return {
        "shape": list(value.shape),
        "dtype": "float64-le",
        "encoding": "canonical-samples-v1",
        "source_sha256": _array_fingerprint(value),
        "sample_indices": indices,
        "sample_values": _encode_full_array_reference(samples),
        "summary": _array_reference_summary(value),
    }


def _decode_zlib_payload(data: object, expected_bytes: int) -> bytes:
    if not isinstance(data, str):
        raise ExperimentStateError("Numerical reference array data are malformed.")
    maximum_encoded_length = ((expected_bytes + 1024) * 4 // 3) + 16
    if len(data) > maximum_encoded_length:
        raise ExperimentStateError("Numerical reference array payload is unexpectedly large.")
    try:
        compressed = base64.b64decode(data.encode("ascii"), validate=True)
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, expected_bytes + 1)
    except (ValueError, zlib.error) as exc:
        raise ExperimentStateError("Numerical reference array data are corrupt.") from exc
    if (
        len(raw) != expected_bytes
        or decompressor.unconsumed_tail
        or not decompressor.eof
        or decompressor.unused_data
    ):
        raise ExperimentStateError("Numerical reference array byte length is invalid.")
    return raw


def _decode_array_reference(reference: Mapping[str, Any]) -> np.ndarray:
    """Decode and validate an embedded numerical reference array."""
    try:
        shape_raw = reference["shape"]
        dtype = reference["dtype"]
        encoding = reference["encoding"]
    except KeyError as exc:
        raise ExperimentStateError(
            f"Numerical reference array is missing field: {exc.args[0]}."
        ) from exc
    if not isinstance(shape_raw, list) or any(
        not isinstance(item, int) or item < 0 for item in shape_raw
    ):
        raise ExperimentStateError("Numerical reference array shape is invalid.")
    if len(shape_raw) > 4:
        raise ExperimentStateError("Numerical reference array has unsupported dimensionality.")
    shape = tuple(shape_raw)
    element_count = math.prod(shape) if shape else 1
    expected_bytes = element_count * 8
    if expected_bytes > _MAX_REFERENCE_ARRAY_BYTES:
        raise ExperimentStateError("Numerical reference array exceeds the safe size limit.")
    if dtype != "float64-le":
        raise ExperimentStateError("Numerical reference array encoding is unsupported.")
    if encoding == "zlib+base64":
        raw = _decode_zlib_payload(reference.get("data"), expected_bytes)
    elif encoding == "zlib+base64-chunks":
        chunks = reference.get("chunks")
        chunk_raw_bytes = reference.get("chunk_raw_bytes")
        if (
            not isinstance(chunks, list)
            or not chunks
            or chunk_raw_bytes != _REFERENCE_ARRAY_CHUNK_BYTES
            or len(chunks)
            > math.ceil(max(expected_bytes, 1) / _REFERENCE_ARRAY_CHUNK_BYTES)
        ):
            raise ExperimentStateError("Numerical reference array chunks are malformed.")
        raw_parts: list[bytes] = []
        total = 0
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Mapping):
                raise ExperimentStateError("Numerical reference array chunk is malformed.")
            raw_size = chunk.get("raw_size")
            if (
                isinstance(raw_size, bool)
                or not isinstance(raw_size, int)
                or raw_size <= 0
                or raw_size > _REFERENCE_ARRAY_CHUNK_BYTES
                or (index < len(chunks) - 1 and raw_size != _REFERENCE_ARRAY_CHUNK_BYTES)
            ):
                raise ExperimentStateError("Numerical reference array chunk size is invalid.")
            total += raw_size
            if total > expected_bytes:
                raise ExperimentStateError("Numerical reference array chunks exceed their shape.")
            raw_parts.append(_decode_zlib_payload(chunk.get("data"), raw_size))
        if total != expected_bytes:
            raise ExperimentStateError("Numerical reference array chunks are incomplete.")
        raw = b"".join(raw_parts)
    else:
        raise ExperimentStateError("Numerical reference array encoding is unsupported.")
    value = np.frombuffer(raw, dtype="<f8").reshape(shape).copy()
    value[value == 0.0] = 0.0
    value[np.isnan(value)] = np.nan
    return value


def _diagnostic_codes(diagnostics: tuple[NumericalDiagnostic, ...]) -> list[str]:
    return [item.code for item in diagnostics]


def _diagnostic_reference_matches(
    reference: Mapping[str, Any],
    diagnostics: tuple[NumericalDiagnostic, ...],
) -> bool:
    """Compare the scientific identity of saved and current diagnostics.

    Newer references preserve complete diagnostic records for reporting, but numerical
    reproduction depends only on diagnostic ``code``, ``severity`` and canonical
    structured ``details``.  Human-facing message wording is presentation metadata.
    Detail-pair order is irrelevant because details represent a mapping.

    Early Phase-A bundles stored only diagnostic codes; those remain compatible using
    the historical code-level comparison.
    """
    if "diagnostic_records" in reference:
        saved_records = reference.get("diagnostic_records")
        if not isinstance(saved_records, list):
            return False
        saved = [_saved_diagnostic_scientific_payload(item) for item in saved_records]
        if any(item is None for item in saved):
            return False
        current = [_diagnostic_scientific_payload(item) for item in diagnostics]
        return saved == current
    return reference.get("diagnostics") == _diagnostic_codes(diagnostics)


def _encode_reference_scalar(value: float) -> float | str:
    numeric = float(value)
    if math.isnan(numeric):
        return "NaN"
    if math.isinf(numeric):
        return "Infinity" if numeric > 0 else "-Infinity"
    return numeric


def _decode_reference_scalar(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numerical reference value")
    if isinstance(value, (int, float)):
        return float(value)
    if value == "NaN":
        return math.nan
    if value == "Infinity":
        return math.inf
    if value == "-Infinity":
        return -math.inf
    raise ValueError("invalid numerical reference scalar")


def _point_reference(
    point: OneDimensionalCriticalPoint | TwoDimensionalCriticalPoint,
) -> dict[str, object]:
    if isinstance(point, OneDimensionalCriticalPoint):
        return {
            "type": "1d",
            "x": _encode_reference_scalar(point.x),
            "value": _encode_reference_scalar(point.value),
            "second_derivative": _encode_reference_scalar(point.second_derivative),
            "classification": point.classification,
        }
    return {
        "type": "2d",
        "x": _encode_reference_scalar(point.x),
        "y": _encode_reference_scalar(point.y),
        "value": _encode_reference_scalar(point.value),
        "eigenvalues": [_encode_reference_scalar(value) for value in point.eigenvalues],
        "classification": point.classification,
    }


def numerical_reference_evaluation(
    result: FunctionEvaluation | SurfaceEvaluation,
) -> dict[str, object]:
    if isinstance(result, FunctionEvaluation):
        return {
            "kind": "FunctionEvaluation",
            "x_name": result.x_name,
            "y_name": result.y_name,
            "arrays": {
                "x": _encode_array_reference(result.x),
                "y": _encode_array_reference(result.y),
            },
            "status": result.numerical_status.value,
            "diagnostics": _diagnostic_codes(result.diagnostics),
            "diagnostic_records": [_diagnostic_payload(item) for item in result.diagnostics],
        }
    return {
        "kind": "SurfaceEvaluation",
        "x_name": result.x_name,
        "y_name": result.y_name,
        "z_name": result.z_name,
        "arrays": {
            "x": _encode_array_reference(result.x),
            "y": _encode_array_reference(result.y),
            "z": _encode_array_reference(result.z),
        },
        "status": result.numerical_status.value,
        "diagnostics": _diagnostic_codes(result.diagnostics),
        "diagnostic_records": [_diagnostic_payload(item) for item in result.diagnostics],
    }


def numerical_reference_stationary_result(
    result: StationaryPointAnalysisResult,
) -> dict[str, object]:
    return {
        "kind": "StationaryPointAnalysisResult",
        "parameter_values": [
            {"name": name, "value": float(value)} for name, value in result.parameter_values
        ],
        "points": [_point_reference(point) for point in result.points],
        "status": result.numerical_status.value,
        "diagnostics": _diagnostic_codes(result.diagnostics),
        "diagnostic_records": [_diagnostic_payload(item) for item in result.diagnostics],
    }


def numerical_reference_parameter_sweep(result: ParameterSweepResult) -> dict[str, object]:
    return {
        "kind": "ParameterSweepResult",
        "parameter_name": result.parameter_name,
        "function_name": result.function_name,
        "variable_names": list(result.variable_names),
        "start": float(result.start),
        "end": float(result.end),
        "step_count": int(result.step_count),
        "fixed_parameters": [
            {"name": name, "value": float(value)} for name, value in result.fixed_parameters
        ],
        "steps": [
            {
                "parameter_value": float(step.parameter_value),
                "points": [_point_reference(point) for point in step.points],
                "status": step.numerical_status.value,
                "diagnostics": _diagnostic_codes(step.diagnostics),
                "diagnostic_records": [_diagnostic_payload(item) for item in step.diagnostics],
                "error": step.error,
            }
            for step in result.steps
        ],
    }


def numerical_reference_symbolic(digest: str) -> dict[str, object]:
    return {"kind": "SymbolicAnalysis", "sha256": digest}


def _scalar_deviation(
    saved: float,
    current: float,
    settings: NumericalReproductionSettings,
) -> tuple[bool, float | None, float | None]:
    saved_value = float(saved)
    current_value = float(current)
    match = bool(
        np.isclose(
            current_value,
            saved_value,
            rtol=settings.relative_tolerance,
            atol=settings.absolute_tolerance,
            equal_nan=True,
        )
    )
    if np.isnan(saved_value) and np.isnan(current_value):
        return match, 0.0, 0.0
    if not np.isfinite(saved_value) or not np.isfinite(current_value):
        return match, 0.0 if match else math.inf, 0.0 if match else math.inf
    absolute = abs(current_value - saved_value)
    relative = absolute / max(abs(saved_value), settings.absolute_tolerance)
    return match, absolute, relative


def _array_deviation(
    saved: np.ndarray,
    current: np.ndarray,
    settings: NumericalReproductionSettings,
) -> tuple[bool, float | None, float | None, str | None]:
    saved_value = np.asarray(saved, dtype=np.float64)
    current_value = np.asarray(current, dtype=np.float64)
    if saved_value.shape != current_value.shape:
        return False, None, None, f"shape differs: saved {saved_value.shape}, current {current_value.shape}"
    close = np.isclose(
        current_value,
        saved_value,
        rtol=settings.relative_tolerance,
        atol=settings.absolute_tolerance,
        equal_nan=True,
    )
    matches = bool(np.all(close))
    finite_pairs = np.isfinite(saved_value) & np.isfinite(current_value)
    nonfinite_mismatch = (~finite_pairs) & (~close)
    if np.any(finite_pairs):
        differences = np.abs(current_value[finite_pairs] - saved_value[finite_pairs])
        max_absolute = float(np.max(differences))
        denominators = np.maximum(
            np.abs(saved_value[finite_pairs]), settings.absolute_tolerance
        )
        max_relative = float(np.max(differences / denominators))
    else:
        max_absolute = 0.0
        max_relative = 0.0
    if np.any(nonfinite_mismatch):
        max_absolute = math.inf
        max_relative = math.inf
    if matches:
        return True, max_absolute, max_relative, None
    mismatch_count = int(close.size - np.count_nonzero(close))
    return (
        False,
        max_absolute,
        max_relative,
        f"{mismatch_count} of {close.size} numerical values exceed the saved tolerances",
    )


def _sampled_array_deviation(
    reference: Mapping[str, Any],
    current: np.ndarray,
    settings: NumericalReproductionSettings,
) -> tuple[bool, float | None, float | None, str | None]:
    """Compare a large array against its deterministic samples and summaries."""
    try:
        shape_raw = reference["shape"]
        dtype = reference["dtype"]
        source_sha256 = reference["source_sha256"]
        indices_raw = reference["sample_indices"]
        samples_raw = reference["sample_values"]
        summary_raw = reference["summary"]
    except KeyError as exc:
        raise ExperimentStateError(
            f"Sampled numerical reference is missing field: {exc.args[0]}."
        ) from exc
    if (
        not isinstance(shape_raw, list)
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape_raw)
        or len(shape_raw) > 4
        or dtype != "float64-le"
    ):
        raise ExperimentStateError("Sampled numerical reference shape or dtype is invalid.")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise ExperimentStateError("Sampled numerical reference fingerprint is invalid.")
    shape = tuple(shape_raw)
    current_value = _normalised_array(current)
    if current_value.shape != shape:
        return False, None, None, f"shape differs: saved {shape}, current {current_value.shape}"
    element_count = math.prod(shape) if shape else 1
    expected_indices = _canonical_sample_indices(element_count)
    if indices_raw != expected_indices:
        raise ExperimentStateError("Sampled numerical reference indices are not canonical.")
    if not isinstance(samples_raw, Mapping):
        raise ExperimentStateError("Sampled numerical reference values are malformed.")
    saved_samples = _decode_array_reference(samples_raw)
    if saved_samples.shape != (len(expected_indices),):
        raise ExperimentStateError("Sampled numerical reference value count is invalid.")

    current_samples = np.asarray(
        current_value.reshape(-1)[expected_indices], dtype="<f8"
    )
    matches, maximum_absolute, maximum_relative, sample_detail = _array_deviation(
        saved_samples, current_samples, settings
    )
    details: list[str] = []
    if sample_detail is not None:
        details.append(f"canonical samples: {sample_detail}")

    if not isinstance(summary_raw, Mapping):
        raise ExperimentStateError("Sampled numerical reference summary is malformed.")
    current_summary = _array_reference_summary(current_value)
    count_fields = (
        "element_count",
        "finite_count",
        "nan_count",
        "positive_infinity_count",
        "negative_infinity_count",
    )
    for field in count_fields:
        saved_count = summary_raw.get(field)
        if (
            isinstance(saved_count, bool)
            or not isinstance(saved_count, int)
            or saved_count < 0
        ):
            raise ExperimentStateError("Sampled numerical reference summary is malformed.")
        if saved_count != current_summary[field]:
            matches = False
            details.append(f"summary {field} differs")

    metrics = [(maximum_absolute, maximum_relative)]
    for field in ("finite_minimum", "finite_maximum", "finite_mean"):
        saved_value = summary_raw.get(field)
        current_value_summary = current_summary[field]
        if saved_value is None or current_value_summary is None:
            if saved_value is not None or current_value_summary is not None:
                matches = False
                details.append(f"summary {field} differs")
            continue
        if isinstance(saved_value, bool) or not isinstance(saved_value, (int, float)):
            raise ExperimentStateError("Sampled numerical reference summary is malformed.")
        scalar_match, absolute, relative = _scalar_deviation(
            float(saved_value), float(current_value_summary), settings
        )
        metrics.append((absolute, relative))
        if not scalar_match:
            matches = False
            details.append(f"summary {field} exceeds the saved tolerances")

    maximum_absolute, maximum_relative = _merge_metrics(metrics)
    return matches, maximum_absolute, maximum_relative, "; ".join(details) or None


def _array_reference_deviation(
    reference: Mapping[str, Any],
    current: np.ndarray,
    settings: NumericalReproductionSettings,
) -> tuple[bool, float | None, float | None, str | None, str]:
    encoding = reference.get("encoding")
    if encoding == "canonical-samples-v1":
        match, absolute, relative, detail = _sampled_array_deviation(
            reference, current, settings
        )
        return match, absolute, relative, detail, "canonical samples and summaries"
    saved = _decode_array_reference(reference)
    match, absolute, relative, detail = _array_deviation(saved, current, settings)
    return match, absolute, relative, detail, "full array"


def _merge_metrics(
    metrics: list[tuple[float | None, float | None]],
) -> tuple[float | None, float | None]:
    absolutes = [value for value, _ in metrics if value is not None]
    relatives = [value for _, value in metrics if value is not None]
    return (
        max(absolutes) if absolutes else None,
        max(relatives) if relatives else None,
    )


def _compare_named_values(
    saved: object,
    current: tuple[tuple[str, float], ...],
    settings: NumericalReproductionSettings,
    *,
    label: str,
) -> tuple[bool, list[tuple[float | None, float | None]], list[str]]:
    if not isinstance(saved, list):
        return False, [], [f"saved {label} are malformed"]
    saved_map: dict[str, float] = {}
    try:
        for item in saved:
            if not isinstance(item, dict):
                raise TypeError
            name = str(item["name"])
            if name in saved_map:
                return False, [], [f"saved {label} contain duplicate name '{name}'"]
            saved_map[name] = float(item["value"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False, [], [f"saved {label} are malformed"]
    current_map = dict(current)
    if set(saved_map) != set(current_map):
        return False, [], [f"{label} names differ"]
    metrics: list[tuple[float | None, float | None]] = []
    details: list[str] = []
    matches = True
    for name in sorted(saved_map):
        match, absolute, relative = _scalar_deviation(
            saved_map[name], current_map[name], settings
        )
        metrics.append((absolute, relative))
        if not match:
            matches = False
            details.append(f"{label} '{name}' exceeds tolerance")
    return matches, metrics, details


def _point_compatibility(
    saved: Mapping[str, Any],
    current: OneDimensionalCriticalPoint | TwoDimensionalCriticalPoint,
    settings: NumericalReproductionSettings,
) -> tuple[bool, float | None, float | None]:
    current_reference = _point_reference(current)
    if saved.get("type") != current_reference.get("type"):
        return False, None, None
    if saved.get("classification") != current_reference.get("classification"):
        return False, None, None
    scalar_fields = ["x", "value", "second_derivative"] if saved.get("type") == "1d" else ["x", "y", "value"]
    metrics: list[tuple[float | None, float | None]] = []
    for field in scalar_fields:
        try:
            match, absolute, relative = _scalar_deviation(
                _decode_reference_scalar(saved[field]),
                _decode_reference_scalar(current_reference[field]),
                settings,
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return False, None, None
        metrics.append((absolute, relative))
        if not match:
            return False, *_merge_metrics(metrics)
    if saved.get("type") == "2d":
        saved_eigenvalues = saved.get("eigenvalues")
        current_eigenvalues = current_reference.get("eigenvalues")
        if not isinstance(saved_eigenvalues, list) or not isinstance(current_eigenvalues, list):
            return False, None, None
        if len(saved_eigenvalues) != len(current_eigenvalues):
            return False, None, None
        try:
            saved_decoded = [_decode_reference_scalar(value) for value in saved_eigenvalues]
            current_decoded = [_decode_reference_scalar(value) for value in current_eigenvalues]
        except ValueError:
            return False, None, None
        eigen_key = lambda value: (math.isnan(value), value if not math.isnan(value) else 0.0)
        for saved_value, current_value in zip(
            sorted(saved_decoded, key=eigen_key),
            sorted(current_decoded, key=eigen_key),
        ):
            match, absolute, relative = _scalar_deviation(
                saved_value, current_value, settings
            )
            metrics.append((absolute, relative))
            if not match:
                return False, *_merge_metrics(metrics)
    return True, *_merge_metrics(metrics)


def _match_stationary_points(
    saved_points: object,
    current_points: tuple[OneDimensionalCriticalPoint | TwoDimensionalCriticalPoint, ...],
    settings: NumericalReproductionSettings,
) -> tuple[bool, float | None, float | None, list[str]]:
    """Semantically match unordered stationary-point collections within saved tolerances.

    Pairing is solved globally rather than by sorted list position.  This avoids false
    non-reproduction when two valid points are returned in a different order or nearby
    branches exchange ordering while remaining individually within tolerance.
    """
    if not isinstance(saved_points, list) or any(not isinstance(item, dict) for item in saved_points):
        return False, None, None, ["saved stationary points are malformed"]
    if len(saved_points) != len(current_points):
        return (
            False,
            None,
            None,
            [f"stationary-point count differs: saved {len(saved_points)}, current {len(current_points)}"],
        )
    if not saved_points:
        return True, 0.0, 0.0, []

    count = len(saved_points)
    costs = np.full((count, count), np.inf, dtype=float)
    pair_metrics: dict[tuple[int, int], tuple[float | None, float | None]] = {}
    for saved_index, saved in enumerate(saved_points):
        for current_index, current in enumerate(current_points):
            compatible, absolute, relative = _point_compatibility(saved, current, settings)
            if not compatible:
                continue
            # The exact value of the cost is only used to choose among multiple admissible
            # tolerance-compatible pairings.  Relative deviation dominates; absolute
            # deviation breaks ties near zero.
            cost = float(relative or 0.0) + float(absolute or 0.0)
            costs[saved_index, current_index] = cost
            pair_metrics[(saved_index, current_index)] = (absolute, relative)

    try:
        saved_indices, current_indices = linear_sum_assignment(costs)
    except ValueError:
        return (
            False,
            None,
            None,
            ["stationary points cannot be paired by type, classification and numerical tolerance"],
        )

    metrics: list[tuple[float | None, float | None]] = []
    for saved_index, current_index in zip(saved_indices, current_indices):
        if not np.isfinite(costs[saved_index, current_index]):
            return (
                False,
                *_merge_metrics(metrics),
                ["stationary points cannot be paired by type, classification and numerical tolerance"],
            )
        metrics.append(pair_metrics[(int(saved_index), int(current_index))])

    max_absolute, max_relative = _merge_metrics(metrics)
    return True, max_absolute, max_relative, []


def _compare_evaluation_reference(
    reference: Mapping[str, Any],
    current: FunctionEvaluation | SurfaceEvaluation,
    settings: NumericalReproductionSettings,
) -> NumericalComparison:
    kind = reference.get("kind")
    expected_kind = "FunctionEvaluation" if isinstance(current, FunctionEvaluation) else "SurfaceEvaluation"
    details: list[str] = []
    metrics: list[tuple[float | None, float | None]] = []
    comparison_modes: set[str] = set()
    matches = kind == expected_kind
    if not matches:
        details.append(f"result kind differs: saved {kind}, current {expected_kind}")
    expected_names = (
        ("x_name", current.x_name),
        ("y_name", current.y_name),
    ) if isinstance(current, FunctionEvaluation) else (
        ("x_name", current.x_name),
        ("y_name", current.y_name),
        ("z_name", current.z_name),
    )
    for field, current_name in expected_names:
        if reference.get(field) != current_name:
            matches = False
            details.append(f"{field} differs")
    if reference.get("status") != current.numerical_status.value:
        matches = False
        details.append("numerical status differs")
    if not _diagnostic_reference_matches(reference, current.diagnostics):
        matches = False
        details.append("numerical diagnostics differ")
    arrays = reference.get("arrays")
    if not isinstance(arrays, dict):
        return NumericalComparison("evaluation", False, "explicit tolerance comparison", details=("saved evaluation arrays are malformed",))
    current_arrays = {"x": current.x, "y": current.y}
    if isinstance(current, SurfaceEvaluation):
        current_arrays["z"] = current.z
    for name, current_array in current_arrays.items():
        saved_array_raw = arrays.get(name)
        if not isinstance(saved_array_raw, dict):
            matches = False
            details.append(f"saved array '{name}' is missing or malformed")
            continue
        try:
            array_match, absolute, relative, detail, mode = _array_reference_deviation(
                saved_array_raw, current_array, settings
            )
        except ExperimentStateError as exc:
            matches = False
            details.append(str(exc))
            continue
        comparison_modes.add(mode)
        metrics.append((absolute, relative))
        if not array_match:
            matches = False
            details.append(f"{name}: {detail}")
    max_absolute, max_relative = _merge_metrics(metrics)
    return NumericalComparison(
        "evaluation",
        matches,
        "explicit rtol/atol comparison using "
        + (" and ".join(sorted(comparison_modes)) if comparison_modes else "saved arrays"),
        max_absolute,
        max_relative,
        tuple(details),
    )


def _compare_stationary_reference(
    reference: Mapping[str, Any],
    current: StationaryPointAnalysisResult,
    settings: NumericalReproductionSettings,
    *,
    name: str = "stationary_points",
) -> NumericalComparison:
    details: list[str] = []
    metrics: list[tuple[float | None, float | None]] = []
    matches = reference.get("kind") == "StationaryPointAnalysisResult"
    if not matches:
        details.append("result kind differs")
    parameters_match, parameter_metrics, parameter_details = _compare_named_values(
        reference.get("parameter_values"), current.parameter_values, settings, label="parameter values"
    )
    metrics.extend(parameter_metrics)
    if not parameters_match:
        matches = False
        details.extend(parameter_details)
    if reference.get("status") != current.numerical_status.value:
        matches = False
        details.append("numerical status differs")
    if not _diagnostic_reference_matches(reference, current.diagnostics):
        matches = False
        details.append("numerical diagnostics differ")
    points_match, absolute, relative, point_details = _match_stationary_points(
        reference.get("points"), current.points, settings
    )
    metrics.append((absolute, relative))
    if not points_match:
        matches = False
        details.extend(point_details)
    max_absolute, max_relative = _merge_metrics(metrics)
    return NumericalComparison(
        name,
        matches,
        "semantic stationary-point tolerance comparison",
        max_absolute,
        max_relative,
        tuple(details),
    )


def _compare_sweep_reference(
    reference: Mapping[str, Any],
    current: ParameterSweepResult,
    settings: NumericalReproductionSettings,
) -> NumericalComparison:
    details: list[str] = []
    metrics: list[tuple[float | None, float | None]] = []
    matches = reference.get("kind") == "ParameterSweepResult"
    if not matches:
        details.append("result kind differs")
    for field, current_value in (
        ("parameter_name", current.parameter_name),
        ("function_name", current.function_name),
    ):
        if reference.get(field) != current_value:
            matches = False
            details.append(f"{field} differs")
    if reference.get("variable_names") != list(current.variable_names):
        matches = False
        details.append("variable names differ")
    if reference.get("step_count") != current.step_count:
        matches = False
        details.append("step count differs")
    for field, current_value in (("start", current.start), ("end", current.end)):
        try:
            match, absolute, relative = _scalar_deviation(
                float(reference[field]), current_value, settings
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            match, absolute, relative = False, None, None
        metrics.append((absolute, relative))
        if not match:
            matches = False
            details.append(f"{field} differs beyond tolerance")
    fixed_match, fixed_metrics, fixed_details = _compare_named_values(
        reference.get("fixed_parameters"), current.fixed_parameters, settings, label="fixed parameters"
    )
    metrics.extend(fixed_metrics)
    if not fixed_match:
        matches = False
        details.extend(fixed_details)
    saved_steps = reference.get("steps")
    if not isinstance(saved_steps, list) or len(saved_steps) != len(current.steps):
        matches = False
        details.append("sweep step count differs or saved steps are malformed")
    elif any(not isinstance(item, dict) for item in saved_steps):
        matches = False
        details.append("one or more saved sweep steps are malformed")
    else:
        try:
            saved_ordered = sorted(saved_steps, key=lambda item: float(item["parameter_value"]))
        except (KeyError, TypeError, ValueError, OverflowError):
            saved_ordered = []
            matches = False
            details.append("saved sweep parameter values are malformed")
        current_ordered = sorted(current.steps, key=lambda item: float(item.parameter_value))
        for index, (saved_step, current_step) in enumerate(zip(saved_ordered, current_ordered)):
            try:
                parameter_match, absolute, relative = _scalar_deviation(
                    float(saved_step["parameter_value"]), current_step.parameter_value, settings
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                parameter_match, absolute, relative = False, None, None
            metrics.append((absolute, relative))
            if not parameter_match:
                matches = False
                details.append(f"sweep step {index} parameter value exceeds tolerance")
            if saved_step.get("status") != current_step.numerical_status.value:
                matches = False
                details.append(f"sweep step {index} numerical status differs")
            if not _diagnostic_reference_matches(saved_step, current_step.diagnostics):
                matches = False
                details.append(f"sweep step {index} numerical diagnostics differ")
            if "error" in saved_step and saved_step.get("error") != current_step.error:
                matches = False
                details.append(f"sweep step {index} error detail differs")
            points_match, point_absolute, point_relative, point_details = _match_stationary_points(
                saved_step.get("points"), current_step.points, settings
            )
            metrics.append((point_absolute, point_relative))
            if not points_match:
                matches = False
                details.extend(f"sweep step {index}: {detail}" for detail in point_details)
    max_absolute, max_relative = _merge_metrics(metrics)
    return NumericalComparison(
        "parameter_sweep",
        matches,
        "step-aligned semantic tolerance comparison",
        max_absolute,
        max_relative,
        tuple(details),
    )


def _compare_symbolic_reference(reference: Mapping[str, Any], current_digest: str | None) -> NumericalComparison:
    matches = (
        reference.get("kind") == "SymbolicAnalysis"
        and isinstance(reference.get("sha256"), str)
        and current_digest == reference.get("sha256")
    )
    details = () if matches else ("symbolic analysis differs",)
    return NumericalComparison(
        "symbolic",
        matches,
        "exact symbolic comparison",
        None,
        None,
        details,
    )



def _validate_result_alignment(
    model: ModelIR,
    parameter_values: dict[str, float],
    evaluation: FunctionEvaluation | SurfaceEvaluation,
    stationary_result: StationaryPointAnalysisResult | None,
    sweep_configuration: SweepConfiguration | None,
    sweep_result: ParameterSweepResult | None,
) -> None:
    """Ensure supplied computed results belong to the supplied validated model."""
    function = model.functions[0] if model.functions else None
    variable_names = tuple(variable.name for variable in model.variables)
    if function is None:
        raise ExperimentStateError("Experiment creation requires one scalar function.")

    if isinstance(evaluation, FunctionEvaluation):
        expected = (variable_names[0], function.name) if len(variable_names) == 1 else None
        actual = (evaluation.x_name, evaluation.y_name)
    else:
        expected = (variable_names[0], variable_names[1], function.name) if len(variable_names) == 2 else None
        actual = (evaluation.x_name, evaluation.y_name, evaluation.z_name)
    if expected is None or actual != expected:
        raise ExperimentStateError(
            "Evaluation result does not correspond to the supplied Model IR."
        )

    if stationary_result is not None:
        stationary_map = dict(stationary_result.parameter_values)
        if tuple(name for name, _ in stationary_result.parameter_values) != tuple(
            parameter.name for parameter in model.parameters
        ) or any(
            float(stationary_map[name]) != float(parameter_values[name])
            for name in stationary_map
            if name in parameter_values
        ):
            raise ExperimentStateError(
                "Stationary-point result parameter state does not correspond to the supplied experiment parameters."
            )

    if sweep_result is not None:
        if sweep_result.function_name != function.name or sweep_result.variable_names != variable_names:
            raise ExperimentStateError(
                "Parameter-sweep result does not correspond to the supplied Model IR."
            )
        try:
            model.parameter(sweep_result.parameter_name)
        except KeyError as exc:
            raise ExperimentStateError(
                "Parameter-sweep result refers to a parameter absent from the supplied Model IR."
            ) from exc
        if sweep_configuration is None:
            raise ExperimentStateError("A parameter-sweep result requires a sweep configuration.")
        if (
            sweep_result.parameter_name != sweep_configuration.parameter_name
            or float(sweep_result.start) != float(sweep_configuration.start)
            or float(sweep_result.end) != float(sweep_configuration.end)
            or int(sweep_result.step_count) != int(sweep_configuration.step_count)
        ):
            raise ExperimentStateError(
                "Parameter-sweep result does not correspond to the supplied sweep configuration."
            )
        expected_fixed = {
            name: float(value)
            for name, value in parameter_values.items()
            if name != sweep_result.parameter_name
        }
        if dict(sweep_result.fixed_parameters) != expected_fixed:
            raise ExperimentStateError(
                "Parameter-sweep fixed parameters do not correspond to the supplied experiment parameters."
            )


def create_experiment_state(
    *,
    model_source: str,
    model: ModelIR,
    parameter_values: dict[str, float],
    selected_model_visualisation: ModelVisualisation | str,
    evaluation: FunctionEvaluation | SurfaceEvaluation,
    stationary_result: StationaryPointAnalysisResult | None,
    evaluation_settings: EvaluationSettings = EvaluationSettings(),
    stationary_settings: StationaryPointSettings = StationaryPointSettings(),
    sweep_configuration: SweepConfiguration | None = None,
    sweep_result: ParameterSweepResult | None = None,
    numerical_reproduction_settings: NumericalReproductionSettings = NumericalReproductionSettings(),
    interpreter_acceptances: tuple[Mapping[str, Any], ...] = (),
) -> ExperimentState:
    """Create a complete experiment state with a corruption-detection checksum."""
    try:
        source_model = validate_model(parse_model_text(model_source))
    except (ModelParseError, ModelValidationError) as exc:
        raise ExperimentStateError(
            f"model_source cannot be compiled to a valid model: {exc}"
        ) from exc
    if fingerprint_model_ir(source_model) != fingerprint_model_ir(model):
        raise ExperimentStateError(
            "model_source does not compile to the supplied Model IR."
        )

    _validate_result_alignment(
        model,
        parameter_values,
        evaluation,
        stationary_result,
        sweep_configuration,
        sweep_result,
    )

    try:
        enforce_experiment_workload_budget(
            model,
            evaluation_settings,
            stationary_settings,
            sweep_configuration,
        )
    except WorkloadBudgetError as exc:
        raise ExperimentStateError(f"Experiment exceeds the computational budget: {exc}") from exc

    expected_names = {parameter.name for parameter in model.parameters}
    if set(parameter_values) != expected_names:
        missing = expected_names - set(parameter_values)
        extra = set(parameter_values) - expected_names
        pieces = []
        if missing:
            pieces.append("missing " + ", ".join(sorted(missing)))
        if extra:
            pieces.append("unknown " + ", ".join(sorted(extra)))
        raise ExperimentStateError("Parameter state does not match the model: " + "; ".join(pieces) + ".")

    if sweep_configuration is None and sweep_result is not None:
        raise ExperimentStateError("A sweep result requires a sweep configuration.")
    if sweep_configuration is not None and sweep_result is None:
        raise ExperimentStateError("A sweep configuration must correspond to a completed sweep result.")

    fingerprints: list[tuple[str, str]] = [
        ("evaluation", fingerprint_evaluation(evaluation)),
    ]
    numerical_references: list[tuple[str, dict[str, Any]]] = [
        ("evaluation", numerical_reference_evaluation(evaluation)),
    ]
    symbolic = fingerprint_symbolic_analysis(model)
    if symbolic is not None:
        fingerprints.append(("symbolic", symbolic))
    if stationary_result is not None:
        fingerprints.append(("stationary_points", fingerprint_stationary_result(stationary_result)))
        numerical_references.append(
            ("stationary_points", numerical_reference_stationary_result(stationary_result))
        )
    if sweep_result is not None:
        fingerprints.append(("parameter_sweep", fingerprint_parameter_sweep(sweep_result)))
        numerical_references.append(
            ("parameter_sweep", numerical_reference_parameter_sweep(sweep_result))
        )

    state = ExperimentState(
        format_version=EXPERIMENT_FORMAT_VERSION,
        laboratory_version=__version__,
        created_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_source=model_source,
        model_sha256=_sha256_text(model_source),
        parameter_values=tuple(
            (parameter.name, float(parameter_values[parameter.name]))
            for parameter in model.parameters
        ),
        selected_model_visualisation=(
            selected_model_visualisation.value
            if isinstance(selected_model_visualisation, ModelVisualisation)
            else str(selected_model_visualisation)
        ),
        evaluation_settings=evaluation_settings,
        stationary_settings=stationary_settings,
        sweep=sweep_configuration,
        result_fingerprints=tuple(fingerprints),
        numerical_reproduction_settings=numerical_reproduction_settings,
        numerical_reference_data=tuple(numerical_references),
        numerical_result_fingerprints=(),
        interpreter_acceptances=tuple(dict(item) for item in interpreter_acceptances),
        environment=current_environment(),
    )
    return state.with_checksum()


def create_run_experiment_state(
    *,
    model_source: str,
    model: ModelIR,
    parameter_values: Mapping[str, float],
    run_outcomes: tuple[RunOutcome, ...],
    views: tuple[ArtifactView, ...] = (),
    numerical_reproduction_settings: NumericalReproductionSettings = NumericalReproductionSettings(),
    interpreter_acceptances: tuple[Mapping[str, Any], ...] = (),
) -> ExperimentState:
    """Freeze a Model IR 3.0 experiment from generic runs and typed artifacts."""
    if not run_outcomes:
        raise ExperimentStateError("A reproducible run experiment requires at least one completed run.")
    try:
        source_model = validate_model(parse_model_text(model_source))
    except (ModelParseError, ModelValidationError) as exc:
        raise ExperimentStateError(f"model_source cannot be compiled to a valid model: {exc}") from exc
    if fingerprint_model_ir(source_model) != fingerprint_model_ir(model):
        raise ExperimentStateError("model_source does not compile to the supplied Model IR.")
    expected_parameters = {item.name for item in model.parameters}
    if set(parameter_values) != expected_parameters:
        raise ExperimentStateError("Parameter state does not exactly match the model parameters.")
    model_hash = fingerprint_model_ir(model)
    run_records: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}
    for outcome in run_outcomes:
        run = outcome.run.payload()
        if run["status"] != "completed" or run["model_ir_sha256"] != model_hash:
            raise ExperimentStateError("Only completed runs for the supplied Model IR may be frozen.")
        run_records.append(run)
        for artifact in outcome.artifacts:
            payload = artifact.payload()
            if payload["model_ir_sha256"] != model_hash:
                raise ExperimentStateError("A scientific artifact belongs to a different model.")
            existing = artifacts.get(artifact.artifact_id)
            if existing is not None and existing != payload:
                raise ExperimentStateError("A content-addressed artifact identifier is inconsistent.")
            artifacts[artifact.artifact_id] = payload
    view_payloads = tuple(item.payload() for item in views)
    known = set(artifacts)
    if any(not set(item["artifact_ids"]).issubset(known) for item in view_payloads):
        raise ExperimentStateError("A view refers to an artifact outside this experiment.")
    state = ExperimentState(
        format_version=EXPERIMENT_FORMAT_VERSION,
        laboratory_version=__version__,
        created_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_source=model_source,
        model_sha256=_sha256_text(model_source),
        parameter_values=tuple(
            (item.name, float(parameter_values[item.name])) for item in model.parameters
        ),
        selected_model_visualisation="",
        evaluation_settings=EvaluationSettings(),
        stationary_settings=StationaryPointSettings(),
        sweep=None,
        result_fingerprints=tuple(
            (artifact_id, payload["artifact_sha256"])
            for artifact_id, payload in sorted(artifacts.items())
        ),
        numerical_reproduction_settings=numerical_reproduction_settings,
        numerical_reference_data=(),
        numerical_result_fingerprints=(),
        interpreter_acceptances=tuple(dict(item) for item in interpreter_acceptances),
        run_records=tuple(run_records),
        artifacts=tuple(artifacts[key] for key in sorted(artifacts)),
        views=view_payloads,
        environment=current_environment(),
    )
    validate_experiment_state_for_model(
        state,
        model,
        enforce_workload_budget=True,
        reconstructed_embedded_model=source_model,
    )
    return state.with_checksum()


def compare_reproduced_results(
    state: ExperimentState,
    *,
    model: ModelIR,
    evaluation: FunctionEvaluation | SurfaceEvaluation,
    stationary_result: StationaryPointAnalysisResult | None,
    sweep_result: ParameterSweepResult | None,
) -> ReproducibilityCheck:
    """Compare strict fingerprints and explicit saved numerical values with reproduced results.

    Strict reproduction uses deterministic SHA-256 fingerprints. Numerical reproduction uses
    direct rtol/atol comparisons against numerical reference data embedded in format-v2
    experiment states. Older format-v1 tolerance-normalised fingerprints are retained only as
    a legacy compatibility signal and are never treated as proof of numerical reproduction.
    """
    actual: dict[str, str] = {"evaluation": fingerprint_evaluation(evaluation)}
    symbolic = fingerprint_symbolic_analysis(model)
    if symbolic is not None:
        actual["symbolic"] = symbolic
    if stationary_result is not None:
        actual["stationary_points"] = fingerprint_stationary_result(stationary_result)
    if sweep_result is not None:
        actual["parameter_sweep"] = fingerprint_parameter_sweep(sweep_result)

    result_matches = tuple(
        (name, actual.get(name) == digest) for name, digest in state.result_fingerprints
    )

    comparisons: list[NumericalComparison] = []
    for name, reference in state.numerical_reference_data:
        if name == "evaluation":
            comparisons.append(
                _compare_evaluation_reference(
                    reference, evaluation, state.numerical_reproduction_settings
                )
            )
        elif name == "symbolic":
            comparisons.append(_compare_symbolic_reference(reference, symbolic))
        elif name == "stationary_points":
            if stationary_result is None:
                comparisons.append(
                    NumericalComparison(
                        name,
                        False,
                        "semantic stationary-point tolerance comparison",
                        details=("stationary-point result was not reproduced",),
                    )
                )
            else:
                comparisons.append(
                    _compare_stationary_reference(
                        reference,
                        stationary_result,
                        state.numerical_reproduction_settings,
                    )
                )
        elif name == "parameter_sweep":
            if sweep_result is None:
                comparisons.append(
                    NumericalComparison(
                        name,
                        False,
                        "step-aligned semantic tolerance comparison",
                        details=("parameter-sweep result was not reproduced",),
                    )
                )
            else:
                comparisons.append(
                    _compare_sweep_reference(
                        reference, sweep_result, state.numerical_reproduction_settings
                    )
                )
        else:
            comparisons.append(
                NumericalComparison(
                    name,
                    False,
                    "unsupported numerical reference",
                    details=(f"unsupported numerical reference result '{name}'",),
                )
            )

    numerical_result_matches = tuple(
        (comparison.name, comparison.matches) for comparison in comparisons
    )

    # Format-v1 states may contain quantisation-based numerical fingerprints. They are
    # preserved for compatibility, but closeness is not an equivalence relation and these
    # hashes therefore cannot establish genuine tolerance agreement.
    legacy_actual: dict[str, str] = {}
    if state.numerical_result_fingerprints:
        legacy_actual["evaluation"] = numerical_fingerprint_evaluation(
            evaluation, state.numerical_reproduction_settings
        )
        if symbolic is not None:
            legacy_actual["symbolic"] = symbolic
        if stationary_result is not None:
            legacy_actual["stationary_points"] = numerical_fingerprint_stationary_result(
                stationary_result, state.numerical_reproduction_settings
            )
        if sweep_result is not None:
            legacy_actual["parameter_sweep"] = numerical_fingerprint_parameter_sweep(
                sweep_result, state.numerical_reproduction_settings
            )
    legacy_matches = tuple(
        (name, legacy_actual.get(name) == digest)
        for name, digest in state.numerical_result_fingerprints
    )

    current_env = dict(current_environment())
    saved_env = state.environment_map
    all_components = sorted(set(current_env) | set(saved_env))
    differences = tuple(
        f"{component}: saved {saved_env.get(component, 'missing')}, current {current_env.get(component, 'missing')}"
        for component in all_components
        if saved_env.get(component) != current_env.get(component)
    )

    try:
        embedded_model = compile_experiment_model(state)
        model_ir_matches = fingerprint_model_ir(embedded_model) == fingerprint_model_ir(model)
    except ExperimentStateError:
        model_ir_matches = False

    return ReproducibilityCheck(
        model_source_matches=_sha256_text(state.model_source) == state.model_sha256,
        laboratory_version_matches=state.laboratory_version == __version__,
        environment_matches=not differences,
        result_matches=result_matches,
        model_ir_matches=model_ir_matches,
        numerical_result_matches=numerical_result_matches,
        numerical_comparisons=tuple(comparisons),
        legacy_tolerance_fingerprint_matches=legacy_matches,
        environment_differences=differences,
    )
