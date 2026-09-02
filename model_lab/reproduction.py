"""Reproduction semantics and reports for Model Laboratory experiments.

Phase B treats reproduction as a deterministic, front-end-independent operation.  Exact
reproduction is established by strict canonical result fingerprints.  Numerical
reproduction is established only by explicit result-specific comparisons against saved
reference values using the experiment's recorded ``rtol``/``atol`` settings.  The recorded
software/hardware environment is compared and reported independently of result identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
import re
from typing import Any, Mapping

from . import __version__
from .analysis import AnalysisError, ParameterSweepResult, StationaryPointAnalysisResult, run_parameter_sweep, run_stationary_point_analysis
from .bundle import canonical_provenance_document
from .canonical import canonical_model_ir_sha256
from .evaluator import EvaluationError, FunctionEvaluation, SurfaceEvaluation, evaluate_single_variable_function, evaluate_two_variable_function
from .experiment import (
    ExperimentState,
    ExperimentStateError,
    NumericalComparison,
    ReproducibilityCheck,
    compare_reproduced_results,
    compile_experiment_model,
    current_environment,
    diagnostic_record_payload,
    fingerprint_evaluation,
    fingerprint_parameter_sweep,
    fingerprint_stationary_result,
    fingerprint_symbolic_analysis,
    validate_experiment_state_for_model,
)
from .model import ModelIR
from .protocol import (
    CapabilityPackRegistry,
    ProtocolError,
    RunOutcome,
    ScientificArtifact,
    artifact_from_document,
    comparator_registry,
    run_capability_sweep,
)


REPRODUCTION_REPORT_SCHEMA = "model-laboratory-reproduction-report"
REPRODUCTION_REPORT_SCHEMA_VERSION = "1.2"
ENVIRONMENT_COMPATIBILITY_POLICY_VERSION = "1.0"




def _json_number(value: float | None) -> float | str | None:
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric):
        return "NaN"
    if math.isinf(numeric):
        return "Infinity" if numeric > 0 else "-Infinity"
    return numeric


class ReproductionStatus(str, Enum):
    """Overall or per-result reproduction status."""

    EXACT = "EXACT REPRODUCTION"
    NUMERICAL = "NUMERICALLY REPRODUCED"
    STATISTICAL = "STATISTICALLY REPRODUCED"
    PARTIAL = "PARTIALLY REPRODUCED"
    NOT_REPRODUCED = "NOT REPRODUCED"
    UNABLE = "UNABLE TO REPRODUCE"


class EnvironmentComparisonStatus(str, Enum):
    """Relationship between the saved and current numerical environments."""

    IDENTICAL = "IDENTICAL"
    COMPATIBLE = "COMPATIBLE"
    DIFFERENT = "DIFFERENT"


@dataclass(frozen=True, slots=True)
class ResultReproductionReport:
    """Reproduction evidence for one recorded result."""

    name: str
    status: ReproductionStatus
    saved_strict_fingerprint: str | None
    current_strict_fingerprint: str | None
    strict_fingerprint_matches: bool
    numerical_reference_available: bool
    numerical_comparison_matches: bool | None
    comparison_kind: str | None = None
    max_absolute_deviation: float | None = None
    max_relative_deviation: float | None = None
    details: tuple[str, ...] = ()
    saved_diagnostics: tuple[dict[str, Any], ...] = ()
    current_diagnostics: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "strict": {
                "saved_sha256": self.saved_strict_fingerprint,
                "current_sha256": self.current_strict_fingerprint,
                "matches": self.strict_fingerprint_matches,
            },
            "numerical": {
                "reference_available": self.numerical_reference_available,
                "matches": self.numerical_comparison_matches,
                "comparison_kind": self.comparison_kind,
                "maximum_absolute_deviation": _json_number(self.max_absolute_deviation),
                "maximum_relative_deviation": _json_number(self.max_relative_deviation),
            },
            "diagnostics": {
                "saved": [dict(item) for item in self.saved_diagnostics],
                "current": [dict(item) for item in self.current_diagnostics],
            },
            "details": list(self.details),
        }


@dataclass(frozen=True, slots=True)
class ReproductionReport:
    """Front-end-independent report for one reproduction attempt."""

    status: ReproductionStatus
    experiment_id: str | None
    experiment_state_sha256: str
    model_validated: bool
    model_source_integrity: bool
    model_ir_matches: bool
    saved_laboratory_version: str
    current_laboratory_version: str
    laboratory_version_matches: bool
    environment_status: EnvironmentComparisonStatus
    environment_differences: tuple[str, ...]
    relative_tolerance: float
    absolute_tolerance: float
    provenance_document: Mapping[str, Any]
    results: tuple[ResultReproductionReport, ...]
    warnings: tuple[str, ...] = ()
    execution_errors: tuple[str, ...] = ()

    @property
    def environment_matches(self) -> bool:
        return self.environment_status is EnvironmentComparisonStatus.IDENTICAL

    @property
    def exact_result_count(self) -> int:
        return sum(item.status is ReproductionStatus.EXACT for item in self.results)

    @property
    def numerical_result_count(self) -> int:
        return sum(item.status is ReproductionStatus.NUMERICAL for item in self.results)

    @property
    def statistical_result_count(self) -> int:
        return sum(item.status is ReproductionStatus.STATISTICAL for item in self.results)

    @property
    def failed_result_count(self) -> int:
        return sum(
            item.status in (ReproductionStatus.NOT_REPRODUCED, ReproductionStatus.UNABLE)
            for item in self.results
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPRODUCTION_REPORT_SCHEMA,
            "schema_version": REPRODUCTION_REPORT_SCHEMA_VERSION,
            "status": self.status.value,
            "experiment_id": self.experiment_id,
            "experiment_state_sha256": self.experiment_state_sha256,
            "model": {
                "validated": self.model_validated,
                "source_integrity": self.model_source_integrity,
                "canonical_ir_matches": self.model_ir_matches,
            },
            "laboratory": {
                "saved_version": self.saved_laboratory_version,
                "current_version": self.current_laboratory_version,
                "version_matches": self.laboratory_version_matches,
            },
            "environment": {
                "status": self.environment_status.value,
                "compatibility_policy_version": ENVIRONMENT_COMPATIBILITY_POLICY_VERSION,
                "differences": list(self.environment_differences),
            },
            "numerical_tolerances": {
                "relative_tolerance": self.relative_tolerance,
                "absolute_tolerance": self.absolute_tolerance,
            },
            "provenance": dict(self.provenance_document),
            "warnings": list(self.warnings),
            "summary": {
                "result_count": len(self.results),
                "exact_result_count": self.exact_result_count,
                "numerical_result_count": self.numerical_result_count,
                "statistical_result_count": self.statistical_result_count,
                "failed_result_count": self.failed_result_count,
            },
            "results": [item.to_dict() for item in self.results],
            "execution_errors": list(self.execution_errors),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"

    def to_text(self) -> str:
        lines = [
            "MODEL LABORATORY REPRODUCTION REPORT",
            "",
            f"Status: {self.status.value}",
        ]
        if self.experiment_id is not None:
            lines.append(f"Experiment ID: {self.experiment_id}")
        lines.extend(
            [
                f"Experiment state SHA-256: {self.experiment_state_sha256}",
                "",
                "Model",
                f"  Validated: {'yes' if self.model_validated else 'no'}",
                f"  Source integrity: {'yes' if self.model_source_integrity else 'no'}",
                f"  Canonical IR matches: {'yes' if self.model_ir_matches else 'no'}",
                "",
                "Environment",
                f"  Status: {self.environment_status.value}",
                f"  Saved laboratory version: {self.saved_laboratory_version}",
                f"  Current laboratory version: {self.current_laboratory_version}",
                f"  Laboratory version matches: {'yes' if self.laboratory_version_matches else 'no'}",
                "",
                "Numerical tolerances",
                f"  rtol: {self.relative_tolerance:.17g}",
                f"  atol: {self.absolute_tolerance:.17g}",
                "",
                "Results",
            ]
        )
        for item in self.results:
            lines.append(f"  {item.name}: {item.status.value}")
            lines.append(
                "    Strict fingerprint: "
                + ("match" if item.strict_fingerprint_matches else "different")
            )
            if item.numerical_reference_available:
                lines.append(
                    "    Numerical tolerance comparison: "
                    + ("match" if item.numerical_comparison_matches else "different")
                )
                if item.max_absolute_deviation is not None:
                    lines.append(
                        f"    Maximum absolute deviation: {item.max_absolute_deviation:.17g}"
                    )
                if item.max_relative_deviation is not None:
                    lines.append(
                        f"    Maximum relative deviation: {item.max_relative_deviation:.17g}"
                    )
            for detail in item.details:
                lines.append(f"    Detail: {detail}")
            for label, records in (
                ("Saved", item.saved_diagnostics),
                ("Current", item.current_diagnostics),
            ):
                for diagnostic in records:
                    scope = diagnostic.get("scope")
                    scope_text = f"; scope={scope}" if scope is not None else ""
                    lines.append(
                        f"    {label} diagnostic [{diagnostic.get('severity', 'unknown')}] "
                        f"{diagnostic.get('code', 'unknown')}: "
                        f"{diagnostic.get('message', '')}{scope_text}"
                    )
        if self.warnings:
            lines.extend(["", "Warnings"])
            lines.extend(f"  {item}" for item in self.warnings)
        provenance_model = self.provenance_document.get("model", [])
        provenance_results = self.provenance_document.get("results", [])
        provenance_interpreter = self.provenance_document.get("interpreter_acceptances", [])
        lines.extend(
            [
                "",
                "Provenance",
                f"  Model records: {len(provenance_model) if isinstance(provenance_model, list) else 0}",
                f"  Result records: {len(provenance_results) if isinstance(provenance_results, list) else 0}",
                f"  Accepted interpreter proposals: {len(provenance_interpreter) if isinstance(provenance_interpreter, list) else 0}",
            ]
        )
        if isinstance(provenance_results, list):
            for record in provenance_results:
                if isinstance(record, Mapping):
                    lines.append(
                        f"  {record.get('reference', 'unknown')}: "
                        f"{record.get('operation') or record.get('source_location') or 'recorded'}"
                    )
        if isinstance(provenance_interpreter, list):
            for record in provenance_interpreter:
                if isinstance(record, Mapping):
                    provider = record.get("provider", {})
                    if not isinstance(provider, Mapping):
                        provider = {}
                    compiler = record.get("compiler", {})
                    if not isinstance(compiler, Mapping):
                        compiler = {}
                    compiler_text = (
                        f" / edit program {compiler.get('edit_program_sha256')}"
                        if compiler.get("edit_program_sha256")
                        else " / legacy whole-source proposal"
                    )
                    lines.append(
                        "  Interpreter acceptance "
                        f"{record.get('acceptance_sha256', 'unknown')}: "
                        f"{provider.get('provider', 'unknown')} / "
                        f"{provider.get('model_tag', provider.get('model', 'unknown'))} / "
                        "observed digest "
                        f"{provider.get('observed_model_digest', provider.get('model_digest', 'unknown'))}"
                        f"{compiler_text}"
                    )
        if self.environment_differences:
            lines.extend(["", "Environment differences"])
            lines.extend(f"  {item}" for item in self.environment_differences)
        if self.execution_errors:
            lines.extend(["", "Execution errors"])
            lines.extend(f"  {item}" for item in self.execution_errors)
        return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class ReproductionOutcome:
    """Reproduction report plus newly computed deterministic result objects."""

    report: ReproductionReport
    evaluation: FunctionEvaluation | SurfaceEvaluation | None = None
    stationary_result: StationaryPointAnalysisResult | None = None
    sweep_result: ParameterSweepResult | None = None
    run_outcomes: tuple[RunOutcome, ...] = ()
    artifacts: tuple[ScientificArtifact, ...] = ()


def reproduce_run_experiment(
    state: ExperimentState,
    *,
    model: ModelIR,
    registry: CapabilityPackRegistry | None = None,
    experiment_id: str | None = None,
) -> ReproductionOutcome:
    """Re-execute a format-6 Run -> Artifact experiment through installed packs."""
    from .builtin_packs import run_registry

    active_registry = registry or run_registry
    try:
        validate_experiment_state_for_model(state, model, enforce_workload_budget=True)
    except ExperimentStateError as exc:
        return ReproductionOutcome(
            report=_unable_report(state, model, experiment_id=experiment_id, message=str(exc))
        )
    saved_artifacts = {
        item.artifact_id: item for item in map(artifact_from_document, state.artifacts)
    }
    result_reports: list[ResultReproductionReport] = []
    outcomes: list[RunOutcome] = []
    current_artifacts: list[ScientificArtifact] = []
    execution_errors: list[str] = []
    backend_warnings: list[str] = []
    rtol = state.numerical_reproduction_settings.relative_tolerance
    atol = state.numerical_reproduction_settings.absolute_tolerance
    for saved_run in state.run_records:
        capability_id = str(saved_run["capability_id"])
        try:
            if capability_id == "org.modellab.protocol.capability-sweep":
                settings = saved_run["settings"]
                outcome = run_capability_sweep(
                    active_registry,
                    model,
                    target_capability_id=str(settings["target_capability_id"]),
                    target_capability_version=settings.get("target_capability_version"),
                    coordinate_name=str(settings["coordinate_name"]),
                    coordinate_values=settings["coordinate_values"],
                    setting_path=tuple(settings["setting_path"]),
                    base_settings=settings.get("base_settings", {}),
                )
                output_types = (outcome.artifacts[0].artifact_type,)
                comparators = ("org.modellab.comparator.numeric",)
            else:
                descriptor = active_registry.descriptor(
                    capability_id, str(saved_run["capability_version"])
                )
                outcome = active_registry.run(
                    capability_id,
                    model,
                    saved_run["settings"],
                    version=str(saved_run["capability_version"]),
                )
                output_types = tuple(item.identifier for item in descriptor.output_types)
                comparators = tuple(item.comparator_id for item in descriptor.output_types)
            outcomes.append(outcome)
            current_artifacts.extend(outcome.artifacts)
        except (ProtocolError, KeyError, TypeError, ValueError) as exc:
            message = f"{capability_id}: {exc}"
            execution_errors.append(message)
            for artifact_id in saved_run["artifact_ids"]:
                saved = saved_artifacts.get(str(artifact_id))
                result_reports.append(
                    ResultReproductionReport(
                        name=str(artifact_id),
                        status=ReproductionStatus.UNABLE,
                        saved_strict_fingerprint=None if saved is None else saved.artifact_sha256,
                        current_strict_fingerprint=None,
                        strict_fingerprint_matches=False,
                        numerical_reference_available=False,
                        numerical_comparison_matches=None,
                        details=(message,),
                    )
                )
            continue
        saved_backend = saved_run.get("backend_identity", {})
        current_backend = outcome.run.backend_identity
        backend_details: tuple[str, ...] = ()
        if saved_backend != current_backend:
            warning = (
                f"{capability_id}: saved and current capability backend identities differ; "
                "result identity is reported separately."
            )
            backend_warnings.append(warning)
            backend_details = (warning,)
        saved_ids = tuple(str(item) for item in saved_run["artifact_ids"])
        if len(saved_ids) != len(outcome.artifacts):
            message = f"{capability_id}: output artifact count changed"
            execution_errors.append(message)
        for index, saved_id in enumerate(saved_ids):
            saved = saved_artifacts.get(saved_id)
            current = outcome.artifacts[index] if index < len(outcome.artifacts) else None
            if saved is None or current is None:
                result_reports.append(
                    ResultReproductionReport(
                        name=saved_id,
                        status=ReproductionStatus.UNABLE,
                        saved_strict_fingerprint=None if saved is None else saved.artifact_sha256,
                        current_strict_fingerprint=None if current is None else current.artifact_sha256,
                        strict_fingerprint_matches=False,
                        numerical_reference_available=False,
                        numerical_comparison_matches=None,
                        details=("Saved or current artifact is unavailable.",),
                    )
                )
                continue
            if index >= len(output_types) or saved.artifact_type != output_types[index] or current.artifact_type != saved.artifact_type:
                result_reports.append(
                    ResultReproductionReport(
                        name=saved_id,
                        status=ReproductionStatus.NOT_REPRODUCED,
                        saved_strict_fingerprint=saved.artifact_sha256,
                        current_strict_fingerprint=current.artifact_sha256,
                        strict_fingerprint_matches=False,
                        numerical_reference_available=False,
                        numerical_comparison_matches=None,
                        details=("Artifact type changed.",),
                    )
                )
                continue
            exact = saved.artifact_sha256 == current.artifact_sha256
            if exact:
                status = ReproductionStatus.EXACT
                comparison = None
            else:
                comparison = comparator_registry.compare(
                    comparators[index], saved, current, rtol=rtol, atol=atol
                )
                status = (
                    (
                        ReproductionStatus.STATISTICAL
                        if comparison.comparator_id == "org.modellab.comparator.stochastic"
                        else ReproductionStatus.NUMERICAL
                    )
                    if comparison.reproduced
                    else ReproductionStatus.NOT_REPRODUCED
                )
            result_reports.append(
                ResultReproductionReport(
                    name=saved_id,
                    status=status,
                    saved_strict_fingerprint=saved.artifact_sha256,
                    current_strict_fingerprint=current.artifact_sha256,
                    strict_fingerprint_matches=exact,
                    numerical_reference_available=not exact,
                    numerical_comparison_matches=None if comparison is None else comparison.reproduced,
                    comparison_kind=None if comparison is None else comparison.comparator_id,
                    max_absolute_deviation=(
                        None if comparison is None else comparison.maximum_absolute_deviation
                    ),
                    max_relative_deviation=(
                        None if comparison is None else comparison.maximum_relative_deviation
                    ),
                    details=backend_details
                    + (
                        ()
                        if comparison is None
                        else tuple(
                            f"{key}: {value}" for key, value in sorted(comparison.details.items())
                        )
                    ),
                )
            )
    model_validated, source_integrity, model_ir_matches = _model_evidence(state, model)
    environment_status, environment_differences = _environment_evidence(state)
    reports = tuple(result_reports)
    report = ReproductionReport(
        status=(
            _overall_status(reports, tuple(execution_errors))
            if model_validated and source_integrity and model_ir_matches
            else ReproductionStatus.UNABLE
        ),
        experiment_id=experiment_id,
        experiment_state_sha256=state.with_checksum().state_sha256,
        model_validated=model_validated,
        model_source_integrity=source_integrity,
        model_ir_matches=model_ir_matches,
        saved_laboratory_version=state.laboratory_version,
        current_laboratory_version=__version__,
        laboratory_version_matches=state.laboratory_version == __version__,
        environment_status=environment_status,
        environment_differences=environment_differences,
        relative_tolerance=rtol,
        absolute_tolerance=atol,
        provenance_document=canonical_provenance_document(state, model),
        results=reports,
        warnings=tuple(backend_warnings),
        execution_errors=tuple(execution_errors),
    )
    return ReproductionOutcome(
        report=report,
        run_outcomes=tuple(outcomes),
        artifacts=tuple(current_artifacts),
    )


def _current_fingerprints(
    model: ModelIR,
    evaluation: FunctionEvaluation | SurfaceEvaluation,
    stationary_result: StationaryPointAnalysisResult | None,
    sweep_result: ParameterSweepResult | None,
) -> dict[str, str]:
    values: dict[str, str] = {"evaluation": fingerprint_evaluation(evaluation)}
    symbolic = fingerprint_symbolic_analysis(model)
    if symbolic is not None:
        values["symbolic"] = symbolic
    if stationary_result is not None:
        values["stationary_points"] = fingerprint_stationary_result(stationary_result)
    if sweep_result is not None:
        values["parameter_sweep"] = fingerprint_parameter_sweep(sweep_result)
    return values


def _model_evidence(state: ExperimentState, model: ModelIR) -> tuple[bool, bool, bool]:
    source_integrity = False
    ir_matches = False
    validated = False
    try:
        compiled = compile_experiment_model(state)
        source_integrity = True
        validated = True
        ir_matches = canonical_model_ir_sha256(compiled) == canonical_model_ir_sha256(model)
    except ExperimentStateError:
        pass
    return validated, source_integrity, ir_matches


def _version_prefix(value: str, length: int) -> tuple[int, ...] | None:
    numbers = tuple(int(item) for item in re.findall(r"\d+", value))
    if len(numbers) < length:
        return None
    return numbers[:length]


def _environment_difference_kind(
    component: str,
    saved: str | None,
    current: str | None,
) -> str:
    """Classify one changed component under the documented compatibility policy."""
    if saved is None or current is None:
        return "unassessed"
    if component in {
        "python_implementation",
        "float_radix",
        "float_mantissa_bits",
        "float_max_exponent",
        "source_tree_sha256",
    }:
        return "incompatible"
    if component == "python":
        return (
            "compatible"
            if _version_prefix(saved, 2) == _version_prefix(current, 2)
            else "incompatible"
        )
    if component in {"numpy", "scipy", "sympy", "pydantic", "PyYAML"}:
        return (
            "compatible"
            if _version_prefix(saved, 1) == _version_prefix(current, 1)
            else "incompatible"
        )
    if component == "laboratory_version":
        return (
            "compatible"
            if _version_prefix(saved, 2) == _version_prefix(current, 2)
            else "incompatible"
        )
    # OS, architecture, compiler, BLAS/LAPACK, patch dependency releases and build
    # metadata may change while remaining eligible for numerical reproduction.
    return "compatible"


def compare_environment_records(
    saved: Mapping[str, str],
    current: Mapping[str, str],
    *,
    saved_laboratory_version: str,
    current_laboratory_version: str,
) -> tuple[EnvironmentComparisonStatus, tuple[str, ...]]:
    """Compare environments as identical, compatible, or materially different.

    Compatibility is an eligibility statement, never a reproduction result.  Exact and
    numerical result identity remain independently established by their fingerprints and
    reference comparisons.
    """
    saved_values = dict(saved)
    current_values = dict(current)
    saved_values["laboratory_version"] = saved_laboratory_version
    current_values["laboratory_version"] = current_laboratory_version
    differences: list[str] = []
    incompatible = False
    for component in sorted(set(saved_values) | set(current_values)):
        saved_value = saved_values.get(component)
        current_value = current_values.get(component)
        if saved_value == current_value:
            continue
        kind = _environment_difference_kind(component, saved_value, current_value)
        incompatible = incompatible or kind == "incompatible"
        differences.append(
            f"{component} [{kind}]: saved {saved_value or 'missing'}, "
            f"current {current_value or 'missing'}"
        )
    if not differences:
        return EnvironmentComparisonStatus.IDENTICAL, ()
    return (
        EnvironmentComparisonStatus.DIFFERENT
        if incompatible
        else EnvironmentComparisonStatus.COMPATIBLE,
        tuple(differences),
    )


def _environment_evidence(
    state: ExperimentState,
) -> tuple[EnvironmentComparisonStatus, tuple[str, ...]]:
    return compare_environment_records(
        state.environment_map,
        dict(current_environment()),
        saved_laboratory_version=state.laboratory_version,
        current_laboratory_version=__version__,
    )


def _overall_status(results: tuple[ResultReproductionReport, ...], execution_errors: tuple[str, ...]) -> ReproductionStatus:
    if not results:
        return ReproductionStatus.UNABLE
    statuses = tuple(item.status for item in results)
    if all(status is ReproductionStatus.EXACT for status in statuses):
        return ReproductionStatus.EXACT
    if all(status in (ReproductionStatus.EXACT, ReproductionStatus.STATISTICAL) for status in statuses):
        return ReproductionStatus.STATISTICAL
    if all(status in (ReproductionStatus.EXACT, ReproductionStatus.NUMERICAL) for status in statuses):
        return ReproductionStatus.NUMERICAL
    reproduced = sum(
        status in (
            ReproductionStatus.EXACT,
            ReproductionStatus.NUMERICAL,
            ReproductionStatus.STATISTICAL,
        )
        for status in statuses
    )
    if reproduced == len(statuses):
        return ReproductionStatus.PARTIAL
    if reproduced and reproduced < len(statuses):
        return ReproductionStatus.PARTIAL
    if all(status is ReproductionStatus.UNABLE for status in statuses) or (
        execution_errors and not reproduced
    ):
        return ReproductionStatus.UNABLE
    return ReproductionStatus.NOT_REPRODUCED


def _saved_diagnostic_records(
    name: str,
    reference: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    """Extract complete saved diagnostics without making message text scientific."""
    if reference is None:
        return ()
    if name != "parameter_sweep":
        records = reference.get("diagnostic_records")
        if not isinstance(records, list):
            return ()
        return tuple(dict(item) for item in records if isinstance(item, Mapping))

    result: list[dict[str, Any]] = []
    steps = reference.get("steps")
    if not isinstance(steps, list):
        return ()
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        records = step.get("diagnostic_records")
        if not isinstance(records, list):
            continue
        for item in records:
            if not isinstance(item, Mapping):
                continue
            record = dict(item)
            record["scope"] = {"parameter_value": step.get("parameter_value")}
            result.append(record)
    return tuple(result)


def _current_diagnostic_records(
    name: str,
    evaluation: FunctionEvaluation | SurfaceEvaluation,
    stationary_result: StationaryPointAnalysisResult | None,
    sweep_result: ParameterSweepResult | None,
) -> tuple[dict[str, Any], ...]:
    """Return complete current diagnostics for report presentation."""
    if name == "evaluation":
        return tuple(diagnostic_record_payload(item) for item in evaluation.diagnostics)
    if name == "stationary_points" and stationary_result is not None:
        return tuple(
            diagnostic_record_payload(item) for item in stationary_result.diagnostics
        )
    if name != "parameter_sweep" or sweep_result is None:
        return ()

    result: list[dict[str, Any]] = []
    for step in sweep_result.steps:
        for item in step.diagnostics:
            record = diagnostic_record_payload(item)
            record["scope"] = {"parameter_value": float(step.parameter_value)}
            result.append(record)
    return tuple(result)


def _warning_inventory(
    results: tuple[ResultReproductionReport, ...],
) -> tuple[str, ...]:
    """Return deterministic human-facing warnings from current diagnostics."""
    warnings: set[str] = set()
    for result in results:
        if result.comparison_kind and "canonical samples and summaries" in result.comparison_kind:
            warnings.add(
                f"{result.name}: numerical reproduction was assessed against the frozen "
                "selective large-result reference, not every source-array element"
            )
        for diagnostic in result.current_diagnostics:
            if diagnostic.get("severity") != "warning":
                continue
            scope = diagnostic.get("scope")
            scope_text = f" ({scope})" if scope is not None else ""
            warnings.add(
                f"{result.name}{scope_text}: "
                f"{diagnostic.get('message') or diagnostic.get('code', 'warning')}"
            )
    return tuple(sorted(warnings))


def build_reproduction_report(
    state: ExperimentState,
    *,
    model: ModelIR,
    evaluation: FunctionEvaluation | SurfaceEvaluation,
    stationary_result: StationaryPointAnalysisResult | None,
    sweep_result: ParameterSweepResult | None,
    experiment_id: str | None = None,
    execution_errors: Mapping[str, str] | None = None,
) -> ReproductionReport:
    """Build a structured reproduction report from already-computed results."""
    errors = dict(execution_errors or {})
    check = compare_reproduced_results(
        state,
        model=model,
        evaluation=evaluation,
        stationary_result=stationary_result,
        sweep_result=sweep_result,
    )
    current_fingerprints = _current_fingerprints(
        model, evaluation, stationary_result, sweep_result
    )
    strict_matches = dict(check.result_matches)
    numerical_matches = dict(check.numerical_result_matches)
    numerical_comparisons = {item.name: item for item in check.numerical_comparisons}
    saved_fingerprints = state.fingerprint_map
    saved_references = dict(state.numerical_reference_data)

    result_names = [name for name, _ in state.result_fingerprints]
    for name, _ in state.numerical_reference_data:
        if name not in result_names:
            result_names.append(name)

    result_reports: list[ResultReproductionReport] = []
    for name in result_names:
        comparison: NumericalComparison | None = numerical_comparisons.get(name)
        strict_match = bool(strict_matches.get(name, False))
        numerical_available = name in numerical_matches
        numerical_match = numerical_matches.get(name)
        details: list[str] = []
        if comparison is not None:
            details.extend(comparison.details)
        if name in errors:
            details.append(errors[name])
            status = ReproductionStatus.UNABLE
        elif strict_match:
            status = ReproductionStatus.EXACT
        elif numerical_available and numerical_match:
            status = ReproductionStatus.NUMERICAL
        else:
            status = ReproductionStatus.NOT_REPRODUCED
        result_reports.append(
            ResultReproductionReport(
                name=name,
                status=status,
                saved_strict_fingerprint=saved_fingerprints.get(name),
                current_strict_fingerprint=current_fingerprints.get(name),
                strict_fingerprint_matches=strict_match,
                numerical_reference_available=numerical_available,
                numerical_comparison_matches=numerical_match,
                comparison_kind=None if comparison is None else comparison.comparison_kind,
                max_absolute_deviation=None if comparison is None else comparison.max_absolute_deviation,
                max_relative_deviation=None if comparison is None else comparison.max_relative_deviation,
                details=tuple(details),
                saved_diagnostics=_saved_diagnostic_records(
                    name, saved_references.get(name)
                ),
                current_diagnostics=_current_diagnostic_records(
                    name,
                    evaluation,
                    stationary_result,
                    sweep_result,
                ),
            )
        )

    model_validated, source_integrity, model_ir_matches = _model_evidence(state, model)
    environment_status, environment_differences = _environment_evidence(state)
    execution_error_values = tuple(f"{name}: {message}" for name, message in sorted(errors.items()))
    results_tuple = tuple(result_reports)
    provenance_document = canonical_provenance_document(state, model)
    warnings = _warning_inventory(results_tuple)
    status = _overall_status(results_tuple, execution_error_values)
    if not (model_validated and source_integrity and model_ir_matches):
        status = ReproductionStatus.UNABLE

    return ReproductionReport(
        status=status,
        experiment_id=experiment_id,
        experiment_state_sha256=state.with_checksum().state_sha256,
        model_validated=model_validated,
        model_source_integrity=source_integrity,
        model_ir_matches=model_ir_matches,
        saved_laboratory_version=state.laboratory_version,
        current_laboratory_version=__version__,
        laboratory_version_matches=state.laboratory_version == __version__,
        environment_status=environment_status,
        environment_differences=environment_differences,
        relative_tolerance=state.numerical_reproduction_settings.relative_tolerance,
        absolute_tolerance=state.numerical_reproduction_settings.absolute_tolerance,
        provenance_document=provenance_document,
        results=results_tuple,
        warnings=warnings,
        execution_errors=execution_error_values,
    )


def _unable_report(
    state: ExperimentState,
    model: ModelIR,
    *,
    experiment_id: str | None,
    message: str,
) -> ReproductionReport:
    model_validated, source_integrity, model_ir_matches = _model_evidence(state, model)
    environment_status, environment_differences = _environment_evidence(state)
    saved_references = dict(state.numerical_reference_data)
    results = tuple(
        ResultReproductionReport(
            name=name,
            status=ReproductionStatus.UNABLE,
            saved_strict_fingerprint=digest,
            current_strict_fingerprint=None,
            strict_fingerprint_matches=False,
            numerical_reference_available=name in {item_name for item_name, _ in state.numerical_reference_data},
            numerical_comparison_matches=None,
            details=(message,),
            saved_diagnostics=_saved_diagnostic_records(
                name, saved_references.get(name)
            ),
        )
        for name, digest in state.result_fingerprints
    )
    return ReproductionReport(
        status=ReproductionStatus.UNABLE,
        experiment_id=experiment_id,
        experiment_state_sha256=state.with_checksum().state_sha256,
        model_validated=model_validated,
        model_source_integrity=source_integrity,
        model_ir_matches=model_ir_matches,
        saved_laboratory_version=state.laboratory_version,
        current_laboratory_version=__version__,
        laboratory_version_matches=state.laboratory_version == __version__,
        environment_status=environment_status,
        environment_differences=environment_differences,
        relative_tolerance=state.numerical_reproduction_settings.relative_tolerance,
        absolute_tolerance=state.numerical_reproduction_settings.absolute_tolerance,
        provenance_document=canonical_provenance_document(state, model),
        results=results,
        execution_errors=(message,),
    )


def reproduce_experiment(
    state: ExperimentState,
    *,
    model: ModelIR | None = None,
    experiment_id: str | None = None,
) -> ReproductionOutcome:
    """Execute a saved deterministic experiment and return a structured report.

    Calling this function is the explicit execution action.  Loading or validating an
    experiment never invokes it.
    """
    if model is None:
        try:
            model = compile_experiment_model(state)
        except ExperimentStateError as exc:
            # There is no trustworthy Model IR available with which to reproduce anything.
            raise ExperimentStateError(f"Experiment cannot be reconstructed for reproduction: {exc}") from exc

    if state.format_version >= 6 and state.artifacts:
        return reproduce_run_experiment(
            state,
            model=model,
            experiment_id=experiment_id,
        )

    try:
        validate_experiment_state_for_model(state, model, enforce_workload_budget=True)
    except ExperimentStateError as exc:
        return ReproductionOutcome(
            report=_unable_report(
                state,
                model,
                experiment_id=experiment_id,
                message=str(exc),
            )
        )

    parameters = state.parameter_map
    try:
        if len(model.variables) == 1:
            evaluation: FunctionEvaluation | SurfaceEvaluation = evaluate_single_variable_function(
                model,
                parameters,
                points=state.evaluation_settings.points_1d,
            )
        elif len(model.variables) == 2:
            evaluation = evaluate_two_variable_function(
                model,
                parameters,
                points_per_axis=state.evaluation_settings.points_per_axis_2d,
            )
        else:
            raise EvaluationError(
                "Saved direct evaluation requires one or two continuous variables."
            )
    except EvaluationError as exc:
        return ReproductionOutcome(
            report=_unable_report(
                state,
                model,
                experiment_id=experiment_id,
                message=f"evaluation: {exc}",
            )
        )

    execution_errors: dict[str, str] = {}
    stationary_result: StationaryPointAnalysisResult | None = None
    if "stationary_points" in state.fingerprint_map:
        try:
            stationary_result = run_stationary_point_analysis(
                model,
                parameters,
                samples_1d=state.stationary_settings.samples_1d,
                seeds_per_axis_2d=state.stationary_settings.seeds_per_axis_2d,
                root_tolerance=state.stationary_settings.root_tolerance,
            )
        except AnalysisError as exc:
            execution_errors["stationary_points"] = str(exc)

    sweep_result: ParameterSweepResult | None = None
    if "parameter_sweep" in state.fingerprint_map:
        if state.sweep is None:
            execution_errors["parameter_sweep"] = "Saved parameter-sweep result has no sweep configuration."
        else:
            fixed = {
                name: value
                for name, value in parameters.items()
                if name != state.sweep.parameter_name
            }
            try:
                sweep_result = run_parameter_sweep(
                    model,
                    state.sweep.parameter_name,
                    start=state.sweep.start,
                    end=state.sweep.end,
                    step_count=state.sweep.step_count,
                    fixed_parameter_values=fixed,
                    samples_1d=state.stationary_settings.samples_1d,
                    seeds_per_axis_2d=state.stationary_settings.seeds_per_axis_2d,
                    root_tolerance=state.stationary_settings.root_tolerance,
                )
            except AnalysisError as exc:
                execution_errors["parameter_sweep"] = str(exc)

    report = build_reproduction_report(
        state,
        model=model,
        evaluation=evaluation,
        stationary_result=stationary_result,
        sweep_result=sweep_result,
        experiment_id=experiment_id,
        execution_errors=execution_errors,
    )
    return ReproductionOutcome(
        report=report,
        evaluation=evaluation,
        stationary_result=stationary_result,
        sweep_result=sweep_result,
    )


def reproduce_mlab_bundle(bundle: object) -> ReproductionOutcome:
    """Explicitly reproduce an already validated :class:`MlabBundle`.

    The import is intentionally avoided at module load time because ``bundle.py`` depends on
    experiment primitives that are also used here.  A structural check keeps accidental
    misuse clear while preserving the one-way bundle -> reproduction execution path.
    """
    try:
        state = bundle.state
        model = bundle.model
        experiment_id = bundle.experiment_id
    except AttributeError as exc:
        raise TypeError("reproduce_mlab_bundle requires a validated MlabBundle.") from exc
    if not isinstance(state, ExperimentState) or not isinstance(model, ModelIR):
        raise TypeError("reproduce_mlab_bundle requires a validated MlabBundle.")
    return reproduce_experiment(state, model=model, experiment_id=str(experiment_id))
