"""Portable ``.mlab`` experiment bundles for Model Laboratory.

Phase A of reproducible mathematical experiments defines a versioned ZIP-based container
that separates the supplied model, canonical validated Model IR, experiment configuration,
provenance, recorded environment, strict fingerprints and numerical reference data.
Loading a bundle validates and reconstructs the experiment state but never runs numerical
or symbolic analyses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Any, Mapping
import uuid
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile, ZipInfo

from .analysis import ParameterSweepResult, StationaryPointAnalysisResult
from .canonical import (
    canonical_json_bytes,
    canonical_json_sha256,
    canonical_model_ir_payload,
    canonical_model_ir_sha256,
    compact_model_ir_document_matches_model,
    hyperbolic_model_ir_document_matches_model,
    legacy_model_ir_document_matches_model,
    previous_model_ir_document_matches_model,
    pretty_json_bytes,
    provenance_payload,
)
from .evaluator import FunctionEvaluation, SurfaceEvaluation
from .experiment import (
    ExperimentState,
    ExperimentStateError,
    fingerprint_evaluation,
    fingerprint_parameter_sweep,
    fingerprint_stationary_result,
    fingerprint_symbolic_analysis,
    validate_experiment_state_for_model,
)
from .model import ModelIR
from .parser import ModelParseError, parse_model_text
from .provenance import Provenance, analysis_ref, model_ref
from .validator import ModelValidationError, validate_legacy_model, validate_model
from .workload import (
    MAX_DIRECT_EVALUATION_SAMPLES,
    MAX_TOTAL_1D_SEARCH_SAMPLES,
    MAX_TOTAL_2D_SOLVER_STARTS,
    estimate_experiment_workload,
)


MLAB_FORMAT_NAME = "Model Laboratory Experiment Bundle"
MLAB_FORMAT_VERSION = "2.0"
_HYPERBOLIC_MLAB_FORMAT_VERSION = "1.6"
_DETERMINISTIC_COMPILER_MLAB_FORMAT_VERSION = "1.5"
_INTERPRETER_LINEAGE_MLAB_FORMAT_VERSION = "1.4"
_AUTHORING_MLAB_FORMAT_VERSION = "1.3"
_COMPACT_DECIMAL_MLAB_FORMAT_VERSION = "1.2"
_EXPRESSION_AST_1_0_MLAB_FORMAT_VERSION = "1.1"
_SUPPORTED_MLAB_FORMAT_VERSIONS = (
    "1.0",
    _EXPRESSION_AST_1_0_MLAB_FORMAT_VERSION,
    _COMPACT_DECIMAL_MLAB_FORMAT_VERSION,
    _AUTHORING_MLAB_FORMAT_VERSION,
    _INTERPRETER_LINEAGE_MLAB_FORMAT_VERSION,
    _DETERMINISTIC_COMPILER_MLAB_FORMAT_VERSION,
    _HYPERBOLIC_MLAB_FORMAT_VERSION,
    MLAB_FORMAT_VERSION,
)
EXPERIMENT_DOCUMENT_SCHEMA = "model-laboratory-experiment"
EXPERIMENT_DOCUMENT_SCHEMA_VERSION = "2.0"
_HYPERBOLIC_EXPERIMENT_DOCUMENT_SCHEMA_VERSION = "1.3"
_DETERMINISTIC_COMPILER_EXPERIMENT_DOCUMENT_SCHEMA_VERSION = "1.2"
_INTERPRETER_EXPERIMENT_DOCUMENT_SCHEMA_VERSION = "1.1"
_PREVIOUS_EXPERIMENT_DOCUMENT_SCHEMA_VERSION = "1.0"
PROVENANCE_DOCUMENT_SCHEMA = "model-laboratory-provenance"
PROVENANCE_DOCUMENT_SCHEMA_VERSION = "2.0"
_HYPERBOLIC_PROVENANCE_DOCUMENT_SCHEMA_VERSION = "1.3"
_DETERMINISTIC_COMPILER_PROVENANCE_DOCUMENT_SCHEMA_VERSION = "1.2"
_INTERPRETER_PROVENANCE_DOCUMENT_SCHEMA_VERSION = "1.1"
_PREVIOUS_PROVENANCE_DOCUMENT_SCHEMA_VERSION = "1.0"
REFERENCE_DOCUMENT_SCHEMA = "model-laboratory-reference-results"
REFERENCE_DOCUMENT_SCHEMA_VERSION = "1.0"
FINGERPRINT_DOCUMENT_SCHEMA = "model-laboratory-result-fingerprints"
FINGERPRINT_DOCUMENT_SCHEMA_VERSION = "1.0"
ENVIRONMENT_DOCUMENT_SCHEMA = "model-laboratory-environment"
ENVIRONMENT_DOCUMENT_SCHEMA_VERSION = "1.0"
AUTHORING_DOCUMENT_SCHEMA = "model-laboratory-authoring-review"
AUTHORING_DOCUMENT_SCHEMA_VERSION = "1.0"
RUN_INDEX_SCHEMA = "model-laboratory-run-index"
ARTIFACT_INDEX_SCHEMA = "model-laboratory-artifact-index"
ASSET_INDEX_SCHEMA = "model-laboratory-asset-index"
VIEW_INDEX_SCHEMA = "model-laboratory-view-index"
INDEX_SCHEMA_VERSION = "1.0"
ARTIFACT_INDEX_SCHEMA_VERSION = "1.1"
ASSET_INDEX_SCHEMA_VERSION = "1.1"

_REQUIRED_MEMBERS = (
    "model.yaml",
    "model_ir.json",
    "experiment.json",
    "provenance.json",
    "environment.json",
    "results/references.json",
    "results/fingerprints.json",
    "metadata/README.txt",
)
_OPTIONAL_MEMBERS = ("metadata/authoring.json",)
_ALLOWED_MEMBERS = ("manifest.json", *_REQUIRED_MEMBERS, *_OPTIONAL_MEMBERS)
_CURRENT_INDEX_MEMBERS = (
    "runs/index.json",
    "artifacts/index.json",
    "assets/index.json",
    "views/index.json",
)
_CURRENT_REQUIRED_MEMBERS = (*_REQUIRED_MEMBERS, *_CURRENT_INDEX_MEMBERS)
_MAX_BUNDLE_BYTES = 128 * 1024 * 1024
_MAX_MEMBER_BYTES = 96 * 1024 * 1024
_MAX_TOTAL_UNCOMPRESSED_BYTES = 192 * 1024 * 1024
_MAX_MANIFEST_BYTES = 512 * 1024
_DEFAULT_CONTENT_CHUNK_BYTES = 1024 * 1024


class MlabBundleError(ValueError):
    """Raised when a portable ``.mlab`` bundle is malformed or internally inconsistent."""


def _experiment_document_version(state_format_version: int) -> str:
    if state_format_version >= 6:
        return EXPERIMENT_DOCUMENT_SCHEMA_VERSION
    if state_format_version >= 5:
        return _HYPERBOLIC_EXPERIMENT_DOCUMENT_SCHEMA_VERSION
    if state_format_version >= 4:
        return _DETERMINISTIC_COMPILER_EXPERIMENT_DOCUMENT_SCHEMA_VERSION
    if state_format_version >= 3:
        return _INTERPRETER_EXPERIMENT_DOCUMENT_SCHEMA_VERSION
    return _PREVIOUS_EXPERIMENT_DOCUMENT_SCHEMA_VERSION


def _provenance_document_version(state_format_version: int) -> str:
    if state_format_version >= 6:
        return PROVENANCE_DOCUMENT_SCHEMA_VERSION
    if state_format_version >= 5:
        return _HYPERBOLIC_PROVENANCE_DOCUMENT_SCHEMA_VERSION
    if state_format_version >= 4:
        return _DETERMINISTIC_COMPILER_PROVENANCE_DOCUMENT_SCHEMA_VERSION
    if state_format_version >= 3:
        return _INTERPRETER_PROVENANCE_DOCUMENT_SCHEMA_VERSION
    return _PREVIOUS_PROVENANCE_DOCUMENT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class MlabBundle:
    """Validated portable experiment bundle loaded without executing the experiment."""

    experiment_id: str
    manifest: Mapping[str, Any]
    experiment_document: Mapping[str, Any]
    provenance_document: Mapping[str, Any]
    reference_document: Mapping[str, Any]
    fingerprint_document: Mapping[str, Any]
    environment_document: Mapping[str, Any]
    authoring_document: Mapping[str, Any] | None
    run_documents: tuple[Mapping[str, Any], ...]
    artifact_documents: tuple[Mapping[str, Any], ...]
    content_descriptors: tuple[Mapping[str, Any], ...]
    asset_documents: tuple[Mapping[str, Any], ...]
    view_documents: tuple[Mapping[str, Any], ...]
    content_members: Mapping[str, bytes]
    asset_members: Mapping[str, bytes]
    model: ModelIR
    state: ExperimentState

    @property
    def author_approved_for_publication(self) -> bool:
        return (
            self.authoring_document is not None
            and self.authoring_document.get("status") == "FROZEN"
        )



def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()



def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))



def _experiment_id(state: ExperimentState) -> str:
    checked = state.with_checksum()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"model-laboratory:{checked.state_sha256}"))



def _iso_utc(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MlabBundleError("Experiment creation time is not valid ISO-8601.") from exc
    if parsed.tzinfo is None:
        raise MlabBundleError("Experiment creation time must include a timezone.")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")



def _analysis_names(state: ExperimentState) -> list[str]:
    return [name for name, _ in state.result_fingerprints]



def canonical_experiment_document(state: ExperimentState, model: ModelIR) -> dict[str, Any]:
    """Return the canonical portable experiment configuration for ``state`` and ``model``."""
    checked = state.with_checksum()
    assumptions = [item.name for item in model.assumptions]
    ambiguity_resolutions = [
        {"name": item.name, "resolution": item.resolution}
        for item in model.ambiguities
        if item.resolution is not None
    ]
    initial_conditions = [
        {"name": item.name, "value": item.initial_value}
        for item in model.variables
        if item.initial_value is not None
    ]
    common = {
        "schema": EXPERIMENT_DOCUMENT_SCHEMA,
        "schema_version": _experiment_document_version(checked.format_version),
        "experiment_id": _experiment_id(checked),
        "experiment_state_format_version": checked.format_version,
        "experiment_state_sha256": checked.state_sha256,
        "laboratory_version": checked.laboratory_version,
        "created_at_utc": _iso_utc(checked.created_at_utc),
        "model": {
            "source_sha256": checked.model_sha256,
            "canonical_ir_sha256": canonical_model_ir_sha256(model),
        },
        "parameter_state": [
            {"name": name, "value": value} for name, value in checked.parameter_values
        ],
        "initial_conditions": initial_conditions,
        "assumptions": assumptions,
        "ambiguity_resolutions": ambiguity_resolutions,
    }
    if checked.format_version >= 6 and checked.artifacts:
        total_workload = sum(int(item["workload_units"]) for item in checked.run_records)
        return {
            **common,
            "reproduction_policy": {
                "relative_tolerance": checked.numerical_reproduction_settings.relative_tolerance,
                "absolute_tolerance": checked.numerical_reproduction_settings.absolute_tolerance,
            },
            "run_protocol": {
                "runs": [item["run_id"] for item in checked.run_records],
                "artifacts": [item["artifact_id"] for item in checked.artifacts],
                "views": [item["view_id"] for item in checked.views],
            },
            "workload": {
                "total_units": total_workload,
                "within_recorded_budget": total_workload <= 10_000_000,
                "maximum_total_units": 10_000_000,
                "runs": [
                    {
                        "run_id": item["run_id"],
                        "capability_id": item["capability_id"],
                        "units": item["workload_units"],
                    }
                    for item in checked.run_records
                ],
            },
            "interpreter_acceptances": list(checked.interpreter_acceptances),
        }

    workload = estimate_experiment_workload(
        model,
        checked.evaluation_settings,
        checked.stationary_settings,
        checked.sweep,
    )
    document = {
        **common,
        "analyses": [
            {
                "name": name,
                "strict_fingerprint_recorded": True,
                "numerical_reference_recorded": name
                in {reference_name for reference_name, _ in checked.numerical_reference_data},
            }
            for name in _analysis_names(checked)
        ],
        "numerical_settings": {
            "evaluation": {
                "points_1d": checked.evaluation_settings.points_1d,
                "points_per_axis_2d": checked.evaluation_settings.points_per_axis_2d,
            },
            "stationary_points": {
                "samples_1d": checked.stationary_settings.samples_1d,
                "seeds_per_axis_2d": checked.stationary_settings.seeds_per_axis_2d,
                "root_tolerance": checked.stationary_settings.root_tolerance,
            },
            "reproduction_tolerances": {
                "relative_tolerance": checked.numerical_reproduction_settings.relative_tolerance,
                "absolute_tolerance": checked.numerical_reproduction_settings.absolute_tolerance,
            },
            "random_seed": None,
        },
        "sweep": None
        if checked.sweep is None
        else {
            "parameter_name": checked.sweep.parameter_name,
            "start": float(checked.sweep.start),
            "end": float(checked.sweep.end),
            "step_count": checked.sweep.step_count,
        },
        "visualisation": {
            "model": checked.selected_model_visualisation,
            "parameter_sweep": (
                None if checked.sweep is None else checked.sweep.selected_visualisation
            ),
        },
        "workload": {
            "evaluation_samples": workload.evaluation_samples,
            "stationary_analysis_count": workload.stationary_analysis_count,
            "one_dimensional_search_samples": workload.one_dimensional_search_samples,
            "two_dimensional_solver_starts": workload.two_dimensional_solver_starts,
            "sweep_steps": workload.sweep_steps,
            "within_recorded_budget": workload.within_budget,
            "budget_limits": {
                "direct_evaluation_samples": MAX_DIRECT_EVALUATION_SAMPLES,
                "total_1d_search_samples": MAX_TOTAL_1D_SEARCH_SAMPLES,
                "total_2d_solver_starts": MAX_TOTAL_2D_SOLVER_STARTS,
            },
        },
    }
    if checked.format_version >= 3:
        document["interpreter_acceptances"] = list(checked.interpreter_acceptances)
    if checked.format_version >= 6:
        document["run_protocol"] = {
            "runs": [item["run_id"] for item in checked.run_records],
            "artifacts": [item["artifact_id"] for item in checked.artifacts],
            "views": [item["view_id"] for item in checked.views],
        }
    return document


def canonical_authoring_review(state: ExperimentState, model: ModelIR) -> dict[str, Any]:
    """Return the complete review an author must approve before publication."""
    checked = state.with_checksum()
    references: list[dict[str, object]] = []
    for name, reference in checked.numerical_reference_data:
        storage_modes: list[str] = []
        arrays = reference.get("arrays")
        if isinstance(arrays, dict):
            storage_modes = sorted(
                {
                    str(item.get("encoding"))
                    for item in arrays.values()
                    if isinstance(item, dict)
                }
            )
        references.append(
            {
                "name": name,
                "reference_sha256": canonical_json_sha256(reference),
                "storage_modes": storage_modes,
            }
        )
    review = {
        "experiment": canonical_experiment_document(checked, model),
        "strict_result_fingerprints": [
            {"name": name, "sha256": digest}
            for name, digest in checked.result_fingerprints
        ],
        "portable_references": references,
        "recorded_environment": [
            {"component": name, "version": version}
            for name, version in checked.environment
        ],
        "author_confirmation": {
            "reference_results_are_frozen": True,
            "tolerances_were_selected_before_reproduction": True,
            "publication_requires_this_exact_review_sha256": True,
        },
    }
    if checked.format_version >= 6 and checked.artifacts:
        review["typed_artifacts"] = [
            {
                "artifact_id": item["artifact_id"],
                "artifact_type": item["artifact_type"],
                "artifact_type_version": item["artifact_type_version"],
                "capability_id": item["capability_id"],
                "scientific_sha256": item["artifact_sha256"],
            }
            for item in checked.artifacts
        ]
        review["run_records"] = [
            {
                "run_id": item["run_id"],
                "capability_id": item["capability_id"],
                "capability_version": item["capability_version"],
                "settings": item["settings"],
                "backend_identity": item.get("backend_identity", {}),
                "artifact_ids": item["artifact_ids"],
            }
            for item in checked.run_records
        ]
    return review



def _provenance_entry(reference: str, provenance: Provenance) -> dict[str, object]:
    return {"reference": reference, **provenance_payload(provenance)}



def _symbolic_provenance_entries(model: ModelIR) -> list[dict[str, object]]:
    if len(model.functions) != 1:
        return []
    function = model.functions[0]
    derived_refs = tuple(
        model_ref("derived_quantity", name)
        for name in model.transitive_derived_dependencies(function.dependencies)
    )
    if len(model.variables) == 1:
        variable = model.variables[0]
        first_ref = analysis_ref("first_derivative", function.name, variable.name)
        return [
            _provenance_entry(
                first_ref,
                Provenance.derived(
                    (model_ref("function", function.name), *derived_refs, model_ref("variable", variable.name)),
                    f"symbolic differentiation of {function.name} with respect to {variable.name}",
                ),
            ),
            _provenance_entry(
                analysis_ref("second_derivative", function.name, variable.name),
                Provenance.derived(
                    (first_ref, model_ref("variable", variable.name)),
                    f"symbolic differentiation of the first derivative with respect to {variable.name}",
                ),
            ),
        ]
    if len(model.variables) == 2:
        first, second = model.variables
        gradient_ref = analysis_ref("gradient", function.name)
        return [
            _provenance_entry(
                gradient_ref,
                Provenance.derived(
                    (
                        model_ref("function", function.name),
                        *derived_refs,
                        model_ref("variable", first.name),
                        model_ref("variable", second.name),
                    ),
                    f"symbolic gradient of {function.name}",
                ),
            ),
            _provenance_entry(
                analysis_ref("hessian", function.name),
                Provenance.derived((gradient_ref,), f"symbolic Hessian of {function.name}"),
            ),
        ]
    return []



def _provenance_document(
    state: ExperimentState,
    model: ModelIR,
    evaluation: FunctionEvaluation | SurfaceEvaluation,
    stationary_result: StationaryPointAnalysisResult | None,
    sweep_result: ParameterSweepResult | None,
) -> dict[str, Any]:
    model_entries = [
        _provenance_entry(reference, provenance)
        for reference, provenance in model.provenance_entries()
    ]
    result_entries: list[dict[str, object]] = [
        _provenance_entry(analysis_ref("evaluation", model.functions[0].name), evaluation.provenance)
    ]
    if "symbolic" in state.fingerprint_map:
        result_entries.extend(_symbolic_provenance_entries(model))
    if stationary_result is not None:
        result_entries.append(
            _provenance_entry(analysis_ref("stationary_points", model.functions[0].name), stationary_result.provenance)
        )
    if sweep_result is not None:
        result_entries.append(
            _provenance_entry(analysis_ref("parameter_sweep", sweep_result.function_name, sweep_result.parameter_name), sweep_result.provenance)
        )
    document = {
        "schema": PROVENANCE_DOCUMENT_SCHEMA,
        "schema_version": _provenance_document_version(state.format_version),
        "model": model_entries,
        "results": result_entries,
    }
    if state.format_version >= 3:
        document["interpreter_acceptances"] = list(state.interpreter_acceptances)
    if state.format_version >= 6:
        document["runs"] = [
            {
                "run_id": item["run_id"],
                "capability_id": item["capability_id"],
                "capability_version": item["capability_version"],
                "artifact_ids": list(item["artifact_ids"]),
            }
            for item in state.run_records
        ]
    return document




def _expected_result_provenance(state: ExperimentState, model: ModelIR) -> list[dict[str, object]]:
    """Reconstruct expected result provenance without executing an analysis."""
    if len(model.functions) != 1:
        return []
    function = model.functions[0]
    derived_refs = [
        model_ref("derived_quantity", name)
        for name in model.transitive_derived_dependencies(function.dependencies)
    ]
    evaluation_refs = [model_ref("function", function.name)]
    evaluation_refs.extend(model_ref("variable", item.name) for item in model.variables)
    evaluation_refs.extend(derived_refs)
    evaluation_refs.extend(model_ref("parameter", item.name) for item in model.parameters)
    evaluation_refs.extend(model_ref("constant", item.name) for item in model.constants)

    stationary_refs = [model_ref("function", function.name), *derived_refs]
    stationary_refs.extend(model_ref("variable", item.name) for item in model.variables)
    stationary_refs.extend(model_ref("parameter", item.name) for item in model.parameters)
    stationary_refs.extend(model_ref("constant", item.name) for item in model.constants)

    if len(model.variables) == 1:
        evaluation_operation = (
            f"numerical evaluation of {function.name} over the declared {model.variables[0].name} domain"
        )
    else:
        evaluation_operation = (
            f"numerical grid evaluation of {function.name} over the declared variable domains"
        )
    entries = [
        _provenance_entry(
            analysis_ref("evaluation", function.name),
            Provenance.derived(tuple(evaluation_refs), evaluation_operation),
        )
    ]
    if "symbolic" in state.fingerprint_map:
        entries.extend(_symbolic_provenance_entries(model))
    if "stationary_points" in state.fingerprint_map:
        entries.append(
            _provenance_entry(
                analysis_ref("stationary_points", function.name),
                Provenance.derived(
                    tuple(stationary_refs),
                    "numerical stationary-point search and local classification",
                ),
            )
        )
    if "parameter_sweep" in state.fingerprint_map:
        if state.sweep is None:
            raise MlabBundleError("Parameter-sweep provenance requires a saved sweep configuration.")
        entries.append(
            _provenance_entry(
                analysis_ref("parameter_sweep", function.name, state.sweep.parameter_name),
                Provenance.derived(
                    (
                        analysis_ref("stationary_points", function.name),
                        model_ref("parameter", state.sweep.parameter_name),
                    ),
                    f"parameter sweep of stationary-point analysis over {state.sweep.parameter_name}",
                ),
            )
        )
    return entries


def canonical_provenance_document(
    state: ExperimentState,
    model: ModelIR,
) -> dict[str, Any]:
    """Return the canonical provenance inventory for a recorded experiment."""
    document = {
        "schema": PROVENANCE_DOCUMENT_SCHEMA,
        "schema_version": _provenance_document_version(state.format_version),
        "model": [
            _provenance_entry(reference, provenance)
            for reference, provenance in model.provenance_entries()
        ],
        "results": _expected_result_provenance(state, model),
    }
    if state.format_version >= 3:
        document["interpreter_acceptances"] = list(state.interpreter_acceptances)
    if state.format_version >= 6:
        document["runs"] = [
            {
                "run_id": item["run_id"],
                "capability_id": item["capability_id"],
                "capability_version": item["capability_version"],
                "artifact_ids": list(item["artifact_ids"]),
            }
            for item in state.run_records
        ]
    return document

def _reference_storage_policy() -> dict[str, object]:
    """Describe the portable reference representation used by current .mlab bundles."""
    return {
        "evaluation": {
            "representation": (
                "adaptive full arrays up to 8 MiB; deterministic canonical samples, "
                "summaries and a complete-array fingerprint above 8 MiB"
            ),
            "array_dtype": "IEEE-754 float64 little-endian",
            "array_encoding": "zlib+base64 or bounded zlib+base64 chunks",
            "full_array_limit_bytes": 8 * 1024 * 1024,
            "chunk_raw_bytes": 1024 * 1024,
            "maximum_canonical_samples": 4096,
            "comparison": "explicit rtol/atol over full arrays or canonical samples and summaries",
        },
        "stationary_points": {
            "representation": "full structured point records",
            "comparison": "semantic point matching plus explicit rtol/atol",
        },
        "parameter_sweep": {
            "representation": "full step-wise structured stationary-point records",
            "comparison": "step alignment plus semantic point matching and explicit rtol/atol",
        },
        "symbolic": {
            "representation": "strict structural fingerprint only",
            "comparison": "exact fingerprint",
        },
    }


def _legacy_reference_storage_policy() -> dict[str, object]:
    """Return the storage declaration written by bundle formats 1.0 and 1.1."""
    return {
        "evaluation": {
            "representation": "full sampled numerical arrays",
            "array_dtype": "IEEE-754 float64 little-endian",
            "array_encoding": "zlib+base64",
            "comparison": "explicit rtol/atol",
        },
        "stationary_points": {
            "representation": "full structured point records",
            "comparison": "semantic point matching plus explicit rtol/atol",
        },
        "parameter_sweep": {
            "representation": "full step-wise structured stationary-point records",
            "comparison": "step alignment plus semantic point matching and explicit rtol/atol",
        },
        "symbolic": {
            "representation": "strict structural fingerprint only",
            "comparison": "exact fingerprint",
        },
    }


def _artifact_reference_storage_policy() -> dict[str, object]:
    """Describe references carried by the content-addressed Artifact protocol."""
    return {
        "typed_artifacts": {
            "representation": "manifest-declared content-addressed typed artifacts",
            "identity": "artifact-schema scientific projection SHA-256",
            "integrity": "independent bundle-member SHA-256",
            "comparison": "artifact-type comparator registry",
            "presentation_fields": "preserved but excluded where the schema marks them presentation-only",
        }
    }



def _reference_document(state: ExperimentState) -> dict[str, Any]:
    return {
        "schema": REFERENCE_DOCUMENT_SCHEMA,
        "schema_version": REFERENCE_DOCUMENT_SCHEMA_VERSION,
        "storage_policy": (
            _artifact_reference_storage_policy()
            if state.artifacts
            else _reference_storage_policy()
        ),
        "references": [
            {"name": name, "reference": reference}
            for name, reference in state.numerical_reference_data
        ],
    }



def _fingerprint_document(state: ExperimentState) -> dict[str, Any]:
    return {
        "schema": FINGERPRINT_DOCUMENT_SCHEMA,
        "schema_version": FINGERPRINT_DOCUMENT_SCHEMA_VERSION,
        "strict": [
            {"name": name, "sha256": digest} for name, digest in state.result_fingerprints
        ],
        "legacy_tolerance_normalised": [
            {"name": name, "sha256": digest}
            for name, digest in state.numerical_result_fingerprints
        ],
    }



def _environment_document(state: ExperimentState) -> dict[str, Any]:
    return {
        "schema": ENVIRONMENT_DOCUMENT_SCHEMA,
        "schema_version": ENVIRONMENT_DOCUMENT_SCHEMA_VERSION,
        "components": [
            {"component": component, "version": version}
            for component, version in state.environment
        ],
    }



def _validate_result_objects(
    state: ExperimentState,
    model: ModelIR,
    evaluation: FunctionEvaluation | SurfaceEvaluation,
    stationary_result: StationaryPointAnalysisResult | None,
    sweep_result: ParameterSweepResult | None,
) -> None:
    expected = state.fingerprint_map
    if expected.get("evaluation") != fingerprint_evaluation(evaluation):
        raise MlabBundleError("Evaluation result does not match the saved experiment state.")
    symbolic = fingerprint_symbolic_analysis(model)
    if "symbolic" in expected and expected["symbolic"] != symbolic:
        raise MlabBundleError("Symbolic result does not match the saved experiment state.")
    if "stationary_points" in expected:
        if stationary_result is None:
            raise MlabBundleError("Saved experiment requires a stationary-point result.")
        if expected["stationary_points"] != fingerprint_stationary_result(stationary_result):
            raise MlabBundleError("Stationary-point result does not match the saved experiment state.")
    elif stationary_result is not None:
        raise MlabBundleError("A stationary-point result was supplied but is absent from the experiment state.")
    if "parameter_sweep" in expected:
        if sweep_result is None:
            raise MlabBundleError("Saved experiment requires a parameter-sweep result.")
        if expected["parameter_sweep"] != fingerprint_parameter_sweep(sweep_result):
            raise MlabBundleError("Parameter-sweep result does not match the saved experiment state.")
    elif sweep_result is not None:
        raise MlabBundleError("A parameter-sweep result was supplied but is absent from the experiment state.")



def _readme_text() -> str:
    return (
        "Model Laboratory .mlab experiment bundle\n"
        f"Format version: {MLAB_FORMAT_VERSION}\n\n"
        "This archive contains a supplied model, canonical validated Model IR, experiment\n"
        "configuration, provenance, environment information, typed runs, content-addressed\n"
        "scientific artifacts, renderer-owned views, fingerprints and portable references.\n"
        "Expressions in model_ir.json use the versioned\n"
        "Model Laboratory expression AST rather than a SymPy-internal representation. SHA-256\n"
        "values in manifest.json are integrity\n"
        "checksums; they do not authenticate the creator of the bundle.\n"
        "Format 2.0 introduces Model IR 3.0 and the generic Run -> Artifact protocol.\n"
        "Format 1.5 records the deterministic interpreter context/compiler identity and\n"
        "edit-program fingerprint for accepted proposals. AI output remains reviewed input\n"
        "to a deterministic compiler, never scientific-result authority.\n"
    )



def _member_semantics(path: str, data: bytes) -> dict[str, Any]:
    media_type = "application/octet-stream"
    schema: str | None = None
    schema_version: str | None = None
    role = "content"
    if path.endswith(".json"):
        media_type = "application/json"
        try:
            document = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            document = None
        if isinstance(document, dict):
            schema = document.get("schema") if isinstance(document.get("schema"), str) else None
            schema_version = (
                document.get("schema_version")
                if isinstance(document.get("schema_version"), str)
                else None
            )
        role = "document"
    elif path.endswith((".yaml", ".yml")):
        media_type = "application/yaml"
        schema = "model-laboratory-model-source"
        schema_version = "3.0"
        role = "model-source"
    elif path.endswith(".txt"):
        media_type = "text/plain; charset=utf-8"
        role = "metadata"
    if path.startswith("artifacts/sha256/"):
        role = "scientific-artifact" if path.endswith(".json") else "artifact-content"
    elif path.startswith("assets/sha256/"):
        role = "model-asset"
    return {
        "media_type": media_type,
        "schema": schema,
        "schema_version": schema_version,
        "role": role,
    }


def _manifest_for_members(
    state: ExperimentState,
    members: Mapping[str, bytes],
    semantic_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    overrides = semantic_overrides or {}
    return {
        "format": MLAB_FORMAT_NAME,
        "format_version": MLAB_FORMAT_VERSION,
        "experiment_id": _experiment_id(state),
        "created_at_utc": _iso_utc(state.created_at_utc),
        "laboratory_version": state.laboratory_version,
        "members": [
            {
                "path": path,
                "sha256": _sha256_bytes(data),
                "size": len(data),
                **dict(overrides.get(path, _member_semantics(path, data))),
            }
            for path, data in sorted(members.items())
        ],
    }


def _chunk_records(blob: bytes, chunk_size: int) -> list[dict[str, Any]]:
    """Return content-addressed, contiguous chunk metadata for ``blob``."""
    return [
        {
            "offset": offset,
            "size": len(blob[offset : offset + chunk_size]),
            "sha256": _sha256_bytes(blob[offset : offset + chunk_size]),
        }
        for offset in range(0, len(blob), chunk_size)
    ]


def _validate_chunk_records(raw: bytes, value: object, *, path: str) -> None:
    """Validate a complete, non-overlapping chunk map without reading beyond ``raw``."""
    if not isinstance(value, list):
        raise MlabBundleError(f"Bundle member '{path}' chunks field must be a list.")
    next_offset = 0
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise MlabBundleError(f"Bundle member '{path}' chunk {index} must be an object.")
        _require_keys(item, {"offset", "size", "sha256"}, path=f"{path}.chunks[{index}]")
        offset = item["offset"]
        size = item["size"]
        digest = item["sha256"]
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or offset != next_offset
            or not 1 <= size <= 8 * 1024 * 1024
            or offset + size > len(raw)
        ):
            raise MlabBundleError(
                f"Bundle member '{path}' has a non-contiguous or invalid chunk map."
            )
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or digest != _sha256_bytes(raw[offset : offset + size])
        ):
            raise MlabBundleError(f"Bundle member '{path}' chunk checksum is invalid.")
        next_offset += size
    if next_offset != len(raw):
        raise MlabBundleError(f"Bundle member '{path}' chunk map does not cover the member exactly.")



def _zip_info(path: str) -> ZipInfo:
    info = ZipInfo(path)
    # Stable timestamp and ordinary-file permissions make repeated bundle serialisation
    # deterministic for an identical experiment state.
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info



def create_mlab_bundle(
    *,
    state: ExperimentState,
    model: ModelIR,
    evaluation: FunctionEvaluation | SurfaceEvaluation,
    stationary_result: StationaryPointAnalysisResult | None,
    sweep_result: ParameterSweepResult | None,
    asset_blobs: Mapping[str, bytes] | None = None,
) -> bytes:
    """Create a deterministic draft/interchange bundle from a completed experiment.

    Author-approved publication bundles are intentionally created only through
    :func:`model_lab.authoring.create_publishable_mlab_bundle`.
    """
    return _create_mlab_bundle(
        state=state,
        model=model,
        evaluation=evaluation,
        stationary_result=stationary_result,
        sweep_result=sweep_result,
        authoring_document=None,
        asset_blobs=asset_blobs,
    )


def create_run_mlab_bundle(
    *,
    state: ExperimentState,
    model: ModelIR,
    content_blobs: Mapping[str, bytes] | None = None,
    content_descriptors: Mapping[str, Mapping[str, Any]] | None = None,
    asset_blobs: Mapping[str, bytes] | None = None,
) -> bytes:
    """Create a format-2.0 bundle from frozen generic runs and typed artifacts."""
    if not state.artifacts or not state.run_records:
        raise MlabBundleError("A Run -> Artifact bundle requires frozen runs and artifacts.")
    return _create_mlab_bundle(
        state=state,
        model=model,
        evaluation=None,
        stationary_result=None,
        sweep_result=None,
        authoring_document=None,
        content_blobs=content_blobs,
        content_descriptors=content_descriptors,
        asset_blobs=asset_blobs,
    )


def _create_mlab_bundle(
    *,
    state: ExperimentState,
    model: ModelIR,
    evaluation: FunctionEvaluation | SurfaceEvaluation | None,
    stationary_result: StationaryPointAnalysisResult | None,
    sweep_result: ParameterSweepResult | None,
    authoring_document: Mapping[str, Any] | None,
    content_blobs: Mapping[str, bytes] | None = None,
    content_descriptors: Mapping[str, Mapping[str, Any]] | None = None,
    asset_blobs: Mapping[str, bytes] | None = None,
) -> bytes:
    """Internal deterministic bundle writer with optional frozen-review metadata."""
    checked = state.with_checksum()
    if checked.format_version < 6:
        raise MlabBundleError(
            f".mlab format {MLAB_FORMAT_VERSION} requires experiment-state format version 6 or later."
        )
    if _sha256_text(checked.model_source) != checked.model_sha256:
        raise MlabBundleError("Experiment model-source checksum is invalid.")
    try:
        reconstructed = validate_model(parse_model_text(checked.model_source))
    except (ModelParseError, ModelValidationError) as exc:
        raise MlabBundleError(f"Experiment model source cannot be validated: {exc}") from exc
    if canonical_model_ir_payload(reconstructed) != canonical_model_ir_payload(model):
        raise MlabBundleError("Experiment model source does not reconstruct the supplied canonical Model IR.")
    try:
        validate_experiment_state_for_model(
            checked,
            model,
            enforce_workload_budget=False,
            reconstructed_embedded_model=reconstructed,
        )
    except ExperimentStateError as exc:
        raise MlabBundleError(str(exc)) from exc
    if checked.artifacts:
        if evaluation is not None or stationary_result is not None or sweep_result is not None:
            raise MlabBundleError("Typed Run -> Artifact bundles do not accept legacy result objects.")
    else:
        if evaluation is None:
            raise MlabBundleError("A legacy scalar experiment requires an evaluation result.")
        _validate_result_objects(checked, model, evaluation, stationary_result, sweep_result)
    if authoring_document is not None:
        _validate_authoring_document(authoring_document, checked, model)

    semantic_overrides: dict[str, dict[str, Any]] = {}
    members: dict[str, bytes] = {
        "model.yaml": checked.model_source.encode("utf-8"),
        "model_ir.json": pretty_json_bytes(canonical_model_ir_payload(model)),
        "experiment.json": pretty_json_bytes(canonical_experiment_document(checked, model)),
        "provenance.json": pretty_json_bytes(
            canonical_provenance_document(checked, model)
            if checked.artifacts
            else _provenance_document(checked, model, evaluation, stationary_result, sweep_result)  # type: ignore[arg-type]
        ),
        "environment.json": pretty_json_bytes(_environment_document(checked)),
        "results/references.json": pretty_json_bytes(_reference_document(checked)),
        "results/fingerprints.json": pretty_json_bytes(_fingerprint_document(checked)),
        "metadata/README.txt": _readme_text().encode("utf-8"),
        "runs/index.json": pretty_json_bytes(
            {
                "schema": RUN_INDEX_SCHEMA,
                "schema_version": INDEX_SCHEMA_VERSION,
                "runs": list(checked.run_records),
            }
        ),
        "views/index.json": pretty_json_bytes(
            {
                "schema": VIEW_INDEX_SCHEMA,
                "schema_version": INDEX_SCHEMA_VERSION,
                "views": list(checked.views),
            }
        ),
    }
    artifact_entries: list[dict[str, Any]] = []
    for artifact in checked.artifacts:
        digest = str(artifact["artifact_sha256"])
        path = f"artifacts/sha256/{digest}.json"
        raw = pretty_json_bytes(artifact)
        members[path] = raw
        artifact_entries.append(
            {
                "artifact_id": artifact["artifact_id"],
                "artifact_type": artifact["artifact_type"],
                "artifact_type_version": artifact["artifact_type_version"],
                "path": path,
                "sha256": _sha256_bytes(raw),
                "size": len(raw),
            }
        )
    content_entries: list[dict[str, Any]] = []
    descriptor_map = dict(content_descriptors or {})
    if set(descriptor_map) != set(content_blobs or {}):
        raise MlabBundleError(
            "Every artifact content blob requires exactly one typed content descriptor."
        )
    known_artifact_ids = {str(item["artifact_id"]) for item in checked.artifacts}
    for digest, blob in sorted((content_blobs or {}).items()):
        if not isinstance(blob, bytes) or digest != _sha256_bytes(blob):
            raise MlabBundleError("Artifact content blobs must be keyed by their SHA-256 digest.")
        descriptor = descriptor_map[digest]
        if not isinstance(descriptor, Mapping):
            raise MlabBundleError("Artifact content descriptors must be objects.")
        required_descriptor_fields = {
            "media_type",
            "schema",
            "schema_version",
            "artifact_ids",
            "provenance",
        }
        if set(descriptor) - required_descriptor_fields - {"chunk_size"} or not required_descriptor_fields.issubset(descriptor):
            raise MlabBundleError("Artifact content descriptor fields are invalid.")
        media_type = descriptor["media_type"]
        schema_name = descriptor["schema"]
        schema_version = descriptor["schema_version"]
        artifact_ids = descriptor["artifact_ids"]
        provenance = descriptor["provenance"]
        chunk_size = descriptor.get("chunk_size", _DEFAULT_CONTENT_CHUNK_BYTES)
        if not isinstance(media_type, str) or "/" not in media_type:
            raise MlabBundleError("Artifact content descriptor media_type must be a MIME type.")
        if schema_name is not None and not isinstance(schema_name, str):
            raise MlabBundleError("Artifact content descriptor schema must be a string or null.")
        if schema_version is not None and not isinstance(schema_version, str):
            raise MlabBundleError("Artifact content descriptor schema_version must be a string or null.")
        if (
            not isinstance(artifact_ids, (list, tuple))
            or not artifact_ids
            or any(not isinstance(item, str) for item in artifact_ids)
            or not set(artifact_ids).issubset(known_artifact_ids)
        ):
            raise MlabBundleError("Artifact content descriptor must reference existing artifact IDs.")
        if not isinstance(provenance, Mapping):
            raise MlabBundleError("Artifact content descriptor provenance must be an object.")
        try:
            portable_provenance = json.loads(canonical_json_bytes(provenance).decode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise MlabBundleError(
                "Artifact content descriptor provenance must contain finite portable JSON data."
            ) from exc
        if not isinstance(portable_provenance, dict):
            raise MlabBundleError("Artifact content descriptor provenance must be an object.")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 1 <= chunk_size <= 8 * 1024 * 1024:
            raise MlabBundleError("Artifact content chunk_size must be between 1 byte and 8 MiB.")
        if len(set(artifact_ids)) != len(artifact_ids):
            raise MlabBundleError("Artifact content descriptor artifact_ids must be unique.")
        path = f"artifacts/sha256/{digest}.bin"
        members[path] = blob
        chunks = _chunk_records(blob, chunk_size)
        semantic_overrides[path] = {
            "media_type": media_type,
            "schema": schema_name,
            "schema_version": schema_version,
            "role": "artifact-content",
        }
        content_entries.append(
            {
                "sha256": digest,
                "path": path,
                "size": len(blob),
                "media_type": media_type,
                "schema": schema_name,
                "schema_version": schema_version,
                "artifact_ids": sorted(set(artifact_ids)),
                "provenance": portable_provenance,
                "chunks": chunks,
            }
        )
    members["artifacts/index.json"] = pretty_json_bytes(
        {
            "schema": ARTIFACT_INDEX_SCHEMA,
            "schema_version": ARTIFACT_INDEX_SCHEMA_VERSION,
            "artifacts": artifact_entries,
            "content": content_entries,
        }
    )
    asset_entries: list[dict[str, Any]] = []
    embedded_asset_digests: set[str] = set()
    supplied_assets = dict(asset_blobs or {})
    for digest, blob in sorted(supplied_assets.items()):
        if not isinstance(blob, bytes) or digest != _sha256_bytes(blob):
            raise MlabBundleError("Asset blobs must be keyed by their SHA-256 digest.")
    for asset in sorted(model.graph.assets, key=lambda item: item.identifier):
        path: str | None = None
        chunks: list[dict[str, Any]] = []
        if asset.sha256 is not None and asset.sha256 in supplied_assets:
            blob = supplied_assets[asset.sha256]
            if asset.size is not None and asset.size != len(blob):
                raise MlabBundleError(
                    f"Embedded asset '{asset.identifier}' does not match its declared byte size."
                )
            path = f"assets/sha256/{asset.sha256}.bin"
            if asset.bundle_path is not None and asset.bundle_path != path:
                raise MlabBundleError(
                    f"Embedded asset '{asset.identifier}' must use content-addressed bundle_path '{path}'."
                )
            members.setdefault(path, blob)
            embedded_asset_digests.add(asset.sha256)
            chunks = _chunk_records(blob, _DEFAULT_CONTENT_CHUNK_BYTES)
            semantic_overrides[path] = {
                "media_type": asset.media_type,
                "schema": asset.kind,
                "schema_version": asset.kind_version,
                "role": "model-asset",
            }
        elif asset.bundle_path is not None:
            raise MlabBundleError(
                f"Asset '{asset.identifier}' declares an embedded bundle_path but its bytes were not supplied."
            )
        asset_entries.append(
            {
                "asset_id": asset.identifier,
                "kind": asset.kind,
                "kind_version": asset.kind_version,
                "media_type": asset.media_type,
                "sha256": asset.sha256,
                "size": asset.size,
                "path": path,
                "metadata": dict(asset.metadata),
                "opaque": asset.opaque,
                "chunks": chunks,
            }
        )
    unreferenced_assets = set(supplied_assets) - embedded_asset_digests
    if unreferenced_assets:
        raise MlabBundleError(
            "Every supplied asset blob must be referenced by a Model Graph asset: "
            + ", ".join(sorted(unreferenced_assets))
            + "."
        )
    members["assets/index.json"] = pretty_json_bytes(
        {
            "schema": ASSET_INDEX_SCHEMA,
            "schema_version": ASSET_INDEX_SCHEMA_VERSION,
            "assets": asset_entries,
        }
    )
    if authoring_document is not None:
        members["metadata/authoring.json"] = pretty_json_bytes(authoring_document)
    manifest = _manifest_for_members(checked, members, semantic_overrides)
    manifest_bytes = pretty_json_bytes(manifest)

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr(_zip_info("manifest.json"), manifest_bytes, compress_type=ZIP_DEFLATED, compresslevel=9)
        for path in sorted(members):
            archive.writestr(_zip_info(path), members[path], compress_type=ZIP_DEFLATED, compresslevel=9)
    data = buffer.getvalue()
    if len(data) > _MAX_BUNDLE_BYTES:
        raise MlabBundleError("The .mlab bundle exceeds the portable bundle size limit.")
    return data



def _json_member(raw: bytes, path: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise MlabBundleError(f"Bundle member '{path}' is not UTF-8.") from exc
    except json.JSONDecodeError as exc:
        raise MlabBundleError(f"Bundle member '{path}' is not valid JSON: {exc.msg}.") from exc
    if not isinstance(value, dict):
        raise MlabBundleError(f"Bundle member '{path}' must contain a JSON object.")
    return value



def _require_keys(value: Mapping[str, Any], required: set[str], *, path: str, optional: set[str] | None = None) -> None:
    optional = optional or set()
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing:
        raise MlabBundleError(f"Bundle member '{path}' is missing field(s): {', '.join(sorted(missing))}.")
    if extra:
        raise MlabBundleError(f"Bundle member '{path}' contains unknown field(s): {', '.join(sorted(extra))}.")



def _validate_zip_structure(data: bytes) -> tuple[ZipFile, BytesIO]:
    if len(data) > _MAX_BUNDLE_BYTES:
        raise MlabBundleError("The .mlab file exceeds the portable bundle size limit.")
    stream = BytesIO(data)
    try:
        archive = ZipFile(stream, "r")
    except BadZipFile as exc:
        raise MlabBundleError("The .mlab file is not a valid ZIP container.") from exc
    infos = archive.infolist()
    if len(infos) > 10000:
        archive.close()
        raise MlabBundleError("The .mlab container contains too many members.")
    names = [item.filename for item in infos]
    if len(names) != len(set(names)):
        archive.close()
        raise MlabBundleError("The .mlab container contains duplicate member names.")
    name_set = set(names)
    if not set(_REQUIRED_MEMBERS).issubset(name_set) or "manifest.json" not in name_set:
        missing = {"manifest.json", *_REQUIRED_MEMBERS} - name_set
        pieces = []
        if missing:
            pieces.append("missing " + ", ".join(sorted(missing)))
        archive.close()
        raise MlabBundleError("Invalid .mlab member set: " + "; ".join(pieces) + ".")
    total = 0
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            archive.close()
            raise MlabBundleError("The .mlab container contains an unsafe member path.")
        if info.flag_bits & 0x1:
            archive.close()
            raise MlabBundleError("Encrypted .mlab members are not supported.")
        if info.compress_type not in (ZIP_STORED, ZIP_DEFLATED):
            archive.close()
            raise MlabBundleError("The .mlab container uses an unsupported compression method.")
        if info.file_size > _MAX_MEMBER_BYTES:
            archive.close()
            raise MlabBundleError(f"Bundle member '{info.filename}' exceeds the safe size limit.")
        total += info.file_size
    if total > _MAX_TOTAL_UNCOMPRESSED_BYTES:
        archive.close()
        raise MlabBundleError("The .mlab container exceeds the safe uncompressed size limit.")
    manifest_info = archive.getinfo("manifest.json")
    if manifest_info.file_size > _MAX_MANIFEST_BYTES:
        archive.close()
        raise MlabBundleError("The .mlab manifest exceeds the safe size limit.")
    return archive, stream



def _manifest_member_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    _require_keys(
        manifest,
        {"format", "format_version", "experiment_id", "created_at_utc", "laboratory_version", "members"},
        path="manifest.json",
    )
    if manifest["format"] != MLAB_FORMAT_NAME or manifest["format_version"] not in _SUPPORTED_MLAB_FORMAT_VERSIONS:
        raise MlabBundleError("Unsupported .mlab format or format version.")
    if not isinstance(manifest["experiment_id"], str) or not manifest["experiment_id"]:
        raise MlabBundleError("The .mlab manifest has an invalid experiment identifier.")
    try:
        uuid.UUID(manifest["experiment_id"])
    except (ValueError, AttributeError) as exc:
        raise MlabBundleError("The .mlab manifest experiment identifier is not a UUID.") from exc
    _iso_utc(str(manifest["created_at_utc"]))
    members_raw = manifest["members"]
    if not isinstance(members_raw, list):
        raise MlabBundleError("The .mlab manifest members field must be a list.")
    member_map: dict[str, Mapping[str, Any]] = {}
    current = manifest["format_version"] == MLAB_FORMAT_VERSION
    for item in members_raw:
        if not isinstance(item, dict):
            raise MlabBundleError("Each .mlab manifest member entry must be an object.")
        if not current and set(item) & {"media_type", "schema", "schema_version", "role"}:
            raise MlabBundleError(
                "The declared .mlab format version does not match the manifest member schema."
            )
        required_member_fields = {"path", "sha256", "size"}
        if current:
            required_member_fields.update(
                {"media_type", "schema", "schema_version", "role"}
            )
        _require_keys(item, required_member_fields, path="manifest.json.members")
        path = item["path"]
        if not isinstance(path, str) or path in member_map:
            raise MlabBundleError("The .mlab manifest contains an invalid or duplicate member path.")
        if current:
            fixed = {*_CURRENT_REQUIRED_MEMBERS, *_OPTIONAL_MEMBERS}
            content_path = re.fullmatch(
                r"(?:artifacts|assets)/sha256/[0-9a-f]{64}\.(?:json|bin)", path
            )
            if path not in fixed and content_path is None:
                raise MlabBundleError(
                    f"The .mlab manifest describes a member outside controlled namespaces: '{path}'."
                )
            if not isinstance(item["media_type"], str) or "/" not in item["media_type"]:
                raise MlabBundleError(f"The .mlab manifest has an invalid media type for '{path}'.")
            if item["schema"] is not None and not isinstance(item["schema"], str):
                raise MlabBundleError(f"The .mlab manifest has an invalid schema for '{path}'.")
            if item["schema_version"] is not None and not isinstance(item["schema_version"], str):
                raise MlabBundleError(f"The .mlab manifest has an invalid schema version for '{path}'.")
            if not isinstance(item["role"], str) or not item["role"]:
                raise MlabBundleError(f"The .mlab manifest has an invalid role for '{path}'.")
        elif path not in (*_REQUIRED_MEMBERS, *_OPTIONAL_MEMBERS):
            raise MlabBundleError(f"The .mlab manifest describes unexpected member '{path}'.")
        if not isinstance(item["size"], int) or item["size"] < 0:
            raise MlabBundleError(f"The .mlab manifest has an invalid size for '{path}'.")
        digest = item["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise MlabBundleError(f"The .mlab manifest has an invalid SHA-256 for '{path}'.")
        member_map[path] = item
    required_members = _CURRENT_REQUIRED_MEMBERS if current else _REQUIRED_MEMBERS
    if not set(required_members).issubset(member_map):
        raise MlabBundleError("The .mlab manifest does not describe every required bundle member.")
    return member_map


def _validate_authoring_document(
    document: Mapping[str, Any], state: ExperimentState, model: ModelIR
) -> None:
    """Validate a frozen author review without trusting presentation-layer state."""
    _require_keys(
        document,
        {
            "schema",
            "schema_version",
            "status",
            "review_sha256",
            "frozen_experiment_state_sha256",
            "model_ir_sha256",
            "review",
        },
        path="metadata/authoring.json",
    )
    if (
        document["schema"] != AUTHORING_DOCUMENT_SCHEMA
        or document["schema_version"] != AUTHORING_DOCUMENT_SCHEMA_VERSION
        or document["status"] != "FROZEN"
    ):
        raise MlabBundleError("Unsupported or non-frozen experiment authoring record.")
    review = document["review"]
    if not isinstance(review, dict):
        raise MlabBundleError("Frozen experiment authoring review is malformed.")
    review_sha256 = document["review_sha256"]
    if review_sha256 != canonical_json_sha256(review):
        raise MlabBundleError("Frozen experiment authoring review checksum is invalid.")
    checked = state.with_checksum()
    if document["frozen_experiment_state_sha256"] != checked.state_sha256:
        raise MlabBundleError("Frozen authoring review does not match the experiment state.")
    if document["model_ir_sha256"] != canonical_model_ir_sha256(model):
        raise MlabBundleError("Frozen authoring review does not match the canonical Model IR.")
    if review != canonical_authoring_review(checked, model):
        raise MlabBundleError(
            "Frozen authoring review is not the canonical review for this experiment."
        )



def _state_from_documents(
    model_source: str,
    experiment: Mapping[str, Any],
    references: Mapping[str, Any],
    fingerprints: Mapping[str, Any],
    environment: Mapping[str, Any],
    run_documents: tuple[Mapping[str, Any], ...] = (),
    artifact_documents: tuple[Mapping[str, Any], ...] = (),
    view_documents: tuple[Mapping[str, Any], ...] = (),
) -> ExperimentState:
    if experiment.get("schema") != EXPERIMENT_DOCUMENT_SCHEMA:
        raise MlabBundleError("Unsupported experiment document schema.")
    schema_version = experiment.get("schema_version")
    typed_protocol = (
        schema_version == EXPERIMENT_DOCUMENT_SCHEMA_VERSION
        and bool(run_documents or artifact_documents or view_documents)
    )
    required_experiment_keys = {
        "schema", "schema_version", "experiment_id", "experiment_state_format_version",
        "experiment_state_sha256", "laboratory_version", "created_at_utc", "model",
        "parameter_state", "initial_conditions", "assumptions", "ambiguity_resolutions",
        "workload",
    }
    if typed_protocol:
        required_experiment_keys.update({"reproduction_policy", "run_protocol"})
    else:
        required_experiment_keys.update(
            {"analyses", "numerical_settings", "sweep", "visualisation"}
        )
    if schema_version in {
        EXPERIMENT_DOCUMENT_SCHEMA_VERSION,
        _HYPERBOLIC_EXPERIMENT_DOCUMENT_SCHEMA_VERSION,
        _DETERMINISTIC_COMPILER_EXPERIMENT_DOCUMENT_SCHEMA_VERSION,
        _INTERPRETER_EXPERIMENT_DOCUMENT_SCHEMA_VERSION,
    }:
        required_experiment_keys.add("interpreter_acceptances")
    elif schema_version != _PREVIOUS_EXPERIMENT_DOCUMENT_SCHEMA_VERSION:
        raise MlabBundleError("Unsupported experiment document schema.")
    if schema_version == EXPERIMENT_DOCUMENT_SCHEMA_VERSION and not typed_protocol:
        required_experiment_keys.add("run_protocol")
    _require_keys(
        experiment,
        required_experiment_keys,
        path="experiment.json",
    )
    state_format_version = int(experiment["experiment_state_format_version"])
    if schema_version != _experiment_document_version(state_format_version):
        raise MlabBundleError(
            "Experiment document schema does not match its experiment-state format version."
        )
    model_info = experiment["model"]
    if not isinstance(model_info, dict):
        raise MlabBundleError("Experiment document contains a malformed model field.")
    _require_keys(model_info, {"source_sha256", "canonical_ir_sha256"}, path="experiment.json.model")
    if state_format_version >= 6:
        run_protocol = experiment.get("run_protocol")
        if not isinstance(run_protocol, dict):
            raise MlabBundleError("Experiment run_protocol must be an object.")
        _require_keys(
            run_protocol,
            {"runs", "artifacts", "views"},
            path="experiment.json.run_protocol",
        )
        if run_protocol["runs"] != [item.get("run_id") for item in run_documents]:
            raise MlabBundleError("Experiment run index does not match experiment.json.")
        if run_protocol["artifacts"] != [item.get("artifact_id") for item in artifact_documents]:
            raise MlabBundleError("Experiment artifact index does not match experiment.json.")
        if run_protocol["views"] != [item.get("view_id") for item in view_documents]:
            raise MlabBundleError("Experiment view index does not match experiment.json.")
    if typed_protocol:
        tolerances = experiment.get("reproduction_policy")
        workload = experiment.get("workload")
        if not isinstance(tolerances, dict) or not isinstance(workload, dict):
            raise MlabBundleError("Typed experiment reproduction/workload policy is malformed.")
        _require_keys(
            tolerances,
            {"relative_tolerance", "absolute_tolerance"},
            path="experiment.json.reproduction_policy",
        )
        _require_keys(
            workload,
            {"total_units", "within_recorded_budget", "maximum_total_units", "runs"},
            path="experiment.json.workload",
        )
        recorded_units = sum(int(item.get("workload_units", 0)) for item in run_documents)
        if workload["total_units"] != recorded_units:
            raise MlabBundleError("Typed experiment workload does not match its run records.")
        evaluation = {"points_1d": 1000, "points_per_axis_2d": 150}
        stationary = {
            "samples_1d": 2001,
            "seeds_per_axis_2d": 11,
            "root_tolerance": 1e-9,
        }
        visualisation = {"model": "", "parameter_sweep": None}
    else:
        numerical = experiment["numerical_settings"]
        visualisation = experiment["visualisation"]
        if not isinstance(numerical, dict) or not isinstance(visualisation, dict):
            raise MlabBundleError("Experiment document contains malformed structured fields.")
        _require_keys(
            numerical,
            {"evaluation", "stationary_points", "reproduction_tolerances", "random_seed"},
            path="experiment.json.numerical_settings",
        )
        _require_keys(visualisation, {"model", "parameter_sweep"}, path="experiment.json.visualisation")
        if numerical["random_seed"] is not None:
            raise MlabBundleError("The current deterministic .mlab experiment system requires random_seed to be null.")
        evaluation = numerical.get("evaluation")
        stationary = numerical.get("stationary_points")
        tolerances = numerical.get("reproduction_tolerances")
        if not isinstance(evaluation, dict) or not isinstance(stationary, dict) or not isinstance(tolerances, dict):
            raise MlabBundleError("Experiment numerical settings are malformed.")
        _require_keys(evaluation, {"points_1d", "points_per_axis_2d"}, path="experiment.json.numerical_settings.evaluation")
        _require_keys(stationary, {"samples_1d", "seeds_per_axis_2d", "root_tolerance"}, path="experiment.json.numerical_settings.stationary_points")
        _require_keys(tolerances, {"relative_tolerance", "absolute_tolerance"}, path="experiment.json.numerical_settings.reproduction_tolerances")

    _require_keys(references, {"schema", "schema_version", "storage_policy", "references"}, path="results/references.json")
    if references["schema"] != REFERENCE_DOCUMENT_SCHEMA or references["schema_version"] != REFERENCE_DOCUMENT_SCHEMA_VERSION:
        raise MlabBundleError("Unsupported reference-result document schema.")
    if references["storage_policy"] not in (
        _reference_storage_policy(),
        _legacy_reference_storage_policy(),
        _artifact_reference_storage_policy(),
    ):
        raise MlabBundleError("The reference-result storage policy is not supported by this Model Laboratory version.")

    _require_keys(fingerprints, {"schema", "schema_version", "strict", "legacy_tolerance_normalised"}, path="results/fingerprints.json")
    if fingerprints["schema"] != FINGERPRINT_DOCUMENT_SCHEMA or fingerprints["schema_version"] != FINGERPRINT_DOCUMENT_SCHEMA_VERSION:
        raise MlabBundleError("Unsupported result-fingerprint document schema.")

    _require_keys(environment, {"schema", "schema_version", "components"}, path="environment.json")
    if environment["schema"] != ENVIRONMENT_DOCUMENT_SCHEMA or environment["schema_version"] != ENVIRONMENT_DOCUMENT_SCHEMA_VERSION:
        raise MlabBundleError("Unsupported environment document schema.")

    sweep_raw = None if typed_protocol else experiment["sweep"]
    sweep_state = None
    if sweep_raw is not None:
        if not isinstance(sweep_raw, dict):
            raise MlabBundleError("Experiment sweep configuration must be an object or null.")
        sweep_visualisation = visualisation.get("parameter_sweep")
        sweep_state = {
            "parameter_name": sweep_raw["parameter_name"],
            "start": sweep_raw["start"],
            "end": sweep_raw["end"],
            "step_count": sweep_raw["step_count"],
            "selected_visualisation": sweep_visualisation,
        }

    raw_state = {
        "format_version": experiment["experiment_state_format_version"],
        "laboratory_version": experiment["laboratory_version"],
        "created_at_utc": experiment["created_at_utc"],
        "model_source": model_source,
        "model_sha256": model_info.get("source_sha256"),
        "parameter_values": experiment["parameter_state"],
        "selected_model_visualisation": visualisation.get("model"),
        "evaluation_settings": {
            "points_1d": evaluation.get("points_1d"),
            "points_per_axis_2d": evaluation.get("points_per_axis_2d"),
        },
        "stationary_settings": {
            "samples_1d": stationary.get("samples_1d"),
            "seeds_per_axis_2d": stationary.get("seeds_per_axis_2d"),
            "root_tolerance": stationary.get("root_tolerance"),
        },
        "sweep": sweep_state,
        "result_fingerprints": fingerprints["strict"],
        "numerical_reproduction_settings": {
            "relative_tolerance": tolerances.get("relative_tolerance"),
            "absolute_tolerance": tolerances.get("absolute_tolerance"),
        },
        "numerical_reference_data": references["references"],
        "numerical_result_fingerprints": fingerprints["legacy_tolerance_normalised"],
        **(
            {"interpreter_acceptances": experiment["interpreter_acceptances"]}
            if state_format_version >= 3
            else {}
        ),
        **(
            {
                "run_records": list(run_documents),
                "artifacts": list(artifact_documents),
                "views": list(view_documents),
            }
            if state_format_version >= 6
            else {}
        ),
        "environment": environment["components"],
        "state_sha256": experiment["experiment_state_sha256"],
    }
    return ExperimentState.from_json(json.dumps(raw_state, ensure_ascii=False, allow_nan=False))



def load_mlab_bundle(data: bytes) -> MlabBundle:
    """Validate and load a portable experiment bundle without reproducing its analyses."""
    archive, stream = _validate_zip_structure(data)
    try:
        manifest_bytes = archive.read("manifest.json")
        manifest = _json_member(manifest_bytes, "manifest.json")
        manifest_members = _manifest_member_map(manifest)
        container_members = set(archive.namelist()) - {"manifest.json"}
        if set(manifest_members) != container_members:
            unexpected = sorted(container_members - set(manifest_members))
            missing = sorted(set(manifest_members) - container_members)
            detail = []
            if unexpected:
                detail.append("unexpected: " + ", ".join(unexpected))
            if missing:
                detail.append("missing: " + ", ".join(missing))
            raise MlabBundleError(
                "The .mlab manifest does not describe the exact container member set"
                + (f" ({'; '.join(detail)})" if detail else "")
                + "."
            )
        member_bytes: dict[str, bytes] = {}
        for path in manifest_members:
            raw = archive.read(path)
            recorded = manifest_members[path]
            if len(raw) != recorded["size"]:
                raise MlabBundleError(f"Bundle member '{path}' size does not match manifest.json.")
            if _sha256_bytes(raw) != recorded["sha256"]:
                raise MlabBundleError(f"Bundle member '{path}' checksum does not match manifest.json.")
            if manifest.get("format_version") == MLAB_FORMAT_VERSION:
                dynamic_binary = re.fullmatch(
                    r"(?:artifacts|assets)/sha256/[0-9a-f]{64}\.bin", path
                ) is not None
                semantics = _member_semantics(path, raw)
                if dynamic_binary:
                    expected_role = (
                        "artifact-content" if path.startswith("artifacts/") else "model-asset"
                    )
                    if recorded.get("role") != expected_role:
                        raise MlabBundleError(
                            f"Bundle member '{path}' has an invalid manifest role."
                        )
                elif any(recorded.get(key) != value for key, value in semantics.items()):
                    raise MlabBundleError(
                        f"Bundle member '{path}' declared type/schema metadata does not match its contents."
                    )
            member_bytes[path] = raw
    finally:
        archive.close()
        stream.close()

    try:
        model_source = member_bytes["model.yaml"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MlabBundleError("Bundle member 'model.yaml' is not UTF-8.") from exc
    bundle_format_version = str(manifest["format_version"])
    try:
        specification = parse_model_text(model_source)
        model = (
            validate_model(specification)
            if bundle_format_version
            in {
                MLAB_FORMAT_VERSION,
                _HYPERBOLIC_MLAB_FORMAT_VERSION,
                _DETERMINISTIC_COMPILER_MLAB_FORMAT_VERSION,
            }
            else validate_legacy_model(specification)
        )
    except (ModelParseError, ModelValidationError) as exc:
        raise MlabBundleError(f"Bundled model cannot be validated: {exc}") from exc

    model_ir_document = _json_member(member_bytes["model_ir.json"], "model_ir.json")
    canonical_model = canonical_model_ir_payload(model)
    if bundle_format_version == MLAB_FORMAT_VERSION:
        if model_ir_document != canonical_model:
            raise MlabBundleError(
                "model_ir.json does not match the canonical Model IR compiled from model.yaml."
            )
        recorded_model_ir_sha256 = canonical_model_ir_sha256(model)
    elif bundle_format_version == _HYPERBOLIC_MLAB_FORMAT_VERSION:
        if not hyperbolic_model_ir_document_matches_model(model_ir_document, model):
            raise MlabBundleError(
                "Historical model_ir.json does not match Model IR 2.2 / expression AST 1.2."
            )
        recorded_model_ir_sha256 = canonical_json_sha256(model_ir_document)
    elif bundle_format_version in (
        _DETERMINISTIC_COMPILER_MLAB_FORMAT_VERSION,
        _INTERPRETER_LINEAGE_MLAB_FORMAT_VERSION,
        _AUTHORING_MLAB_FORMAT_VERSION,
        _COMPACT_DECIMAL_MLAB_FORMAT_VERSION,
    ):
        if not compact_model_ir_document_matches_model(model_ir_document, model):
            raise MlabBundleError(
                "Historical model_ir.json does not match the Model IR compiled under its versioned expression language."
            )
        recorded_model_ir_sha256 = canonical_json_sha256(model_ir_document)
    elif bundle_format_version == _EXPRESSION_AST_1_0_MLAB_FORMAT_VERSION:
        if not previous_model_ir_document_matches_model(model_ir_document, model):
            raise MlabBundleError(
                "Previous model_ir.json does not match the Model Laboratory-owned structure compiled from model.yaml."
            )
        # Format 1.1 used canonical Model IR 2.0 with expression-AST 1.0 real
        # literals.  Validate its historical checksum against the stored document, then
        # migrate the mathematical expressions independently of decimal context.
        recorded_model_ir_sha256 = canonical_json_sha256(model_ir_document)
    elif bundle_format_version == "1.0":
        if not legacy_model_ir_document_matches_model(model_ir_document, model):
            raise MlabBundleError(
                "Legacy model_ir.json does not match the Model Laboratory-owned structure compiled from model.yaml."
            )
        # Version 1.0 hashed the complete legacy document, including SymPy-owned
        # expression printer strings.  Validate that historical checksum against the
        # stored document itself, but do not regenerate those strings with the current
        # SymPy version.
        recorded_model_ir_sha256 = canonical_json_sha256(model_ir_document)
    else:  # defensive; _manifest_member_map has already rejected unsupported versions
        raise MlabBundleError("Unsupported .mlab format version.")

    experiment = _json_member(member_bytes["experiment.json"], "experiment.json")
    provenance = _json_member(member_bytes["provenance.json"], "provenance.json")
    references = _json_member(member_bytes["results/references.json"], "results/references.json")
    fingerprints = _json_member(member_bytes["results/fingerprints.json"], "results/fingerprints.json")
    environment = _json_member(member_bytes["environment.json"], "environment.json")
    authoring = (
        None
        if "metadata/authoring.json" not in member_bytes
        else _json_member(member_bytes["metadata/authoring.json"], "metadata/authoring.json")
    )
    run_documents: tuple[Mapping[str, Any], ...] = ()
    artifact_documents: tuple[Mapping[str, Any], ...] = ()
    content_descriptors: tuple[Mapping[str, Any], ...] = ()
    asset_documents: tuple[Mapping[str, Any], ...] = ()
    view_documents: tuple[Mapping[str, Any], ...] = ()
    content_members: dict[str, bytes] = {}
    asset_members: dict[str, bytes] = {}
    if bundle_format_version == MLAB_FORMAT_VERSION:
        run_index = _json_member(member_bytes["runs/index.json"], "runs/index.json")
        artifact_index = _json_member(
            member_bytes["artifacts/index.json"], "artifacts/index.json"
        )
        asset_index = _json_member(member_bytes["assets/index.json"], "assets/index.json")
        view_index = _json_member(member_bytes["views/index.json"], "views/index.json")
        _require_keys(run_index, {"schema", "schema_version", "runs"}, path="runs/index.json")
        _require_keys(
            artifact_index,
            {"schema", "schema_version", "artifacts", "content"},
            path="artifacts/index.json",
        )
        _require_keys(asset_index, {"schema", "schema_version", "assets"}, path="assets/index.json")
        _require_keys(view_index, {"schema", "schema_version", "views"}, path="views/index.json")
        if (
            run_index["schema"] != RUN_INDEX_SCHEMA
            or run_index["schema_version"] != INDEX_SCHEMA_VERSION
            or artifact_index["schema"] != ARTIFACT_INDEX_SCHEMA
            or artifact_index["schema_version"]
            not in {INDEX_SCHEMA_VERSION, ARTIFACT_INDEX_SCHEMA_VERSION}
            or asset_index["schema"] != ASSET_INDEX_SCHEMA
            or asset_index["schema_version"]
            not in {INDEX_SCHEMA_VERSION, ASSET_INDEX_SCHEMA_VERSION}
            or view_index["schema"] != VIEW_INDEX_SCHEMA
            or view_index["schema_version"] != INDEX_SCHEMA_VERSION
        ):
            raise MlabBundleError("Unsupported Run -> Artifact index schema.")
        if not isinstance(run_index["runs"], list) or not all(
            isinstance(item, dict) for item in run_index["runs"]
        ):
            raise MlabBundleError("Run index entries must be objects.")
        if not isinstance(view_index["views"], list) or not all(
            isinstance(item, dict) for item in view_index["views"]
        ):
            raise MlabBundleError("View index entries must be objects.")
        run_documents = tuple(run_index["runs"])
        view_documents = tuple(view_index["views"])
        artifact_entries = artifact_index["artifacts"]
        content_entries = artifact_index["content"]
        if not isinstance(artifact_entries, list) or not isinstance(content_entries, list):
            raise MlabBundleError("Artifact index entries must be lists.")
        loaded_artifacts: list[Mapping[str, Any]] = []
        indexed_paths: set[str] = set()
        for entry in artifact_entries:
            if not isinstance(entry, dict):
                raise MlabBundleError("Artifact index entry must be an object.")
            _require_keys(
                entry,
                {"artifact_id", "artifact_type", "artifact_type_version", "path", "sha256", "size"},
                path="artifacts/index.json.artifacts",
            )
            path = entry["path"]
            if not isinstance(path, str) or re.fullmatch(
                r"artifacts/sha256/[0-9a-f]{64}\.json", path
            ) is None or path not in member_bytes:
                raise MlabBundleError("Artifact index contains an invalid or missing artifact path.")
            indexed_paths.add(path)
            raw = member_bytes[path]
            if entry["sha256"] != _sha256_bytes(raw) or entry["size"] != len(raw):
                raise MlabBundleError("Artifact index checksum/size does not match its member.")
            document = _json_member(raw, path)
            if (
                document.get("artifact_id") != entry["artifact_id"]
                or document.get("artifact_type") != entry["artifact_type"]
                or document.get("artifact_type_version") != entry["artifact_type_version"]
            ):
                raise MlabBundleError("Artifact index metadata does not match its document.")
            loaded_artifacts.append(document)
        for entry in content_entries:
            if not isinstance(entry, dict):
                raise MlabBundleError("Artifact content index entry must be an object.")
            typed_content = artifact_index["schema_version"] == ARTIFACT_INDEX_SCHEMA_VERSION
            content_fields = {"sha256", "path", "size"}
            if typed_content:
                content_fields.update(
                    {
                        "media_type",
                        "schema",
                        "schema_version",
                        "artifact_ids",
                        "provenance",
                        "chunks",
                    }
                )
            _require_keys(entry, content_fields, path="artifacts/index.json.content")
            path = entry["path"]
            if not isinstance(path, str) or re.fullmatch(
                r"artifacts/sha256/[0-9a-f]{64}\.bin", path
            ) is None or path not in member_bytes:
                raise MlabBundleError("Artifact content index contains an invalid or missing path.")
            raw = member_bytes[path]
            if (
                path != f"artifacts/sha256/{entry['sha256']}.bin"
                or entry["sha256"] != _sha256_bytes(raw)
                or entry["size"] != len(raw)
            ):
                raise MlabBundleError("Artifact content checksum/size does not match its member.")
            if typed_content:
                if not isinstance(entry["media_type"], str) or "/" not in entry["media_type"]:
                    raise MlabBundleError("Artifact content index contains an invalid media type.")
                if entry["schema"] is not None and not isinstance(entry["schema"], str):
                    raise MlabBundleError("Artifact content index contains an invalid schema.")
                if entry["schema_version"] is not None and not isinstance(
                    entry["schema_version"], str
                ):
                    raise MlabBundleError("Artifact content index contains an invalid schema version.")
                artifact_ids = entry["artifact_ids"]
                known_ids = {str(item.get("artifact_id")) for item in loaded_artifacts}
                if (
                    not isinstance(artifact_ids, list)
                    or not artifact_ids
                    or artifact_ids != sorted(set(artifact_ids))
                    or any(not isinstance(item, str) for item in artifact_ids)
                    or not set(artifact_ids).issubset(known_ids)
                ):
                    raise MlabBundleError(
                        "Artifact content index must canonically reference existing artifact IDs."
                    )
                if not isinstance(entry["provenance"], dict):
                    raise MlabBundleError("Artifact content provenance must be an object.")
                try:
                    canonical_json_bytes(entry["provenance"])
                except (TypeError, ValueError) as exc:
                    raise MlabBundleError(
                        "Artifact content provenance is not finite portable JSON data."
                    ) from exc
                _validate_chunk_records(raw, entry["chunks"], path="artifacts/index.json.content")
                declared = manifest_members[path]
                expected_semantics = {
                    "media_type": entry["media_type"],
                    "schema": entry["schema"],
                    "schema_version": entry["schema_version"],
                    "role": "artifact-content",
                }
                if any(declared.get(key) != value for key, value in expected_semantics.items()):
                    raise MlabBundleError(
                        "Artifact content manifest metadata does not match its typed descriptor."
                    )
            indexed_paths.add(path)
            content_members[path] = raw
        content_descriptors = tuple(content_entries)
        dynamic_paths = {
            path for path in member_bytes if path.startswith("artifacts/sha256/")
        }
        if indexed_paths != dynamic_paths:
            raise MlabBundleError("Artifact index does not describe the exact content-addressed member set.")
        artifact_documents = tuple(loaded_artifacts)
        raw_asset_entries = asset_index["assets"]
        if not isinstance(raw_asset_entries, list):
            raise MlabBundleError("Asset index entries must be a list.")
        expected_assets = {item.identifier: item for item in model.graph.assets}
        seen_asset_ids: set[str] = set()
        indexed_asset_paths: set[str] = set()
        typed_assets = asset_index["schema_version"] == ASSET_INDEX_SCHEMA_VERSION
        loaded_assets: list[Mapping[str, Any]] = []
        for entry in raw_asset_entries:
            if not isinstance(entry, dict):
                raise MlabBundleError("Asset index entry must be an object.")
            asset_fields = {
                    "asset_id",
                    "kind",
                    "kind_version",
                    "media_type",
                    "sha256",
                    "size",
                    "path",
                    "metadata",
                    "opaque",
                }
            if typed_assets:
                asset_fields.add("chunks")
            _require_keys(entry, asset_fields, path="assets/index.json.assets")
            asset_id = entry["asset_id"]
            if not isinstance(asset_id, str) or asset_id in seen_asset_ids:
                raise MlabBundleError("Asset index contains an invalid or duplicate asset identifier.")
            expected = expected_assets.get(asset_id)
            if expected is None:
                raise MlabBundleError("Asset index references an asset absent from Model IR.")
            seen_asset_ids.add(asset_id)
            if (
                entry["kind"] != expected.kind
                or entry["kind_version"] != expected.kind_version
                or entry["media_type"] != expected.media_type
                or entry["sha256"] != expected.sha256
                or entry["size"] != expected.size
                or entry["metadata"] != dict(expected.metadata)
                or entry["opaque"] != expected.opaque
            ):
                raise MlabBundleError("Asset index metadata does not match canonical Model IR.")
            path = entry["path"]
            if path is None:
                if expected.bundle_path is not None:
                    raise MlabBundleError("A Model IR embedded asset is missing from the bundle.")
                if typed_assets and entry["chunks"] != []:
                    raise MlabBundleError("An external asset cannot declare embedded chunks.")
                loaded_assets.append(entry)
                continue
            if (
                not isinstance(path, str)
                or re.fullmatch(r"assets/sha256/[0-9a-f]{64}\.bin", path) is None
                or path not in member_bytes
            ):
                raise MlabBundleError("Asset index contains an invalid or missing asset path.")
            if expected.bundle_path is not None and expected.bundle_path != path:
                raise MlabBundleError("Asset index path does not match canonical Model IR.")
            raw = member_bytes[path]
            if (
                path != f"assets/sha256/{entry['sha256']}.bin"
                or entry["sha256"] != _sha256_bytes(raw)
                or entry["size"] != len(raw)
            ):
                raise MlabBundleError("Asset index checksum/size does not match its member.")
            if typed_assets:
                _validate_chunk_records(raw, entry["chunks"], path="assets/index.json.assets")
                declared = manifest_members[path]
                expected_semantics = {
                    "media_type": entry["media_type"],
                    "schema": entry["kind"],
                    "schema_version": entry["kind_version"],
                    "role": "model-asset",
                }
                if any(declared.get(key) != value for key, value in expected_semantics.items()):
                    raise MlabBundleError(
                        "Asset manifest metadata does not match its typed descriptor."
                    )
            indexed_asset_paths.add(path)
            asset_members[path] = raw
            loaded_assets.append(entry)
        if seen_asset_ids != set(expected_assets):
            raise MlabBundleError("Asset index does not describe every Model Graph asset.")
        dynamic_asset_paths = {
            path for path in member_bytes if path.startswith("assets/sha256/")
        }
        if indexed_asset_paths != dynamic_asset_paths:
            raise MlabBundleError("Asset index does not describe the exact content-addressed asset set.")
        asset_documents = tuple(loaded_assets)
    if authoring is not None and bundle_format_version not in (
        MLAB_FORMAT_VERSION,
        _HYPERBOLIC_MLAB_FORMAT_VERSION,
        _DETERMINISTIC_COMPILER_MLAB_FORMAT_VERSION,
        _INTERPRETER_LINEAGE_MLAB_FORMAT_VERSION,
        _AUTHORING_MLAB_FORMAT_VERSION,
    ):
        raise MlabBundleError(
            "Frozen authoring metadata is supported only by .mlab format 1.3 or later."
        )

    if provenance.get("schema") != PROVENANCE_DOCUMENT_SCHEMA:
        raise MlabBundleError("Unsupported provenance document schema.")
    provenance_version = provenance.get("schema_version")
    provenance_keys = {"schema", "schema_version", "model", "results"}
    if provenance_version in {
        PROVENANCE_DOCUMENT_SCHEMA_VERSION,
        _HYPERBOLIC_PROVENANCE_DOCUMENT_SCHEMA_VERSION,
        _DETERMINISTIC_COMPILER_PROVENANCE_DOCUMENT_SCHEMA_VERSION,
        _INTERPRETER_PROVENANCE_DOCUMENT_SCHEMA_VERSION,
    }:
        provenance_keys.add("interpreter_acceptances")
    elif provenance_version != _PREVIOUS_PROVENANCE_DOCUMENT_SCHEMA_VERSION:
        raise MlabBundleError("Unsupported provenance document schema.")
    if provenance_version == PROVENANCE_DOCUMENT_SCHEMA_VERSION:
        provenance_keys.add("runs")
    _require_keys(provenance, provenance_keys, path="provenance.json")
    if not isinstance(provenance["model"], list) or not isinstance(provenance["results"], list):
        raise MlabBundleError("Provenance document entries must be lists.")

    try:
        state = _state_from_documents(
            model_source,
            experiment,
            references,
            fingerprints,
            environment,
            run_documents,
            artifact_documents,
            view_documents,
        )
    except (ExperimentStateError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, MlabBundleError):
            raise
        raise MlabBundleError(f"Portable experiment state is malformed: {exc}") from exc
    if provenance_version != _provenance_document_version(state.format_version):
        raise MlabBundleError(
            "Provenance document schema does not match the experiment-state format version."
        )
    expected_interpreter_bundle = (
        MLAB_FORMAT_VERSION
        if state.format_version >= 6
        else (
            _HYPERBOLIC_MLAB_FORMAT_VERSION
            if state.format_version >= 5
            else (
                _DETERMINISTIC_COMPILER_MLAB_FORMAT_VERSION
                if state.format_version >= 4
                else (
                    _INTERPRETER_LINEAGE_MLAB_FORMAT_VERSION
                    if state.format_version >= 3
                    else None
                )
            )
        )
    )
    if expected_interpreter_bundle is not None and bundle_format_version != expected_interpreter_bundle:
        raise MlabBundleError(
            ".mlab format version does not match the embedded experiment-state generation."
        )
    if expected_interpreter_bundle is None and bundle_format_version in {
        MLAB_FORMAT_VERSION,
        _HYPERBOLIC_MLAB_FORMAT_VERSION,
        _DETERMINISTIC_COMPILER_MLAB_FORMAT_VERSION,
        _INTERPRETER_LINEAGE_MLAB_FORMAT_VERSION,
    }:
        raise MlabBundleError(
            ".mlab format version does not match the embedded experiment-state generation."
        )
    if state.model_sha256 != _sha256_text(model_source):
        raise MlabBundleError("Bundled model source SHA-256 does not match experiment.json.")
    if experiment.get("model", {}).get("canonical_ir_sha256") != recorded_model_ir_sha256:
        raise MlabBundleError("Canonical Model IR SHA-256 does not match experiment.json.")
    try:
        validate_experiment_state_for_model(
            state,
            model,
            enforce_workload_budget=False,
            reconstructed_embedded_model=model,
        )
    except ExperimentStateError as exc:
        raise MlabBundleError(str(exc)) from exc
    if authoring is not None:
        _validate_authoring_document(authoring, state, model)

    expected_provenance = canonical_provenance_document(state, model)
    if provenance != expected_provenance:
        raise MlabBundleError("provenance.json is not consistent with the reconstructed model and experiment configuration.")

    expected_experiment = canonical_experiment_document(state, model)
    if bundle_format_version != MLAB_FORMAT_VERSION:
        # Preserve the historical Model IR checksum when validating an older bundle.
        # Every other experiment field is checked normally.
        expected_experiment["model"]["canonical_ir_sha256"] = recorded_model_ir_sha256
    if experiment != expected_experiment:
        raise MlabBundleError(
            "experiment.json is not the canonical representation of the reconstructed experiment state."
        )
    if manifest["experiment_id"] != experiment["experiment_id"]:
        raise MlabBundleError("Experiment identifier differs between manifest.json and experiment.json.")
    if manifest["laboratory_version"] != state.laboratory_version:
        raise MlabBundleError("Laboratory version differs between manifest.json and experiment.json.")
    if _iso_utc(str(manifest["created_at_utc"])) != _iso_utc(state.created_at_utc):
        raise MlabBundleError("Creation time differs between manifest.json and experiment.json.")

    return MlabBundle(
        experiment_id=str(experiment["experiment_id"]),
        manifest=manifest,
        experiment_document=experiment,
        provenance_document=provenance,
        reference_document=references,
        fingerprint_document=fingerprints,
        environment_document=environment,
        authoring_document=authoring,
        run_documents=run_documents,
        artifact_documents=artifact_documents,
        content_descriptors=content_descriptors,
        asset_documents=asset_documents,
        view_documents=view_documents,
        content_members=content_members,
        asset_members=asset_members,
        model=model,
        state=state,
    )
