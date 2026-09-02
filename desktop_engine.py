"""JSON process boundary used by the Model Laboratory Tauri desktop application.

The desktop shell never imports the scientific package.  Instead it keeps this executable
alive and exchanges bounded newline-delimited JSON request/response pairs with it.  This
keeps numerical work outside the interface process, pays Python and scientific-library
startup once, and makes the engine independently testable.

Opening an experiment is deliberately separate from reproducing it.  The
``inspect_experiment`` action validates and reconstructs saved state but performs no
numerical analysis.  Only ``reproduce_experiment`` crosses the execution boundary.
"""

from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, cast

import sympy as sp

from model_lab import __version__
from model_lab.analysis import (
    AnalysisError,
    ParameterSweepResult,
    StationaryPointAnalysisResult,
)
from model_lab.authoring import (
    ExperimentAuthoringError,
    create_publishable_mlab_bundle,
    create_publishable_run_mlab_bundle,
    freeze_experiment,
    prepare_experiment_review,
)
from model_lab.bundle import (
    MlabBundle,
    MlabBundleError,
    create_mlab_bundle,
    create_run_mlab_bundle,
    load_mlab_bundle,
)
from model_lab.builtin_packs import run_registry
from model_lab.commit import (
    CommittedExperiment,
    CommittedExperimentError,
    commit_experiment_state,
)
from model_lab.capabilities import (
    AnalysisCapability,
    AnalysisVisualisation,
    ModelVisualisation,
    detect_capabilities,
)
from model_lab.evaluator import (
    EvaluationError,
    FunctionEvaluation,
    SurfaceEvaluation,
    evaluate_single_variable_function,
    evaluate_two_variable_function,
)
from model_lab.experiment import (
    EvaluationSettings,
    ExperimentState,
    ExperimentStateError,
    NumericalReproductionSettings,
    StationaryPointSettings,
    SweepConfiguration,
    create_experiment_state,
    create_run_experiment_state,
    current_environment,
    validate_experiment_state_for_model,
)
from model_lab.issues import NumericalDiagnostic, NumericalStatistics
from model_lab.interpreter import (
    MAX_CONTEXT_ROUNDS,
    MAX_CONTEXT_PACKAGE_BYTES,
    InterpreterProposalError,
    accept_interpreter_proposal,
    create_interpreter_context,
    generation_options_for_request,
    process_interpreter_output,
    proposal_diff,
    validate_interpreter_proposal_document,
)
from model_lab.interpreter_attachments import (
    InterpreterAttachmentError,
    InterpreterAttachmentStore,
    ingest_attachment,
)
from model_lab.model import ModelIR
from model_lab.official_packs import official_pack_catalogue
from model_lab.official_packs.renderers import create_official_pack_figure
from model_lab.parser import ModelParseError, parse_model_text
from model_lab.presentation import format_number
from model_lab.provenance import Provenance, analysis_ref
from model_lab.registry import CapabilityRegistryError, capability_registry
from model_lab.protocol import ArtifactView, ProtocolError, portable_value
from model_lab.reproduction import (
    ReproductionOutcome,
    reproduce_experiment,
    reproduce_mlab_bundle,
)
from model_lab.symbolic import (
    OneVariableSymbolicAnalysis,
    SymbolicAnalysisError,
    TwoVariableSymbolicAnalysis,
)
from model_lab.validator import ModelValidationError, validate_model
from model_lab.visualisation import (
    VisualisationError,
    render_analysis_visualisation,
    render_model_visualisation,
    create_vector_field_figure,
)
from model_lab.vector_analysis import VectorFieldEvaluation
from model_lab.workload import WorkloadEstimate, estimate_experiment_workload


ROOT = Path(__file__).resolve().parent
EXAMPLE_MODEL = ROOT / "models" / "quadratic.yaml"
MAX_REQUEST_BYTES = 192 * 1024 * 1024
MAX_CACHE_BYTES = 32 * 1024 * 1024
MAX_CACHE_ITEM_BYTES = 8 * 1024 * 1024
MAX_CACHE_ENTRIES = 32
MAX_CONTENT_STORE_BYTES = 192 * 1024 * 1024
MAX_CONTENT_STORES = 3
MAX_CONTENT_CHUNK_BYTES = 1024 * 1024


class _ResponseCache:
    """A byte-bounded LRU for deterministic desktop responses.

    Entry and total-byte limits prevent a large surface request from turning caching into
    memory amplification.  Values are stored as JSON text so callers cannot mutate a
    cached response object.
    """

    def __init__(self) -> None:
        self._values: OrderedDict[tuple[str, str], tuple[str, int]] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, action: str, key: str) -> dict[str, Any] | None:
        stored = self._values.pop((action, key), None)
        if stored is None:
            self.misses += 1
            return None
        self._values[(action, key)] = stored
        self.hits += 1
        return cast(dict[str, Any], json.loads(stored[0]))

    def put(self, action: str, key: str, value: Mapping[str, Any]) -> None:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        size = len(text.encode("utf-8"))
        if size > MAX_CACHE_ITEM_BYTES:
            return
        previous = self._values.pop((action, key), None)
        if previous is not None:
            self._bytes -= previous[1]
        self._values[(action, key)] = (text, size)
        self._bytes += size
        while self._bytes > MAX_CACHE_BYTES or len(self._values) > MAX_CACHE_ENTRIES:
            _, (_, removed_size) = self._values.popitem(last=False)
            self._bytes -= removed_size

    def clear(self) -> None:
        self._values.clear()
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    def status(self) -> dict[str, int | str]:
        return {
            "mode": "persistent-byte-bounded-lru",
            "entries": len(self._values),
            "bytes": self._bytes,
            "maximum_bytes": MAX_CACHE_BYTES,
            "maximum_item_bytes": MAX_CACHE_ITEM_BYTES,
        }


_RESPONSE_CACHE = _ResponseCache()
_CACHEABLE_ACTIONS = frozenset({"inspect_model", "analyse_model", "run_sweep"})
_INTERPRETER_ATTACHMENTS = InterpreterAttachmentStore()


class _OpenedContentStore:
    """Byte-bounded handles for content-addressed bundle members.

    Large scientific arrays and assets stay as bytes in the persistent engine.  The JSON
    bridge carries only metadata and explicitly requested, bounded chunks.
    """

    def __init__(self) -> None:
        self._values: OrderedDict[str, tuple[dict[str, bytes], int]] = OrderedDict()
        self._bytes = 0

    def put(self, bundle: MlabBundle) -> str | None:
        members = {**bundle.content_members, **bundle.asset_members}
        if not members:
            return None
        size = sum(len(value) for value in members.values())
        if size > MAX_CONTENT_STORE_BYTES:
            raise DesktopEngineError(
                "Opened artifact content exceeds the persistent content-store safety limit."
            )
        identity = json.dumps(
            bundle.manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        handle = hashlib.sha256(identity).hexdigest()
        previous = self._values.pop(handle, None)
        if previous is not None:
            self._bytes -= previous[1]
        self._values[handle] = (members, size)
        self._bytes += size
        while self._bytes > MAX_CONTENT_STORE_BYTES or len(self._values) > MAX_CONTENT_STORES:
            _, (_, removed_size) = self._values.popitem(last=False)
            self._bytes -= removed_size
        return handle

    def read(self, handle: str, path: str, offset: int, length: int) -> tuple[bytes, int]:
        stored = self._values.pop(handle, None)
        if stored is None:
            raise DesktopEngineError("The opened-content handle is unknown or has expired.")
        self._values[handle] = stored
        raw = stored[0].get(path)
        if raw is None:
            raise DesktopEngineError("The requested content path does not belong to this handle.")
        if offset < 0 or offset > len(raw):
            raise DesktopEngineError("Content offset is outside the member.")
        return raw[offset : offset + length], len(raw)

    def status(self) -> dict[str, int | str]:
        return {
            "mode": "content-addressed-bounded-chunks",
            "stores": len(self._values),
            "bytes": self._bytes,
            "maximum_bytes": MAX_CONTENT_STORE_BYTES,
            "maximum_chunk_bytes": MAX_CONTENT_CHUNK_BYTES,
        }


_OPENED_CONTENT = _OpenedContentStore()


class DesktopEngineError(ValueError):
    """A safe, user-facing desktop service error."""


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@lru_cache(maxsize=64)
def _compile_model(source: str) -> ModelIR:
    if not isinstance(source, str) or not source.strip():
        raise DesktopEngineError("The model source is empty.")
    try:
        return validate_model(parse_model_text(source))
    except (ModelParseError, ModelValidationError) as exc:
        raise DesktopEngineError(str(exc)) from exc


def _evaluation_settings(payload: Mapping[str, Any]) -> EvaluationSettings:
    raw = payload.get("evaluation_settings") or {}
    if not isinstance(raw, Mapping):
        raise DesktopEngineError("evaluation_settings must be an object.")
    try:
        return EvaluationSettings(
            points_1d=int(raw.get("points_1d", EvaluationSettings().points_1d)),
            points_per_axis_2d=int(
                raw.get(
                    "points_per_axis_2d",
                    EvaluationSettings().points_per_axis_2d,
                )
            ),
        )
    except (TypeError, ValueError, ExperimentStateError) as exc:
        raise DesktopEngineError(str(exc)) from exc


def _stationary_settings(payload: Mapping[str, Any]) -> StationaryPointSettings:
    raw = payload.get("stationary_settings") or {}
    if not isinstance(raw, Mapping):
        raise DesktopEngineError("stationary_settings must be an object.")
    defaults = StationaryPointSettings()
    try:
        return StationaryPointSettings(
            samples_1d=int(raw.get("samples_1d", defaults.samples_1d)),
            seeds_per_axis_2d=int(
                raw.get("seeds_per_axis_2d", defaults.seeds_per_axis_2d)
            ),
            root_tolerance=float(raw.get("root_tolerance", defaults.root_tolerance)),
        )
    except (TypeError, ValueError, ExperimentStateError) as exc:
        raise DesktopEngineError(str(exc)) from exc


def _reproduction_settings(payload: Mapping[str, Any]) -> NumericalReproductionSettings:
    raw = payload.get("reproduction_settings") or {}
    if not isinstance(raw, Mapping):
        raise DesktopEngineError("reproduction_settings must be an object.")
    defaults = NumericalReproductionSettings()
    try:
        return NumericalReproductionSettings(
            relative_tolerance=float(
                raw.get("relative_tolerance", defaults.relative_tolerance)
            ),
            absolute_tolerance=float(
                raw.get("absolute_tolerance", defaults.absolute_tolerance)
            ),
        )
    except (TypeError, ValueError, ExperimentStateError) as exc:
        raise DesktopEngineError(str(exc)) from exc


def _parameter_values(model: ModelIR, payload: Mapping[str, Any]) -> dict[str, float]:
    raw = payload.get("parameter_values") or {}
    if not isinstance(raw, Mapping):
        raise DesktopEngineError("parameter_values must be an object.")
    unknown = set(raw) - {item.name for item in model.parameters}
    if unknown:
        raise DesktopEngineError("Unknown parameter(s): " + ", ".join(sorted(unknown)) + ".")
    values = model.parameter_defaults()
    try:
        values.update({str(name): float(value) for name, value in raw.items()})
    except (TypeError, ValueError, OverflowError) as exc:
        raise DesktopEngineError("Parameter values must be finite real numbers.") from exc
    for parameter in model.parameters:
        value = values[parameter.name]
        if not math.isfinite(value) or not parameter.domain.contains(value):
            raise DesktopEngineError(
                f"Parameter '{parameter.name}' must lie inside "
                f"[{parameter.domain.lower}, {parameter.domain.upper}]."
            )
    return values


def _diagnostic(item: NumericalDiagnostic) -> dict[str, Any]:
    return {
        "code": item.code,
        "severity": item.severity.value,
        "message": item.message,
        "details": [{"key": key, "value": value} for key, value in item.details],
        "details_text": ", ".join(f"{key}={value}" for key, value in item.details) or "—",
    }


def _statistics(statistics: NumericalStatistics) -> list[dict[str, Any]]:
    return [
        {"measure": name, "value": _finite(value), "display": format_number(value)}
        for name, value in statistics.values
    ]


def _provenance(reference: str, provenance: Provenance) -> dict[str, str]:
    if provenance.source_location:
        source = provenance.source_location
    elif provenance.source_refs:
        source = ", ".join(provenance.source_refs)
    else:
        source = "—"
    return {
        "reference": reference,
        "origin": provenance.kind.value,
        "source": source,
        "operation": provenance.operation or "—",
        "approval": provenance.approval.value,
        "note": provenance.note or "—",
    }


def _workload(value: WorkloadEstimate) -> dict[str, Any]:
    return {
        "within_budget": value.within_budget,
        "rows": [
            {"component": name, "value": display}
            for name, display in value.summary_rows()
        ],
        "violations": list(value.violations),
    }


def _model_document(source: str, model: ModelIR) -> dict[str, Any]:
    report = detect_capabilities(model)
    variables = [
        {
            "name": item.name,
            "label": item.metadata.label or "",
            "description": item.metadata.description or "",
            "lower": float(item.domain.lower),
            "upper": float(item.domain.upper),
            "domain": f"[{format_number(item.domain.lower)}, {format_number(item.domain.upper)}]",
            "initial": None if item.initial_value is None else float(item.initial_value),
            "initial_display": "—" if item.initial_value is None else format_number(item.initial_value),
            "unit": item.metadata.unit or "",
            "tags": list(item.metadata.tags),
        }
        for item in model.variables
    ]
    parameters = [
        {
            "name": item.name,
            "label": item.metadata.label or item.name,
            "description": item.metadata.description or "",
            "default": float(item.default),
            "default_display": format_number(item.default),
            "lower": float(item.domain.lower),
            "upper": float(item.domain.upper),
            "domain": f"[{format_number(item.domain.lower)}, {format_number(item.domain.upper)}]",
            "unit": item.metadata.unit or "",
            "tags": list(item.metadata.tags),
        }
        for item in model.parameters
    ]
    constants = [
        {
            "name": item.name,
            "label": item.metadata.label or "",
            "value": float(item.value),
            "value_display": format_number(item.value),
            "unit": item.metadata.unit or "",
        }
        for item in model.constants
    ]
    derived = [
        {
            "name": item.name,
            "definition": item.source,
            "dependencies": list(item.dependencies),
        }
        for item in model.derived_quantities
    ]
    functions = [
        {
            "name": item.name,
            "definition": item.source,
            "dependencies": list(item.dependencies),
        }
        for item in model.functions
    ]
    vector_functions = [
        {
            "name": item.name,
            "components": [
                {
                    "name": component.name,
                    "definition": component.source,
                    "dependencies": list(component.dependencies),
                }
                for component in item.components
            ],
            "output_dimension": item.output_dimension,
        }
        for item in model.vector_functions
    ]
    matrix_functions = [
        {
            "name": item.name,
            "shape": list(item.shape),
            "entries": [[component.source for component in row] for row in item.entries],
            "row_labels": list(item.row_labels),
            "column_labels": list(item.column_labels),
        }
        for item in model.matrix_functions
    ]
    constraints = [
        {
            "name": item.name,
            "relation": f"{item.left_source} {item.relation.value} {item.right_source}",
            "dependencies": list(item.dependencies),
        }
        for item in model.constraints
    ]
    assumptions = [
        {
            "name": item.name,
            "statement": item.statement,
            "affects": list(item.affects),
        }
        for item in model.assumptions
    ]
    ambiguities = [
        {
            "name": item.name,
            "statement": item.statement,
            "status": item.status.value,
            "blocking": item.blocking,
            "options": list(item.options),
            "resolution": item.resolution,
            "affects": list(item.affects),
        }
        for item in model.ambiguities
    ]
    return {
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "name": model.name,
        "description": model.metadata.description or "",
        "notes": model.metadata.notes or "",
        "tags": list(model.metadata.tags),
        "counts": {
            "variables": len(variables),
            "functions": len(functions),
            "vector_functions": len(vector_functions),
            "matrix_functions": len(matrix_functions),
            "parameters": len(parameters),
            "constants": len(constants),
            "derived_quantities": len(derived),
            "constraints": len(constraints),
            "assumptions": len(assumptions),
            "ambiguities": len(ambiguities),
            "graph_objects": len(model.graph.objects),
            "graph_relationships": len(model.graph.relationships),
            "assets": len(model.graph.assets),
        },
        "structure": {
            "variables": variables,
            "parameters": parameters,
            "constants": constants,
            "derived_quantities": derived,
            "functions": functions,
            "vector_functions": vector_functions,
            "matrix_functions": matrix_functions,
            "constraints": constraints,
            "assumptions": assumptions,
            "ambiguities": ambiguities,
            "model_graph": model.graph.payload(),
            "unavailable_extension_kinds": [
                {"kind": kind, "version": version}
                for kind, version in model.graph.unavailable_extension_kinds
            ],
        },
        "blocking_ambiguities": [item.name for item in model.blocking_ambiguities],
        "capabilities": {
            "visualisations": [item.value for item in report.visualisations],
            "analyses": [item.value for item in report.analyses],
            "controls": [item.value for item in report.controls],
            "installed_packs": run_registry.catalogue(model),
            "official_packs": official_pack_catalogue(model),
        },
        "defaults": {
            "selected_visualisation": (
                report.visualisations[0].value if report.visualisations else None
            ),
            "evaluation_settings": asdict(EvaluationSettings()),
            "stationary_settings": asdict(StationaryPointSettings()),
            "reproduction_settings": asdict(NumericalReproductionSettings()),
        },
        "provenance": [
            _provenance(reference, provenance)
            for reference, provenance in model.provenance_entries()
        ],
    }


def _mathml(expression: sp.Expr) -> str:
    body = sp.printing.mathml(expression, printer="presentation")
    return f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">{body}</math>'


def _symbolic_document(model: ModelIR) -> tuple[dict[str, Any], list[dict[str, str]]]:
    report = detect_capabilities(model)
    entries: list[dict[str, str]] = []
    provenance_rows: list[dict[str, str]] = []
    if len(model.variables) == 1 and AnalysisCapability.FIRST_DERIVATIVE in report.analyses:
        result = cast(
            OneVariableSymbolicAnalysis,
            capability_registry.run_analysis(AnalysisCapability.FIRST_DERIVATIVE, model),
        )
        function = model.functions[0]
        variable = model.variables[0]
        expressions = (
            ("Supplied function", sp.Eq(sp.Symbol(function.name), function.expression)),
            (
                "First derivative",
                sp.Eq(
                    sp.Symbol(f"d{function.name}/d{variable.name}"),
                    result.first_derivative,
                ),
            ),
            (
                "Second derivative",
                sp.Eq(
                    sp.Symbol(f"d2{function.name}/d{variable.name}2"),
                    result.second_derivative,
                ),
            ),
        )
        entries.extend(
            {"label": label, "latex": sp.latex(expr), "mathml": _mathml(expr)}
            for label, expr in expressions
        )
        provenance_rows.extend(
            [
                _provenance(
                    analysis_ref("first_derivative", function.name, variable.name),
                    result.first_derivative_provenance,
                ),
                _provenance(
                    analysis_ref("second_derivative", function.name, variable.name),
                    result.second_derivative_provenance,
                ),
            ]
        )
    elif len(model.variables) == 2 and AnalysisCapability.GRADIENT in report.analyses:
        result = cast(
            TwoVariableSymbolicAnalysis,
            capability_registry.run_analysis(AnalysisCapability.GRADIENT, model),
        )
        function = model.functions[0]
        supplied = sp.Eq(sp.Symbol(function.name), function.expression)
        entries.extend(
            [
                {
                    "label": "Supplied function",
                    "latex": sp.latex(supplied),
                    "mathml": _mathml(supplied),
                },
                {
                    "label": "Gradient",
                    "latex": r"\nabla " + function.name + " = " + sp.latex(sp.Matrix(result.gradient)),
                    "mathml": _mathml(sp.Matrix(result.gradient)),
                },
                {
                    "label": "Hessian",
                    "latex": "H_{" + function.name + "} = " + sp.latex(result.hessian),
                    "mathml": _mathml(result.hessian),
                },
            ]
        )
        provenance_rows.extend(
            [
                _provenance(
                    analysis_ref("gradient", function.name),
                    result.gradient_provenance,
                ),
                _provenance(
                    analysis_ref("hessian", function.name),
                    result.hessian_provenance,
                ),
            ]
        )
    return {"entries": entries}, provenance_rows


def _stationary_document(
    model: ModelIR,
    result: StationaryPointAnalysisResult | None,
    error: str | None = None,
) -> dict[str, Any]:
    if result is None:
        return {
            "available": False,
            "status": "unavailable",
            "error": error,
            "diagnostics": [],
            "statistics": [],
            "columns": [],
            "rows": [],
        }
    rows: list[dict[str, Any]] = []
    if len(model.variables) == 1:
        variable_name = model.variables[0].name
        function_name = model.functions[0].name
        columns = [variable_name, function_name, "second derivative", "classification"]
        for point in result.points:
            rows.append(
                {
                    variable_name: format_number(point.x),
                    function_name: format_number(point.value),
                    "second derivative": format_number(point.second_derivative),
                    "classification": point.classification,
                }
            )
    else:
        x_name = model.variables[0].name
        y_name = model.variables[1].name
        function_name = model.functions[0].name
        columns = [
            x_name,
            y_name,
            function_name,
            "Hessian eigenvalue 1",
            "Hessian eigenvalue 2",
            "classification",
        ]
        for point in result.points:
            rows.append(
                {
                    x_name: format_number(point.x),
                    y_name: format_number(point.y),
                    function_name: format_number(point.value),
                    "Hessian eigenvalue 1": format_number(point.eigenvalues[0]),
                    "Hessian eigenvalue 2": format_number(point.eigenvalues[1]),
                    "classification": point.classification,
                }
            )
    return {
        "available": True,
        "status": result.numerical_status.value,
        "error": error,
        "diagnostics": [_diagnostic(item) for item in result.diagnostics],
        "statistics": _statistics(result.statistics),
        "columns": columns,
        "rows": rows,
    }


def _selected_model_visualisation(model: ModelIR, payload: Mapping[str, Any]) -> ModelVisualisation:
    available = detect_capabilities(model).visualisations
    requested = payload.get("selected_visualisation")
    try:
        selected = available[0] if requested is None else ModelVisualisation(str(requested))
    except (IndexError, ValueError) as exc:
        raise DesktopEngineError("No compatible model visualisation was selected.") from exc
    if selected not in available:
        raise DesktopEngineError(
            f"'{selected.value}' is not compatible with the validated model."
        )
    return selected


def _evaluate(
    model: ModelIR,
    parameters: dict[str, float],
    settings: EvaluationSettings,
) -> FunctionEvaluation | SurfaceEvaluation:
    if len(model.variables) == 1:
        return evaluate_single_variable_function(
            model,
            parameter_values=parameters,
            points=settings.points_1d,
        )
    if len(model.variables) == 2:
        return evaluate_two_variable_function(
            model,
            parameter_values=parameters,
            points_per_axis=settings.points_per_axis_2d,
        )
    raise DesktopEngineError("Direct evaluation requires one or two continuous variables.")


def _stationary(
    model: ModelIR,
    parameters: dict[str, float],
    settings: StationaryPointSettings,
) -> tuple[StationaryPointAnalysisResult | None, str | None]:
    if AnalysisCapability.STATIONARY_POINTS not in detect_capabilities(model).analyses:
        return None, None
    try:
        return (
            cast(
                StationaryPointAnalysisResult,
                capability_registry.run_analysis(
                    AnalysisCapability.STATIONARY_POINTS,
                    model,
                    parameter_values=parameters,
                    samples_1d=settings.samples_1d,
                    seeds_per_axis_2d=settings.seeds_per_axis_2d,
                    root_tolerance=settings.root_tolerance,
                ),
            ),
            None,
        )
    except (AnalysisError, CapabilityRegistryError) as exc:
        return None, str(exc)


def _analysis_objects(
    payload: Mapping[str, Any],
) -> tuple[
    str,
    ModelIR,
    dict[str, float],
    ModelVisualisation,
    EvaluationSettings,
    StationaryPointSettings,
    FunctionEvaluation | SurfaceEvaluation,
    StationaryPointAnalysisResult | None,
    str | None,
]:
    source = payload.get("source")
    if not isinstance(source, str):
        raise DesktopEngineError("A model source string is required.")
    model = _compile_model(source)
    parameters = _parameter_values(model, payload)
    evaluation_settings = _evaluation_settings(payload)
    stationary_settings = _stationary_settings(payload)
    selected = _selected_model_visualisation(model, payload)
    evaluation = _evaluate(model, parameters, evaluation_settings)
    stationary, stationary_error = _stationary(model, parameters, stationary_settings)
    return (
        source,
        model,
        parameters,
        selected,
        evaluation_settings,
        stationary_settings,
        evaluation,
        stationary,
        stationary_error,
    )


def _current_analysis_document(payload: Mapping[str, Any]) -> dict[str, Any]:
    (
        source,
        model,
        parameters,
        selected,
        evaluation_settings,
        stationary_settings,
        evaluation,
        stationary,
        stationary_error,
    ) = _analysis_objects(payload)
    try:
        figure = render_model_visualisation(
            selected,
            evaluation,
            stationary_points=() if stationary is None else stationary.points,
        )
        symbolic, symbolic_provenance = _symbolic_document(model)
    except (VisualisationError, SymbolicAnalysisError, CapabilityRegistryError) as exc:
        raise DesktopEngineError(str(exc)) from exc

    provenance_rows = [
        _provenance(reference, provenance)
        for reference, provenance in model.provenance_entries()
    ]
    provenance_rows.append(
        _provenance(
            analysis_ref("evaluation", model.functions[0].name),
            evaluation.provenance,
        )
    )
    if stationary is not None:
        provenance_rows.append(
            _provenance(
                analysis_ref("stationary_points", model.functions[0].name),
                stationary.provenance,
            )
        )
    provenance_rows.extend(symbolic_provenance)
    workload = estimate_experiment_workload(
        model,
        evaluation_settings,
        stationary_settings,
    )
    return {
        "model": _model_document(source, model),
        "parameter_values": parameters,
        "selected_visualisation": selected.value,
        "figure": json.loads(figure.to_json()),
        "evaluation": {
            "kind": type(evaluation).__name__,
            "status": evaluation.numerical_status.value,
            "diagnostics": [_diagnostic(item) for item in evaluation.diagnostics],
        },
        "stationary": _stationary_document(model, stationary, stationary_error),
        "symbolic": symbolic,
        "provenance": provenance_rows,
        "workload": _workload(workload),
    }


def _sweep_configuration(
    model: ModelIR,
    payload: Mapping[str, Any],
) -> tuple[SweepConfiguration, AnalysisVisualisation]:
    raw = payload.get("sweep")
    if not isinstance(raw, Mapping):
        raise DesktopEngineError("A sweep configuration is required.")
    parameter_name = str(raw.get("parameter_name") or "")
    try:
        parameter = model.parameter(parameter_name)
    except KeyError as exc:
        raise DesktopEngineError(f"Unknown sweep parameter: {parameter_name}.") from exc
    try:
        configuration = SweepConfiguration(
            parameter_name=parameter_name,
            start=float(raw.get("start", parameter.domain.lower)),
            end=float(raw.get("end", parameter.domain.upper)),
            step_count=int(raw.get("step_count", 21)),
            selected_visualisation=(
                None
                if raw.get("selected_visualisation") is None
                else str(raw.get("selected_visualisation"))
            ),
        )
    except (TypeError, ValueError, ExperimentStateError) as exc:
        raise DesktopEngineError(str(exc)) from exc
    available = capability_registry.compatible_analysis_visualisations(
        AnalysisCapability.PARAMETER_SWEEP,
        model,
    )
    try:
        selected = (
            available[0]
            if configuration.selected_visualisation is None
            else AnalysisVisualisation(configuration.selected_visualisation)
        )
    except (IndexError, ValueError) as exc:
        raise DesktopEngineError("No compatible parameter-sweep visualisation is available.") from exc
    if selected not in available:
        raise DesktopEngineError(
            f"'{selected.value}' is not compatible with the parameter sweep."
        )
    configuration = SweepConfiguration(
        parameter_name=configuration.parameter_name,
        start=configuration.start,
        end=configuration.end,
        step_count=configuration.step_count,
        selected_visualisation=selected.value,
    )
    return configuration, selected


def _run_sweep(
    model: ModelIR,
    parameters: dict[str, float],
    stationary_settings: StationaryPointSettings,
    configuration: SweepConfiguration,
) -> ParameterSweepResult:
    fixed = {
        name: value
        for name, value in parameters.items()
        if name != configuration.parameter_name
    }
    try:
        return cast(
            ParameterSweepResult,
            capability_registry.run_analysis(
                AnalysisCapability.PARAMETER_SWEEP,
                model,
                parameter_name=configuration.parameter_name,
                start=configuration.start,
                end=configuration.end,
                step_count=configuration.step_count,
                fixed_parameter_values=fixed,
                samples_1d=stationary_settings.samples_1d,
                seeds_per_axis_2d=stationary_settings.seeds_per_axis_2d,
                root_tolerance=stationary_settings.root_tolerance,
            ),
        )
    except (AnalysisError, CapabilityRegistryError) as exc:
        raise DesktopEngineError(str(exc)) from exc


def _sweep_document(
    model: ModelIR,
    result: ParameterSweepResult,
    selected: AnalysisVisualisation,
) -> dict[str, Any]:
    try:
        figure = render_analysis_visualisation(selected, result)
    except VisualisationError as exc:
        raise DesktopEngineError(str(exc)) from exc
    diagnostic_rows = [
        {
            result.parameter_name: format_number(step.parameter_value),
            "status": step.numerical_status.value,
            **_diagnostic(diagnostic),
        }
        for step in result.steps
        for diagnostic in step.diagnostics
    ]
    summary_rows: list[dict[str, Any]] = []
    for step in result.steps:
        if step.error is not None:
            summary_rows.append(
                {
                    result.parameter_name: format_number(step.parameter_value),
                    "local minima": "—",
                    "saddles": "—",
                    "local maxima": "—",
                    "degenerate / inconclusive": "—",
                    "status": step.error,
                }
            )
            continue
        summary_rows.append(
            {
                result.parameter_name: format_number(step.parameter_value),
                "local minima": sum(
                    point.classification == "local minimum" for point in step.points
                ),
                "saddles": sum(point.classification == "saddle" for point in step.points),
                "local maxima": sum(
                    point.classification == "local maximum" for point in step.points
                ),
                "degenerate / inconclusive": sum(
                    point.classification == "degenerate / inconclusive"
                    for point in step.points
                ),
                "status": (
                    "analysed with warning"
                    if step.numerical_status.value == "partial"
                    else "analysed"
                ),
            }
        )
    return {
        "parameter_name": result.parameter_name,
        "start": result.start,
        "end": result.end,
        "step_count": result.step_count,
        "successful_step_count": result.successful_step_count,
        "failed_step_count": result.failed_step_count,
        "partial_step_count": result.partial_step_count,
        "available_visualisations": [
            item.value
            for item in capability_registry.compatible_analysis_visualisations(
                AnalysisCapability.PARAMETER_SWEEP,
                model,
            )
        ],
        "selected_visualisation": selected.value,
        "figure": json.loads(figure.to_json()),
        "diagnostics": diagnostic_rows,
        "summary_columns": (
            [result.parameter_name]
            + [
                "local minima",
                "saddles",
                "local maxima",
                "degenerate / inconclusive",
                "status",
            ]
        ),
        "summary_rows": summary_rows,
    }


def _sweep_action(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = payload.get("source")
    if not isinstance(source, str):
        raise DesktopEngineError("A model source string is required.")
    model = _compile_model(source)
    parameters = _parameter_values(model, payload)
    stationary_settings = _stationary_settings(payload)
    configuration, selected = _sweep_configuration(model, payload)
    evaluation_settings = _evaluation_settings(payload)
    estimate = estimate_experiment_workload(
        model,
        evaluation_settings,
        stationary_settings,
        configuration,
    )
    if not estimate.within_budget:
        raise DesktopEngineError(
            "The requested experiment exceeds the computational budget: "
            + "; ".join(estimate.violations)
            + "."
        )
    result = _run_sweep(model, parameters, stationary_settings, configuration)
    document = _sweep_document(model, result, selected)
    document["workload"] = _workload(estimate)
    return document


def _run_capability_action(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one explicitly selected installed capability."""
    source = payload.get("source")
    capability_id = payload.get("capability_id")
    settings = payload.get("settings") or {}
    if not isinstance(source, str) or not isinstance(capability_id, str):
        raise DesktopEngineError("source and capability_id are required.")
    if not isinstance(settings, Mapping):
        raise DesktopEngineError("Capability settings must be an object.")
    model = _compile_model(source)
    try:
        outcome = run_registry.run(
            capability_id,
            model,
            settings,
            version=(
                None
                if payload.get("capability_version") is None
                else str(payload["capability_version"])
            ),
            maximum_workload_units=int(payload.get("maximum_workload_units", 10_000_000)),
        )
    except (ProtocolError, TypeError, ValueError) as exc:
        raise DesktopEngineError(str(exc)) from exc
    figure = None
    for result in outcome.results:
        if isinstance(result, VectorFieldEvaluation):
            figure = json.loads(create_vector_field_figure(result).to_json())
            break
        rendered = create_official_pack_figure(result)
        if rendered is not None:
            figure = json.loads(rendered.to_json())
            break
    return {
        "execution_performed": True,
        "run": outcome.run.payload(),
        "artifacts": [item.payload() for item in outcome.artifacts],
        "results": [portable_value(item) for item in outcome.results],
        "figure": figure,
    }


def _prepare_run_experiment(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Execute explicitly selected generic runs and freeze a reviewable reference state."""
    source = payload.get("source")
    requested_runs = payload.get("runs")
    if not isinstance(source, str) or not isinstance(requested_runs, list) or not requested_runs:
        raise DesktopEngineError("A model source and at least one run request are required.")
    if len(requested_runs) > 64 or any(not isinstance(item, Mapping) for item in requested_runs):
        raise DesktopEngineError("A run experiment may contain at most 64 run request objects.")
    model = _compile_model(source)
    parameters = _parameter_values(model, payload)
    outcomes = []
    for request in requested_runs:
        capability_id = request.get("capability_id")
        settings = request.get("settings") or {}
        if not isinstance(capability_id, str) or not isinstance(settings, Mapping):
            raise DesktopEngineError("Every run requires capability_id and object settings.")
        settings = dict(settings)
        descriptor = run_registry.descriptor(
            capability_id,
            None if request.get("capability_version") is None else str(request["capability_version"]),
        )
        properties = descriptor.settings_schema.get("properties", {})
        if isinstance(properties, Mapping) and "parameter_values" in properties and "parameter_values" not in settings:
            settings["parameter_values"] = parameters
        outcomes.append(
            run_registry.run(
                capability_id,
                model,
                settings,
                version=descriptor.version,
                maximum_workload_units=10_000_000,
            )
        )
    artifact_ids = [item.artifact_id for outcome in outcomes for item in outcome.artifacts]
    views_raw = payload.get("views") or []
    if not isinstance(views_raw, list) or any(not isinstance(item, Mapping) for item in views_raw):
        raise DesktopEngineError("views must be a list of objects.")
    views: list[ArtifactView] = []
    for index, item in enumerate(views_raw):
        renderer_id = item.get("renderer_id")
        selected_artifacts = item.get("artifact_ids", artifact_ids)
        configuration = item.get("configuration", {})
        if not isinstance(renderer_id, str) or not isinstance(selected_artifacts, list) or not isinstance(configuration, Mapping):
            raise DesktopEngineError("Every view requires renderer_id, artifact_ids, and object configuration.")
        views.append(
            ArtifactView(
                view_id=str(item.get("view_id") or f"view-{index + 1}"),
                renderer_id=renderer_id,
                renderer_version=str(item.get("renderer_version") or "1.0"),
                artifact_ids=tuple(str(value) for value in selected_artifacts),
                configuration=dict(configuration),
            )
        )
    raw_acceptances = payload.get("interpreter_acceptances", [])
    if not isinstance(raw_acceptances, list) or any(not isinstance(item, Mapping) for item in raw_acceptances):
        raise DesktopEngineError("interpreter_acceptances must be a list of objects.")
    try:
        state = create_run_experiment_state(
            model_source=source,
            model=model,
            parameter_values=parameters,
            run_outcomes=tuple(outcomes),
            views=tuple(views),
            numerical_reproduction_settings=_reproduction_settings(payload),
            interpreter_acceptances=tuple(dict(item) for item in raw_acceptances),
        )
        draft = create_run_mlab_bundle(state=state, model=model)
        review = prepare_experiment_review(state, model)
    except (ExperimentStateError, MlabBundleError, ExperimentAuthoringError, ProtocolError) as exc:
        raise DesktopEngineError(str(exc)) from exc
    return {
        "state_json": state.to_json(),
        "state_sha256": state.state_sha256,
        "draft_bundle_base64": base64.b64encode(draft).decode("ascii"),
        "review": dict(review.review),
        "review_sha256": review.review_sha256,
        "runs": [outcome.run.payload() for outcome in outcomes],
        "artifacts": [item.payload() for outcome in outcomes for item in outcome.artifacts],
    }


def _decode_experiment(payload: Mapping[str, Any]) -> tuple[ExperimentState, ModelIR, MlabBundle | None]:
    encoded = payload.get("data_base64")
    if not isinstance(encoded, str) or not encoded:
        raise DesktopEngineError("Experiment bytes are required.")
    try:
        data = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise DesktopEngineError("Experiment data is not valid base64.") from exc
    filename = str(payload.get("filename") or "experiment.mlab").lower()
    try:
        if filename.endswith(".mlab") or data.startswith(b"PK\x03\x04"):
            bundle = load_mlab_bundle(data)
            return bundle.state, bundle.model, bundle
        state = ExperimentState.from_json(data.decode("utf-8"))
        model = _compile_model(state.model_source)
        validate_experiment_state_for_model(state, model, enforce_workload_budget=False)
        return state, model, None
    except (UnicodeDecodeError, ExperimentStateError, MlabBundleError) as exc:
        raise DesktopEngineError(str(exc)) from exc


def _experiment_summary(
    state: ExperimentState,
    model: ModelIR,
    bundle: MlabBundle | None,
) -> dict[str, Any]:
    content_handle = None if bundle is None else _OPENED_CONTENT.put(bundle)
    typed_protocol = bool(state.run_records or state.artifacts or state.views)
    if typed_protocol:
        total_units = sum(int(item.get("workload_units", 0)) for item in state.run_records)
        workload = {
            "within_budget": total_units <= 10_000_000,
            "rows": [
                {"component": "Typed Run -> Artifact records", "value": len(state.run_records)},
                {"component": "Declared workload units", "value": total_units},
            ],
            "violations": (
                []
                if total_units <= 10_000_000
                else ["Typed runs exceed the default 10,000,000-unit inspection limit"]
            ),
        }
    else:
        estimate = estimate_experiment_workload(
            model,
            state.evaluation_settings,
            state.stationary_settings,
            state.sweep,
        )
        workload = _workload(estimate)
    return {
        "execution_performed": False,
        "kind": ".mlab bundle" if bundle is not None else "legacy JSON experiment",
        "experiment_id": None if bundle is None else bundle.experiment_id,
        "state_sha256": state.with_checksum().state_sha256,
        "created_at_utc": state.created_at_utc,
        "saved_laboratory_version": state.laboratory_version,
        "author_approved_for_publication": (
            False if bundle is None else bundle.author_approved_for_publication
        ),
        "bundle_format": (
            None
            if bundle is None
            else {
                "name": bundle.manifest.get("format"),
                "version": bundle.manifest.get("format_version"),
                "member_count": len(bundle.manifest.get("members", [])),
            }
        ),
        "model": _model_document(state.model_source, model),
        "model_source": state.model_source,
        "parameter_values": state.parameter_map,
        "selected_visualisation": state.selected_model_visualisation,
        "evaluation_settings": asdict(state.evaluation_settings),
        "stationary_settings": asdict(state.stationary_settings),
        "reproduction_settings": asdict(state.numerical_reproduction_settings),
        "sweep": None if state.sweep is None else asdict(state.sweep),
        "recorded_results": [
            {"name": name, "sha256": digest}
            for name, digest in state.result_fingerprints
        ],
        "run_protocol": {
            "enabled": typed_protocol,
            "content_handle": content_handle,
            "runs": list(state.run_records),
            "artifacts": [
                {
                    "artifact_id": item.get("artifact_id"),
                    "artifact_type": item.get("artifact_type"),
                    "artifact_type_version": item.get("artifact_type_version"),
                    "capability_id": item.get("capability_id"),
                    "artifact_sha256": item.get("artifact_sha256"),
                }
                for item in state.artifacts
            ],
            "views": list(state.views),
            "content_descriptors": (
                [] if bundle is None else list(bundle.content_descriptors)
            ),
            "asset_descriptors": (
                [] if bundle is None else list(bundle.asset_documents)
            ),
            "content_members": (
                []
                if bundle is None
                else [
                    {"path": path, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                    for path, content in sorted(bundle.content_members.items())
                ]
            ),
            "asset_members": (
                []
                if bundle is None
                else [
                    {"path": path, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                    for path, content in sorted(bundle.asset_members.items())
                ]
            ),
        },
        "interpreter_acceptances": list(state.interpreter_acceptances),
        "environment": [
            {"component": name, "value": value}
            for name, value in state.environment
        ],
        "workload": workload,
    }


def _read_content_chunk_action(payload: Mapping[str, Any]) -> dict[str, Any]:
    handle = payload.get("handle")
    path = payload.get("path")
    offset = payload.get("offset", 0)
    length = payload.get("length", MAX_CONTENT_CHUNK_BYTES)
    if not isinstance(handle, str) or not handle:
        raise DesktopEngineError("An opened-content handle is required.")
    if not isinstance(path, str) or not path:
        raise DesktopEngineError("A content-addressed member path is required.")
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise DesktopEngineError("Content offset must be an integer.")
    if isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= MAX_CONTENT_CHUNK_BYTES:
        raise DesktopEngineError(
            f"Content chunk length must be between 1 and {MAX_CONTENT_CHUNK_BYTES} bytes."
        )
    chunk, total = _OPENED_CONTENT.read(handle, path, offset, length)
    return {
        "handle": handle,
        "path": path,
        "offset": offset,
        "length": len(chunk),
        "total_size": total,
        "eof": offset + len(chunk) >= total,
        "data_base64": base64.b64encode(chunk).decode("ascii"),
    }


def _reproduction_document(
    state: ExperimentState,
    model: ModelIR,
    outcome: ReproductionOutcome,
) -> dict[str, Any]:
    report = outcome.report
    result_document = report.to_dict()
    figure = None
    stationary_document = _stationary_document(model, outcome.stationary_result)
    if outcome.evaluation is not None:
        try:
            selected = ModelVisualisation(state.selected_model_visualisation)
            rendered = render_model_visualisation(
                selected,
                outcome.evaluation,
                stationary_points=(
                    ()
                    if outcome.stationary_result is None
                    else outcome.stationary_result.points
                ),
            )
            figure = json.loads(rendered.to_json())
        except (ValueError, VisualisationError):
            figure = None
    sweep_document = None
    if outcome.sweep_result is not None and state.sweep is not None:
        available = capability_registry.compatible_analysis_visualisations(
            AnalysisCapability.PARAMETER_SWEEP,
            model,
        )
        try:
            selected_sweep = (
                available[0]
                if state.sweep.selected_visualisation is None
                else AnalysisVisualisation(state.sweep.selected_visualisation)
            )
            sweep_document = _sweep_document(model, outcome.sweep_result, selected_sweep)
        except (IndexError, ValueError, DesktopEngineError):
            sweep_document = None
    if figure is None:
        for run_outcome in outcome.run_outcomes:
            for result in run_outcome.results:
                if isinstance(result, VectorFieldEvaluation):
                    figure = json.loads(create_vector_field_figure(result).to_json())
                    break
                rendered = create_official_pack_figure(result)
                if rendered is not None:
                    figure = json.loads(rendered.to_json())
                    break
            if figure is not None:
                break
    return {
        "report": result_document,
        "report_json": report.to_json(),
        "report_text": report.to_text(),
        "figure": figure,
        "stationary": stationary_document,
        "sweep": sweep_document,
        "runs": [item.run.payload() for item in outcome.run_outcomes],
        "artifacts": [item.payload() for item in outcome.artifacts],
    }


def _reproduce_action(payload: Mapping[str, Any]) -> dict[str, Any]:
    state, model, bundle = _decode_experiment(payload)
    try:
        outcome = (
            reproduce_mlab_bundle(bundle)
            if bundle is not None
            else reproduce_experiment(state, model=model)
        )
    except ExperimentStateError as exc:
        raise DesktopEngineError(str(exc)) from exc
    return _reproduction_document(state, model, outcome)


def _prepared_experiment(payload: Mapping[str, Any]) -> dict[str, Any]:
    (
        source,
        model,
        parameters,
        selected,
        evaluation_settings,
        stationary_settings,
        evaluation,
        stationary,
        stationary_error,
    ) = _analysis_objects(payload)
    if stationary_error is not None and AnalysisCapability.STATIONARY_POINTS in detect_capabilities(model).analyses:
        raise DesktopEngineError(
            "Experiment references cannot be frozen because stationary-point analysis failed: "
            + stationary_error
        )
    sweep_configuration = None
    sweep_result = None
    sweep_document = None
    raw_sweep = payload.get("sweep")
    if isinstance(raw_sweep, Mapping) and bool(raw_sweep.get("enabled", True)):
        sweep_configuration, selected_sweep = _sweep_configuration(model, payload)
        estimate = estimate_experiment_workload(
            model,
            evaluation_settings,
            stationary_settings,
            sweep_configuration,
        )
        if not estimate.within_budget:
            raise DesktopEngineError(
                "The requested experiment exceeds the computational budget: "
                + "; ".join(estimate.violations)
                + "."
            )
        sweep_result = _run_sweep(
            model,
            parameters,
            stationary_settings,
            sweep_configuration,
        )
        sweep_document = _sweep_document(model, sweep_result, selected_sweep)
    raw_acceptances = payload.get("interpreter_acceptances", [])
    if not isinstance(raw_acceptances, list) or any(
        not isinstance(item, Mapping) for item in raw_acceptances
    ):
        raise DesktopEngineError("interpreter_acceptances must be a list of objects.")
    try:
        state = create_experiment_state(
            model_source=source,
            model=model,
            parameter_values=parameters,
            selected_model_visualisation=selected,
            evaluation=evaluation,
            stationary_result=stationary,
            evaluation_settings=evaluation_settings,
            stationary_settings=stationary_settings,
            sweep_configuration=sweep_configuration,
            sweep_result=sweep_result,
            numerical_reproduction_settings=_reproduction_settings(payload),
            interpreter_acceptances=tuple(dict(item) for item in raw_acceptances),
        )
        draft = create_mlab_bundle(
            state=state,
            model=model,
            evaluation=evaluation,
            stationary_result=stationary,
            sweep_result=sweep_result,
        )
        review = prepare_experiment_review(state, model)
    except (ExperimentStateError, MlabBundleError, ExperimentAuthoringError) as exc:
        raise DesktopEngineError(str(exc)) from exc
    return {
        "state_json": state.to_json(),
        "state_sha256": state.state_sha256,
        "draft_bundle_base64": base64.b64encode(draft).decode("ascii"),
        "review": dict(review.review),
        "review_sha256": review.review_sha256,
        "sweep": sweep_document,
    }


def _finalize_experiment(payload: Mapping[str, Any]) -> dict[str, Any]:
    state_json = payload.get("state_json")
    approved = payload.get("approved_review_sha256")
    if not isinstance(state_json, str) or not isinstance(approved, str):
        raise DesktopEngineError("Frozen experiment state and approved review SHA-256 are required.")
    try:
        state = ExperimentState.from_json(state_json)
        model = _compile_model(state.model_source)
        outcome = reproduce_experiment(state, model=model)
        if outcome.report.status.value != "EXACT REPRODUCTION":
            raise DesktopEngineError(
                "The frozen reference state no longer reproduces exactly; prepare a new review."
            )
        review = prepare_experiment_review(state, model)
        frozen = freeze_experiment(review, approved_review_sha256=approved)
        if state.artifacts:
            bundle_data = create_publishable_run_mlab_bundle(
                frozen=frozen,
                state=state,
                model=model,
            )
        else:
            if outcome.evaluation is None:
                raise DesktopEngineError(
                    "The frozen reference evaluation could not be reconstructed."
                )
            bundle_data = create_publishable_mlab_bundle(
                frozen=frozen,
                state=state,
                model=model,
                evaluation=outcome.evaluation,
                stationary_result=outcome.stationary_result,
                sweep_result=outcome.sweep_result,
            )
        validated = load_mlab_bundle(bundle_data)
    except DesktopEngineError:
        raise
    except (ExperimentStateError, ExperimentAuthoringError, MlabBundleError) as exc:
        raise DesktopEngineError(str(exc)) from exc
    return {
        "bundle_base64": base64.b64encode(bundle_data).decode("ascii"),
        "experiment_id": validated.experiment_id,
        "state_sha256": validated.state.state_sha256,
        "review_sha256": approved,
        "author_approved_for_publication": validated.author_approved_for_publication,
    }



def _commit_experiment_action(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Commit an exact prepared experiment state without consulting mutable UI state."""
    state_json = payload.get("state_json")
    expected_state_sha256 = payload.get("expected_state_sha256")
    parent_commit_sha256 = payload.get("parent_commit_sha256")
    if not isinstance(state_json, str) or not isinstance(expected_state_sha256, str):
        raise DesktopEngineError(
            "state_json and expected_state_sha256 are required to commit an experiment state."
        )
    if parent_commit_sha256 is not None and not isinstance(parent_commit_sha256, str):
        raise DesktopEngineError("parent_commit_sha256 must be null or a SHA-256 string.")
    try:
        state = ExperimentState.from_json(state_json)
        if state.state_sha256 != expected_state_sha256:
            raise DesktopEngineError(
                "The prepared experiment state changed before commit; prepare it again."
            )
        model = _compile_model(state.model_source)
        committed = commit_experiment_state(
            state,
            model,
            parent_commit_sha256=parent_commit_sha256,
        )
        # Re-parse the serialized envelope before returning it. This makes the desktop action
        # exercise the same validation path a future physical executor will use.
        checked = CommittedExperiment.from_json(committed.to_json())
        checked.verify_for_model(model)
    except DesktopEngineError:
        raise
    except (ExperimentStateError, CommittedExperimentError) as exc:
        raise DesktopEngineError(str(exc)) from exc
    return {
        "status": "COMMITTED",
        "commit_sha256": checked.commit_sha256,
        "experiment_state_sha256": checked.experiment_state_sha256,
        "model_source_sha256": checked.model_source_sha256,
        "model_ir_sha256": checked.model_ir_sha256,
        "parent_commit_sha256": checked.parent_commit_sha256,
        "committed_at_utc": checked.committed_at_utc,
        "commit_json": checked.to_json(),
        "execution_policy": dict(checked.to_document()["execution_policy"]),
    }


def _inspect_committed_experiment_action(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a committed-state receipt without executing numerical analysis."""
    commit_json = payload.get("commit_json")
    expected_commit_sha256 = payload.get("expected_commit_sha256")
    if not isinstance(commit_json, str):
        raise DesktopEngineError("commit_json is required.")
    if expected_commit_sha256 is not None and not isinstance(expected_commit_sha256, str):
        raise DesktopEngineError("expected_commit_sha256 must be a SHA-256 string when supplied.")
    try:
        committed = CommittedExperiment.from_json(commit_json)
        if expected_commit_sha256 is not None and committed.commit_sha256 != expected_commit_sha256:
            raise DesktopEngineError("Committed experiment identity does not match the expected SHA-256.")
        state = committed.experiment()
        model = _compile_model(state.model_source)
        committed.verify_for_model(model)
    except DesktopEngineError:
        raise
    except (ExperimentStateError, CommittedExperimentError) as exc:
        raise DesktopEngineError(str(exc)) from exc
    state_document = state.payload_dict()
    return {
        "status": "COMMITTED",
        "commit_json": committed.to_json(),
        "commit_sha256": committed.commit_sha256,
        "experiment_state_sha256": committed.experiment_state_sha256,
        "model_source_sha256": committed.model_source_sha256,
        "model_ir_sha256": committed.model_ir_sha256,
        "parent_commit_sha256": committed.parent_commit_sha256,
        "committed_at_utc": committed.committed_at_utc,
        "execution_policy": dict(committed.to_document()["execution_policy"]),
        "model_source": state.model_source,
        "parameter_values": state.parameter_map,
        "selected_model_visualisation": state.selected_model_visualisation,
        "evaluation_settings": state_document["evaluation_settings"],
        "stationary_settings": state_document["stationary_settings"],
        "reproduction_settings": state_document["numerical_reproduction_settings"],
        "sweep": state_document["sweep"],
        "interpreter_acceptances": list(state.interpreter_acceptances),
        "model": _model_document(state.model_source, model),
    }

def _health() -> dict[str, Any]:
    environment = dict(current_environment())
    return {
        "application": "Model Laboratory",
        "version": __version__,
        "engine": "python-sidecar",
        "protocol_version": 7,
        "process_mode": "persistent",
        "cache": _RESPONSE_CACHE.status(),
        "content_store": _OPENED_CONTENT.status(),
        "state_boundary": {
            "live_state": "mutable interactive authoring state",
            "committed_state_schema": "model-laboratory.committed-experiment/1.0",
            "external_execution_policy": "committed_only",
        },
        "build_identity": {
            "source_tree_sha256": environment["source_tree_sha256"],
            "git_commit_or_build_id": environment["git_commit_or_build_id"],
        },
        "interpreter": {
            "status": "deterministic_edit_compiler",
            "provider": "ollama",
            "configured_model_tag": "qwen3:4b-instruct-2507-q4_K_M",
            "identity_claim": "exact local manifest and model-blob SHA-256 evidence",
            "output": "typed edit programs; never complete model YAML",
            "execution_policy": "edits are compiled, validated, reviewed, and explicitly accepted",
        },
        "model_ir": {"schema_version": "3.0", "graph": True},
        "run_protocol": {
            "schema_version": "1.1",
            "installed_capabilities": run_registry.catalogue(),
            "official_packs": official_pack_catalogue(),
        },
    }


def _active_interpreter_attachments(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    try:
        return _INTERPRETER_ATTACHMENTS.resolve(payload.get("attachment_ids", []))
    except InterpreterAttachmentError as exc:
        raise DesktopEngineError(str(exc)) from exc


def _ingest_interpreter_attachment(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Decode/extract one local reference without exposing its path to inference."""
    try:
        manifest = _INTERPRETER_ATTACHMENTS.add(
            ingest_attachment(payload.get("name"), payload.get("data_base64"))
        )
    except InterpreterAttachmentError as exc:
        raise DesktopEngineError(str(exc)) from exc
    return {
        "attachment": manifest,
        "execution_performed": False,
    }


def _release_interpreter_attachment(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        removed = _INTERPRETER_ATTACHMENTS.remove(payload.get("attachment_id"))
    except InterpreterAttachmentError as exc:
        raise DesktopEngineError(str(exc)) from exc
    return {"removed": removed, "execution_performed": False}


def _prepare_interpreter_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build the bounded deterministic model context supplied to local inference."""
    instruction = payload.get("instruction")
    current_source = payload.get("current_model_source")
    clarification_history = payload.get("clarification_history", [])
    if not isinstance(instruction, str) or not isinstance(current_source, str):
        raise DesktopEngineError(
            "Interpreter instruction and current_model_source must be strings."
        )
    try:
        attachments = _active_interpreter_attachments(payload)
        context = create_interpreter_context(
            instruction=instruction,
            current_model_source=current_source,
            clarification_history=clarification_history,
            attachments=attachments,
        )
    except InterpreterProposalError as exc:
        raise DesktopEngineError(str(exc)) from exc
    return {
        "status": (
            "unable"
            if context.get("contract", {}).get("request_boundary", {}).get("required_action") == "unable"
            else "ready"
        ),
        "explanation": context.get("contract", {}).get("request_boundary", {}).get("reason", ""),
        "context": context,
        "context_bytes": len(
            json.dumps(
                context,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ),
        "maximum_context_bytes": MAX_CONTEXT_PACKAGE_BYTES,
        "maximum_context_rounds": MAX_CONTEXT_ROUNDS,
        "generation_options": generation_options_for_request(
            instruction=instruction, context_document=context
        ),
        "execution_performed": False,
    }


def _process_interpreter_output(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Negotiate exact context or compile a typed edit program into a proposal."""
    instruction = payload.get("instruction")
    current_source = payload.get("current_model_source")
    clarification_history = payload.get("clarification_history", [])
    if not isinstance(instruction, str) or not isinstance(current_source, str):
        raise DesktopEngineError(
            "Interpreter instruction and current_model_source must be strings."
        )
    try:
        attachments = _active_interpreter_attachments(payload)
        result = process_interpreter_output(
            raw_output=payload.get("output"),
            context_document=payload.get("context"),
            provider_identity=payload.get("provider_identity"),
            instruction=instruction,
            current_model_source=current_source,
            clarification_history=clarification_history,
            attachments=attachments,
            conversation_id=payload.get("conversation_id"),
            conversation_lineage=payload.get("conversation_lineage", []),
        )
        if result["status"] != "proposal":
            return result
        proposal = result["proposal"]
        source = cast(str, proposal["proposed_model"]["source"])
        model = _compile_model(source)
    except InterpreterProposalError as exc:
        raise DesktopEngineError(str(exc)) from exc
    return {
        **result,
        "diff": proposal_diff(current_source, source),
        "model": _model_document(source, model),
    }


def _accept_interpreter_proposal(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate and accept only the exact proposal currently shown to the user."""
    expected_sha = payload.get("expected_proposal_sha256")
    current_source = payload.get("current_model_source")
    if not isinstance(expected_sha, str) or not isinstance(current_source, str):
        raise DesktopEngineError(
            "The reviewed proposal SHA-256 and current_model_source are required."
        )
    try:
        proposal = validate_interpreter_proposal_document(payload.get("proposal"))
        current_sha = hashlib.sha256(current_source.encode("utf-8")).hexdigest()
        if proposal["request"]["current_model_source_sha256"] != current_sha:
            raise InterpreterProposalError(
                "The editor source changed after this proposal was requested; request a new proposal."
            )
        acceptance = accept_interpreter_proposal(
            proposal,
            expected_proposal_sha256=expected_sha,
        )
        source = cast(str, proposal["proposed_model"]["source"])
        model = _compile_model(source)
    except InterpreterProposalError as exc:
        raise DesktopEngineError(str(exc)) from exc
    return {
        "acceptance": acceptance,
        "source": source,
        "model": _model_document(source, model),
        "execution_performed": False,
    }


def _payload_cache_key(payload: Mapping[str, Any]) -> str | None:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None


def _dispatch_cacheable(action: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if action == "inspect_model":
        source = payload.get("source")
        if not isinstance(source, str):
            raise DesktopEngineError("A model source string is required.")
        return _model_document(source, _compile_model(source))
    if action == "analyse_model":
        return _current_analysis_document(payload)
    if action == "run_sweep":
        return _sweep_action(payload)
    raise DesktopEngineError(f"Action '{action}' is not cacheable.")


def dispatch(request: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one validated desktop request and return a JSON-safe result."""
    action = request.get("action")
    payload = request.get("payload") or {}
    if not isinstance(action, str):
        raise DesktopEngineError("A request action is required.")
    if not isinstance(payload, Mapping):
        raise DesktopEngineError("The request payload must be an object.")
    if action == "health":
        return _health()
    if action == "example_model":
        source = EXAMPLE_MODEL.read_text(encoding="utf-8")
        model = _compile_model(source)
        return {"source": source, "model": _model_document(source, model)}
    if action in _CACHEABLE_ACTIONS:
        cache_key = _payload_cache_key(payload)
        if cache_key is not None:
            cached = _RESPONSE_CACHE.get(action, cache_key)
            if cached is not None:
                return cached
        result = _dispatch_cacheable(action, payload)
        if cache_key is not None:
            _RESPONSE_CACHE.put(action, cache_key, result)
        return result
    if action == "inspect_experiment":
        state, model, bundle = _decode_experiment(payload)
        return _experiment_summary(state, model, bundle)
    if action == "read_content_chunk":
        return _read_content_chunk_action(payload)
    if action == "reproduce_experiment":
        return _reproduce_action(payload)
    if action == "run_capability":
        return _run_capability_action(payload)
    if action == "prepare_experiment":
        return _prepared_experiment(payload)
    if action == "prepare_run_experiment":
        return _prepare_run_experiment(payload)
    if action == "finalize_experiment":
        return _finalize_experiment(payload)
    if action == "commit_experiment_state":
        return _commit_experiment_action(payload)
    if action == "inspect_committed_experiment":
        return _inspect_committed_experiment_action(payload)
    if action == "ingest_interpreter_attachment":
        return _ingest_interpreter_attachment(payload)
    if action == "release_interpreter_attachment":
        return _release_interpreter_attachment(payload)
    if action == "prepare_interpreter_context":
        return _prepare_interpreter_context(payload)
    if action == "process_interpreter_output":
        return _process_interpreter_output(payload)
    if action == "accept_interpreter_proposal":
        return _accept_interpreter_proposal(payload)
    raise DesktopEngineError(f"Unknown desktop engine action: {action}.")


def handle_request(raw: bytes) -> bytes:
    """Parse and handle one request without writing incidental data to stdout."""
    request_id: Any = None
    try:
        if len(raw) > MAX_REQUEST_BYTES:
            raise DesktopEngineError("The desktop request exceeds the safe size limit.")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise DesktopEngineError("A desktop request must be a JSON object.")
        request_id = value.get("id")
        response = {"id": request_id, "ok": True, "result": dispatch(value)}
    except (
        DesktopEngineError,
        AnalysisError,
        EvaluationError,
        VisualisationError,
        CapabilityRegistryError,
        ExperimentStateError,
        MlabBundleError,
        ExperimentAuthoringError,
        InterpreterProposalError,
        ProtocolError,
    ) as exc:
        response = {
            "id": request_id,
            "ok": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        response = {
            "id": request_id,
            "ok": False,
            "error": {"type": "InvalidRequest", "message": str(exc)},
        }
    except Exception as exc:  # pragma: no cover - last-resort process boundary
        traceback.print_exc(file=sys.stderr)
        response = {
            "id": request_id,
            "ok": False,
            "error": {
                "type": "InternalEngineError",
                "message": "The scientific engine encountered an unexpected internal error.",
            },
        }
    return (
        json.dumps(response, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Model Laboratory desktop scientific engine")
    parser.add_argument("--once", action="store_true", help="handle one JSON request")
    args = parser.parse_args()
    while True:
        raw = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 2)
        if not raw:
            break
        if raw.endswith(b"\n"):
            raw = raw[:-1]
            if raw.endswith(b"\r"):
                raw = raw[:-1]
        sys.stdout.buffer.write(handle_request(raw))
        sys.stdout.buffer.flush()
        if args.once:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
