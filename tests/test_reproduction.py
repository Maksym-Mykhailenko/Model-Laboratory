from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from model_lab.analysis import run_parameter_sweep, run_stationary_point_analysis
from model_lab.capabilities import AnalysisVisualisation, ModelVisualisation
from model_lab.bundle import create_mlab_bundle, load_mlab_bundle
from model_lab.evaluator import FunctionEvaluation, evaluate_single_variable_function
from model_lab.experiment import (
    EvaluationSettings,
    ExperimentStateError,
    NumericalReproductionSettings,
    StationaryPointSettings,
    SweepConfiguration,
    create_experiment_state,
    validate_experiment_state_for_model,
)
from model_lab.parser import parse_model_text
from model_lab.reproduction import (
    EnvironmentComparisonStatus,
    ReproductionStatus,
    build_reproduction_report,
    compare_environment_records,
    reproduce_experiment,
    reproduce_mlab_bundle,
)
from model_lab.validator import validate_model


MODEL_TEXT = """name: Phase B example
variables:
  x:
    domain: [-3, 3]
parameters:
  a:
    default: 1
    domain: [-2, 2]
functions:
  y: x**2 + a*x
"""


def _experiment(*, with_sweep: bool = True):
    model = validate_model(parse_model_text(MODEL_TEXT))
    parameters = {"a": 1.25}
    evaluation_settings = EvaluationSettings(points_1d=129, points_per_axis_2d=45)
    stationary_settings = StationaryPointSettings(
        samples_1d=801,
        seeds_per_axis_2d=7,
        root_tolerance=1e-10,
    )
    evaluation = evaluate_single_variable_function(
        model, parameters, points=evaluation_settings.points_1d
    )
    stationary = run_stationary_point_analysis(
        model,
        parameters,
        samples_1d=stationary_settings.samples_1d,
        seeds_per_axis_2d=stationary_settings.seeds_per_axis_2d,
        root_tolerance=stationary_settings.root_tolerance,
    )
    sweep = None
    sweep_configuration = None
    if with_sweep:
        sweep = run_parameter_sweep(
            model,
            "a",
            start=-2,
            end=2,
            step_count=7,
            fixed_parameter_values={},
            samples_1d=stationary_settings.samples_1d,
            seeds_per_axis_2d=stationary_settings.seeds_per_axis_2d,
            root_tolerance=stationary_settings.root_tolerance,
        )
        sweep_configuration = SweepConfiguration(
            parameter_name="a",
            start=-2,
            end=2,
            step_count=7,
            selected_visualisation=AnalysisVisualisation.SWEEP_POSITIONS.value,
        )
    state = create_experiment_state(
        model_source=MODEL_TEXT,
        model=model,
        parameter_values=parameters,
        selected_model_visualisation=ModelVisualisation.TWO_D_FUNCTION_PLOT,
        evaluation=evaluation,
        stationary_result=stationary,
        evaluation_settings=evaluation_settings,
        stationary_settings=stationary_settings,
        sweep_configuration=sweep_configuration,
        sweep_result=sweep,
        numerical_reproduction_settings=NumericalReproductionSettings(
            relative_tolerance=1e-8,
            absolute_tolerance=1e-11,
        ),
    )
    return state, model, evaluation, stationary, sweep


def test_exact_reproduction_report_is_independent_of_environment_identity() -> None:
    state, model, evaluation, stationary, sweep = _experiment()
    altered_environment = replace(
        state,
        laboratory_version="0.0-test",
        environment=(("python", "0.0-test"),),
    )

    report = build_reproduction_report(
        altered_environment,
        model=model,
        evaluation=evaluation,
        stationary_result=stationary,
        sweep_result=sweep,
        experiment_id="example-id",
    )

    assert report.status is ReproductionStatus.EXACT
    assert report.environment_status is EnvironmentComparisonStatus.DIFFERENT
    assert not report.laboratory_version_matches
    assert all(item.status is ReproductionStatus.EXACT for item in report.results)


def test_environment_policy_distinguishes_compatible_platform_changes() -> None:
    saved = {
        "python": "3.12.1",
        "python_implementation": "CPython",
        "numpy": "2.3.0",
        "scipy": "1.16.0",
        "sympy": "1.14.0",
        "operating_system": "Linux",
        "numpy_blas": "OpenBLAS",
        "source_tree_sha256": "a" * 64,
    }
    current = {
        **saved,
        "operating_system": "Windows",
        "numpy_blas": "MKL",
        "numpy": "2.4.0",
    }

    status, differences = compare_environment_records(
        saved,
        current,
        saved_laboratory_version="1.4.1",
        current_laboratory_version="1.4.2",
    )

    assert status is EnvironmentComparisonStatus.COMPATIBLE
    assert any("operating_system [compatible]" in item for item in differences)
    assert any("numpy_blas [compatible]" in item for item in differences)


def test_environment_policy_rejects_material_runtime_changes() -> None:
    saved = {
        "python": "3.12.1",
        "python_implementation": "CPython",
        "numpy": "2.3.0",
        "source_tree_sha256": "a" * 64,
    }
    current = {**saved, "numpy": "3.0.0"}

    status, differences = compare_environment_records(
        saved,
        current,
        saved_laboratory_version="1.4.1",
        current_laboratory_version="1.4.1",
    )

    assert status is EnvironmentComparisonStatus.DIFFERENT
    assert any("numpy [incompatible]" in item for item in differences)


def test_numerically_reproduced_requires_explicit_reference_comparison() -> None:
    state, model, evaluation, stationary, sweep = _experiment()
    changed_y = evaluation.y.copy()
    changed_y[0] += 1e-9  # within rtol for this non-zero value, but changes strict bytes
    changed = replace(evaluation, y=changed_y)

    report = build_reproduction_report(
        state,
        model=model,
        evaluation=changed,
        stationary_result=stationary,
        sweep_result=sweep,
    )

    assert report.status is ReproductionStatus.NUMERICAL
    evaluation_report = next(item for item in report.results if item.name == "evaluation")
    assert not evaluation_report.strict_fingerprint_matches
    assert evaluation_report.numerical_reference_available
    assert evaluation_report.numerical_comparison_matches is True
    assert evaluation_report.status is ReproductionStatus.NUMERICAL
    assert evaluation_report.max_absolute_deviation == pytest.approx(1e-9)


def test_outside_tolerance_yields_partial_reproduction_when_other_results_match() -> None:
    state, model, evaluation, stationary, sweep = _experiment()
    changed_y = evaluation.y.copy()
    changed_y[0] += 1e-2
    changed = replace(evaluation, y=changed_y)

    report = build_reproduction_report(
        state,
        model=model,
        evaluation=changed,
        stationary_result=stationary,
        sweep_result=sweep,
    )

    assert report.status is ReproductionStatus.PARTIAL
    evaluation_report = next(item for item in report.results if item.name == "evaluation")
    assert evaluation_report.status is ReproductionStatus.NOT_REPRODUCED
    assert any(item.status is ReproductionStatus.EXACT for item in report.results if item.name != "evaluation")


def test_strict_only_symbolic_result_must_match_for_numerical_reproduction() -> None:
    state, model, evaluation, stationary, sweep = _experiment()
    fingerprints = tuple(
        (name, "0" * 64 if name == "symbolic" else digest)
        for name, digest in state.result_fingerprints
    )
    altered = replace(state, result_fingerprints=fingerprints).with_checksum()

    report = build_reproduction_report(
        altered,
        model=model,
        evaluation=evaluation,
        stationary_result=stationary,
        sweep_result=sweep,
    )

    symbolic = next(item for item in report.results if item.name == "symbolic")
    assert symbolic.status is ReproductionStatus.NOT_REPRODUCED
    assert not symbolic.numerical_reference_available
    assert report.status is ReproductionStatus.PARTIAL


def test_validate_experiment_rejects_different_model_ir_even_with_compatible_shape() -> None:
    state, _, *_ = _experiment(with_sweep=False)
    other = validate_model(
        parse_model_text(
            """name: Other
variables:
  x:
    domain: [-3, 3]
parameters:
  a:
    default: 1
    domain: [-2, 2]
functions:
  y: x**3 + a*x
"""
        )
    )

    with pytest.raises(ExperimentStateError, match="does not match the model embedded"):
        validate_experiment_state_for_model(state, other)


def test_reproduce_experiment_executes_saved_settings_and_returns_exact_report() -> None:
    state, model, *_ = _experiment()

    outcome = reproduce_experiment(state, model=model, experiment_id="example-id")

    assert outcome.report.status is ReproductionStatus.EXACT
    assert outcome.report.experiment_id == "example-id"
    assert outcome.evaluation is not None
    assert outcome.stationary_result is not None
    assert outcome.sweep_result is not None


def test_reproduce_experiment_returns_unable_report_when_saved_workload_exceeds_budget() -> None:
    state, model, *_ = _experiment()
    oversized = replace(
        state,
        stationary_settings=StationaryPointSettings(
            samples_1d=200001,
            seeds_per_axis_2d=7,
            root_tolerance=1e-10,
        ),
        sweep=SweepConfiguration(
            parameter_name="a",
            start=-2,
            end=2,
            step_count=201,
            selected_visualisation=AnalysisVisualisation.SWEEP_POSITIONS.value,
        ),
    ).with_checksum()

    outcome = reproduce_experiment(oversized, model=model)

    assert outcome.report.status is ReproductionStatus.UNABLE
    assert outcome.evaluation is None
    assert outcome.report.execution_errors
    assert "budget" in outcome.report.execution_errors[0].lower()


def test_reproduction_report_json_and_text_are_portable_and_explicit() -> None:
    state, model, evaluation, stationary, sweep = _experiment()
    report = build_reproduction_report(
        state,
        model=model,
        evaluation=evaluation,
        stationary_result=stationary,
        sweep_result=sweep,
        experiment_id="example-id",
    )

    document = json.loads(report.to_json())
    assert document["schema"] == "model-laboratory-reproduction-report"
    assert document["schema_version"] == "1.2"
    assert document["status"] == "EXACT REPRODUCTION"
    assert document["environment"]["status"] in {
        "IDENTICAL",
        "COMPATIBLE",
        "DIFFERENT",
    }
    assert document["numerical_tolerances"]["relative_tolerance"] == pytest.approx(1e-8)
    text = report.to_text()
    assert "MODEL LABORATORY REPRODUCTION REPORT" in text
    assert "EXACT REPRODUCTION" in text
    assert "rtol:" in text
    assert "atol:" in text


def test_report_preserves_diagnostic_messages_warnings_and_provenance() -> None:
    from model_lab.experiment import (
        fingerprint_evaluation,
        numerical_reference_evaluation,
    )
    from model_lab.issues import NumericalDiagnostic, NumericalStatus

    state, model, evaluation, stationary, sweep = _experiment()
    saved_evaluation = replace(
        evaluation,
        numerical_status=NumericalStatus.PARTIAL,
        diagnostics=(NumericalDiagnostic.warning("nonfinite", "Old wording", count=1),),
    )
    current_evaluation = replace(
        evaluation,
        numerical_status=NumericalStatus.PARTIAL,
        diagnostics=(NumericalDiagnostic.warning("nonfinite", "New wording", count=1),),
    )
    state = replace(
        state,
        result_fingerprints=tuple(
            (name, fingerprint_evaluation(saved_evaluation) if name == "evaluation" else digest)
            for name, digest in state.result_fingerprints
        ),
        numerical_reference_data=tuple(
            (
                name,
                numerical_reference_evaluation(saved_evaluation)
                if name == "evaluation"
                else reference,
            )
            for name, reference in state.numerical_reference_data
        ),
        state_sha256="",
    )

    report = build_reproduction_report(
        state,
        model=model,
        evaluation=current_evaluation,
        stationary_result=stationary,
        sweep_result=sweep,
    )
    document = json.loads(report.to_json())
    evaluation_report = next(
        item for item in document["results"] if item["name"] == "evaluation"
    )

    assert evaluation_report["strict"]["matches"] is True
    assert evaluation_report["numerical"]["matches"] is True
    assert evaluation_report["diagnostics"]["saved"][0]["message"] == "Old wording"
    assert evaluation_report["diagnostics"]["current"][0]["message"] == "New wording"
    assert any("New wording" in item for item in document["warnings"])
    assert document["provenance"]["model"]
    assert document["provenance"]["results"]
    assert "Old wording" in report.to_text()
    assert "New wording" in report.to_text()
    assert "Provenance" in report.to_text()


def test_report_json_encodes_nonfinite_deviation_without_invalid_json() -> None:
    state, model, evaluation, stationary, sweep = _experiment()
    changed_y = evaluation.y.copy()
    changed_y[0] = np.inf
    changed = replace(evaluation, y=changed_y)

    report = build_reproduction_report(
        state,
        model=model,
        evaluation=changed,
        stationary_result=stationary,
        sweep_result=sweep,
    )
    document = json.loads(report.to_json())
    evaluation_document = next(item for item in document["results"] if item["name"] == "evaluation")
    assert evaluation_document["numerical"]["maximum_absolute_deviation"] == "Infinity"


def test_partial_execution_error_is_reported_as_unable_for_that_result() -> None:
    state, model, evaluation, stationary, sweep = _experiment()

    report = build_reproduction_report(
        state,
        model=model,
        evaluation=evaluation,
        stationary_result=None,
        sweep_result=sweep,
        execution_errors={"stationary_points": "solver unavailable"},
    )

    stationary_report = next(item for item in report.results if item.name == "stationary_points")
    assert stationary_report.status is ReproductionStatus.UNABLE
    assert "solver unavailable" in stationary_report.details
    assert report.status is ReproductionStatus.PARTIAL


def test_validated_mlab_bundle_reproduces_through_core_reproduction_api() -> None:
    state, model, evaluation, stationary, sweep = _experiment()
    data = create_mlab_bundle(
        state=state,
        model=model,
        evaluation=evaluation,
        stationary_result=stationary,
        sweep_result=sweep,
    )
    bundle = load_mlab_bundle(data)

    outcome = reproduce_mlab_bundle(bundle)

    assert outcome.report.status is ReproductionStatus.EXACT
    assert outcome.report.experiment_id == bundle.experiment_id
    assert outcome.evaluation is not None
