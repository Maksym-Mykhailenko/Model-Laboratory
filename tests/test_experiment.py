from __future__ import annotations

from dataclasses import replace
import json

import pytest

from model_lab.analysis import run_parameter_sweep, run_stationary_point_analysis
from model_lab.capabilities import AnalysisVisualisation, ModelVisualisation
from model_lab.evaluator import FunctionEvaluation, evaluate_single_variable_function
from model_lab.experiment import (
    EvaluationSettings,
    ExperimentState,
    ExperimentStateError,
    NumericalReproductionSettings,
    StationaryPointSettings,
    SweepConfiguration,
    compare_reproduced_results,
    compile_experiment_model,
    create_experiment_state,
    validate_experiment_state_for_model,
)
from model_lab.parser import parse_model_text
from model_lab.validator import validate_model


MODEL_TEXT = """# exact source text is intentionally retained
name: Reproducibility example
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


def _model():
    return validate_model(parse_model_text(MODEL_TEXT))


def _state():
    model = _model()
    parameters = {"a": 1.25}
    evaluation_settings = EvaluationSettings(points_1d=257, points_per_axis_2d=75)
    stationary_settings = StationaryPointSettings(
        samples_1d=1001,
        seeds_per_axis_2d=9,
        root_tolerance=1e-10,
    )
    evaluation = evaluate_single_variable_function(
        model, parameter_values=parameters, points=evaluation_settings.points_1d
    )
    stationary = run_stationary_point_analysis(
        model,
        parameters,
        samples_1d=stationary_settings.samples_1d,
        seeds_per_axis_2d=stationary_settings.seeds_per_axis_2d,
        root_tolerance=stationary_settings.root_tolerance,
    )
    sweep = run_parameter_sweep(
        model,
        "a",
        start=-2,
        end=2,
        step_count=9,
        fixed_parameter_values={},
        samples_1d=stationary_settings.samples_1d,
        seeds_per_axis_2d=stationary_settings.seeds_per_axis_2d,
        root_tolerance=stationary_settings.root_tolerance,
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
        sweep_configuration=SweepConfiguration(
            parameter_name="a",
            start=-2,
            end=2,
            step_count=9,
            selected_visualisation=AnalysisVisualisation.SWEEP_POSITIONS.value,
        ),
        sweep_result=sweep,
    )
    return state, model, evaluation, stationary, sweep


def test_experiment_state_round_trip_preserves_exact_model_source() -> None:
    state, *_ = _state()
    restored = ExperimentState.from_json(state.to_json())

    assert restored.model_source == MODEL_TEXT
    assert restored.parameter_map == {"a": 1.25}
    assert restored.evaluation_settings.points_1d == 257
    assert restored.stationary_settings.root_tolerance == pytest.approx(1e-10)
    assert restored.sweep is not None
    assert restored.sweep.selected_visualisation == AnalysisVisualisation.SWEEP_POSITIONS.value
    assert restored.state_sha256


def test_experiment_state_detects_document_tampering() -> None:
    state, *_ = _state()
    document = json.loads(state.to_json())
    document["parameter_values"][0]["value"] = 1.5

    with pytest.raises(ExperimentStateError, match="checksum"):
        ExperimentState.from_json(json.dumps(document))


def test_embedded_model_reconstructs_and_validates_semantically() -> None:
    state, *_ = _state()
    restored = ExperimentState.from_json(state.to_json())
    model = compile_experiment_model(restored)

    validate_experiment_state_for_model(restored, model)
    assert model.name == "Reproducibility example"
    assert model.parameter("a").default == pytest.approx(1.0)


def test_reproduction_matches_all_recorded_result_fingerprints() -> None:
    state, model, evaluation, stationary, sweep = _state()

    check = compare_reproduced_results(
        state,
        model=model,
        evaluation=evaluation,
        stationary_result=stationary,
        sweep_result=sweep,
    )

    assert check.all_results_match
    assert check.model_source_matches
    assert all(matches for _, matches in check.result_matches)


def test_changed_evaluation_is_detected_as_non_reproduction() -> None:
    state, model, _, stationary, sweep = _state()
    changed = evaluate_single_variable_function(model, {"a": 1.5}, points=257)

    check = compare_reproduced_results(
        state,
        model=model,
        evaluation=changed,
        stationary_result=stationary,
        sweep_result=sweep,
    )

    assert dict(check.result_matches)["evaluation"] is False
    assert check.all_results_match is False


def test_environment_difference_is_reported_separately_from_result_match() -> None:
    state, model, evaluation, stationary, sweep = _state()
    altered = replace(state, environment=(("python", "0.0"),))

    check = compare_reproduced_results(
        altered,
        model=model,
        evaluation=evaluation,
        stationary_result=stationary,
        sweep_result=sweep,
    )

    assert check.all_results_match
    assert not check.environment_matches
    assert check.environment_differences


def test_semantic_validation_rejects_inapplicable_saved_visualisation() -> None:
    state, model, *_ = _state()
    bad = replace(state, selected_model_visualisation=ModelVisualisation.THREE_D_SURFACE.value)

    with pytest.raises(ExperimentStateError, match="not applicable"):
        validate_experiment_state_for_model(bad, model)


def test_state_requires_sweep_configuration_and_result_together() -> None:
    model = _model()
    evaluation = evaluate_single_variable_function(model, {"a": 1.0}, points=51)
    stationary = run_stationary_point_analysis(model, {"a": 1.0})

    with pytest.raises(ExperimentStateError, match="configuration"):
        create_experiment_state(
            model_source=MODEL_TEXT,
            model=model,
            parameter_values={"a": 1.0},
            selected_model_visualisation=ModelVisualisation.TWO_D_FUNCTION_PLOT,
            evaluation=evaluation,
            stationary_result=stationary,
            sweep_configuration=SweepConfiguration("a", -2, 2, 5),
            sweep_result=None,
        )


def test_environment_records_platform_build_and_numerical_backend_metadata() -> None:
    state, *_ = _state()
    environment = state.environment_map

    assert environment["operating_system"]
    assert environment["cpu_architecture"]
    assert environment["python_implementation"]
    assert environment["source_tree_sha256"]
    assert environment["git_commit_or_build_id"]
    assert environment["numpy_blas"]
    assert environment["machine_byteorder"] in {"little", "big"}
    assert environment["float_radix"] == "2"
    assert environment["float_mantissa_bits"]
    assert environment["numpy_default_bit_generator"]


def test_create_experiment_rejects_model_source_model_ir_mismatch() -> None:
    model = _model()
    other_source = MODEL_TEXT.replace("x**2 + a*x", "x**3 + a*x")
    evaluation = evaluate_single_variable_function(model, {"a": 1.0}, points=51)
    stationary = run_stationary_point_analysis(model, {"a": 1.0})

    with pytest.raises(ExperimentStateError, match="does not compile to the supplied Model IR"):
        create_experiment_state(
            model_source=other_source,
            model=model,
            parameter_values={"a": 1.0},
            selected_model_visualisation=ModelVisualisation.TWO_D_FUNCTION_PLOT,
            evaluation=evaluation,
            stationary_result=stationary,
        )


def test_explicit_tolerance_reproduction_is_distinct_from_exact_reproduction() -> None:
    state, model, evaluation, stationary, sweep = _state()
    perturbed_y = evaluation.y.copy()
    perturbed_y[10] += 1e-12
    perturbed = FunctionEvaluation(
        x_name=evaluation.x_name,
        y_name=evaluation.y_name,
        x=evaluation.x.copy(),
        y=perturbed_y,
        provenance=evaluation.provenance,
        numerical_status=evaluation.numerical_status,
        diagnostics=evaluation.diagnostics,
    )

    check = compare_reproduced_results(
        state,
        model=model,
        evaluation=perturbed,
        stationary_result=stationary,
        sweep_result=sweep,
    )

    assert dict(check.result_matches)["evaluation"] is False
    assert dict(check.numerical_result_matches)["evaluation"] is True
    assert check.numerically_reproduced
    assert not check.exact_reproduction


def test_old_experiment_without_numerical_fingerprints_remains_loadable() -> None:
    state, *_ = _state()
    document = json.loads(state.to_json())
    document["format_version"] = 1
    document.pop("interpreter_acceptances")
    document.pop("numerical_reproduction_settings")
    document.pop("numerical_reference_data")
    document.pop("numerical_result_fingerprints")
    document.pop("run_records")
    document.pop("artifacts")
    document.pop("views")

    import hashlib
    payload = {key: value for key, value in document.items() if key != "state_sha256"}
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    document["state_sha256"] = hashlib.sha256(canonical).hexdigest()

    restored = ExperimentState.from_json(json.dumps(document))
    assert restored.format_version == 1
    assert restored.numerical_reference_data == ()
    assert restored.numerical_result_fingerprints == ()


def test_saved_experiment_cross_setting_workload_is_rejected_before_reproduction() -> None:
    state, model, *_ = _state()
    # Use a 2D model because nonlinear solver starts scale as seeds_per_axis squared.
    surface_text = """
name: Heavy surface
variables:
  x: {domain: [-2, 2]}
  y: {domain: [-2, 2]}
parameters:
  a: {default: 1, domain: [0, 2]}
functions:
  z: a*(x**2+y**2)
"""
    surface_model = validate_model(parse_model_text(surface_text))
    heavy = replace(
        state,
        model_source=surface_text,
        model_sha256=__import__("hashlib").sha256(surface_text.encode("utf-8")).hexdigest(),
        parameter_values=(("a", 1.0),),
        selected_model_visualisation=ModelVisualisation.THREE_D_SURFACE.value,
        evaluation_settings=EvaluationSettings(points_1d=1000, points_per_axis_2d=1000),
        stationary_settings=StationaryPointSettings(
            samples_1d=2001, seeds_per_axis_2d=101, root_tolerance=1e-9
        ),
        sweep=SweepConfiguration("a", 0, 2, 201),
    )

    with pytest.raises(ExperimentStateError, match="computational budget"):
        validate_experiment_state_for_model(heavy, surface_model)


def test_numerical_reproduction_tolerances_are_bounded() -> None:
    with pytest.raises(ExperimentStateError, match="relative_tolerance"):
        NumericalReproductionSettings(relative_tolerance=0.1, absolute_tolerance=1e-11)
    with pytest.raises(ExperimentStateError, match="absolute_tolerance"):
        NumericalReproductionSettings(relative_tolerance=1e-8, absolute_tolerance=1e-3)


def test_numerical_reproduction_uses_direct_tolerance_comparison_across_quantisation_boundary() -> None:
    import numpy as np
    from model_lab.experiment import (
        _compare_evaluation_reference,
        numerical_fingerprint_evaluation,
        numerical_reference_evaluation,
    )
    from model_lab.provenance import Provenance

    settings = NumericalReproductionSettings(relative_tolerance=1e-8, absolute_tolerance=1e-11)
    saved = FunctionEvaluation(
        x_name="x",
        y_name="y",
        x=np.array([0.0, 1.0]),
        y=np.array([1.0000000049, 2.0]),
        provenance=Provenance.derived(("test:source",), "test reference"),
    )
    current = FunctionEvaluation(
        x_name="x",
        y_name="y",
        x=np.array([0.0, 1.0]),
        y=np.array([1.0000000051, 2.0]),
        provenance=saved.provenance,
    )

    # This is the exact boundary failure that motivated format-v2 numerical references.
    assert numerical_fingerprint_evaluation(saved, settings) != numerical_fingerprint_evaluation(
        current, settings
    )

    comparison = _compare_evaluation_reference(
        numerical_reference_evaluation(saved), current, settings
    )
    assert comparison.matches
    assert comparison.max_absolute_deviation == pytest.approx(2e-10, rel=1e-6)


def test_direct_tolerance_comparison_is_stable_across_power_of_ten_boundary() -> None:
    import numpy as np
    from model_lab.experiment import _compare_evaluation_reference, numerical_reference_evaluation
    from model_lab.provenance import Provenance

    settings = NumericalReproductionSettings(relative_tolerance=1e-8, absolute_tolerance=1e-11)
    saved = FunctionEvaluation(
        x_name="x",
        y_name="y",
        x=np.array([0.0]),
        y=np.array([9.99999999]),
        provenance=Provenance.derived(("test:source",), "test reference"),
    )
    current = FunctionEvaluation(
        x_name="x",
        y_name="y",
        x=np.array([0.0]),
        y=np.array([10.00000001]),
        provenance=saved.provenance,
    )

    comparison = _compare_evaluation_reference(
        numerical_reference_evaluation(saved), current, settings
    )
    assert comparison.matches


def test_direct_tolerance_comparison_rejects_values_outside_tolerance() -> None:
    import numpy as np
    from model_lab.experiment import _compare_evaluation_reference, numerical_reference_evaluation
    from model_lab.provenance import Provenance

    settings = NumericalReproductionSettings(relative_tolerance=1e-8, absolute_tolerance=1e-11)
    saved = FunctionEvaluation(
        x_name="x",
        y_name="y",
        x=np.array([0.0]),
        y=np.array([1.0]),
        provenance=Provenance.derived(("test:source",), "test reference"),
    )
    current = FunctionEvaluation(
        x_name="x",
        y_name="y",
        x=np.array([0.0]),
        y=np.array([1.0001]),
        provenance=saved.provenance,
    )

    comparison = _compare_evaluation_reference(
        numerical_reference_evaluation(saved), current, settings
    )
    assert not comparison.matches
    assert comparison.details


def test_new_experiment_embeds_numerical_reference_values_not_quantised_hashes() -> None:
    state, *_ = _state()

    assert state.format_version == 6
    assert state.numerical_reference_data
    assert state.numerical_result_fingerprints == ()
    reference_names = {name for name, _ in state.numerical_reference_data}
    assert reference_names == {"evaluation", "stationary_points", "parameter_sweep"}

    restored = ExperimentState.from_json(state.to_json())
    assert restored.numerical_reference_data == state.numerical_reference_data


def test_medium_evaluation_arrays_use_bounded_independent_chunks() -> None:
    import numpy as np
    from model_lab.experiment import _compare_evaluation_reference, numerical_reference_evaluation
    from model_lab.provenance import Provenance

    values = np.linspace(-1.0, 1.0, 200_000)
    result = FunctionEvaluation(
        x_name="x",
        y_name="y",
        x=values,
        y=values**2,
        provenance=Provenance.derived(("test:source",), "chunked reference"),
    )
    reference = numerical_reference_evaluation(result)

    assert reference["arrays"]["x"]["encoding"] == "zlib+base64-chunks"
    assert all(
        chunk["raw_size"] <= 1024 * 1024
        for chunk in reference["arrays"]["x"]["chunks"]
    )
    comparison = _compare_evaluation_reference(
        reference, result, NumericalReproductionSettings()
    )
    assert comparison.matches
    assert "full array" in comparison.comparison_kind


def test_large_evaluation_arrays_use_compact_canonical_samples_and_summaries() -> None:
    import numpy as np
    from model_lab.experiment import _compare_evaluation_reference, numerical_reference_evaluation
    from model_lab.provenance import Provenance

    values = np.linspace(-2.0, 2.0, 1_100_000)
    result = FunctionEvaluation(
        x_name="x",
        y_name="y",
        x=values,
        y=np.sin(values),
        provenance=Provenance.derived(("test:source",), "selective reference"),
    )
    reference = numerical_reference_evaluation(result)
    x_reference = reference["arrays"]["x"]

    assert x_reference["encoding"] == "canonical-samples-v1"
    assert len(x_reference["sample_indices"]) == 4096
    assert len(json.dumps(reference)) < 500_000
    identical = _compare_evaluation_reference(
        reference, result, NumericalReproductionSettings()
    )
    assert identical.matches
    assert "canonical samples and summaries" in identical.comparison_kind

    changed_values = values.copy()
    changed_values[x_reference["sample_indices"][100]] += 0.01
    changed = replace(result, x=changed_values)
    different = _compare_evaluation_reference(
        reference, changed, NumericalReproductionSettings()
    )
    assert not different.matches
    assert any("canonical samples" in detail for detail in different.details)


def test_chunked_array_reference_rejects_invalid_declared_chunk_size() -> None:
    import numpy as np
    from model_lab.experiment import _compare_evaluation_reference, numerical_reference_evaluation
    from model_lab.provenance import Provenance

    values = np.linspace(0.0, 1.0, 200_000)
    result = FunctionEvaluation(
        x_name="x",
        y_name="y",
        x=values,
        y=values,
        provenance=Provenance.derived(("test:source",), "chunk validation"),
    )
    reference = numerical_reference_evaluation(result)
    reference["arrays"]["x"]["chunks"][0]["raw_size"] = 1024 * 1024 + 1

    comparison = _compare_evaluation_reference(
        reference, result, NumericalReproductionSettings()
    )
    assert not comparison.matches
    assert any("chunk size" in detail for detail in comparison.details)


def test_legacy_tolerance_fingerprint_agreement_does_not_claim_numerical_reproduction() -> None:
    from model_lab.experiment import numerical_fingerprint_evaluation

    state, model, evaluation, stationary, sweep = _state()
    legacy = replace(
        state,
        format_version=1,
        numerical_reference_data=(),
        numerical_result_fingerprints=(
            (
                "evaluation",
                numerical_fingerprint_evaluation(
                    evaluation, state.numerical_reproduction_settings
                ),
            ),
        ),
    )

    check = compare_reproduced_results(
        legacy,
        model=model,
        evaluation=evaluation,
        stationary_result=stationary,
        sweep_result=sweep,
    )

    assert not check.numerical_reproduction_available
    assert check.legacy_tolerance_fingerprint_available
    assert dict(check.legacy_tolerance_fingerprint_matches)["evaluation"]
    assert not check.numerically_reproduced


def test_stationary_point_numerical_comparison_is_order_independent_and_semantic() -> None:
    from model_lab.critical_points import OneDimensionalCriticalPoint
    from model_lab.experiment import (
        _compare_stationary_reference,
        numerical_reference_stationary_result,
    )

    model_text = """
name: Two stationary points
variables:
  x: {domain: [-2, 2]}
functions:
  y: x**3 - 3*x
"""
    model = validate_model(parse_model_text(model_text))
    result = run_stationary_point_analysis(model)
    assert len(result.points) == 2
    reference = numerical_reference_stationary_result(result)

    reversed_points = tuple(reversed(result.points))
    perturbed_points = tuple(
        OneDimensionalCriticalPoint(
            x=point.x + (2e-10 if index == 0 else -2e-10),
            value=point.value,
            second_derivative=point.second_derivative,
            classification=point.classification,
        )
        for index, point in enumerate(reversed_points)
    )
    reproduced = replace(result, points=perturbed_points)

    comparison = _compare_stationary_reference(
        reference,
        reproduced,
        NumericalReproductionSettings(relative_tolerance=1e-8, absolute_tolerance=1e-9),
    )
    assert comparison.matches

    wrong_classification = replace(
        reproduced,
        points=(
            OneDimensionalCriticalPoint(
                x=reproduced.points[0].x,
                value=reproduced.points[0].value,
                second_derivative=reproduced.points[0].second_derivative,
                classification="local maximum"
                if reproduced.points[0].classification != "local maximum"
                else "local minimum",
            ),
            reproduced.points[1],
        ),
    )
    comparison = _compare_stationary_reference(
        reference,
        wrong_classification,
        NumericalReproductionSettings(relative_tolerance=1e-8, absolute_tolerance=1e-9),
    )
    assert not comparison.matches


def test_numerical_reference_handles_nonfinite_stationary_metadata_without_invalid_json() -> None:
    import math
    from model_lab.critical_points import OneDimensionalCriticalPoint
    from model_lab.experiment import (
        _compare_stationary_reference,
        numerical_reference_stationary_result,
    )
    from model_lab.analysis import StationaryPointAnalysisResult
    from model_lab.issues import NumericalStatus
    from model_lab.provenance import Provenance

    result = StationaryPointAnalysisResult(
        parameter_values=(),
        points=(
            OneDimensionalCriticalPoint(
                x=0.0,
                value=math.nan,
                second_derivative=math.inf,
                classification="degenerate / inconclusive",
            ),
        ),
        provenance=Provenance.derived(("test:source",), "test reference"),
        numerical_status=NumericalStatus.PARTIAL,
    )
    reference = numerical_reference_stationary_result(result)
    # The reference must remain valid strict JSON despite non-finite numerical metadata.
    json.dumps(reference, allow_nan=False)

    comparison = _compare_stationary_reference(
        reference,
        result,
        NumericalReproductionSettings(),
    )
    assert comparison.matches


def test_corrupt_compressed_numerical_reference_is_rejected_safely() -> None:
    import numpy as np
    from model_lab.experiment import (
        _compare_evaluation_reference,
        numerical_reference_evaluation,
    )
    from model_lab.provenance import Provenance

    result = FunctionEvaluation(
        x_name="x",
        y_name="y",
        x=np.array([0.0, 1.0]),
        y=np.array([1.0, 2.0]),
        provenance=Provenance.derived(("test:source",), "test reference"),
    )
    reference = numerical_reference_evaluation(result)
    reference["arrays"]["y"]["data"] = "not-valid-base64"

    comparison = _compare_evaluation_reference(
        reference,
        result,
        NumericalReproductionSettings(),
    )
    assert not comparison.matches
    assert any("corrupt" in detail.lower() for detail in comparison.details)


def test_sweep_configuration_canonicalises_numeric_bounds_for_stable_state_checksum() -> None:
    state, *_ = _state()
    restored = ExperimentState.from_json(state.to_json())

    assert restored.state_sha256 == state.state_sha256
    assert restored.with_checksum().state_sha256 == state.state_sha256


def test_numerical_diagnostic_message_wording_does_not_affect_reproduction() -> None:
    import numpy as np
    from model_lab.evaluator import FunctionEvaluation
    from model_lab.experiment import (
        _compare_evaluation_reference,
        fingerprint_evaluation,
        numerical_reference_evaluation,
    )
    from model_lab.issues import NumericalDiagnostic
    from model_lab.provenance import Provenance

    saved = FunctionEvaluation(
        x_name="x",
        y_name="y",
        x=np.array([0.0]),
        y=np.array([float("nan")]),
        provenance=Provenance.derived(("test:source",), "test reference"),
        diagnostics=(NumericalDiagnostic.warning("nonfinite", "Old wording", count=1),),
    )
    current = FunctionEvaluation(
        x_name="x",
        y_name="y",
        x=saved.x.copy(),
        y=saved.y.copy(),
        provenance=saved.provenance,
        diagnostics=(NumericalDiagnostic.warning("nonfinite", "New wording", count=1),),
    )

    comparison = _compare_evaluation_reference(
        numerical_reference_evaluation(saved), current, NumericalReproductionSettings()
    )
    assert fingerprint_evaluation(saved) == fingerprint_evaluation(current)
    assert comparison.matches


def test_strict_numerical_fingerprints_use_scientific_diagnostic_identity() -> None:
    from model_lab.experiment import (
        fingerprint_evaluation,
        fingerprint_parameter_sweep,
        fingerprint_stationary_result,
    )
    from model_lab.issues import NumericalDiagnostic

    _, _, evaluation, stationary, sweep = _state()
    old = NumericalDiagnostic.warning("nonfinite", "Old wording", b=2, a=1)
    new = NumericalDiagnostic.warning("nonfinite", "New wording", a=1, b=2)

    assert fingerprint_evaluation(
        replace(evaluation, diagnostics=(old,))
    ) == fingerprint_evaluation(replace(evaluation, diagnostics=(new,)))
    assert fingerprint_stationary_result(
        replace(stationary, diagnostics=(old,))
    ) == fingerprint_stationary_result(replace(stationary, diagnostics=(new,)))

    assert sweep is not None
    old_step = replace(sweep.steps[0], diagnostics=(old,))
    new_step = replace(sweep.steps[0], diagnostics=(new,))
    assert fingerprint_parameter_sweep(
        replace(sweep, steps=(old_step, *sweep.steps[1:]))
    ) == fingerprint_parameter_sweep(
        replace(sweep, steps=(new_step, *sweep.steps[1:]))
    )

    scientifically_different = replace(
        evaluation,
        diagnostics=(NumericalDiagnostic.warning("nonfinite", "Any wording", a=1, b=3),),
    )
    assert fingerprint_evaluation(
        replace(evaluation, diagnostics=(old,))
    ) != fingerprint_evaluation(scientifically_different)


def test_numerical_diagnostic_detail_order_does_not_affect_reproduction() -> None:
    import numpy as np
    from model_lab.evaluator import FunctionEvaluation
    from model_lab.experiment import _compare_evaluation_reference, numerical_reference_evaluation
    from model_lab.issues import DiagnosticSeverity, NumericalDiagnostic
    from model_lab.provenance import Provenance

    saved_diagnostic = NumericalDiagnostic(
        code="example",
        severity=DiagnosticSeverity.WARNING,
        message="Saved",
        details=(("a", "1"), ("b", "2")),
    )
    current_diagnostic = NumericalDiagnostic(
        code="example",
        severity=DiagnosticSeverity.WARNING,
        message="Current",
        details=(("b", "2"), ("a", "1")),
    )
    saved = FunctionEvaluation(
        x_name="x", y_name="y", x=np.array([0.0]), y=np.array([1.0]),
        provenance=Provenance.derived(("test:source",), "test reference"),
        diagnostics=(saved_diagnostic,),
    )
    current = FunctionEvaluation(
        x_name="x", y_name="y", x=saved.x.copy(), y=saved.y.copy(),
        provenance=saved.provenance, diagnostics=(current_diagnostic,),
    )

    reference = numerical_reference_evaluation(saved)
    # Full presentation records remain preserved in the experiment reference.
    assert reference["diagnostic_records"][0]["message"] == "Saved"
    comparison = _compare_evaluation_reference(reference, current, NumericalReproductionSettings())
    assert comparison.matches


def test_numerical_diagnostic_scientific_identity_still_rejects_real_differences() -> None:
    import numpy as np
    from model_lab.evaluator import FunctionEvaluation
    from model_lab.experiment import _compare_evaluation_reference, numerical_reference_evaluation
    from model_lab.issues import NumericalDiagnostic
    from model_lab.provenance import Provenance

    saved = FunctionEvaluation(
        x_name="x", y_name="y", x=np.array([0.0]), y=np.array([1.0]),
        provenance=Provenance.derived(("test:source",), "test reference"),
        diagnostics=(NumericalDiagnostic.warning("example", "Text", count=1),),
    )
    current = FunctionEvaluation(
        x_name="x", y_name="y", x=saved.x.copy(), y=saved.y.copy(),
        provenance=saved.provenance,
        diagnostics=(NumericalDiagnostic.warning("example", "Text", count=2),),
    )
    comparison = _compare_evaluation_reference(
        numerical_reference_evaluation(saved), current, NumericalReproductionSettings()
    )
    assert not comparison.matches
    assert any("diagnostic" in detail for detail in comparison.details)
