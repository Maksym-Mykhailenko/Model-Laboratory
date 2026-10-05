from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest

import desktop_engine
from model_lab.bundle import create_run_mlab_bundle, load_mlab_bundle
from model_lab.builtin_packs import run_registry
from model_lab.experiment import create_run_experiment_state
from model_lab.official_packs.composition import (
    AnalysisRecipeError,
    COMPOSED_NUMERIC,
    ComposedAnalysisResult,
    evaluate_safe_expression,
)
from model_lab.official_packs.renderers import create_official_pack_figure
from model_lab.parser import parse_model_text
from model_lab.protocol import ScientificArtifact, comparator_registry
from model_lab.reproduction import ReproductionStatus, reproduce_run_experiment
from model_lab.validator import ModelValidationError, validate_model


ROOT = Path(__file__).resolve().parents[1]
FLAGSHIP = ROOT / "models" / "gaussian-hierarchy-dispersion.yaml"
FLAGSHIP_BUNDLE = ROOT / "examples" / "gaussian-hierarchy-dispersion.mlab"


def _flagship():
    source = FLAGSHIP.read_text(encoding="utf-8")
    return source, validate_model(parse_model_text(source))


def _run_flagship():
    _source, model = _flagship()
    return run_registry.run(
        "org.modellab.composition.run-analysis-recipe",
        model,
        {"object_id": "matched-relaxation-profiles"},
    )


def test_flagship_recipe_reproduces_the_six_level_gaussian_results():
    outcome = _run_flagship()
    result = outcome.results[0]
    assert isinstance(result, ComposedAnalysisResult)
    assert len(result.steps) == 27
    np.testing.assert_allclose(
        result.outputs["Profile A relaxation rates"].value,
        [1.000, 5.129, 9.482, 12.402, 15.325, 24.160],
        atol=5e-4,
    )
    np.testing.assert_allclose(
        result.outputs["Profile B relaxation rates"].value,
        [1.000, 1.730, 15.015, 15.094, 23.575, 24.160],
        atol=5e-4,
    )
    assert result.outputs["Profile A dispersion"].value == pytest.approx(1.031, abs=5e-4)
    assert result.outputs["Profile B dispersion"].value == pytest.approx(1.282, abs=5e-4)
    for profile in ("Profile A", "Profile B"):
        assert result.outputs[f"{profile} minimum rate"].value == pytest.approx(1.000, abs=5e-4)
        assert result.outputs[f"{profile} maximum rate"].value == pytest.approx(24.160, abs=5e-4)
        assert result.outputs[f"{profile} geometric mean"].value == pytest.approx(7.789, abs=5e-4)
        assert result.outputs[f"{profile} condition number"].value == pytest.approx(24.160, abs=5e-4)
    assert result.outputs["Profile A t0.90"].value == pytest.approx(0.00479, abs=5e-5)
    assert result.outputs["Profile B t0.90"].value == pytest.approx(0.00403, abs=5e-5)
    assert result.outputs["Profile A t0.10"].value == pytest.approx(0.299, abs=5e-4)
    assert result.outputs["Profile B t0.10"].value == pytest.approx(0.461, abs=5e-4)
    assert result.outputs["Profile A recovery breadth"].value == pytest.approx(4.134, abs=5e-4)
    assert result.outputs["Profile B recovery breadth"].value == pytest.approx(4.741, abs=5e-4)

    assert result.outputs["Profile A effective dimension at t=0.4"].value == pytest.approx(1.076, abs=5e-4)
    assert result.outputs["Profile B effective dimension at t=0.4"].value == pytest.approx(1.851, abs=5e-4)
    assert all(len(step.value_sha256) == 64 for step in result.steps)


def test_composed_result_is_rendered_and_exactly_reproduced():
    source, model = _flagship()
    first = _run_flagship()
    second = _run_flagship()
    assert first.artifacts[0].artifact_sha256 == second.artifacts[0].artifact_sha256
    figure = create_official_pack_figure(first.results[0])
    assert figure is not None and len(figure.data) == 4
    assert len(figure.layout.annotations) == 2
    assert figure.layout.xaxis.type == figure.layout.xaxis2.type == "log"
    assert figure.data[1].line.dash == figure.data[3].line.dash == "dash"

    state = create_run_experiment_state(
        model_source=source,
        model=model,
        parameter_values=model.parameter_defaults(),
        run_outcomes=(first,),
    )
    reproduced = reproduce_run_experiment(state, model=model)
    assert reproduced.report.status is ReproductionStatus.EXACT
    bundle = create_run_mlab_bundle(state=state, model=model)
    loaded = load_mlab_bundle(bundle)
    assert loaded.state.state_sha256 == state.state_sha256
    assert loaded.state.artifacts[0]["artifact_sha256"] == first.artifacts[0].artifact_sha256

    checked_in = load_mlab_bundle(FLAGSHIP_BUNDLE.read_bytes())
    assert checked_in.state.model_source == source
    assert checked_in.state.laboratory_version == "1.19.0"
    assert checked_in.state.artifacts[0]["artifact_type"] == "org.modellab.artifact.composed-analysis"


def test_composed_numeric_reproduction_compares_values_not_derived_hashes():
    reference = _run_flagship().artifacts[0]
    descriptor = run_registry.descriptor(
        "org.modellab.composition.run-analysis-recipe", "1.0"
    ).output_types[0]

    def changed_artifact(delta: float) -> ScientificArtifact:
        data = deepcopy(reference.data)
        selected = data["outputs"]["Profile A dispersion"]
        step_id = selected["step_id"]
        selected["value"] += delta
        for step in data["steps"]:
            step["value_sha256"] = "0" * 64
            if step["step_id"] == step_id:
                step["value"] += delta
        for output in data["outputs"].values():
            output["value_sha256"] = "0" * 64
        return ScientificArtifact.create(
            artifact_type=descriptor,
            capability_id=reference.capability_id,
            capability_version=reference.capability_version,
            model_ir_sha256=reference.model_ir_sha256,
            data=data,
        )

    within = comparator_registry.compare(
        COMPOSED_NUMERIC,
        reference,
        changed_artifact(1e-12),
        rtol=1e-8,
        atol=1e-11,
    )
    assert within.reproduced is True
    assert within.maximum_absolute_deviation == pytest.approx(1e-12)
    assert within.details["semantic_structure_matches"] is True
    assert within.details["ignored_derived_fields"] == ["value_sha256"]

    outside = comparator_registry.compare(
        COMPOSED_NUMERIC,
        reference,
        changed_artifact(1e-3),
        rtol=1e-8,
        atol=1e-11,
    )
    assert outside.reproduced is False
    assert outside.maximum_absolute_deviation == pytest.approx(1e-3)


def test_generalized_metric_and_safe_formula_primitives_are_reusable():
    source = """
name: Generalized spectrum
variables:
  x: {domain: [-1, 1], initial: 0}
matrix_functions:
  curvature:
    entries: [["2", "0"], ["0", "8"]]
  metric:
    entries: [["2", "0"], ["0", "4"]]
objects:
  spectrum:
    kind: org.modellab.composition.analysis-recipe
    properties:
      steps:
        - id: H
          operation: matrix.evaluate
          settings: {matrix: curvature, point: [0]}
        - id: G
          operation: matrix.evaluate
          settings: {matrix: metric, point: [0]}
        - id: mu
          operation: matrix.generalized-eigenvalues
          inputs: {matrix: H, metric: G}
          settings: {require_positive: true}
        - id: ratio
          operation: array.expression
          inputs: {rates: mu}
          settings: {expression: "max(rates) / min(rates)"}
      outputs: {rates: mu, ratio: ratio}
"""
    model = validate_model(parse_model_text(source))
    result = run_registry.run(
        "org.modellab.composition.run-analysis-recipe", model, {"object_id": "spectrum"}
    ).results[0]
    np.testing.assert_allclose(result.outputs["rates"].value, [1.0, 2.0])
    assert result.outputs["ratio"].value == pytest.approx(2.0)


def test_recipe_validation_rejects_forward_references_and_python_syntax():
    forward_reference = """
name: Invalid recipe
objects:
  invalid:
    kind: org.modellab.composition.analysis-recipe
    properties:
      steps:
        - id: result
          operation: array.expression
          inputs: {x: later}
          settings: {expression: "x + 1"}
        - id: later
          operation: array.literal
          settings: {value: 1}
      outputs: {result: result}
"""
    with pytest.raises(ModelValidationError, match="earlier step"):
        validate_model(parse_model_text(forward_reference))

    executable_syntax = forward_reference.replace(
        "inputs: {x: later}", "inputs: {}"
    ).replace(
        'expression: "x + 1"', 'expression: "__import__(1)"'
    ).replace(
        "        - id: later\n          operation: array.literal\n          settings: {value: 1}\n",
        "",
    )
    with pytest.raises(ModelValidationError, match="unknown names"):
        validate_model(parse_model_text(executable_syntax))


def test_expression_engine_is_bounded_and_fails_closed():
    with pytest.raises(AnalysisRecipeError, match="bounded intermediate"):
        evaluate_safe_expression(
            "outer(left, right)",
            {"left": np.ones(500), "right": np.ones(500)},
        )
    with pytest.raises(AnalysisRecipeError, match="non-finite"):
        evaluate_safe_expression("log(values)", {"values": np.asarray([-1.0])})
    with pytest.raises(AnalysisRecipeError, match="unsupported syntax"):
        evaluate_safe_expression("values.__class__", {"values": np.asarray([1.0])})
    with pytest.raises(AnalysisRecipeError, match="intermediate limit"):
        evaluate_safe_expression(
            "left + right",
            {"left": np.ones((1000, 1)), "right": np.ones((1, 1000))},
        )
    with pytest.raises(AnalysisRecipeError, match="intermediate limit"):
        evaluate_safe_expression(
            "left @ right",
            {"left": np.ones((1000, 100)), "right": np.ones((100, 1000))},
        )
    with pytest.raises(AnalysisRecipeError, match="linear-algebra work limit"):
        evaluate_safe_expression(
            "left @ right",
            {"left": np.ones((300, 300)), "right": np.ones((300, 300))},
        )
    with pytest.raises(AnalysisRecipeError, match="linear-algebra work limit"):
        evaluate_safe_expression("det(matrix)", {"matrix": np.eye(200)})


def test_recipe_enforces_a_cumulative_stored_value_limit():
    steps = "\n".join(
        f"""        - id: values_{index}
          operation: array.linspace
          settings: {{start: 0, stop: 1, count: 200000}}"""
        for index in range(6)
    )
    source = f"""
name: Oversized recipe
objects:
  oversized:
    kind: org.modellab.composition.analysis-recipe
    properties:
      steps:
{steps}
      outputs: {{result: values_5}}
"""
    model = validate_model(parse_model_text(source))
    with pytest.raises(AnalysisRecipeError, match="cumulative stored-value limit"):
        run_registry.run(
            "org.modellab.composition.run-analysis-recipe",
            model,
            {"object_id": "oversized"},
        )


def test_desktop_boundary_discovers_runs_and_renders_the_recipe():
    source = FLAGSHIP.read_text(encoding="utf-8")
    inspected = desktop_engine.dispatch(
        {"action": "inspect_model", "payload": {"source": source}}
    )
    capability = next(
        item for item in inspected["capabilities"]["installed_packs"]
        if item["id"] == "org.modellab.composition.run-analysis-recipe"
    )
    assert capability["applicable"] is True
    result = desktop_engine.dispatch({
        "action": "run_capability",
        "payload": {
            "source": source,
            "capability_id": capability["id"],
            "settings": {"object_id": "matched-relaxation-profiles"},
        },
    })
    assert result["artifacts"][0]["artifact_type"] == "org.modellab.artifact.composed-analysis"
    assert len(result["figure"]["data"]) == 4
