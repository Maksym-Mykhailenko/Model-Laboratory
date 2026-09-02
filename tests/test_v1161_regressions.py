from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from model_lab.builtin_packs import run_registry
from model_lab.interpreter import _official_pack_context_selection
from model_lab.official_packs import OFFICIAL_KIND_DESCRIPTORS
from model_lab.parser import parse_model_text
from model_lab.protocol import ProtocolError, portable_value
from model_lab.validator import ModelValidationError, validate_model


ROOT = Path(__file__).resolve().parents[1]


def compiled(source: str):
    return validate_model(parse_model_text(source))


def test_workload_uses_the_selected_ode_instead_of_the_first_object() -> None:
    model = compiled(
        """
name: selected workload
objects:
  small:
    kind: org.modellab.dynamics.ode-system
    kind_version: '1.0'
    properties:
      states: [x]
      initial_state: [0]
      time_span: [0, 1]
      equations: ['0']
  selected:
    kind: org.modellab.dynamics.ode-system
    kind_version: '1.0'
    properties:
      states: [a, b, c, d, e]
      initial_state: [0, 0, 0, 0, 0]
      time_span: [0, 1]
      equations: ['0', '0', '0', '0', '0']
"""
    )
    descriptor = run_registry.descriptor("org.modellab.dynamics.integrate-ode")
    settings = {"object_id": "selected", "samples": 100}
    assert descriptor.workload_estimator(settings, model) == 10_000
    with pytest.raises(ProtocolError, match="workload"):
        run_registry.run(
            descriptor.identifier,
            model,
            settings,
            maximum_workload_units=3_000,
        )


def test_selected_pomdp_controls_exponential_workload_estimate() -> None:
    model = compiled(
        """
name: selected POMDP workload
objects:
  tiny:
    kind: org.modellab.generative.pomdp
    kind_version: '1.0'
    properties:
      states: [s]
      observations: [o]
      actions: [a]
      initial: [1]
      transitions: [[[1]]]
      emissions: [[1]]
      rewards: [[0]]
      discount: 0.9
  selected:
    kind: org.modellab.generative.pomdp
    kind_version: '1.0'
    properties:
      states: [good, bad]
      observations: [green, red]
      actions: [wait, repair]
      initial: [0.8, 0.2]
      transitions:
        - [[0.9, 0.1], [0.2, 0.8]]
        - [[0.98, 0.02], [0.8, 0.2]]
      emissions: [[0.9, 0.1], [0.15, 0.85]]
      rewards: [[4, 0], [-6, -1]]
      discount: 0.95
"""
    )
    descriptor = run_registry.descriptor("org.modellab.generative.solve-pomdp")
    assert descriptor.workload_estimator({"object_id": "tiny", "horizon": 5}, model) == 5
    assert descriptor.workload_estimator({"object_id": "selected", "horizon": 5}, model) == 1_364
    with pytest.raises(ProtocolError, match="workload"):
        run_registry.run(
            descriptor.identifier,
            model,
            {"object_id": "selected", "horizon": 5},
            maximum_workload_units=100,
        )


@pytest.mark.parametrize(
    ("instruction", "pack_id"),
    (
        ("Create a 3x3 matrix", "org.modellab.pack.multidimensional-mathematics"),
        ("Create an HMM", "org.modellab.pack.generative-inference-decision-systems"),
        ("Create a graph", "org.modellab.pack.graphs-networks-discrete"),
        ("Create a PDE for heat conduction", "org.modellab.pack.spatial-fields-continuum-pdes"),
        ("Fit kmeans clusters", "org.modellab.pack.machine-learning-computational-intelligence"),
        ("Create a logistic classifier", "org.modellab.pack.machine-learning-computational-intelligence"),
    ),
)
def test_compact_routing_vocabulary_covers_natural_pack_requests(
    instruction: str, pack_id: str
) -> None:
    assert pack_id in _official_pack_context_selection(instruction, None)


def test_routing_vocabulary_is_word_bounded() -> None:
    assert _official_pack_context_selection("Write a paragraph", None) == ()
    assert _official_pack_context_selection("Check the odometer", None) == ()


def test_multinomial_regularisation_is_invariant_to_class_renaming() -> None:
    source = """
name: renamed classes
objects:
  study:
    kind: org.modellab.learning.supervised-study
    kind_version: '1.0'
    properties:
      feature_names: [x, y]
      features: [[-2,-1],[-1,-2],[-1,0],[0,2],[1,1],[2,0],[2,2],[0,-2],[1,-1]]
      target_name: class
      task: classification
      targets: [%s]
"""
    original = run_registry.run(
        "org.modellab.learning.fit-supervised-model",
        compiled(source % "alpha, alpha, alpha, beta, beta, gamma, gamma, gamma, beta"),
        {"regularisation": 1.0},
    ).results[0]
    renamed = run_registry.run(
        "org.modellab.learning.fit-supervised-model",
        compiled(source % "zeta, zeta, zeta, aardvark, aardvark, middle, middle, middle, aardvark"),
        {"regularisation": 1.0},
    ).results[0]
    mapping = {"alpha": "zeta", "beta": "aardvark", "gamma": "middle"}
    reordered = [renamed.class_names.index(mapping[name]) for name in original.class_names]
    assert np.allclose(
        original.class_probabilities,
        renamed.class_probabilities[:, reordered],
        rtol=1e-10,
        atol=1e-12,
    )


def test_self_contraction_uses_correlated_first_order_uncertainty() -> None:
    result = run_registry.run(
        "org.modellab.multidimensional.contract-tensors",
        compiled(
            """
name: self contraction
objects:
  x:
    kind: org.modellab.multidimensional.array
    kind_version: '1.1'
    properties:
      shape: [1]
      values: [2.0]
      standard_uncertainties: [0.1]
"""
        ),
        {"left_object_id": "x", "right_object_id": "x"},
    ).results[0]
    assert float(result.values) == pytest.approx(4.0)
    assert float(result.standard_uncertainties) == pytest.approx(0.4)
    assert result.uncertainty_model == "correlated-first-order-shared-input"


def test_operational_unit_semantics_have_a_new_kind_version_and_legacy_is_opaque() -> None:
    legacy = compiled(
        """
name: legacy units
objects:
  points:
    kind: org.modellab.geometry.point-cloud
    kind_version: '1.0'
    properties: {points: [[0, 0], [1, 1]], coordinate_unit: furlong}
"""
    )
    assert legacy.graph.objects[0].opaque is True
    assert legacy.graph.objects[0].executable is False
    with pytest.raises(ModelValidationError, match="unsupported unit 'furlong'"):
        compiled(
            """
name: operational units
objects:
  points:
    kind: org.modellab.geometry.point-cloud
    kind_version: '1.1'
    properties: {points: [[0, 0], [1, 1]], coordinate_unit: furlong}
"""
        )


def test_geometry_fields_and_arrays_preserve_source_unit_provenance() -> None:
    model = compiled(
        """
name: source units
objects:
  points:
    kind: org.modellab.geometry.point-cloud
    kind_version: '1.1'
    properties: {points: [[0, 0], [100, 0]], coordinate_unit: cm}
  voltage:
    kind: org.modellab.field.structured-scalar-field
    kind_version: '1.1'
    properties:
      axes: [{name: x, coordinates: [0, 50, 100], unit: cm}]
      values: [0, 500, 1000]
      value_unit: mV
  lengths:
    kind: org.modellab.multidimensional.array
    kind_version: '1.1'
    properties:
      shape: [2]
      values: [10, 20]
      unit: cm
      dimension_exponents: [1, 0, 0, 0, 0, 0, 0]
"""
    )
    geometry = run_registry.run("org.modellab.geometry.analyse-point-cloud", model).results[0]
    field = run_registry.run("org.modellab.field.analyse-scalar-field", model).results[0]
    array = run_registry.run("org.modellab.multidimensional.analyse-array", model).results[0]
    assert geometry.coordinate_unit == "m"
    assert geometry.source_units == {"coordinate": "cm"}
    assert geometry.points[1, 0] == pytest.approx(1.0)
    assert field.source_units == {"x": "cm", "value": "mV"}
    assert field.axis_units == ("m",)
    assert field.values[-1] == pytest.approx(1.0)
    assert array.source_units == {"value": "cm"}
    assert array.values.tolist() == pytest.approx([0.1, 0.2])


def test_degenerate_group_tests_are_explicitly_undefined_without_nan_payloads() -> None:
    result = run_registry.run(
        "org.modellab.statistics.compare-groups",
        compiled(
            """
name: tied samples
objects:
  samples:
    kind: org.modellab.statistics.grouped-samples
    kind_version: '1.0'
    properties:
      measure_name: score
      groups:
        - {name: a, values: [1, 1, 1]}
        - {name: b, values: [1, 1, 1]}
"""
        ),
    ).results[0]
    assert result.anova_f is None
    assert result.kruskal_h is None
    assert result.eta_squared is None
    assert all(entry["status"] == "undefined" for entry in result.test_statuses.values())
    assert result.pairwise_welch[0]["status"] == "undefined"
    assert '"special_float":"nan"' not in json.dumps(portable_value(result), separators=(",", ":"))


def test_pca_defaults_to_scale_invariant_standardized_coordinates() -> None:
    template = """
name: PCA scaling
objects:
  data:
    kind: org.modellab.statistics.dataset
    kind_version: '1.0'
    properties:
      columns: [x, y]
      column_units: [m, %s]
      data: %s
"""
    metres = [[0, 0], [1, 1], [2, 1.5], [3, 3], [4, 2.5]]
    centimetres = [[x, y * 100] for x, y in metres]
    first_model = compiled(template % ("m", metres))
    second_model = compiled(template % ("cm", centimetres))
    first = run_registry.run("org.modellab.statistics.analyse-dataset", first_model).results[0]
    second = run_registry.run("org.modellab.statistics.analyse-dataset", second_model).results[0]
    assert first.principal_coordinate_mode == second.principal_coordinate_mode == "standardized"
    assert np.allclose(first.explained_variance, second.explained_variance)
    assert np.allclose(
        np.abs(first.principal_components.display_basis),
        np.abs(second.principal_components.display_basis),
    )
    raw_first = run_registry.run(
        "org.modellab.statistics.analyse-dataset", first_model, {"feature_scaling": "raw"}
    ).results[0]
    raw_second = run_registry.run(
        "org.modellab.statistics.analyse-dataset", second_model, {"feature_scaling": "raw"}
    ).results[0]
    assert not np.allclose(raw_first.explained_variance, raw_second.explained_variance)


def test_kmeans_defaults_to_scale_invariant_standardized_coordinates() -> None:
    template = """
name: clustering scaling
objects:
  data:
    kind: org.modellab.learning.feature-dataset
    kind_version: '1.0'
    properties:
      feature_names: [x, y]
      feature_units: [m, %s]
      features: %s
"""
    metres = [[0, 0], [0.2, 0.1], [0.1, 0.3], [5, 5], [5.2, 5.1], [4.9, 5.3]]
    centimetres = [[x, y * 100] for x, y in metres]
    first = run_registry.run(
        "org.modellab.learning.cluster-kmeans",
        compiled(template % ("m", metres)),
        {"clusters": 2, "seed": 9},
    ).results[0]
    second = run_registry.run(
        "org.modellab.learning.cluster-kmeans",
        compiled(template % ("cm", centimetres)),
        {"clusters": 2, "seed": 9},
    ).results[0]
    assert first.feature_scaling == second.feature_scaling == "standardized"
    assert np.array_equal(first.assignments, second.assignments)
    assert np.allclose(first.analysis_centres, second.analysis_centres)


def test_corpus_and_benchmark_cover_every_official_pack_and_kind() -> None:
    corpus = ROOT / "training" / "interpreter_corpus_v1.5"
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    official_families = {
        "official-multidimensional", "official-probability", "official-graphs",
        "official-generative", "official-dynamics", "official-fields", "official-geometry",
        "official-mechanics", "official-statistics", "official-optimisation",
        "official-electrical", "official-reactions", "official-learning",
    }
    assert manifest["summary"]["case_count"] == 6_300
    assert {family for family in manifest["summary"]["family_counts"] if family.startswith("official-")} == official_families
    assert all(manifest["summary"]["family_counts"][family] == 100 for family in official_families)

    kinds_by_split = {"train": set(), "validation": set()}
    for split in kinds_by_split:
        with (corpus / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record["family"] not in official_families:
                    continue
                for operation in record["target"]["operations"]:
                    value = operation.get("value")
                    if operation.get("path", [None])[0] == "objects" and isinstance(value, dict):
                        kinds_by_split[split].add(value.get("kind"))
    official_kinds = {descriptor.kind for descriptor in OFFICIAL_KIND_DESCRIPTORS}
    assert kinds_by_split["train"] == official_kinds
    assert kinds_by_split["validation"] == official_kinds

    benchmark = json.loads(
        (ROOT / "verification" / "interpreter_baseline_v1.6.json").read_text(encoding="utf-8")
    )
    assert len(benchmark["cases"]) == 150
    benchmark_families = {
        case["family"] for case in benchmark["cases"] if case["family"].startswith("official-")
    }
    assert benchmark_families == official_families
    benchmark_kinds = set()
    for case in benchmark["cases"]:
        if not case["family"].startswith("official-"):
            continue
        source = case["expected"].get("model_source")
        if not source:
            source = (case["expected"].get("clarification_followup") or {}).get("model_source")
        if not source:
            continue
        expected = compiled(source)
        benchmark_kinds.update(item.kind for item in expected.graph.objects)
    assert benchmark_kinds == official_kinds


def test_v116_release_note_no_longer_claims_unimplemented_sample_weights() -> None:
    notes = (ROOT / "docs" / "releases" / "RELEASE_NOTES_1.16.0.md").read_text(encoding="utf-8").casefold()
    assert "optional targets and sample weights" not in notes
