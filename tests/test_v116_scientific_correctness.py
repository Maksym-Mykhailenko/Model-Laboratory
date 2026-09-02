from dataclasses import replace
import math

import numpy as np
import pytest

from model_lab.builtin_packs import run_registry
from model_lab.official_packs import OFFICIAL_PACK_MANIFESTS
from model_lab.official_packs.common import subspace_representation
from model_lab.official_packs.reactions import _reaction_network, _reaction_rates
from model_lab.parser import parse_model_text
from model_lab.protocol import ArtifactTypeDescriptor, ProtocolError, ScientificArtifact
from model_lab.validator import validate_model


def compiled(source: str):
    return validate_model(parse_model_text(source))


def test_corrected_scientific_contracts_are_not_relabelled_as_version_1_0():
    corrected = {
        "org.modellab.multidimensional.contract-tensors": "1.2",
        "org.modellab.probability.evolve-markov-chain": "1.1",
        "org.modellab.graph.analyse-network": "1.1",
        "org.modellab.generative.infer-hidden-markov-model": "1.1",
        "org.modellab.statistics.fit-linear-model": "1.1",
        "org.modellab.estimation.fit-nonlinear-least-squares": "1.1",
        "org.modellab.electromagnetics.analyse-point-charges": "1.2",
        "org.modellab.chemistry.simulate-reaction-network": "1.2",
    }
    for identifier, version in corrected.items():
        descriptor = run_registry.descriptor(identifier)
        assert descriptor.version == version
        assert all(item.version == version for item in descriptor.output_types)
        with pytest.raises(ProtocolError, match="unavailable"):
            run_registry.descriptor(identifier, "1.0")
    assert run_registry.descriptor("org.modellab.electronics.evaluate-shockley-diode").version == "1.0"
    versions = {item.identifier: item.version for item in OFFICIAL_PACK_MANIFESTS}
    assert versions["org.modellab.pack.electrical-electronic-electromagnetic-systems"] == "1.2"
    assert versions["org.modellab.pack.machine-learning-computational-intelligence"] == "1.1"


def test_hmm_long_smoothing_is_scaled_and_finite():
    source = """
name: stable HMM
objects:
  hmm:
    kind: org.modellab.generative.hidden-markov-model
    kind_version: '1.0'
    properties:
      states: [a, b]
      observations: [x, y]
      initial: [0.7, 0.3]
      transition: [[0.97, 0.03], [0.15, 0.85]]
      emission: [[0.8, 0.2], [0.25, 0.75]]
"""
    outcome = run_registry.run(
        "org.modellab.generative.infer-hidden-markov-model", compiled(source),
        {"observed_sequence": ["x", "y"] * 2500},
    )
    result = outcome.results[0]
    assert np.all(np.isfinite(result.smoothed_probabilities))
    np.testing.assert_allclose(result.smoothed_probabilities.sum(axis=1), 1.0, atol=1e-13)
    assert 0.0 < result.smoothed_probabilities[0, 1] < 1.0
    assert outcome.run.workload_units == 5000 * 4


def test_viterbi_never_crosses_zero_probability_transition():
    source = """
name: structural zeros
objects:
  hmm:
    kind: org.modellab.generative.hidden-markov-model
    kind_version: '1.0'
    properties:
      states: [locked, unreachable]
      observations: [x, y]
      initial: [1.0, 0.0]
      transition: [[1.0, 0.0], [0.0, 1.0]]
      emission: [[1.0, 1.0e-100], [0.0, 1.0]]
"""
    result = run_registry.run(
        "org.modellab.generative.infer-hidden-markov-model", compiled(source),
        {"observed_sequence": ["x", *(["y"] * 100)]},
    ).results[0]
    assert result.viterbi_path == ("locked",) * 101


def test_capability_settings_are_schema_enforced_before_work_estimation():
    source = """
name: HMM settings
objects:
  hmm:
    kind: org.modellab.generative.hidden-markov-model
    kind_version: '1.0'
    properties:
      states: [a, b]
      observations: [x]
      initial: [1.0, 0.0]
      transition: [[1.0, 0.0], [0.0, 1.0]]
      emission: [[1.0], [1.0]]
"""
    model = compiled(source)
    with pytest.raises(ProtocolError, match="unknown settings"):
        run_registry.run(
            "org.modellab.generative.infer-hidden-markov-model", model,
            {"observed_sequence": ["x"], "not_a_setting": True},
        )
    with pytest.raises(ProtocolError, match="too many items"):
        run_registry.run(
            "org.modellab.generative.infer-hidden-markov-model", model,
            {"observed_sequence": ["x"] * 100001},
        )
    with pytest.raises(ProtocolError, match="exceeds the configured limit"):
        run_registry.run(
            "org.modellab.generative.infer-hidden-markov-model", model,
            {"observed_sequence": ["x"] * 100}, maximum_workload_units=100,
        )


def test_effective_schema_defaults_are_frozen_in_run_record():
    source = """
name: chain
objects:
  chain:
    kind: org.modellab.probability.markov-chain
    kind_version: '1.0'
    properties:
      states: [a, b]
      initial: [1.0, 0.0]
      transition: [[0.8, 0.2], [0.1, 0.9]]
"""
    outcome = run_registry.run("org.modellab.probability.evolve-markov-chain", compiled(source))
    assert outcome.run.settings == {"steps": 20}


def test_reducible_chain_reports_stationary_family_not_arbitrary_member():
    source = """
name: identity chain
objects:
  chain:
    kind: org.modellab.probability.markov-chain
    kind_version: '1.0'
    properties:
      states: [a, b, c]
      initial: [0.2, 0.3, 0.5]
      transition: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
"""
    result = run_registry.run("org.modellab.probability.evolve-markov-chain", compiled(source)).results[0]
    assert result.stationary_distribution is None and not result.stationary_unique
    np.testing.assert_allclose(result.stationary_distributions, np.eye(3))
    assert result.recurrent_classes == (("a",), ("b",), ("c",))


def test_multigraph_degrees_count_parallel_edges_and_undirected_loop_twice():
    source = """
name: multigraph
objects:
  graph:
    kind: org.modellab.graph.network
    kind_version: '1.0'
    properties:
      nodes: [a, b]
      directed: false
      multigraph: true
      edges:
        - {source: a, target: a, weight: 3}
        - {source: a, target: b, weight: 1}
        - {source: a, target: b, weight: 2}
"""
    result = run_registry.run("org.modellab.graph.analyse-network", compiled(source)).results[0]
    np.testing.assert_array_equal(result.out_degree, [4, 2])
    np.testing.assert_array_equal(result.unique_out_neighbour_count, [2, 1])
    np.testing.assert_allclose(result.weighted_out_degree, [9, 3])


def test_rank_deficient_estimators_report_nonidentifiability_without_covariance():
    source = """
name: unidentifiable estimation
objects:
  nonlinear:
    kind: org.modellab.estimation.nonlinear-least-squares
    kind_version: '1.0'
    properties:
      variables: [a, b]
      initial: [0.0, 0.0]
      bounds: [[-10, 10], [-10, 10]]
      residuals: [a + b - 1, 2*a + 2*b - 2, 3*a + 3*b - 3]
  linear:
    kind: org.modellab.inverse.linear-problem
    kind_version: '1.0'
    properties:
      parameter_names: [a, b]
      observation_names: [o1, o2, o3]
      design_matrix: [[1, 1], [2, 2], [3, 3]]
      observations: [1, 2, 3]
"""
    model = compiled(source)
    nonlinear = run_registry.run("org.modellab.estimation.fit-nonlinear-least-squares", model).results[0]
    linear = run_registry.run("org.modellab.inverse.solve-linear-problem", model).results[0]
    assert not nonlinear.identifiable and nonlinear.covariance is None and nonlinear.standard_errors is None
    assert nonlinear.unidentifiable_directions.dimension == 1
    assert not linear.data_identifiable and linear.covariance is None and linear.standard_errors is None
    assert linear.unidentifiable_directions.dimension == 1


def test_through_origin_regression_uses_uncentred_r_squared_and_adjustment():
    source = """
name: through-origin regression
objects:
  fit:
    kind: org.modellab.statistics.linear-model-study
    kind_version: '1.0'
    properties:
      response_name: y
      predictor_names: [x]
      response: [1.0, 2.0, 1.0]
      predictors: [[1.0], [2.0], [3.0]]
      include_intercept: false
"""
    result = run_registry.run("org.modellab.statistics.fit-linear-model", compiled(source)).results[0]
    observed = result.fitted_values + result.residuals
    expected = 1.0 - result.weighted_residual_sum_squares / float(observed @ observed)
    assert result.r_squared_definition == "uncentred-weighted-through-origin"
    assert result.r_squared == pytest.approx(expected)
    assert result.adjusted_r_squared == pytest.approx(1.0 - (1.0 - expected) * 3 / 2)
    assert "relative inverse-variance" in result.information_criterion_likelihood


def test_electrostatic_units_are_converted_to_si_before_coulomb_equations():
    source = """
name: non-SI electrostatics
objects:
  charges:
    kind: org.modellab.electromagnetics.point-charge-system
    kind_version: '1.1'
    properties:
      dimension: 2
      coordinate_unit: cm
      charge_unit: nC
      charges: [{id: q, charge: 1.0, position: [0.0, 0.0]}]
      evaluation_points: [[1.0, 0.0]]
"""
    result = run_registry.run("org.modellab.electromagnetics.analyse-point-charges", compiled(source)).results[0]
    coulomb = 1.0 / (4.0 * math.pi * 8.8541878128e-12)
    assert result.net_charge == pytest.approx(1e-9)
    np.testing.assert_allclose(result.evaluation_points, [[0.01, 0.0]])
    assert result.field_magnitude[0] == pytest.approx(coulomb * 1e-9 / 0.01**2)
    assert result.coordinate_unit == "m" and result.charge_unit == "C"
    assert result.source_units == {"coordinate": "cm", "charge": "nC", "permittivity": "F/m"}


def test_tensor_contraction_preserves_labels_dimensions_and_uncertainty():
    source = """
name: labelled contraction
objects:
  matrix:
    kind: org.modellab.multidimensional.array
    kind_version: '1.1'
    properties:
      shape: [2, 2]
      values: [1, 2, 3, 4]
      standard_uncertainties: [0.1, 0.1, 0.1, 0.1]
      axis_labels: [[r1, r2], [x, y]]
      unit: V
      dimension_exponents: [2, 1, -3, -1, 0, 0, 0]
  vector:
    kind: org.modellab.multidimensional.array
    kind_version: '1.1'
    properties:
      shape: [2]
      values: [2, 1]
      standard_uncertainties: [0.2, 0.1]
      axis_labels: [[x, y]]
"""
    result = run_registry.run(
        "org.modellab.multidimensional.contract-tensors", compiled(source),
        {"left_object_id": "matrix", "right_object_id": "vector", "left_axes": [1], "right_axes": [0]},
    ).results[0]
    np.testing.assert_allclose(result.values, [4, 10])
    assert result.axis_labels == (("r1", "r2"),)
    assert result.unit == "V" and result.dimension_exponents == (2, 1, -3, -1, 0, 0, 0)
    np.testing.assert_allclose(result.standard_uncertainties[0], math.sqrt(0.1305))


def test_reaction_rate_vector_field_is_not_internally_clipped():
    source = """
name: polynomial reaction
objects:
  reaction:
    kind: org.modellab.chemistry.mass-action-network
    kind_version: '1.1'
    properties:
      species: [A, B]
      initial_concentrations: [1, 0]
      time_span: [0, 1]
      concentration_unit: mol/m^3
      time_unit: s
      reactions:
        - id: conversion
          reactants: [{species: A, stoichiometry: 1}]
          products: [{species: B, stoichiometry: 1}]
          rate_constant: 2
"""
    model = compiled(source)
    network = _reaction_network(model.graph.object("reaction"))
    assert _reaction_rates(network, np.asarray([-0.5, 0.0]))[0] == pytest.approx(-1.0)


def test_rotated_display_basis_does_not_change_subspace_artifact_identity():
    first = subspace_representation(np.eye(2), orientation="columns")
    angle = 0.37
    rotated = subspace_representation(
        np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]),
        orientation="columns",
    )
    descriptor = ArtifactTypeDescriptor(
        "org.modellab.artifact.test-subspace", "1.0", "Test subspace", "org.modellab.comparator.numeric"
    )
    left = ScientificArtifact.create(
        artifact_type=descriptor, capability_id="org.modellab.test.subspace", capability_version="1.0",
        model_ir_sha256="0" * 64, data={"subspace": first},
    )
    right = ScientificArtifact.create(
        artifact_type=descriptor, capability_id="org.modellab.test.subspace", capability_version="1.0",
        model_ir_sha256="0" * 64, data={"subspace": rotated},
    )
    assert left.artifact_sha256 == right.artifact_sha256
    assert left.data != right.data
