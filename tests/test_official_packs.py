from pathlib import Path
import json
from dataclasses import replace

import numpy as np
import pytest

from model_lab.builtin_packs import run_registry
from model_lab.official_packs import (
    OFFICIAL_KIND_REGISTRY, OFFICIAL_PACK_MANIFESTS, official_pack_catalogue,
)
from model_lab.official_packs.generative import ActiveInferenceEvaluation, HiddenMarkovInference
from model_lab.official_packs.dynamics import (
    ODEEquilibriumAnalysis, ODETrajectory, StateSpaceAnalysis, StateSpaceTrajectory,
)
from model_lab.official_packs.fields import (
    DiffusionSolution, PoissonSolution, ScalarFieldAnalysis, VectorFieldAnalysis,
)
from model_lab.official_packs.geometry import (
    MeshShortestPath, PointCloudAnalysis, TriangleMeshAnalysis,
)
from model_lab.official_packs.graphs import NetworkAnalysis
from model_lab.official_packs.mechanics import (
    IsotropicMaterialAnalysis, TrussModalAnalysis, TrussStaticResult,
)
from model_lab.official_packs.multidimensional import ArrayAnalysis, TensorContraction
from model_lab.official_packs.statistics import DatasetAnalysis, GroupComparison, LinearModelFit
from model_lab.official_packs.optimisation import (
    LinearInverseSolution, NonlinearLeastSquaresFit, OptimisationResult,
)
from model_lab.official_packs.electrical import (
    CircuitACAnalysis, CircuitDCAnalysis, DiodeCharacteristic, ElectrostaticAnalysis,
)
from model_lab.official_packs.reactions import (
    CompartmentAnalysis, CompartmentTrajectory, PopulationAnalysis, PopulationTrajectory,
    ReactionNetworkAnalysis, ReactionTrajectory,
)
from model_lab.official_packs.learning import (
    ClusteringResult, FuzzyInference, NeuralNetworkEvaluation, SupervisedLearningFit,
)
from model_lab.official_packs.probability import MarkovEvolution, MarkovSimulation
from model_lab.official_packs.renderers import create_official_pack_figure
from model_lab.parser import parse_model_text
from model_lab.protocol import CapabilityPackRegistry, ScientificArtifact, comparator_registry
from model_lab.experiment import create_run_experiment_state
from model_lab.interpreter import MAX_CONTEXT_PACKAGE_BYTES, create_interpreter_context
from model_lab.reproduction import ReproductionStatus, reproduce_run_experiment
from model_lab.official_packs.probability import CAPABILITY_DESCRIPTORS, simulate_markov_chain
from model_lab.validator import ModelValidationError, validate_model


ROOT = Path(__file__).resolve().parents[1]


def model(name: str):
    return validate_model(parse_model_text((ROOT / "models" / name).read_text(encoding="utf-8")))


def run(capability: str, source: str, settings=None):
    return run_registry.run(capability, model(source), settings or {})


def test_official_catalogue_is_versioned_and_complete():
    catalogue = official_pack_catalogue(model("generative-systems.yaml"))
    assert len(catalogue) == 13
    assert all(item["distribution"] == "official-optional" and item["installed"] for item in catalogue)
    assert sum(item["capability_count"] for item in catalogue) == 45
    assert len(OFFICIAL_KIND_REGISTRY.descriptors) == 43
    assert tuple(item.identifier for item in OFFICIAL_PACK_MANIFESTS) == (
        "org.modellab.pack.multidimensional-mathematics",
        "org.modellab.pack.probability-stochastic-systems",
        "org.modellab.pack.graphs-networks-discrete",
        "org.modellab.pack.generative-inference-decision-systems",
        "org.modellab.pack.dynamics-differential-equations-control",
        "org.modellab.pack.spatial-fields-continuum-pdes",
        "org.modellab.pack.geometry-meshes-spatial-computation",
        "org.modellab.pack.mechanics-structures-materials",
        "org.modellab.pack.statistical-inference-data-modelling",
        "org.modellab.pack.optimisation-estimation-inverse-problems",
        "org.modellab.pack.electrical-electronic-electromagnetic-systems",
        "org.modellab.pack.chemical-reaction-biological-systems",
        "org.modellab.pack.machine-learning-computational-intelligence",
    )


def test_machine_learning_and_computational_intelligence_pack_is_executable():
    fitted_outcome = run("org.modellab.learning.fit-supervised-model", "machine-learning.yaml")
    fitted = fitted_outcome.results[0]
    clustered_outcome = run(
        "org.modellab.learning.cluster-kmeans", "machine-learning.yaml",
        {"object_id": "labelled-observations", "clusters": 2, "seed": 17},
    )
    clustered = clustered_outcome.results[0]
    network = run("org.modellab.learning.evaluate-feedforward-network", "machine-learning.yaml").results[0]
    fuzzy = run(
        "org.modellab.intelligence.evaluate-fuzzy-system", "machine-learning.yaml",
        {"points": [[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]]},
    ).results[0]
    assert isinstance(fitted, SupervisedLearningFit) and fitted.metrics["accuracy"] == pytest.approx(1.0)
    assert isinstance(clustered, ClusteringResult) and tuple(clustered.cluster_sizes) == (4, 4)
    assert fitted_outcome.run.workload_units == 48_000
    assert clustered_outcome.run.workload_units == 9_600
    for cluster in range(clustered.cluster_count):
        np.testing.assert_allclose(
            clustered.centres[cluster],
            np.mean(clustered.features[clustered.assignments == cluster], axis=0),
        )
    assert isinstance(network, NeuralNetworkEvaluation) and network.outputs.shape == (8, 2)
    np.testing.assert_allclose(network.outputs.sum(axis=1), 1.0)
    assert isinstance(fuzzy, FuzzyInference)
    assert np.all(np.diff(fuzzy.outputs) > 0.0) and np.all((fuzzy.outputs >= 0.0) & (fuzzy.outputs <= 1.0))
    assert all(create_official_pack_figure(value) is not None for value in (fitted, clustered, network, fuzzy))


def test_multidimensional_analysis_and_contraction_are_executable():
    analysed = run("org.modellab.multidimensional.analyse-array", "multidimensional.yaml").results[0]
    assert isinstance(analysed, ArrayAnalysis)
    assert analysed.shape == (2, 3) and analysed.matrix_rank == 2
    contracted = run(
        "org.modellab.multidimensional.contract-tensors", "multidimensional.yaml",
        {"left_object_id": "response-matrix", "right_object_id": "projection", "left_axes": [1], "right_axes": [0]},
    ).results[0]
    assert isinstance(contracted, TensorContraction)
    np.testing.assert_allclose(contracted.values, [-0.5, 0.25])
    assert create_official_pack_figure(analysed) is not None


def test_quantities_normalise_without_erasing_dimensions():
    result = run("org.modellab.multidimensional.normalise-quantities", "multidimensional.yaml").results[0]
    assert [row["base_value"] for row in result.quantities] == [1.25, 1.25]
    assert len(result.compatible_groups) == 1


def test_distribution_and_markov_evolution():
    distribution = run("org.modellab.probability.analyse-distribution", "probability.yaml").results[0]
    assert distribution.expectation == pytest.approx(0.0)
    assert distribution.variance == pytest.approx(0.4)
    evolution = run(
        "org.modellab.probability.evolve-markov-chain", "probability.yaml", {"steps": 12}
    ).results[0]
    assert isinstance(evolution, MarkovEvolution)
    np.testing.assert_allclose(evolution.distributions.sum(axis=1), 1.0)
    assert evolution.stationarity_residual < 1e-12


def test_markov_simulation_is_seeded_and_has_stochastic_comparator():
    settings = {"steps": 10, "trajectories": 300, "seed": 42}
    first = run("org.modellab.probability.simulate-markov-chain", "probability.yaml", settings)
    second = run("org.modellab.probability.simulate-markov-chain", "probability.yaml", settings)
    assert isinstance(first.results[0], MarkovSimulation)
    np.testing.assert_array_equal(first.results[0].sampled_state_indices, second.results[0].sampled_state_indices)
    comparison = comparator_registry.compare(
        "org.modellab.comparator.stochastic", first.artifacts[0], second.artifacts[0]
    )
    assert comparison.reproduced
    assert comparison.details["status"] == "EXACT STOCHASTIC REPRODUCTION"


def test_network_analysis_covers_topology_spectrum_and_path():
    result = run(
        "org.modellab.graph.analyse-network", "network.yaml",
        {"source": "hypothesis", "target": "conclusion"},
    ).results[0]
    assert isinstance(result, NetworkAnalysis)
    assert result.shortest_path == ("hypothesis", "method", "observation", "conclusion")
    assert result.shortest_path_length == 3.0
    assert result.is_dag and not result.has_cycle
    assert create_official_pack_figure(result) is not None


def test_hidden_markov_inference_is_normalised():
    result = run(
        "org.modellab.generative.infer-hidden-markov-model", "generative-systems.yaml",
        {"observed_sequence": ["quiet", "alarm", "alarm"]},
    ).results[0]
    assert isinstance(result, HiddenMarkovInference)
    np.testing.assert_allclose(result.filtered_probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(result.smoothed_probabilities.sum(axis=1), 1.0)
    assert len(result.viterbi_path) == 3 and np.isfinite(result.log_evidence)


def test_decision_blanket_and_active_inference_capabilities():
    decision = run("org.modellab.generative.solve-pomdp", "generative-systems.yaml").results[0]
    assert decision.recommended_action in decision.actions
    blanket = run(
        "org.modellab.generative.markov-blanket", "generative-systems.yaml",
        {"network_object_id": "causal-network", "target": "state"},
    ).results[0]
    assert set(blanket.blanket) == {"action", "observation"}
    active = run("org.modellab.generative.evaluate-active-inference", "generative-systems.yaml").results[0]
    assert isinstance(active, ActiveInferenceEvaluation)
    assert active.policy_posterior.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(active.posterior_state_trajectory.sum(axis=1), 1.0)
    assert create_official_pack_figure(active) is not None


def test_ode_integration_equilibria_and_state_space_are_executable():
    trajectory = run("org.modellab.dynamics.integrate-ode", "dynamics-control.yaml", {"samples": 101}).results[0]
    assert isinstance(trajectory, ODETrajectory)
    assert trajectory.states.shape == (101, 2)
    np.testing.assert_allclose(trajectory.states[0], [1.0, 0.0])
    equilibria = run("org.modellab.dynamics.analyse-equilibria", "dynamics-control.yaml").results[0]
    assert isinstance(equilibria, ODEEquilibriumAnalysis)
    assert len(equilibria.points) == 1
    np.testing.assert_allclose(equilibria.points[0].coordinates, [0.0, 0.0], atol=1e-10)
    assert equilibria.points[0].classification == "asymptotically stable"
    analysed = run("org.modellab.control.analyse-state-space", "dynamics-control.yaml").results[0]
    assert isinstance(analysed, StateSpaceAnalysis)
    assert analysed.controllable and analysed.observable and analysed.stability == "asymptotically stable"
    response = run(
        "org.modellab.control.simulate-state-space", "dynamics-control.yaml",
        {"samples": 101, "duration": 4.0, "constant_input": [1.0]},
    ).results[0]
    assert isinstance(response, StateSpaceTrajectory) and response.outputs.shape == (101, 2)
    assert create_official_pack_figure(trajectory) is not None
    assert create_official_pack_figure(analysed) is not None


def test_solver_work_count_is_persisted_but_not_part_of_scientific_identity():
    outcome = run("org.modellab.dynamics.integrate-ode", "dynamics-control.yaml", {"samples": 51})
    trajectory = outcome.results[0]
    changed = replace(
        trajectory,
        solver_record={
            "scientific": {"converged": True},
            "presentation": {
                "function_evaluations": trajectory.solver_record["presentation"]["function_evaluations"] + 7,
                "solver_status": 0,
            },
        },
    )
    descriptor = run_registry.descriptor("org.modellab.dynamics.integrate-ode").output_types[0]
    changed_artifact = ScientificArtifact.create(
        artifact_type=descriptor,
        capability_id="org.modellab.dynamics.integrate-ode",
        capability_version="1.0",
        model_ir_sha256=outcome.artifacts[0].model_ir_sha256,
        data=changed,
    )
    assert changed_artifact.data != outcome.artifacts[0].data
    assert changed_artifact.artifact_sha256 == outcome.artifacts[0].artifact_sha256


def test_spatial_field_operators_and_pde_solvers_are_executable():
    scalar = run("org.modellab.field.analyse-scalar-field", "spatial-fields.yaml").results[0]
    assert isinstance(scalar, ScalarFieldAnalysis)
    np.testing.assert_allclose(scalar.gradient[0][2, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(scalar.gradient[1][2, 2], 0.0, atol=1e-12)
    assert scalar.minimum == 0.0 and scalar.minimum_coordinates == (0.0, 0.0)
    vector = run("org.modellab.field.analyse-vector-field", "spatial-fields.yaml").results[0]
    assert isinstance(vector, VectorFieldAnalysis)
    np.testing.assert_allclose(vector.divergence, 0.0, atol=1e-12)
    np.testing.assert_allclose(vector.curl, 2.0, atol=1e-12)
    diffusion = run(
        "org.modellab.pde.solve-diffusion", "spatial-fields.yaml", {"time_samples": 31},
    ).results[0]
    assert isinstance(diffusion, DiffusionSolution) and diffusion.values.shape == (31, 11)
    np.testing.assert_allclose(diffusion.values[:, [0, -1]], 0.0, atol=1e-12)
    poisson = run("org.modellab.pde.solve-poisson", "spatial-fields.yaml").results[0]
    assert isinstance(poisson, PoissonSolution)
    assert poisson.maximum_residual < 1e-12 and poisson.solution[2, 2] > 0.0
    assert create_official_pack_figure(vector) is not None
    assert create_official_pack_figure(poisson) is not None


def test_geometry_point_cloud_mesh_topology_and_path_are_executable():
    cloud = run("org.modellab.geometry.analyse-point-cloud", "geometry-mesh.yaml").results[0]
    assert isinstance(cloud, PointCloudAnalysis)
    assert cloud.affine_rank == 3 and cloud.diameter == pytest.approx(2**0.5)
    mesh = run("org.modellab.geometry.analyse-triangle-mesh", "geometry-mesh.yaml").results[0]
    assert isinstance(mesh, TriangleMeshAnalysis)
    assert mesh.orientation_consistent and mesh.vertex_manifold
    assert mesh.manifold_with_boundary and mesh.watertight_two_manifold and mesh.euler_characteristic == 2
    assert mesh.enclosed_volume == pytest.approx(1.0 / 6.0)
    path = run(
        "org.modellab.geometry.shortest-mesh-path", "geometry-mesh.yaml",
        {"source_vertex": 0, "target_vertex": 3},
    ).results[0]
    assert isinstance(path, MeshShortestPath)
    assert path.vertex_indices == (0, 3) and path.length == pytest.approx(1.0)
    assert create_official_pack_figure(cloud) is not None
    assert create_official_pack_figure(mesh) is not None


def test_material_and_truss_static_modal_analysis_are_executable():
    material = run("org.modellab.material.analyse-isotropic-elasticity", "mechanics-structure.yaml").results[0]
    assert isinstance(material, IsotropicMaterialAnalysis)
    assert material.shear_modulus == pytest.approx(210e9 / 2.6)
    static = run("org.modellab.mechanics.solve-truss-static", "mechanics-structure.yaml").results[0]
    assert isinstance(static, TrussStaticResult)
    assert static.maximum_displacement > 0.0 and static.strain_energy > 0.0
    np.testing.assert_allclose(np.sum(static.reactions + static.applied_forces, axis=0), 0.0, atol=1e-8)
    modal = run("org.modellab.mechanics.analyse-truss-modes", "mechanics-structure.yaml", {"modes": 3}).results[0]
    assert isinstance(modal, TrussModalAnalysis)
    assert modal.returned_modes == 3 and np.all(modal.frequencies_hz > 0.0)
    assert modal.mechanism_count == 0 and modal.negative_eigenvalue_count == 0
    assert create_official_pack_figure(material) is not None
    assert create_official_pack_figure(static) is not None
    assert create_official_pack_figure(modal) is not None


def test_statistical_data_regression_and_group_inference_are_executable():
    dataset = run("org.modellab.statistics.analyse-dataset", "statistical-data.yaml").results[0]
    assert isinstance(dataset, DatasetAnalysis)
    assert dataset.observation_count == 6 and dataset.column_count == 3
    np.testing.assert_allclose(np.diag(dataset.correlation), 1.0)
    fit = run("org.modellab.statistics.fit-linear-model", "statistical-data.yaml").results[0]
    assert isinstance(fit, LinearModelFit)
    np.testing.assert_allclose(fit.coefficients, [1.01454786, 2.00068062, -0.43390813], rtol=1e-7)
    assert fit.degrees_of_freedom == 5 and fit.r_squared > 0.999
    comparison = run("org.modellab.statistics.compare-groups", "statistical-data.yaml").results[0]
    assert isinstance(comparison, GroupComparison)
    assert comparison.anova_p < 1e-6 and len(comparison.pairwise_welch) == 3
    assert all(create_official_pack_figure(value) is not None for value in (dataset, fit, comparison))


def test_optimisation_estimation_and_inverse_problem_are_executable():
    optimum = run("org.modellab.optimisation.solve-nonlinear-problem", "optimisation-inverse.yaml").results[0]
    assert isinstance(optimum, OptimisationResult) and optimum.feasible
    np.testing.assert_allclose(optimum.optimum, [1.5, -1.5], atol=1e-7)
    estimate = run("org.modellab.estimation.fit-nonlinear-least-squares", "optimisation-inverse.yaml").results[0]
    assert isinstance(estimate, NonlinearLeastSquaresFit)
    np.testing.assert_allclose(estimate.estimate, [1.99, 1.06], atol=1e-8)
    inverse = run(
        "org.modellab.inverse.solve-linear-problem", "optimisation-inverse.yaml",
        {"regularisation": 0.01},
    ).results[0]
    assert isinstance(inverse, LinearInverseSolution)
    assert inverse.design_rank == 2 and inverse.weighted_residual_sum_squares < 3.0
    assert all(create_official_pack_figure(value) is not None for value in (optimum, estimate, inverse))


def test_electrical_electronic_and_electromagnetic_analyses_are_executable():
    dc = run("org.modellab.electrical.analyse-dc-circuit", "electrical-electromagnetic.yaml").results[0]
    assert isinstance(dc, CircuitDCAnalysis)
    np.testing.assert_allclose(dc.node_voltages, [0.0, 5.0, 10.0 / 3.0])
    assert abs(dc.total_element_power) < 1e-12 and dc.maximum_linear_residual < 1e-12
    ac = run("org.modellab.electrical.analyse-ac-circuit", "electrical-electromagnetic.yaml").results[0]
    assert isinstance(ac, CircuitACAnalysis) and ac.node_voltage_phasors.shape == (5, 3)
    diode = run(
        "org.modellab.electronics.evaluate-shockley-diode", "electrical-electromagnetic.yaml",
        {"samples": 101},
    ).results[0]
    assert isinstance(diode, DiodeCharacteristic) and np.all(np.diff(diode.currents) > 0.0)
    field = run("org.modellab.electromagnetics.analyse-point-charges", "electrical-electromagnetic.yaml").results[0]
    assert isinstance(field, ElectrostaticAnalysis) and field.net_charge == pytest.approx(0.0)
    assert all(create_official_pack_figure(value) is not None for value in (dc, ac, diode, field))


def test_chemical_reaction_and_biological_systems_are_executable():
    network = run("org.modellab.chemistry.analyse-reaction-network", "chemical-biological.yaml").results[0]
    assert isinstance(network, ReactionNetworkAnalysis)
    assert network.stoichiometric_rank == 2 and network.deficiency == 0 and network.conservation_laws.shape == (1, 3)
    reaction = run(
        "org.modellab.chemistry.simulate-reaction-network", "chemical-biological.yaml", {"samples": 101},
    ).results[0]
    assert isinstance(reaction, ReactionTrajectory)
    np.testing.assert_allclose(np.sum(reaction.concentrations, axis=1), 1000.0, atol=1e-6)
    assert reaction.concentration_unit == "mol/m^3" and reaction.source_units["concentration"] == "mol/L"
    compartment = run("org.modellab.biological.analyse-compartment-system", "chemical-biological.yaml").results[0]
    trajectory = run(
        "org.modellab.biological.simulate-compartment-system", "chemical-biological.yaml", {"samples": 101},
    ).results[0]
    assert isinstance(compartment, CompartmentAnalysis) and compartment.stability == "asymptotically stable"
    assert isinstance(trajectory, CompartmentTrajectory) and trajectory.amounts.shape == (101, 2)
    population = run("org.modellab.biological.analyse-population-system", "chemical-biological.yaml").results[0]
    population_trajectory = run(
        "org.modellab.biological.simulate-population-system", "chemical-biological.yaml", {"samples": 101},
    ).results[0]
    assert isinstance(population, PopulationAnalysis) and population.coexistence_feasible
    np.testing.assert_allclose(population.coexistence_equilibrium, [0.8, 0.4])
    assert isinstance(population_trajectory, PopulationTrajectory) and np.all(population_trajectory.values >= 0.0)
    assert all(create_official_pack_figure(value) is not None for value in (network, reaction, compartment, trajectory, population, population_trajectory))


def test_semantic_validation_rejects_non_stochastic_transition():
    source = """
name: invalid chain
objects:
  chain:
    kind: org.modellab.probability.markov-chain
    kind_version: '1.0'
    properties:
      states: [a, b]
      initial: [1, 0]
      transition: [[0.8, 0.8], [0, 1]]
"""
    with pytest.raises(ModelValidationError, match="sum to one"):
        validate_model(parse_model_text(source))


@pytest.mark.parametrize(
    ("source", "message"),
    (
        (
            """
name: invalid statistical data
objects:
  data:
    kind: org.modellab.statistics.dataset
    kind_version: '1.0'
    properties: {columns: [x, y], data: [[1, 2], [3]]}
""",
            "rectangular",
        ),
        (
            """
name: invalid optimisation expression
objects:
  problem:
    kind: org.modellab.optimisation.nonlinear-problem
    kind_version: '1.0'
    properties: {variables: [x], initial: [0], bounds: [[-1, 1]], objective: x + z}
""",
            "Undefined symbol",
        ),
        (
            """
name: invalid circuit endpoint
objects:
  circuit:
    kind: org.modellab.electrical.linear-circuit
    kind_version: '1.1'
    properties:
      nodes: [ground, a]
      ground: ground
      elements: [{id: R, type: resistor, from: a, to: missing, value: 10}]
""",
            "distinct declared nodes",
        ),
        (
            """
name: invalid reaction species
objects:
  network:
    kind: org.modellab.chemistry.mass-action-network
    kind_version: '1.1'
    properties:
      species: [A]
      initial_concentrations: [1]
      time_span: [0, 1]
      reactions:
        - id: reaction
          reactants: [{species: missing, stoichiometry: 1}]
          products: [{species: A, stoichiometry: 1}]
          rate_constant: 1
""",
            "declared species",
        ),
    ),
)
def test_v115_pack_semantic_validation_is_strict(source, message):
    with pytest.raises(ModelValidationError, match=message):
        validate_model(parse_model_text(source))


def test_unknown_third_party_kind_remains_opaque():
    source = """
name: unknown extension
objects:
  object-a:
    kind: org.example.future.object
    kind_version: '9.0'
    properties: {meaning: preserved}
"""
    compiled = validate_model(parse_model_text(source))
    assert compiled.graph.object("object-a").opaque
    assert not compiled.graph.object("object-a").executable


def test_official_pack_run_freezes_and_reproduces_through_generic_protocol():
    source = (ROOT / "models" / "probability.yaml").read_text(encoding="utf-8")
    compiled = validate_model(parse_model_text(source))
    outcome = run_registry.run(
        "org.modellab.probability.simulate-markov-chain", compiled,
        {"steps": 8, "trajectories": 300, "seed": 17},
    )
    state = create_run_experiment_state(
        model_source=source, model=compiled, parameter_values={}, run_outcomes=(outcome,),
    )
    reproduced = reproduce_run_experiment(state, model=compiled)
    assert reproduced.report.status is ReproductionStatus.EXACT
    assert reproduced.run_outcomes[0].artifacts[0].artifact_sha256 == outcome.artifacts[0].artifact_sha256


@pytest.mark.parametrize(
    ("source_name", "capability_id", "settings"),
    (
        ("dynamics-control.yaml", "org.modellab.dynamics.integrate-ode", {"samples": 31}),
        ("spatial-fields.yaml", "org.modellab.pde.solve-poisson", {}),
        ("geometry-mesh.yaml", "org.modellab.geometry.analyse-triangle-mesh", {}),
        ("mechanics-structure.yaml", "org.modellab.mechanics.solve-truss-static", {}),
        ("statistical-data.yaml", "org.modellab.statistics.fit-linear-model", {}),
        ("optimisation-inverse.yaml", "org.modellab.inverse.solve-linear-problem", {"regularisation": 0.01}),
        ("electrical-electromagnetic.yaml", "org.modellab.electrical.analyse-ac-circuit", {}),
        ("chemical-biological.yaml", "org.modellab.chemistry.simulate-reaction-network", {"samples": 31}),
    ),
)
def test_later_pack_artifacts_freeze_and_reproduce_exactly(source_name, capability_id, settings):
    source = (ROOT / "models" / source_name).read_text(encoding="utf-8")
    compiled = validate_model(parse_model_text(source))
    outcome = run_registry.run(capability_id, compiled, settings)
    state = create_run_experiment_state(
        model_source=source, model=compiled, parameter_values={}, run_outcomes=(outcome,),
    )
    reproduced = reproduce_run_experiment(state, model=compiled)
    assert reproduced.report.status is ReproductionStatus.EXACT
    assert reproduced.run_outcomes[0].artifacts[0].artifact_sha256 == outcome.artifacts[0].artifact_sha256


def test_interpreter_expands_registry_only_for_official_pack_authoring():
    core = create_interpreter_context(
        instruction="x in [-2,2], f=x**2", current_model_source=""
    )
    official = create_interpreter_context(
        instruction="Create a two-state active inference model.", current_model_source=""
    )
    core_kinds = {item["kind"] for item in core["contract"]["installed_model_object_kinds"]}
    official_kinds = {item["kind"] for item in official["contract"]["installed_model_object_kinds"]}
    assert len(core_kinds) == 9
    assert "org.modellab.generative.active-inference-model" in official_kinds
    assert "org.modellab.electrical.linear-circuit" not in official_kinds
    assert "org.modellab.learning.feedforward-network" not in official_kinds
    assert len(official_kinds) > len(core_kinds)
    assert len(json.dumps(official, ensure_ascii=False).encode("utf-8")) <= MAX_CONTEXT_PACKAGE_BYTES


def test_interpreter_routes_later_pack_schemas_without_exceeding_context_budget():
    cases = {
        "Create a damped ODE system.": "org.modellab.dynamics.ode-system",
        "Create a Poisson equation model.": "org.modellab.pde.poisson-problem",
        "Create a triangle mesh.": "org.modellab.geometry.triangle-mesh",
        "Create a truss structure.": "org.modellab.mechanics.truss-structure",
        "Create a linear regression study.": "org.modellab.statistics.linear-model-study",
        "Create a constrained optimisation problem.": "org.modellab.optimisation.nonlinear-problem",
        "Create an AC electrical circuit.": "org.modellab.electrical.linear-circuit",
        "Create a mass action reaction network.": "org.modellab.chemistry.mass-action-network",
        "Create a machine learning classification study.": "org.modellab.learning.supervised-study",
    }
    for instruction, expected_kind in cases.items():
        context = create_interpreter_context(instruction=instruction, current_model_source="")
        kinds = {item["kind"] for item in context["contract"]["installed_model_object_kinds"]}
        assert expected_kind in kinds
        assert len(
            json.dumps(
                context,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ) <= MAX_CONTEXT_PACKAGE_BYTES


def test_reproduction_report_distinguishes_statistical_equivalence():
    source = (ROOT / "models" / "probability.yaml").read_text(encoding="utf-8")
    compiled = validate_model(parse_model_text(source))
    settings = {"steps": 20, "trajectories": 1000, "seed": 17}
    reference = run_registry.run(
        "org.modellab.probability.simulate-markov-chain", compiled, settings,
    )
    state = create_run_experiment_state(
        model_source=source, model=compiled, parameter_values={}, run_outcomes=(reference,),
    )
    descriptor = next(
        item for item in CAPABILITY_DESCRIPTORS
        if item.identifier == "org.modellab.probability.simulate-markov-chain"
    )

    def shifted_stream(model, **values):
        return simulate_markov_chain(model, **{**values, "seed": int(values.get("seed", 0)) + 1})

    alternative = CapabilityPackRegistry((replace(descriptor, runner=shifted_stream),))
    reproduced = reproduce_run_experiment(state, model=compiled, registry=alternative)
    assert reproduced.report.status is ReproductionStatus.STATISTICAL
    assert reproduced.report.results[0].status is ReproductionStatus.STATISTICAL
    assert reproduced.report.results[0].comparison_kind == "org.modellab.comparator.stochastic"
