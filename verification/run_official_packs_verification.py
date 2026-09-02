"""Independent release checks for the current Model Laboratory official packs."""

from __future__ import annotations

import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from model_lab import __version__
from model_lab.builtin_packs import run_registry
from model_lab.experiment import create_run_experiment_state
from model_lab.interpreter import MAX_CONTEXT_PACKAGE_BYTES, create_interpreter_context
from model_lab.official_packs import (
    OFFICIAL_CAPABILITY_DESCRIPTORS,
    OFFICIAL_KIND_DESCRIPTORS,
    OFFICIAL_PACK_MANIFESTS,
)
from model_lab.official_packs.renderers import create_official_pack_figure
from model_lab.parser import parse_model_text
from model_lab.protocol import comparator_registry
from model_lab.reproduction import ReproductionStatus, reproduce_run_experiment
from model_lab.validator import validate_model


checks: list[dict[str, object]] = []


def check(name: str, condition: bool, evidence: object) -> None:
    checks.append({"name": name, "passed": bool(condition), "evidence": evidence})


def compiled(name: str):
    source = (ROOT / "models" / name).read_text(encoding="utf-8")
    return source, validate_model(parse_model_text(source))


check("official manifest count", len(OFFICIAL_PACK_MANIFESTS) == 13, len(OFFICIAL_PACK_MANIFESTS))
check("official kind count", len(OFFICIAL_KIND_DESCRIPTORS) == 34, len(OFFICIAL_KIND_DESCRIPTORS))
official_capability_count = len(OFFICIAL_CAPABILITY_DESCRIPTORS)
check("official capability count", official_capability_count == 45, official_capability_count)

multi_source, multi = compiled("multidimensional.yaml")
array = run_registry.run("org.modellab.multidimensional.analyse-array", multi).results[0]
check("matrix rank", array.matrix_rank == 2, array.matrix_rank)
tensor = run_registry.run("org.modellab.multidimensional.contract-tensors", multi, {"left_object_id": "response-matrix", "right_object_id": "projection", "left_axes": [1], "right_axes": [0]}).results[0]
check("tensor contraction", np.allclose(tensor.values, [-0.5, 0.25]), tensor.values.tolist())
quantities = run_registry.run("org.modellab.multidimensional.normalise-quantities", multi).results[0]
check("quantity conversion", [row["base_value"] for row in quantities.quantities] == [1.25, 1.25], list(quantities.quantities))

probability_source, probability = compiled("probability.yaml")
distribution = run_registry.run("org.modellab.probability.analyse-distribution", probability).results[0]
check("distribution variance", np.isclose(distribution.variance, 0.4), distribution.variance)
evolution = run_registry.run("org.modellab.probability.evolve-markov-chain", probability, {"steps": 20}).results[0]
check("Markov normalisation", np.allclose(evolution.distributions.sum(axis=1), 1.0), evolution.stationarity_residual)
simulation = run_registry.run("org.modellab.probability.simulate-markov-chain", probability, {"steps": 20, "trajectories": 1000, "seed": 17})
check("stochastic reference", simulation.results[0].stochastic_reference["schema"] == "model-laboratory-stochastic-reference", simulation.artifacts[0].artifact_sha256)
alternative_stream = run_registry.run("org.modellab.probability.simulate-markov-chain", probability, {"steps": 20, "trajectories": 1000, "seed": 18})
statistical = comparator_registry.compare("org.modellab.comparator.stochastic", simulation.artifacts[0], alternative_stream.artifacts[0])
check("statistical equivalence", statistical.reproduced and statistical.details["status"] == "STATISTICALLY EQUIVALENT", statistical.details["status"])
state = create_run_experiment_state(model_source=probability_source, model=probability, parameter_values={}, run_outcomes=(simulation,))
reproduced = reproduce_run_experiment(state, model=probability)
check("generic exact reproduction", reproduced.report.status is ReproductionStatus.EXACT, reproduced.report.status.value)

_, network = compiled("network.yaml")
network_result = run_registry.run("org.modellab.graph.analyse-network", network, {"source": "hypothesis", "target": "conclusion"}).results[0]
check("network shortest path", network_result.shortest_path_length == 3.0, network_result.shortest_path)
check("network DAG", network_result.is_dag, network_result.has_cycle)

generative_source, generative = compiled("generative-systems.yaml")
hmm = run_registry.run("org.modellab.generative.infer-hidden-markov-model", generative, {"observed_sequence": ["quiet", "alarm", "alarm"]}).results[0]
check("HMM posterior normalisation", np.allclose(hmm.smoothed_probabilities.sum(axis=1), 1.0), hmm.viterbi_path)
decision = run_registry.run("org.modellab.generative.solve-pomdp", generative, {"horizon": 4}).results[0]
check("POMDP belief plan", bool(decision.observation_contingencies), decision.recommended_action)
blanket = run_registry.run("org.modellab.generative.markov-blanket", generative, {"network_object_id": "causal-network", "target": "state"}).results[0]
check("Markov blanket", set(blanket.blanket) == {"action", "observation"}, blanket.blanket)
active = run_registry.run("org.modellab.generative.evaluate-active-inference", generative).results[0]
check("active-inference posterior", np.isclose(active.policy_posterior.sum(), 1.0), active.policy_posterior.tolist())
check("active-inference event probability", all(0.0 <= row["reach_probability"] <= 1.0 for row in active.event_probabilities), active.event_probabilities)

_, dynamics = compiled("dynamics-control.yaml")
ode = run_registry.run("org.modellab.dynamics.integrate-ode", dynamics, {"samples": 101}).results[0]
check("ODE trajectory", ode.states.shape == (101, 2) and np.all(np.isfinite(ode.states)), ode.solver_record["presentation"]["function_evaluations"])
equilibria = run_registry.run("org.modellab.dynamics.analyse-equilibria", dynamics).results[0]
check("ODE equilibrium stability", len(equilibria.points) == 1 and equilibria.points[0].classification == "asymptotically stable", [point.classification for point in equilibria.points])
state_space = run_registry.run("org.modellab.control.analyse-state-space", dynamics).results[0]
check("state-space control structure", state_space.controllable and state_space.observable, [state_space.controllability_rank, state_space.observability_rank])
control_response = run_registry.run("org.modellab.control.simulate-state-space", dynamics, {"samples": 101, "constant_input": [1.0]}).results[0]
check("state-space response", control_response.outputs.shape == (101, 2), control_response.solver_record["presentation"]["function_evaluations"])

_, fields = compiled("spatial-fields.yaml")
scalar_field = run_registry.run("org.modellab.field.analyse-scalar-field", fields).results[0]
check("scalar-field differential operators", np.isclose(scalar_field.laplacian[2, 2], 4.0), scalar_field.laplacian[2, 2])
vector_field = run_registry.run("org.modellab.field.analyse-vector-field", fields).results[0]
check("vector-field divergence and curl", np.allclose(vector_field.divergence, 0.0) and np.allclose(vector_field.curl, 2.0), [vector_field.mean_divergence, float(np.mean(vector_field.curl))])
diffusion = run_registry.run("org.modellab.pde.solve-diffusion", fields, {"time_samples": 31}).results[0]
check("diffusion boundary", np.allclose(diffusion.values[:, [0, -1]], 0.0), diffusion.values[-1, [0, -1]].tolist())
poisson = run_registry.run("org.modellab.pde.solve-poisson", fields).results[0]
check("Poisson residual", poisson.maximum_residual < 1e-12, poisson.maximum_residual)

_, geometry = compiled("geometry-mesh.yaml")
cloud = run_registry.run("org.modellab.geometry.analyse-point-cloud", geometry).results[0]
check("point-cloud rank and diameter", cloud.affine_rank == 3 and np.isclose(cloud.diameter, np.sqrt(2.0)), [cloud.affine_rank, cloud.diameter])
mesh = run_registry.run("org.modellab.geometry.analyse-triangle-mesh", geometry).results[0]
check("mesh topology and volume", mesh.watertight_two_manifold and mesh.euler_characteristic == 2 and np.isclose(mesh.enclosed_volume, 1.0 / 6.0), [mesh.euler_characteristic, mesh.enclosed_volume])
mesh_path = run_registry.run("org.modellab.geometry.shortest-mesh-path", geometry, {"source_vertex": 0, "target_vertex": 3}).results[0]
check("mesh shortest path", mesh_path.vertex_indices == (0, 3) and np.isclose(mesh_path.length, 1.0), mesh_path.vertex_indices)

_, mechanics = compiled("mechanics-structure.yaml")
material = run_registry.run("org.modellab.material.analyse-isotropic-elasticity", mechanics).results[0]
check("material constitutive analysis", np.isclose(material.shear_modulus, 210e9 / 2.6), material.shear_modulus)
static = run_registry.run("org.modellab.mechanics.solve-truss-static", mechanics).results[0]
check("truss force balance", np.allclose(np.sum(static.reactions + static.applied_forces, axis=0), 0.0, atol=1e-8), np.sum(static.reactions + static.applied_forces, axis=0).tolist())
modal = run_registry.run("org.modellab.mechanics.analyse-truss-modes", mechanics, {"modes": 3}).results[0]
check("truss modes", modal.returned_modes == 3 and np.all(modal.frequencies_hz > 0.0), modal.frequencies_hz.tolist())

_, statistical = compiled("statistical-data.yaml")
dataset = run_registry.run("org.modellab.statistics.analyse-dataset", statistical).results[0]
check("dataset covariance and PCA", dataset.observation_count == 6 and np.isclose(np.sum(dataset.explained_variance_ratio), 1.0), dataset.explained_variance_ratio.tolist())
linear_fit = run_registry.run("org.modellab.statistics.fit-linear-model", statistical).results[0]
check("linear-model inference", linear_fit.design_rank == 3 and linear_fit.r_squared > 0.999, [linear_fit.design_rank, linear_fit.r_squared])
groups = run_registry.run("org.modellab.statistics.compare-groups", statistical).results[0]
check("group inference", groups.anova_p < 1e-6 and len(groups.pairwise_welch) == 3, groups.anova_p)

_, optimisation = compiled("optimisation-inverse.yaml")
optimum = run_registry.run("org.modellab.optimisation.solve-nonlinear-problem", optimisation).results[0]
check("constrained optimum", optimum.feasible and np.allclose(optimum.optimum, [1.5, -1.5]), [optimum.optimum.tolist(), optimum.maximum_constraint_violation])
nonlinear_fit = run_registry.run("org.modellab.estimation.fit-nonlinear-least-squares", optimisation).results[0]
check("nonlinear parameter estimation", np.allclose(nonlinear_fit.estimate, [1.99, 1.06]), nonlinear_fit.estimate.tolist())
inverse = run_registry.run("org.modellab.inverse.solve-linear-problem", optimisation, {"regularisation": 0.01}).results[0]
check("weighted inverse solution", inverse.design_rank == 2 and inverse.weighted_residual_sum_squares < 3.0, [inverse.design_rank, inverse.weighted_residual_sum_squares])

_, electrical = compiled("electrical-electromagnetic.yaml")
dc = run_registry.run("org.modellab.electrical.analyse-dc-circuit", electrical).results[0]
check("DC modified-nodal solution", np.allclose(dc.node_voltages, [0.0, 5.0, 10.0 / 3.0]) and dc.maximum_linear_residual < 1e-12, dc.node_voltages.tolist())
ac = run_registry.run("org.modellab.electrical.analyse-ac-circuit", electrical).results[0]
check("AC phasor sweep", ac.node_voltage_phasors.shape == (5, 3) and ac.maximum_linear_residual < 1e-12, list(ac.node_voltage_phasors.shape))
diode = run_registry.run("org.modellab.electronics.evaluate-shockley-diode", electrical, {"samples": 101}).results[0]
check("Shockley diode characteristic", np.all(np.diff(diode.currents) > 0.0), [float(diode.currents[0]), float(diode.currents[-1])])
electrostatic = run_registry.run("org.modellab.electromagnetics.analyse-point-charges", electrical).results[0]
check("electrostatic dipole", np.isclose(electrostatic.net_charge, 0.0) and electrostatic.electric_field.shape == (6, 2), electrostatic.dipole_moment.tolist())

_, chemical = compiled("chemical-biological.yaml")
reaction_structure = run_registry.run("org.modellab.chemistry.analyse-reaction-network", chemical).results[0]
check("reaction-network structure", reaction_structure.stoichiometric_rank == 2 and reaction_structure.deficiency == 0, [reaction_structure.stoichiometric_rank, reaction_structure.deficiency])
reaction_trajectory = run_registry.run("org.modellab.chemistry.simulate-reaction-network", chemical, {"samples": 101}).results[0]
check("mass-action conservation", np.allclose(np.sum(reaction_trajectory.concentrations, axis=1), 1000.0, atol=1e-6), reaction_trajectory.concentrations[-1].tolist())
compartment = run_registry.run("org.modellab.biological.analyse-compartment-system", chemical).results[0]
check("compartment stability", compartment.stability == "asymptotically stable", compartment.spectral_abscissa)
compartment_trajectory = run_registry.run("org.modellab.biological.simulate-compartment-system", chemical, {"samples": 101}).results[0]
check("compartment trajectory", compartment_trajectory.amounts.shape == (101, 2) and np.all(compartment_trajectory.amounts >= 0.0), compartment_trajectory.amounts[-1].tolist())
population = run_registry.run("org.modellab.biological.analyse-population-system", chemical).results[0]
check("population coexistence", population.coexistence_feasible and np.allclose(population.coexistence_equilibrium, [0.8, 0.4]), population.coexistence_stability)
population_trajectory = run_registry.run("org.modellab.biological.simulate-population-system", chemical, {"samples": 101}).results[0]
check("population trajectory", population_trajectory.values.shape == (101, 2) and np.all(population_trajectory.values >= 0.0), population_trajectory.values[-1].tolist())

_, learning = compiled("machine-learning.yaml")
supervised = run_registry.run("org.modellab.learning.fit-supervised-model", learning).results[0]
check("supervised learning", supervised.metrics["accuracy"] == 1.0, supervised.metrics)
clustering = run_registry.run("org.modellab.learning.cluster-kmeans", learning, {"clusters": 2, "seed": 17}).results[0]
check("seeded canonical clustering", tuple(clustering.cluster_sizes) == (4, 4), clustering.cluster_sizes.tolist())
neural = run_registry.run("org.modellab.learning.evaluate-feedforward-network", learning).results[0]
check("feed-forward inference", neural.outputs.shape == (8, 2) and np.allclose(neural.outputs.sum(axis=1), 1.0), neural.outputs.tolist())
fuzzy = run_registry.run("org.modellab.intelligence.evaluate-fuzzy-system", learning, {"points": [[1, 1], [5, 5], [9, 9]]}).results[0]
check("fuzzy inference", np.all(np.diff(fuzzy.outputs) > 0.0), fuzzy.outputs.tolist())

rendered = [
    array, evolution, network_result, hmm, decision, blanket, active,
    ode, equilibria, state_space, control_response, scalar_field, vector_field,
    diffusion, poisson, cloud, mesh, mesh_path, material, static, modal,
    dataset, linear_fit, groups, optimum, nonlinear_fit, inverse, dc, ac, diode,
    electrostatic, reaction_structure, reaction_trajectory, compartment,
    compartment_trajectory, population, population_trajectory,
    supervised, clustering, neural, fuzzy,
]
check("official renderers", all(create_official_pack_figure(item) is not None for item in rendered), len(rendered))
contexts = [
    create_interpreter_context(instruction="Create a two-state active inference model.", current_model_source=""),
    create_interpreter_context(instruction="Create a damped ODE system.", current_model_source=""),
    create_interpreter_context(instruction="Create a Poisson equation model.", current_model_source=""),
    create_interpreter_context(instruction="Create a triangle mesh.", current_model_source=""),
    create_interpreter_context(instruction="Create a truss structure.", current_model_source=""),
    create_interpreter_context(instruction="Create a linear regression study.", current_model_source=""),
    create_interpreter_context(instruction="Create a constrained optimisation problem.", current_model_source=""),
    create_interpreter_context(instruction="Create an AC electrical circuit.", current_model_source=""),
    create_interpreter_context(instruction="Create a mass action reaction network.", current_model_source=""),
    create_interpreter_context(instruction="Create a machine learning classification study.", current_model_source=""),
]
context_bytes = [
    len(
        json.dumps(
            context,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )
    for context in contexts
]
check("interpreter context budget", max(context_bytes) <= MAX_CONTEXT_PACKAGE_BYTES, context_bytes)

passed = sum(item["passed"] is True for item in checks)
report = {"suite": "official-packs", "version": __version__, "passed": passed, "total": len(checks), "checks": checks}
rendered_report = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
(ROOT / "verification" / "official-packs-report.json").write_text(rendered_report, encoding="utf-8")
print(rendered_report, end="")
raise SystemExit(0 if passed == len(checks) else 1)
