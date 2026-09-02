#!/usr/bin/env python3
"""Build the frozen v1.3 development benchmark with complete official-kind coverage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "verification" / "interpreter_baseline_v1.2.json"
TARGET = ROOT / "verification" / "interpreter_baseline_v1.3.json"


def _case(
    identifier: str,
    family: str,
    instruction: str,
    model_name: str,
    object_id: str,
    kind: str,
    kind_version: str,
    properties: Mapping[str, Any],
) -> dict[str, Any]:
    source = yaml.safe_dump(
        {
            "name": model_name,
            "objects": {
                object_id: {
                    "kind": kind,
                    "kind_version": kind_version,
                    "properties": dict(properties),
                }
            },
        },
        sort_keys=False,
        allow_unicode=True,
        width=1000,
    )
    return {
        "id": identifier,
        "family": family,
        "instruction": instruction,
        "current_model_source": "",
        "expected": {
            "terminal_status": "proposal",
            "first_action": "propose_edits",
            "model_source": source,
        },
        "notes": "Official-pack authoring coverage; excluded from all interpreter training corpora.",
    }


OFFICIAL_CASES = [
    _case(
        "official-array-matrix", "official-multidimensional",
        "Create a 3x3 identity matrix called basis in a model named Matrix benchmark. Store its nine row-major entries, name every row and column a, b, c, and treat the entries as dimensionless.",
        "Matrix benchmark", "basis", "org.modellab.multidimensional.array", "1.1",
        {"shape": [3, 3], "values": [1, 0, 0, 0, 1, 0, 0, 0, 1], "axis_labels": [["a", "b", "c"], ["a", "b", "c"]]},
    ),
    _case(
        "official-scientific-quantity", "official-multidimensional",
        "In a model named Quantity benchmark, record object span as 12.5 millimetres with 0.1 mm standard uncertainty. It is a length, and one source unit equals 0.001 base metres.",
        "Quantity benchmark", "span", "org.modellab.multidimensional.quantity", "1.1",
        {"value": 12.5, "unit": "mm", "dimension_exponents": [1, 0, 0, 0, 0, 0, 0], "scale_to_base": 0.001, "standard_uncertainty": 0.1},
    ),
    _case(
        "official-discrete-distribution", "official-probability",
        "Build Probability benchmark with a finite distribution object named demand: low, normal and peak have probabilities 0.1, 0.7 and 0.2, and numeric values 1, 2 and 4.",
        "Probability benchmark", "demand", "org.modellab.probability.discrete-distribution", "1.0",
        {"outcomes": ["low", "normal", "peak"], "probabilities": [0.1, 0.7, 0.2], "numeric_values": [1, 2, 4]},
    ),
    _case(
        "official-markov-chain", "official-probability",
        "Create Markov benchmark. Object machine has states running and stopped, starts with probabilities 0.9 and 0.1, and uses transition rows [0.95,0.05] and [0.4,0.6].",
        "Markov benchmark", "machine", "org.modellab.probability.markov-chain", "1.0",
        {"states": ["running", "stopped"], "initial": [0.9, 0.1], "transition": [[0.95, 0.05], [0.4, 0.6]]},
    ),
    _case(
        "official-graph-network", "official-graphs",
        "Create a directed graph in Graph benchmark called workflow. Its nodes are ingest, check and publish; add weighted arcs ingest to check with 2 and check to publish with 3.",
        "Graph benchmark", "workflow", "org.modellab.graph.network", "1.0",
        {"nodes": ["ingest", "check", "publish"], "directed": True, "edges": [{"source": "ingest", "target": "check", "weight": 2.0}, {"source": "check", "target": "publish", "weight": 3.0}]},
    ),
    _case(
        "official-hmm", "official-generative",
        "Create an HMM named sensor in HMM benchmark. Hidden states cool/hot and observations green/red start at [0.7,0.3], transition by [[0.8,0.2],[0.1,0.9]], and emit by [[0.85,0.15],[0.25,0.75]].",
        "HMM benchmark", "sensor", "org.modellab.generative.hidden-markov-model", "1.0",
        {"states": ["cool", "hot"], "observations": ["green", "red"], "initial": [0.7, 0.3], "transition": [[0.8, 0.2], [0.1, 0.9]], "emission": [[0.85, 0.15], [0.25, 0.75]]},
    ),
    _case(
        "official-pomdp", "official-generative",
        "For POMDP benchmark, define controller with states stable/failed, observations ok/alert, actions hold/reset, initial [0.9,0.1], hold and reset transition matrices [[[0.9,0.1],[0.3,0.7]],[[0.99,0.01],[0.9,0.1]]], emissions [[0.95,0.05],[0.2,0.8]], rewards [[3,0],[-9,-2]], and discount 0.92.",
        "POMDP benchmark", "controller", "org.modellab.generative.pomdp", "1.0",
        {"states": ["stable", "failed"], "observations": ["ok", "alert"], "actions": ["hold", "reset"], "initial": [0.9, 0.1], "transitions": [[[0.9, 0.1], [0.3, 0.7]], [[0.99, 0.01], [0.9, 0.1]]], "emissions": [[0.95, 0.05], [0.2, 0.8]], "rewards": [[3, 0], [-9, -2]], "discount": 0.92},
    ),
    _case(
        "official-active-inference", "official-generative",
        "Make Active inference benchmark with object agent. Use states calm/threat, observations quiet/siren, actions observe/respond, prior [0.65,0.35], transition models [[[0.85,0.15],[0.25,0.75]],[[0.97,0.03],[0.8,0.2]]], likelihood [[0.9,0.1],[0.05,0.95]], preferences [1,-5], and policies watch=[observe,observe] and act=[respond,observe].",
        "Active inference benchmark", "agent", "org.modellab.generative.active-inference-model", "1.0",
        {"states": ["calm", "threat"], "observations": ["quiet", "siren"], "actions": ["observe", "respond"], "initial": [0.65, 0.35], "transitions": [[[0.85, 0.15], [0.25, 0.75]], [[0.97, 0.03], [0.8, 0.2]]], "likelihood": [[0.9, 0.1], [0.05, 0.95]], "preferences": [1, -5], "policies": [{"name": "watch", "actions": ["observe", "observe"]}, {"name": "act", "actions": ["respond", "observe"]}]},
    ),
    _case(
        "official-ode-system", "official-dynamics",
        "In ODE benchmark create oscillator: states q and v begin at 0.5 and 0 over time 0 to 6; derivatives are v and -2*q-0.4*v.",
        "ODE benchmark", "oscillator", "org.modellab.dynamics.ode-system", "1.0",
        {"states": ["q", "v"], "initial_state": [0.5, 0.0], "time_span": [0.0, 6.0], "equations": ["v", "-2*q - 0.4*v"]},
    ),
    _case(
        "official-state-space", "official-dynamics",
        "Create Control benchmark with state-space object plant: states q/v, input u, output q, A=[[0,1],[-2,-0.4]], B=[[0],[1]], C=[[1,0]], D=[[0]], initial state [0.5,0].",
        "Control benchmark", "plant", "org.modellab.control.state-space-system", "1.0",
        {"states": ["q", "v"], "inputs": ["u"], "outputs": ["q"], "A": [[0.0, 1.0], [-2.0, -0.4]], "B": [[0.0], [1.0]], "C": [[1.0, 0.0]], "D": [[0.0]], "initial_state": [0.5, 0.0]},
    ),
    _case(
        "official-scalar-field", "official-fields",
        "Create Scalar field benchmark object temperature. Coordinates x=[0,1,2] metres and y=[0,10,20] centimetres; its kelvin grid is [[290,291,292],[293,294,295],[296,297,298]].",
        "Scalar field benchmark", "temperature", "org.modellab.field.structured-scalar-field", "1.1",
        {"axes": [{"name": "x", "coordinates": [0, 1, 2], "unit": "m"}, {"name": "y", "coordinates": [0, 10, 20], "unit": "cm"}], "values": [[290, 291, 292], [293, 294, 295], [296, 297, 298]], "value_name": "temperature", "value_unit": "K"},
    ),
    _case(
        "official-vector-field", "official-fields",
        "In Vector field benchmark add flow on metre axes x,y=[-1,0,1]. Components ux and uy in m/s have grids [[[-1,0,1],[-1,0,1],[-1,0,1]],[[1,1,1],[0,0,0],[-1,-1,-1]]].",
        "Vector field benchmark", "flow", "org.modellab.field.structured-vector-field", "1.1",
        {"axes": [{"name": "x", "coordinates": [-1, 0, 1], "unit": "m"}, {"name": "y", "coordinates": [-1, 0, 1], "unit": "m"}], "component_names": ["ux", "uy"], "values": [[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]], "value_unit": "m/s"},
    ),
    _case(
        "official-diffusion-pde", "official-fields",
        "Create a PDE for heat conduction in Diffusion benchmark. Object rod uses x=[0,.2,.4,.6,.8,1], initial temperature [0,.5,.9,.9,.5,0], diffusivity .03, time 0 to 4, and zero Dirichlet values at both ends.",
        "Diffusion benchmark", "rod", "org.modellab.pde.diffusion-problem", "1.0",
        {"coordinate": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "initial_values": [0.0, 0.5, 0.9, 0.9, 0.5, 0.0], "diffusivity": 0.03, "time_span": [0.0, 4.0], "dirichlet_boundary": {"left": 0.0, "right": 0.0}, "value_name": "temperature"},
    ),
    _case(
        "official-poisson-pde", "official-fields",
        "Build Poisson benchmark object plate on x=y=[0,.25,.5,.75,1]. Use a 5 by 5 source grid filled with 3 and declare all four Dirichlet edges zero.",
        "Poisson benchmark", "plate", "org.modellab.pde.poisson-problem", "1.0",
        {"x_coordinates": [0, 0.25, 0.5, 0.75, 1], "y_coordinates": [0, 0.25, 0.5, 0.75, 1], "source": [[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]], "dirichlet_boundary": {"left": 0, "right": 0, "bottom": 0, "top": 0}},
    ),
    _case(
        "official-point-cloud", "official-geometry",
        "In Point cloud benchmark create survey with metre points (0,0,0), (2,0,0), (0,3,0), (0,0,4), labelled origin, east, north and up.",
        "Point cloud benchmark", "survey", "org.modellab.geometry.point-cloud", "1.1",
        {"points": [[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]], "labels": ["origin", "east", "north", "up"], "coordinate_unit": "m"},
    ),
    _case(
        "official-triangle-mesh", "official-geometry",
        "Create Mesh benchmark object pyramid as a centimetre tetrahedral surface. Vertices are (0,0,0),(200,0,0),(0,200,0),(0,0,200); oriented faces are (0,2,1),(0,1,3),(0,3,2),(1,2,3).",
        "Mesh benchmark", "pyramid", "org.modellab.geometry.triangle-mesh", "1.1",
        {"vertices": [[0, 0, 0], [200, 0, 0], [0, 200, 0], [0, 0, 200]], "faces": [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], "coordinate_unit": "cm"},
    ),
    _case(
        "official-elastic-material", "official-mechanics",
        "Record aluminium in Material benchmark as isotropic elastic object alloy: Young's modulus 69 GPa, Poisson ratio .32, density 2.68 g/cm^3 and yield strength 270 MPa.",
        "Material benchmark", "alloy", "org.modellab.material.isotropic-linear-elastic", "1.1",
        {"youngs_modulus": 69, "poissons_ratio": 0.32, "density": 2.68, "yield_strength": 0.27, "modulus_unit": "GPa", "density_unit": "g/cm^3"},
    ),
    _case(
        "official-truss", "official-mechanics",
        "Create Truss benchmark object frame in 2D metres, newtons and pascals. Nodes A=(0,0), B=(3,0), C=(1.5,1.5); bars AB, AC, BC each have area .002 m2 and E=200e9 Pa. Fix A in x/y, B in y, and add service load (1000,-5000) N at C.",
        "Truss benchmark", "frame", "org.modellab.mechanics.truss-structure", "1.1",
        {"dimension": 2, "coordinate_unit": "m", "force_unit": "N", "stress_unit": "Pa", "nodes": [{"id": "A", "coordinates": [0, 0]}, {"id": "B", "coordinates": [3, 0]}, {"id": "C", "coordinates": [1.5, 1.5]}], "elements": [{"id": "AB", "start": "A", "end": "B", "area": 0.002, "youngs_modulus": 200000000000.0}, {"id": "AC", "start": "A", "end": "C", "area": 0.002, "youngs_modulus": 200000000000.0}, {"id": "BC", "start": "B", "end": "C", "area": 0.002, "youngs_modulus": 200000000000.0}], "supports": [{"node": "A", "fixed": [True, True]}, {"node": "B", "fixed": [False, True]}], "load_cases": [{"name": "service", "loads": [{"node": "C", "force": [1000, -5000]}]}]},
    ),
    _case(
        "official-statistics-dataset", "official-statistics",
        "Create Data benchmark with dataset measurements. Columns distance and duration use km and h, and rows are [1,0.2],[2,0.35],[4,0.7],[8,1.25],[10,1.6].",
        "Data benchmark", "measurements", "org.modellab.statistics.dataset", "1.0",
        {"columns": ["distance", "duration"], "column_units": ["km", "h"], "data": [[1, 0.2], [2, 0.35], [4, 0.7], [8, 1.25], [10, 1.6]]},
    ),
    _case(
        "official-linear-model", "official-statistics",
        "Set up Regression benchmark study calibration: response y=[2.1,3.8,6.2,7.9,10.1], one predictor x with rows 0 through 4, and include an intercept.",
        "Regression benchmark", "calibration", "org.modellab.statistics.linear-model-study", "1.0",
        {"response_name": "y", "predictor_names": ["x"], "response": [2.1, 3.8, 6.2, 7.9, 10.1], "predictors": [[0], [1], [2], [3], [4]], "include_intercept": True},
    ),
    _case(
        "official-grouped-samples", "official-statistics",
        "For Group test benchmark make trial scores with placebo [5.0,5.2,4.9,5.1] and treatment [6.0,6.3,5.8,6.1], measured in points.",
        "Group test benchmark", "trial", "org.modellab.statistics.grouped-samples", "1.0",
        {"measure_name": "score", "unit": "points", "groups": [{"name": "placebo", "values": [5.0, 5.2, 4.9, 5.1]}, {"name": "treatment", "values": [6.0, 6.3, 5.8, 6.1]}]},
    ),
    _case(
        "official-nonlinear-optimisation", "official-optimisation",
        "Build Optimisation benchmark object design with x,y initially 1,1, bounds -3 to 3, objective (x+1)^2+(y-2)^2, and constraint x-y <= 0.",
        "Optimisation benchmark", "design", "org.modellab.optimisation.nonlinear-problem", "1.0",
        {"variables": ["x", "y"], "initial": [1, 1], "bounds": [[-3, 3], [-3, 3]], "objective": "(x+1)**2 + (y-2)**2", "constraints": [{"name": "ordering", "relation": "less-equal", "expression": "x-y"}]},
    ),
    _case(
        "official-nonlinear-estimation", "official-optimisation",
        "In Estimation benchmark define fit with parameters slope,offset, initial [1,1], bounds [-5,5] each, and residuals offset-2, slope+offset-4, 2*slope+offset-6.1, 3*slope+offset-8.",
        "Estimation benchmark", "fit", "org.modellab.estimation.nonlinear-least-squares", "1.0",
        {"variables": ["slope", "offset"], "initial": [1, 1], "bounds": [[-5, 5], [-5, 5]], "residuals": ["offset-2", "slope+offset-4", "2*slope+offset-6.1", "3*slope+offset-8"]},
    ),
    _case(
        "official-linear-inverse", "official-optimisation",
        "Create Inverse benchmark object sources for parameters left/right and sensors p,q,r. Design rows are [1,.1],[.2,1],[1,1], observations [1.2,2.1,3.0], standard deviations [.1,.2,.15].",
        "Inverse benchmark", "sources", "org.modellab.inverse.linear-problem", "1.0",
        {"parameter_names": ["left", "right"], "observation_names": ["p", "q", "r"], "design_matrix": [[1, 0.1], [0.2, 1], [1, 1]], "observations": [1.2, 2.1, 3.0], "standard_deviations": [0.1, 0.2, 0.15]},
    ),
    _case(
        "official-linear-circuit", "official-electrical",
        "Create an electrical circuit in Circuit benchmark called divider. Nodes ground,in,out use ground as reference. Add 9 V source in-to-ground, 1500 ohm resistor in-to-out, and 3000 ohm resistor out-to-ground; voltage is V and current is mA.",
        "Circuit benchmark", "divider", "org.modellab.electrical.linear-circuit", "1.1",
        {"nodes": ["ground", "in", "out"], "ground": "ground", "voltage_unit": "V", "current_unit": "mA", "elements": [{"id": "source", "type": "voltage-source", "from": "in", "to": "ground", "value": 9}, {"id": "upper", "type": "resistor", "from": "in", "to": "out", "value": 1500}, {"id": "lower", "type": "resistor", "from": "out", "to": "ground", "value": 3000}]},
    ),
    _case(
        "official-diode", "official-electrical",
        "In Diode benchmark create device with saturation current 5e-13 A, ideality 1.4, temperature 310 K, voltage span -0.15 to 0.7 V and area scale 2.",
        "Diode benchmark", "device", "org.modellab.electronics.shockley-diode", "1.0",
        {"saturation_current": 5e-13, "ideality_factor": 1.4, "temperature_kelvin": 310, "voltage_span": [-0.15, 0.7], "area_scale": 2.0},
    ),
    _case(
        "official-point-charges", "official-electrical",
        "Make Electrostatic benchmark object dipole in two dimensions. Coordinates are centimetres and charge is nC; plus has +2 at (-2,0), minus has -2 at (2,0), and evaluate at (0,3) and (0,-3).",
        "Electrostatic benchmark", "dipole", "org.modellab.electromagnetics.point-charge-system", "1.1",
        {"dimension": 2, "coordinate_unit": "cm", "charge_unit": "nC", "charges": [{"id": "plus", "charge": 2, "position": [-2, 0]}, {"id": "minus", "charge": -2, "position": [2, 0]}], "evaluation_points": [[0, 3], [0, -3]]},
    ),
    _case(
        "official-reaction-network", "official-reactions",
        "Create Chemistry benchmark network decay. Species X,Y start [2,0] mol/L for 0 to 15 seconds; one reaction converts one X to one Y at rate constant .12.",
        "Chemistry benchmark", "decay", "org.modellab.chemistry.mass-action-network", "1.1",
        {"species": ["X", "Y"], "initial_concentrations": [2, 0], "time_span": [0, 15], "concentration_unit": "mol/L", "time_unit": "s", "reactions": [{"id": "step", "reactants": [{"species": "X", "stoichiometry": 1}], "products": [{"species": "Y", "stoichiometry": 1}], "rate_constant": 0.12}]},
    ),
    _case(
        "official-compartment", "official-reactions",
        "Create Compartment benchmark object drug. Central and tissue start with [60,0] mg; transfer central-to-tissue .15 and tissue-to-central .05, lose .08 from central, simulate 0 to 18 hours.",
        "Compartment benchmark", "drug", "org.modellab.biological.compartment-system", "1.1",
        {"compartments": ["central", "tissue"], "initial_amounts": [60, 0], "transfers": [{"from": "central", "to": "tissue", "rate": 0.15}, {"from": "tissue", "to": "central", "rate": 0.05}], "losses": [{"compartment": "central", "rate": 0.08}], "time_span": [0, 18], "amount_unit": "mg", "time_unit": "h"},
    ),
    _case(
        "official-population", "official-reactions",
        "In Ecology benchmark add system ecosystem. Populations resource/consumer begin [.8,.25], intrinsic growth [.9,-.3], interaction matrix [[-1,-.6],[.4,-.4]], duration 0 to 25 days, population unit scaled-density.",
        "Ecology benchmark", "ecosystem", "org.modellab.biological.population-interaction-system", "1.1",
        {"populations": ["resource", "consumer"], "initial_populations": [0.8, 0.25], "intrinsic_growth": [0.9, -0.3], "interaction_matrix": [[-1, -0.6], [0.4, -0.4]], "time_span": [0, 25], "population_unit": "scaled-density", "time_unit": "day"},
    ),
    _case(
        "official-feature-dataset", "official-learning",
        "Fit kmeans clusters later using ML data benchmark object samples. Features height cm and weight kg have IDs p1-p4 and rows [150,50],[155,52],[185,82],[190,85].",
        "ML data benchmark", "samples", "org.modellab.learning.feature-dataset", "1.0",
        {"feature_names": ["height", "weight"], "feature_units": ["cm", "kg"], "sample_ids": ["p1", "p2", "p3", "p4"], "features": [[150, 50], [155, 52], [185, 82], [190, 85]]},
    ),
    _case(
        "official-supervised-study", "official-learning",
        "Create a logistic classifier study in Classifier benchmark called labelled. Features a,b use rows [-2,0],[-1,1],[0,-1],[1,0],[2,1],[3,2]; target category is [red,red,green,green,blue,blue].",
        "Classifier benchmark", "labelled", "org.modellab.learning.supervised-study", "1.0",
        {"feature_names": ["a", "b"], "features": [[-2, 0], [-1, 1], [0, -1], [1, 0], [2, 1], [3, 2]], "target_name": "category", "task": "classification", "targets": ["red", "red", "green", "green", "blue", "blue"]},
    ),
    _case(
        "official-feedforward-network", "official-learning",
        "For Neural benchmark create network scorer with inputs x,y, output z, first weights [[.5,.2],[-.4,.8]], second weights [[1,-1]], biases [[0,.1],[.2]], and activations relu then linear.",
        "Neural benchmark", "scorer", "org.modellab.learning.feedforward-network", "1.0",
        {"input_names": ["x", "y"], "output_names": ["z"], "weights": [[[0.5, 0.2], [-0.4, 0.8]], [[1, -1]]], "biases": [[0, 0.1], [0.2]], "activations": ["relu", "linear"]},
    ),
    _case(
        "official-fuzzy-system", "official-learning",
        "Create Fuzzy benchmark controller. Input heat ranges 0..100 with cold triangle [0,0,60] and hot [40,100,100]. Output fan has singletons slow=0.2 and fast=1.0; map cold to slow and hot to fast.",
        "Fuzzy benchmark", "controller", "org.modellab.intelligence.fuzzy-rule-system", "1.0",
        {"inputs": [{"name": "heat", "minimum": 0, "maximum": 100, "sets": {"cold": [0, 0, 60], "hot": [40, 100, 100]}}], "output_name": "fan", "output_singletons": {"slow": 0.2, "fast": 1.0}, "rules": [{"antecedents": {"heat": "cold"}, "consequent": "slow"}, {"antecedents": {"heat": "hot"}, "consequent": "fast"}]},
    ),
]


def main() -> None:
    benchmark = json.loads(SOURCE.read_text(encoding="utf-8"))
    benchmark["schema_version"] = "1.3"
    benchmark["title"] = "Model Laboratory v1.16.1 held-out interpreter development benchmark"
    benchmark["review"] = {
        "status": "internal-curated",
        "provenance": (
            "Original 80 cases retained; 34 official-kind cases added and deterministically "
            "validated for v1.16.1. Not claimed to be independently expert-reviewed."
        ),
    }
    existing = {case["id"] for case in benchmark["cases"]}
    if existing.intersection(case["id"] for case in OFFICIAL_CASES):
        raise ValueError("Official benchmark IDs collide with the retained core cases.")
    benchmark["cases"].extend(OFFICIAL_CASES)
    if len(benchmark["cases"]) != 114:
        raise ValueError("The v1.3 benchmark must contain exactly 114 cases.")
    TARGET.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
