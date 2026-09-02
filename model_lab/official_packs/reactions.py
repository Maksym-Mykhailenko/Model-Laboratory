"""Chemical, Reaction and Biological Systems official capability pack."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import null_space

from ..model import ModelIR
from ..model_graph import ModelGraphError, ModelObject, ObjectKindDescriptor
from ..protocol import ArtifactTypeDescriptor, CapabilityDescriptor, SubspaceRepresentation
from .common import PackManifest, finite_matrix, finite_vector, objects_of_kind, select_object, subspace_representation, unique_labels
from .units import CONCENTRATION, DIMENSIONLESS, TIME, canonical_unit, resolve_unit


PACK_ID = "org.modellab.pack.chemical-reaction-biological-systems"
REACTION_NETWORK_KIND = "org.modellab.chemistry.mass-action-network"
COMPARTMENT_KIND = "org.modellab.biological.compartment-system"
POPULATION_KIND = "org.modellab.biological.population-interaction-system"

_PARTICIPANT_SCHEMA = {
    "type": "object", "required": ["species", "stoichiometry"],
    "properties": {
        "species": {"type": "string", "minLength": 1, "maxLength": 128},
        "stoichiometry": {"type": "number", "exclusiveMinimum": 0},
    }, "additionalProperties": False,
}
_REACTION_SCHEMA = {
    "type": "object", "required": ["id", "reactants", "products", "rate_constant"],
    "properties": {
        "id": {"type": "string", "minLength": 1, "maxLength": 128},
        "reactants": {"type": "array", "maxItems": 256, "items": _PARTICIPANT_SCHEMA},
        "products": {"type": "array", "maxItems": 256, "items": _PARTICIPANT_SCHEMA},
        "rate_constant": {"type": "number", "exclusiveMinimum": 0},
    }, "additionalProperties": False,
}

REACTION_NETWORK_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["species", "initial_concentrations", "reactions", "time_span"],
    "properties": {
        "species": {"type": "array", "minItems": 1, "maxItems": 512, "items": {"type": "string"}},
        "initial_concentrations": {"type": "array", "minItems": 1, "maxItems": 512, "items": {"type": "number"}},
        "reactions": {"type": "array", "minItems": 1, "maxItems": 10000, "items": _REACTION_SCHEMA},
        "time_span": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}},
        "concentration_unit": {"type": "string", "maxLength": 128},
        "time_unit": {"type": "string", "maxLength": 128},
    }, "additionalProperties": False,
}

_TRANSFER_SCHEMA = {
    "type": "object", "required": ["from", "to", "rate"],
    "properties": {
        "from": {"type": "string", "minLength": 1, "maxLength": 128},
        "to": {"type": "string", "minLength": 1, "maxLength": 128},
        "rate": {"type": "number", "exclusiveMinimum": 0},
    }, "additionalProperties": False,
}
_LOSS_SCHEMA = {
    "type": "object", "required": ["compartment", "rate"],
    "properties": {
        "compartment": {"type": "string", "minLength": 1, "maxLength": 128},
        "rate": {"type": "number", "exclusiveMinimum": 0},
    }, "additionalProperties": False,
}
COMPARTMENT_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["compartments", "initial_amounts", "transfers", "time_span"],
    "properties": {
        "compartments": {"type": "array", "minItems": 1, "maxItems": 512, "items": {"type": "string"}},
        "initial_amounts": {"type": "array", "minItems": 1, "maxItems": 512, "items": {"type": "number"}},
        "transfers": {"type": "array", "maxItems": 10000, "items": _TRANSFER_SCHEMA},
        "losses": {"type": "array", "maxItems": 512, "items": _LOSS_SCHEMA},
        "constant_inputs": {"type": "array", "maxItems": 512, "items": {"type": "number"}},
        "time_span": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}},
        "amount_unit": {"type": "string", "maxLength": 128},
        "time_unit": {"type": "string", "maxLength": 128},
    }, "additionalProperties": False,
}

_INTERACTION_ROW = {"type": "array", "minItems": 1, "maxItems": 512, "items": {"type": "number"}}
POPULATION_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["populations", "initial_populations", "intrinsic_growth", "interaction_matrix", "time_span"],
    "properties": {
        "populations": {"type": "array", "minItems": 1, "maxItems": 512, "items": {"type": "string"}},
        "initial_populations": {"type": "array", "minItems": 1, "maxItems": 512, "items": {"type": "number"}},
        "intrinsic_growth": {"type": "array", "minItems": 1, "maxItems": 512, "items": {"type": "number"}},
        "interaction_matrix": {"type": "array", "minItems": 1, "maxItems": 512, "items": _INTERACTION_ROW},
        "time_span": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}},
        "population_unit": {"type": "string", "maxLength": 128},
        "time_unit": {"type": "string", "maxLength": 128},
    }, "additionalProperties": False,
}

KIND_DESCRIPTORS = (
    ObjectKindDescriptor(REACTION_NETWORK_KIND, "1.1", "Irreversible mass-action reaction network", REACTION_NETWORK_SCHEMA, True),
    ObjectKindDescriptor(COMPARTMENT_KIND, "1.1", "Linear biological compartment system", COMPARTMENT_SCHEMA, True),
    ObjectKindDescriptor(POPULATION_KIND, "1.1", "Generalised Lotka-Volterra population system", POPULATION_SCHEMA, True),
)


def _finite(value: object, *, field: str, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ModelGraphError(f"{field} must be a finite real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelGraphError(f"{field} must be a finite real number.") from exc
    if not math.isfinite(result) or (positive and result <= 0.0) or (nonnegative and result < 0.0):
        qualifier = "positive " if positive else "non-negative " if nonnegative else ""
        raise ModelGraphError(f"{field} must be {qualifier}and finite.")
    return result


def _time_span(value: object, *, field: str) -> tuple[float, float]:
    span = finite_vector(value, field=field)
    if span.size != 2 or span[0] >= span[1]:
        raise ModelGraphError(f"{field} must contain increasing start/end values.")
    return float(span[0]), float(span[1])


@dataclass(frozen=True, slots=True)
class _Reaction:
    identifier: str
    reactants: np.ndarray
    products: np.ndarray
    rate_constant: float


@dataclass(frozen=True, slots=True)
class _ReactionNetwork:
    object_id: str
    species: tuple[str, ...]
    initial: np.ndarray
    reactions: tuple[_Reaction, ...]
    time_span: tuple[float, float]
    concentration_unit: str
    time_unit: str
    source_units: Mapping[str, str]


def _participants(raw: object, lookup: Mapping[str, int], *, field: str) -> np.ndarray:
    result = np.zeros(len(lookup), dtype=np.float64)
    if not isinstance(raw, list):
        raise ModelGraphError(f"{field} must be a list.")
    seen: set[str] = set()
    for index, participant in enumerate(raw):
        if not isinstance(participant, Mapping):
            raise ModelGraphError(f"{field}[{index}] must be an object.")
        name = str(participant["species"])
        if name not in lookup or name in seen:
            raise ModelGraphError(f"{field} must name distinct declared species.")
        coefficient = _finite(participant["stoichiometry"], field=f"{field}[{index}].stoichiometry", positive=True)
        if not coefficient.is_integer():
            raise ModelGraphError(
                f"{field}[{index}].stoichiometry must be a positive integer for the "
                "unclipped polynomial mass-action vector field."
            )
        result[lookup[name]] = coefficient
        seen.add(name)
    return result


def _reaction_network(item: ModelObject) -> _ReactionNetwork:
    species = unique_labels(item.properties["species"], field=f"{item.identifier}.species")
    concentration_unit = str(item.properties.get("concentration_unit", "mol/m^3"))
    time_unit = str(item.properties.get("time_unit", "s"))
    concentration_definition = resolve_unit(concentration_unit, field=f"{item.identifier}.concentration_unit", expected=CONCENTRATION)
    time_definition = resolve_unit(time_unit, field=f"{item.identifier}.time_unit", expected=TIME)
    initial = finite_vector(item.properties["initial_concentrations"], field=f"{item.identifier}.initial_concentrations") * concentration_definition.scale_to_si
    if initial.size != len(species) or np.any(initial < 0.0):
        raise ModelGraphError(f"{item.identifier}.initial_concentrations must be non-negative and align with species.")
    lookup = {name: index for index, name in enumerate(species)}
    identifiers: set[str] = set(); reactions: list[_Reaction] = []
    for index, raw in enumerate(item.properties["reactions"]):
        if not isinstance(raw, Mapping):
            raise ModelGraphError(f"{item.identifier}.reactions[{index}] must be an object.")
        identifier = str(raw["id"])
        if not identifier.strip() or identifier in identifiers:
            raise ModelGraphError(f"{item.identifier}.reactions must have unique non-empty IDs.")
        reactants = _participants(raw["reactants"], lookup, field=f"{item.identifier}.reactions[{index}].reactants")
        products = _participants(raw["products"], lookup, field=f"{item.identifier}.reactions[{index}].products")
        if np.array_equal(reactants, products):
            raise ModelGraphError(f"{item.identifier}.reactions[{index}] must change at least one stoichiometric coefficient.")
        rate = _finite(raw["rate_constant"], field=f"{item.identifier}.reactions[{index}].rate_constant", positive=True)
        order = int(np.sum(reactants))
        rate *= concentration_definition.scale_to_si ** (1 - order) / time_definition.scale_to_si
        identifiers.add(identifier); reactions.append(_Reaction(identifier, reactants, products, rate))
    return _ReactionNetwork(
        item.identifier, species, initial, tuple(reactions), tuple(value * time_definition.scale_to_si for value in _time_span(item.properties["time_span"], field=f"{item.identifier}.time_span")),
        "mol/m^3", "s", {"concentration": concentration_unit, "time": time_unit},
    )


@dataclass(frozen=True, slots=True)
class _Compartment:
    object_id: str
    names: tuple[str, ...]
    initial: np.ndarray
    system_matrix: np.ndarray
    inputs: np.ndarray
    outgoing_rates: np.ndarray
    time_span: tuple[float, float]
    amount_unit: str
    time_unit: str
    source_units: Mapping[str, str]


def _compartment(item: ModelObject) -> _Compartment:
    names = unique_labels(item.properties["compartments"], field=f"{item.identifier}.compartments")
    amount_unit = str(item.properties.get("amount_unit", "mol"))
    time_unit = str(item.properties.get("time_unit", "s"))
    amount_definition = resolve_unit(amount_unit, field=f"{item.identifier}.amount_unit")
    time_definition = resolve_unit(time_unit, field=f"{item.identifier}.time_unit", expected=TIME)
    initial = finite_vector(item.properties["initial_amounts"], field=f"{item.identifier}.initial_amounts") * amount_definition.scale_to_si
    if initial.size != len(names) or np.any(initial < 0.0):
        raise ModelGraphError(f"{item.identifier}.initial_amounts must be non-negative and align with compartments.")
    lookup = {name: index for index, name in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=np.float64)
    outgoing = np.zeros(len(names), dtype=np.float64)
    pairs: set[tuple[str, str]] = set()
    for index, raw in enumerate(item.properties["transfers"]):
        if not isinstance(raw, Mapping):
            raise ModelGraphError(f"{item.identifier}.transfers[{index}] must be an object.")
        source = str(raw["from"]); target = str(raw["to"])
        if source not in lookup or target not in lookup or source == target or (source, target) in pairs:
            raise ModelGraphError(f"{item.identifier}.transfers must uniquely connect two distinct declared compartments.")
        rate = _finite(raw["rate"], field=f"{item.identifier}.transfers[{index}].rate", positive=True) / time_definition.scale_to_si
        left, right = lookup[source], lookup[target]
        matrix[right, left] += rate; matrix[left, left] -= rate; outgoing[left] += rate; pairs.add((source, target))
    losses: set[str] = set()
    for index, raw in enumerate(item.properties.get("losses", [])):
        if not isinstance(raw, Mapping):
            raise ModelGraphError(f"{item.identifier}.losses[{index}] must be an object.")
        name = str(raw["compartment"])
        if name not in lookup or name in losses:
            raise ModelGraphError(f"{item.identifier}.losses must uniquely name declared compartments.")
        rate = _finite(raw["rate"], field=f"{item.identifier}.losses[{index}].rate", positive=True) / time_definition.scale_to_si
        matrix[lookup[name], lookup[name]] -= rate; outgoing[lookup[name]] += rate; losses.add(name)
    raw_inputs = item.properties.get("constant_inputs")
    inputs = np.zeros(len(names), dtype=np.float64) if raw_inputs is None else finite_vector(raw_inputs, field=f"{item.identifier}.constant_inputs")
    inputs = inputs * amount_definition.scale_to_si / time_definition.scale_to_si
    if inputs.size != len(names) or np.any(inputs < 0.0):
        raise ModelGraphError(f"{item.identifier}.constant_inputs must be non-negative and align with compartments.")
    return _Compartment(
        item.identifier, names, initial, matrix, inputs, outgoing,
        tuple(value * time_definition.scale_to_si for value in _time_span(item.properties["time_span"], field=f"{item.identifier}.time_span")),
        canonical_unit(amount_definition.dimensions), "s", {"amount": amount_unit, "time": time_unit},
    )


@dataclass(frozen=True, slots=True)
class _Population:
    object_id: str
    names: tuple[str, ...]
    initial: np.ndarray
    growth: np.ndarray
    interaction: np.ndarray
    time_span: tuple[float, float]
    population_unit: str
    time_unit: str
    source_units: Mapping[str, str]


def _population(item: ModelObject) -> _Population:
    names = unique_labels(item.properties["populations"], field=f"{item.identifier}.populations")
    population_unit = str(item.properties.get("population_unit", "dimensionless"))
    time_unit = str(item.properties.get("time_unit", "s"))
    population_definition = resolve_unit(population_unit, field=f"{item.identifier}.population_unit", expected=DIMENSIONLESS)
    time_definition = resolve_unit(time_unit, field=f"{item.identifier}.time_unit", expected=TIME)
    initial = finite_vector(item.properties["initial_populations"], field=f"{item.identifier}.initial_populations") * population_definition.scale_to_si
    growth = finite_vector(item.properties["intrinsic_growth"], field=f"{item.identifier}.intrinsic_growth") / time_definition.scale_to_si
    interaction = finite_matrix(item.properties["interaction_matrix"], field=f"{item.identifier}.interaction_matrix") / (time_definition.scale_to_si * population_definition.scale_to_si)
    count = len(names)
    if initial.size != count or growth.size != count or interaction.shape != (count, count) or np.any(initial < 0.0):
        raise ModelGraphError(f"{item.identifier} population vectors/matrix must align and initial values must be non-negative.")
    return _Population(
        item.identifier, names, initial, growth, interaction,
        tuple(value * time_definition.scale_to_si for value in _time_span(item.properties["time_span"], field=f"{item.identifier}.time_span")),
        "1", "s", {"population": population_unit, "time": time_unit},
    )


SEMANTIC_VALIDATORS = {
    REACTION_NETWORK_KIND: lambda item: _reaction_network(item),
    COMPARTMENT_KIND: lambda item: _compartment(item),
    POPULATION_KIND: lambda item: _population(item),
}


def _canonical_basis(basis: np.ndarray) -> np.ndarray:
    result = np.asarray(basis.T, dtype=np.float64).copy()
    order = sorted(range(len(result)), key=lambda index: tuple(np.round(np.abs(result[index]), 14)), reverse=True)
    result = result[order]
    for row in result:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
        row[np.abs(row) < 1e-14] = 0.0
    return result


@dataclass(frozen=True, slots=True)
class ReactionNetworkAnalysis:
    object_id: str
    species: tuple[str, ...]
    reaction_ids: tuple[str, ...]
    reactant_stoichiometry: np.ndarray
    product_stoichiometry: np.ndarray
    stoichiometric_matrix: np.ndarray
    rate_constants: np.ndarray
    stoichiometric_rank: int
    conservation_laws: SubspaceRepresentation
    complexes: np.ndarray
    linkage_class_count: int
    deficiency: int
    concentration_unit: str
    time_unit: str
    source_units: Mapping[str, str]


def analyse_reaction_network(model: ModelIR, object_id: str | None = None) -> ReactionNetworkAnalysis:
    network = _reaction_network(select_object(model, REACTION_NETWORK_KIND, object_id, label="mass-action reaction network"))
    reactants = np.column_stack([reaction.reactants for reaction in network.reactions])
    products = np.column_stack([reaction.products for reaction in network.reactions])
    stoichiometry = products - reactants
    rank = int(np.linalg.matrix_rank(stoichiometry))
    conservation = subspace_representation(_canonical_basis(null_space(stoichiometry.T)), orientation="rows")
    complex_values = sorted({tuple(column) for column in reactants.T} | {tuple(column) for column in products.T})
    complex_lookup = {value: index for index, value in enumerate(complex_values)}
    adjacency = [set() for _ in complex_values]
    for left, right in zip(reactants.T, products.T):
        a, b = complex_lookup[tuple(left)], complex_lookup[tuple(right)]
        adjacency[a].add(b); adjacency[b].add(a)
    unseen = set(range(len(complex_values))); linkage = 0
    while unseen:
        linkage += 1; stack = [unseen.pop()]
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour in unseen:
                    unseen.remove(neighbour); stack.append(neighbour)
    return ReactionNetworkAnalysis(
        network.object_id, network.species, tuple(item.identifier for item in network.reactions),
        reactants, products, stoichiometry, np.asarray([item.rate_constant for item in network.reactions]),
        rank, conservation, np.asarray(complex_values, dtype=np.float64), linkage,
        len(complex_values) - linkage - rank, network.concentration_unit, network.time_unit, network.source_units,
    )


def _reaction_rates(network: _ReactionNetwork, concentrations: np.ndarray) -> np.ndarray:
    # Integer stoichiometries make this a polynomial vector field on all real solver
    # stage values.  Do not silently clip internal stages and thereby solve a different ODE.
    values = np.asarray(concentrations, dtype=np.float64)
    rates = np.asarray([reaction.rate_constant * float(np.prod(np.power(values, reaction.reactants.astype(np.int64)))) for reaction in network.reactions])
    if not np.all(np.isfinite(rates)):
        raise ValueError("Mass-action rate evaluation produced non-finite values.")
    return rates


@dataclass(frozen=True, slots=True)
class ReactionTrajectory:
    object_id: str
    species: tuple[str, ...]
    reaction_ids: tuple[str, ...]
    times: np.ndarray
    concentrations: np.ndarray
    reaction_rates: np.ndarray
    method: str
    relative_tolerance: float
    absolute_tolerance: float
    nonnegative_projection_tolerance: float
    concentration_unit: str
    time_unit: str
    source_units: Mapping[str, str]
    solver_record: Mapping[str, Any]


def simulate_reaction_network(
    model: ModelIR, object_id: str | None = None, samples: int = 501, method: str = "BDF",
    relative_tolerance: float = 1e-9, absolute_tolerance: float = 1e-12,
    nonnegative_projection_tolerance: float = 1e-10,
) -> ReactionTrajectory:
    _validate_solver_settings(samples, method, relative_tolerance, absolute_tolerance)
    if not math.isfinite(float(nonnegative_projection_tolerance)) or float(nonnegative_projection_tolerance) <= 0.0:
        raise ValueError("nonnegative_projection_tolerance must be positive and finite.")
    network = _reaction_network(select_object(model, REACTION_NETWORK_KIND, object_id, label="mass-action reaction network"))
    stoichiometry = np.column_stack([reaction.products - reaction.reactants for reaction in network.reactions])
    times = np.linspace(*network.time_span, samples)
    solution = solve_ivp(
        lambda _time, state: stoichiometry @ _reaction_rates(network, state), network.time_span,
        network.initial, method=method, t_eval=times, rtol=float(relative_tolerance), atol=float(absolute_tolerance),
    )
    if not solution.success:
        raise ValueError(f"Reaction integration failed with solver status {solution.status}.")
    values = solution.y.T.copy()
    if float(np.min(values)) < -float(nonnegative_projection_tolerance):
        raise ValueError("Reaction integration produced concentrations below the declared non-negative tolerance.")
    values[values < 0.0] = 0.0
    rates = np.asarray([_reaction_rates(network, row) for row in values])
    return ReactionTrajectory(
        network.object_id, network.species, tuple(item.identifier for item in network.reactions),
        times, values, rates, method, float(relative_tolerance), float(absolute_tolerance),
        float(nonnegative_projection_tolerance), network.concentration_unit, network.time_unit, network.source_units,
        {"scientific": {
            "converged": True,
            "internal_vector_field": "unclipped-polynomial-mass-action",
            "output_nonnegative_policy": "project-only-within-declared-tolerance",
        }, "presentation": {"function_evaluations": int(solution.nfev), "jacobian_evaluations": int(solution.njev), "solver_status": int(solution.status)}},
    )


@dataclass(frozen=True, slots=True)
class CompartmentAnalysis:
    object_id: str
    compartments: tuple[str, ...]
    system_matrix: np.ndarray
    constant_inputs: np.ndarray
    eigenvalues: tuple[complex, ...]
    spectral_abscissa: float
    stability: str
    outgoing_rates: np.ndarray
    mean_residence_times: np.ndarray
    closed_mass_conserving: bool
    steady_state: np.ndarray | None
    amount_unit: str
    time_unit: str
    source_units: Mapping[str, str]


def analyse_compartment_system(model: ModelIR, object_id: str | None = None, stability_tolerance: float = 1e-10) -> CompartmentAnalysis:
    if not math.isfinite(float(stability_tolerance)) or float(stability_tolerance) <= 0.0:
        raise ValueError("stability_tolerance must be positive and finite.")
    system = _compartment(select_object(model, COMPARTMENT_KIND, object_id, label="compartment system"))
    eigenvalues = np.linalg.eigvals(system.system_matrix)
    abscissa = float(np.max(np.real(eigenvalues)))
    if abscissa < -stability_tolerance:
        stability = "asymptotically stable"
    elif abscissa > stability_tolerance:
        stability = "unstable"
    else:
        stability = "marginal or unresolved"
    residence = np.divide(1.0, system.outgoing_rates, out=np.full_like(system.outgoing_rates, np.inf), where=system.outgoing_rates > 0.0)
    closed = bool(np.allclose(np.sum(system.system_matrix, axis=0), 0.0, rtol=0.0, atol=1e-12) and np.all(system.inputs == 0.0))
    steady = None
    if np.linalg.matrix_rank(system.system_matrix) == len(system.names):
        candidate = np.linalg.solve(system.system_matrix, -system.inputs)
        if np.all(candidate >= -1e-12):
            candidate[candidate < 0.0] = 0.0; steady = candidate
    return CompartmentAnalysis(
        system.object_id, system.names, system.system_matrix, system.inputs,
        tuple(complex(value) for value in eigenvalues), abscissa, stability, system.outgoing_rates,
        residence, closed, steady, system.amount_unit, system.time_unit, system.source_units,
    )


@dataclass(frozen=True, slots=True)
class CompartmentTrajectory:
    object_id: str
    compartments: tuple[str, ...]
    times: np.ndarray
    amounts: np.ndarray
    total_amount: np.ndarray
    method: str
    relative_tolerance: float
    absolute_tolerance: float
    amount_unit: str
    time_unit: str
    source_units: Mapping[str, str]
    solver_record: Mapping[str, Any]


def simulate_compartment_system(
    model: ModelIR, object_id: str | None = None, samples: int = 501, method: str = "DOP853",
    relative_tolerance: float = 1e-10, absolute_tolerance: float = 1e-12,
) -> CompartmentTrajectory:
    _validate_solver_settings(samples, method, relative_tolerance, absolute_tolerance)
    system = _compartment(select_object(model, COMPARTMENT_KIND, object_id, label="compartment system"))
    times = np.linspace(*system.time_span, samples)
    solution = solve_ivp(
        lambda _time, state: system.system_matrix @ state + system.inputs, system.time_span,
        system.initial, method=method, t_eval=times, rtol=float(relative_tolerance), atol=float(absolute_tolerance),
    )
    if not solution.success:
        raise ValueError(f"Compartment integration failed with solver status {solution.status}.")
    amounts = solution.y.T.copy()
    if float(np.min(amounts)) < -1e-9:
        raise ValueError("Compartment integration produced materially negative amounts.")
    amounts[amounts < 0.0] = 0.0
    return CompartmentTrajectory(
        system.object_id, system.names, times, amounts, np.sum(amounts, axis=1), method,
        float(relative_tolerance), float(absolute_tolerance), system.amount_unit, system.time_unit, system.source_units,
        {"scientific": {"converged": True}, "presentation": {"function_evaluations": int(solution.nfev), "solver_status": int(solution.status)}},
    )


@dataclass(frozen=True, slots=True)
class PopulationAnalysis:
    object_id: str
    populations: tuple[str, ...]
    intrinsic_growth: np.ndarray
    interaction_matrix: np.ndarray
    coexistence_equilibrium: np.ndarray | None
    coexistence_feasible: bool
    coexistence_jacobian: np.ndarray | None
    coexistence_eigenvalues: tuple[complex, ...]
    coexistence_stability: str
    population_unit: str
    time_unit: str
    source_units: Mapping[str, str]


def analyse_population_system(model: ModelIR, object_id: str | None = None, stability_tolerance: float = 1e-9) -> PopulationAnalysis:
    if not math.isfinite(float(stability_tolerance)) or float(stability_tolerance) <= 0.0:
        raise ValueError("stability_tolerance must be positive and finite.")
    system = _population(select_object(model, POPULATION_KIND, object_id, label="population-interaction system"))
    if np.linalg.matrix_rank(system.interaction) < len(system.names):
        return PopulationAnalysis(system.object_id, system.names, system.growth, system.interaction, None, False, None, (), "undefined: singular interaction matrix", system.population_unit, system.time_unit, system.source_units)
    equilibrium = np.linalg.solve(system.interaction, -system.growth)
    feasible = bool(np.all(equilibrium > 0.0))
    jacobian = np.diag(equilibrium) @ system.interaction
    eigenvalues = np.linalg.eigvals(jacobian)
    abscissa = float(np.max(np.real(eigenvalues)))
    stability = "asymptotically stable" if abscissa < -stability_tolerance else "unstable" if abscissa > stability_tolerance else "marginal or unresolved"
    if not feasible:
        stability = "not biologically feasible"
    return PopulationAnalysis(
        system.object_id, system.names, system.growth, system.interaction, equilibrium, feasible,
        jacobian, tuple(complex(value) for value in eigenvalues), stability,
        system.population_unit, system.time_unit, system.source_units,
    )


@dataclass(frozen=True, slots=True)
class PopulationTrajectory:
    object_id: str
    populations: tuple[str, ...]
    times: np.ndarray
    values: np.ndarray
    method: str
    relative_tolerance: float
    absolute_tolerance: float
    population_unit: str
    time_unit: str
    source_units: Mapping[str, str]
    solver_record: Mapping[str, Any]


def simulate_population_system(
    model: ModelIR, object_id: str | None = None, samples: int = 501, method: str = "DOP853",
    relative_tolerance: float = 1e-9, absolute_tolerance: float = 1e-12,
) -> PopulationTrajectory:
    _validate_solver_settings(samples, method, relative_tolerance, absolute_tolerance)
    system = _population(select_object(model, POPULATION_KIND, object_id, label="population-interaction system"))
    times = np.linspace(*system.time_span, samples)
    solution = solve_ivp(
        lambda _time, state: state * (system.growth + system.interaction @ state), system.time_span,
        system.initial, method=method, t_eval=times, rtol=float(relative_tolerance), atol=float(absolute_tolerance),
    )
    if not solution.success:
        raise ValueError(f"Population integration failed with solver status {solution.status}.")
    values = solution.y.T.copy()
    if float(np.min(values)) < -1e-9:
        raise ValueError("Population integration produced materially negative populations.")
    values[values < 0.0] = 0.0
    return PopulationTrajectory(
        system.object_id, system.names, times, values, method, float(relative_tolerance),
        float(absolute_tolerance), system.population_unit, system.time_unit, system.source_units,
        {"scientific": {"converged": True}, "presentation": {"function_evaluations": int(solution.nfev), "solver_status": int(solution.status)}},
    )


def _validate_solver_settings(samples: int, method: str, relative_tolerance: float, absolute_tolerance: float) -> None:
    if isinstance(samples, bool) or not isinstance(samples, int) or not 2 <= samples <= 20000:
        raise ValueError("samples must be an integer between 2 and 20000.")
    if method not in {"RK45", "DOP853", "Radau", "BDF"}:
        raise ValueError("method must be RK45, DOP853, Radau, or BDF.")
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in (relative_tolerance, absolute_tolerance)):
        raise ValueError("Solver tolerances must be positive finite numbers.")


def _kind_applicability(kind: str, label: str):
    def applicable(model: ModelIR) -> tuple[bool, str]:
        found = bool(objects_of_kind(model, kind))
        return found, "" if found else f"an executable {label} object is required"
    return applicable


def _network_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(
        model, REACTION_NETWORK_KIND, settings.get("object_id"), label="reaction network"
    )
    return max(1, int(settings.get("samples", 501)) * len(item.properties["species"]) * len(item.properties["reactions"]))


def _system_units(settings: Mapping[str, Any], model: ModelIR, kind: str, field: str) -> int:
    item = select_object(model, kind, settings.get("object_id"), label="biological system")
    count = len(item.properties[field])
    return max(1, int(settings.get("samples", 501)) * count * count)


NUMERIC = "org.modellab.comparator.numeric"
_SOLVER_SETTINGS = {
    "object_id": {"type": ["string", "null"]}, "samples": {"type": "integer", "minimum": 2, "maximum": 20000, "default": 501},
    "method": {"type": "string", "enum": ["RK45", "DOP853", "Radau", "BDF"]},
    "relative_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-9},
    "absolute_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-12},
}
CAPABILITY_DESCRIPTORS = (
    CapabilityDescriptor(
        "org.modellab.chemistry.analyse-reaction-network", "1.2", PACK_ID,
        "Analyse reaction network", "Compute stoichiometry, rank, conservation laws, reaction complexes, linkage classes and deficiency.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}}}, "numpy+scipy",
        (ArtifactTypeDescriptor("org.modellab.artifact.reaction-network-analysis", "1.2", "Reaction-network analysis", NUMERIC),),
        _kind_applicability(REACTION_NETWORK_KIND, "mass-action-reaction-network"), analyse_reaction_network, _network_units,
        ("org.modellab.renderer.plotly-reaction-stoichiometry",),
    ),
    CapabilityDescriptor(
        "org.modellab.chemistry.simulate-reaction-network", "1.2", PACK_ID,
        "Simulate reaction network", "Integrate deterministic irreversible mass-action kinetics with explicit non-negative projection tolerance.",
        {"type": "object", "properties": {**_SOLVER_SETTINGS, "method": {"type": "string", "enum": ["RK45", "DOP853", "Radau", "BDF"], "default": "BDF"}, "nonnegative_projection_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-10}}},
        "scipy+numpy", (ArtifactTypeDescriptor("org.modellab.artifact.reaction-trajectory", "1.2", "Reaction trajectory", NUMERIC),),
        _kind_applicability(REACTION_NETWORK_KIND, "mass-action-reaction-network"), simulate_reaction_network, _network_units,
        ("org.modellab.renderer.plotly-reaction-trajectory",),
    ),
    CapabilityDescriptor(
        "org.modellab.biological.analyse-compartment-system", "1.2", PACK_ID,
        "Analyse compartment system", "Analyse transfer/loss structure, residence times, stability, conservation and any unique non-negative steady state.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "stability_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-10}}},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.compartment-analysis", "1.2", "Compartment analysis", NUMERIC),),
        _kind_applicability(COMPARTMENT_KIND, "compartment-system"), analyse_compartment_system,
        lambda settings, model: _system_units(settings, model, COMPARTMENT_KIND, "compartments"),
        ("org.modellab.renderer.plotly-compartment-spectrum",),
    ),
    CapabilityDescriptor(
        "org.modellab.biological.simulate-compartment-system", "1.2", PACK_ID,
        "Simulate compartment system", "Integrate a linear transfer/loss system under explicit constant inputs and solver tolerances.",
        {"type": "object", "properties": {**_SOLVER_SETTINGS, "method": {"type": "string", "enum": ["RK45", "DOP853", "Radau", "BDF"], "default": "DOP853"}}},
        "scipy+numpy", (ArtifactTypeDescriptor("org.modellab.artifact.compartment-trajectory", "1.2", "Compartment trajectory", NUMERIC),),
        _kind_applicability(COMPARTMENT_KIND, "compartment-system"), simulate_compartment_system,
        lambda settings, model: _system_units(settings, model, COMPARTMENT_KIND, "compartments"),
        ("org.modellab.renderer.plotly-compartment-trajectory",),
    ),
    CapabilityDescriptor(
        "org.modellab.biological.analyse-population-system", "1.2", PACK_ID,
        "Analyse population interaction system", "Solve and classify the full coexistence equilibrium of a generalised Lotka-Volterra system.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "stability_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-9}}},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.population-analysis", "1.2", "Population-system analysis", NUMERIC),),
        _kind_applicability(POPULATION_KIND, "population-interaction-system"), analyse_population_system,
        lambda settings, model: _system_units(settings, model, POPULATION_KIND, "populations"),
        ("org.modellab.renderer.plotly-population-spectrum",),
    ),
    CapabilityDescriptor(
        "org.modellab.biological.simulate-population-system", "1.2", PACK_ID,
        "Simulate population interaction system", "Integrate generalised Lotka-Volterra population dynamics with explicit solver settings.",
        {"type": "object", "properties": {**_SOLVER_SETTINGS, "method": {"type": "string", "enum": ["RK45", "DOP853", "Radau", "BDF"], "default": "DOP853"}}},
        "scipy+numpy", (ArtifactTypeDescriptor("org.modellab.artifact.population-trajectory", "1.2", "Population trajectory", NUMERIC),),
        _kind_applicability(POPULATION_KIND, "population-interaction-system"), simulate_population_system,
        lambda settings, model: _system_units(settings, model, POPULATION_KIND, "populations"),
        ("org.modellab.renderer.plotly-population-trajectory",),
    ),
)

MANIFEST = PackManifest(
    PACK_ID, "1.2", "Chemical, Reaction and Biological Systems",
    "Mass-action chemical networks, linear biological compartments and nonlinear population interactions with structural and temporal analyses.",
    ("org.modellab.pack.multidimensional-mathematics", "org.modellab.pack.dynamics-differential-equations-control"),
    tuple(item.kind for item in KIND_DESCRIPTORS), tuple(item.identifier for item in CAPABILITY_DESCRIPTORS),
)
