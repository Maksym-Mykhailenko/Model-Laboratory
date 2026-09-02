"""Probability, Stochastic Processes and Simulation pack."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from ..model import ModelIR
from ..model_graph import ModelGraphError, ModelObject, ObjectKindDescriptor
from ..protocol import ArtifactTypeDescriptor, CapabilityDescriptor
from .common import (
    PackManifest,
    entropy,
    objects_of_kind,
    probability_vector,
    select_object,
    stochastic_matrix,
    unique_labels,
)


PACK_ID = "org.modellab.pack.probability-stochastic-systems"
DISTRIBUTION_KIND = "org.modellab.probability.discrete-distribution"
MARKOV_CHAIN_KIND = "org.modellab.probability.markov-chain"


DISTRIBUTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["outcomes", "probabilities"],
    "properties": {
        "outcomes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10000,
            "items": {"type": "string"},
        },
        "probabilities": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10000,
            "items": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "numeric_values": {
            "type": "array",
            "maxItems": 10000,
            "items": {"type": "number"},
        },
    },
    "additionalProperties": False,
}

MARKOV_CHAIN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["states", "initial", "transition"],
    "properties": {
        "states": {
            "type": "array",
            "minItems": 1,
            "maxItems": 2048,
            "items": {"type": "string"},
        },
        "initial": {
            "type": "array",
            "minItems": 1,
            "maxItems": 2048,
            "items": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "transition": {
            "type": "array",
            "minItems": 1,
            "maxItems": 2048,
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2048,
                "items": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "state_values": {
            "type": "array",
            "maxItems": 2048,
            "items": {"type": "number"},
        },
    },
    "additionalProperties": False,
}


KIND_DESCRIPTORS = (
    ObjectKindDescriptor(
        DISTRIBUTION_KIND,
        "1.0",
        "Finite discrete probability distribution",
        DISTRIBUTION_SCHEMA,
        True,
    ),
    ObjectKindDescriptor(
        MARKOV_CHAIN_KIND,
        "1.0",
        "Finite row-stochastic Markov chain",
        MARKOV_CHAIN_SCHEMA,
        True,
    ),
)


def _distribution(item: ModelObject) -> tuple[tuple[str, ...], np.ndarray, np.ndarray | None]:
    outcomes = unique_labels(item.properties["outcomes"], field=f"{item.identifier}.outcomes")
    probabilities = probability_vector(
        item.properties["probabilities"], field=f"{item.identifier}.probabilities", size=len(outcomes)
    )
    numeric_raw = item.properties.get("numeric_values")
    numeric = None if numeric_raw is None else np.asarray(numeric_raw, dtype=np.float64)
    if numeric is not None and (numeric.ndim != 1 or numeric.size != len(outcomes) or not np.all(np.isfinite(numeric))):
        raise ModelGraphError(f"{item.identifier}.numeric_values must align with outcomes.")
    return outcomes, probabilities, numeric


def _chain(item: ModelObject) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    states = unique_labels(item.properties["states"], field=f"{item.identifier}.states")
    initial = probability_vector(
        item.properties["initial"], field=f"{item.identifier}.initial", size=len(states)
    )
    transition = stochastic_matrix(
        item.properties["transition"],
        field=f"{item.identifier}.transition",
        rows=len(states),
        columns=len(states),
    )
    state_values = item.properties.get("state_values")
    if state_values is not None:
        values = np.asarray(state_values, dtype=np.float64)
        if values.ndim != 1 or values.size != len(states) or not np.all(np.isfinite(values)):
            raise ModelGraphError(f"{item.identifier}.state_values must align with states.")
    return states, initial, transition


def validate_distribution(item: ModelObject) -> None:
    _distribution(item)


def validate_markov_chain(item: ModelObject) -> None:
    _chain(item)


SEMANTIC_VALIDATORS = {
    DISTRIBUTION_KIND: validate_distribution,
    MARKOV_CHAIN_KIND: validate_markov_chain,
}


@dataclass(frozen=True, slots=True)
class DistributionAnalysis:
    object_id: str
    outcomes: tuple[str, ...]
    probabilities: np.ndarray
    support: tuple[str, ...]
    entropy_nats: float
    concentration: float
    expectation: float | None
    variance: float | None


@dataclass(frozen=True, slots=True)
class MarkovEvolution:
    object_id: str
    states: tuple[str, ...]
    steps: int
    distributions: np.ndarray
    entropies: np.ndarray
    stationary_distribution: np.ndarray | None
    stationary_distributions: np.ndarray
    stationary_unique: bool
    recurrent_classes: tuple[tuple[str, ...], ...]
    stationarity_residual: float
    eigenvalues: tuple[complex, ...]
    spectral_gap: float | None
    communicating_classes: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class MarkovSimulation:
    object_id: str
    states: tuple[str, ...]
    seed: int
    trajectories: int
    steps: int
    sampled_state_indices: np.ndarray
    empirical_distributions: np.ndarray
    rng: Mapping[str, Any]
    stochastic_reference: Mapping[str, Any]


def analyse_distribution(model: ModelIR, object_id: str | None = None) -> DistributionAnalysis:
    item = select_object(model, DISTRIBUTION_KIND, object_id, label="discrete distribution")
    outcomes, probabilities, numeric = _distribution(item)
    expectation: float | None = None
    variance: float | None = None
    if numeric is not None:
        expectation = float(np.dot(probabilities, numeric))
        variance = float(np.dot(probabilities, np.square(numeric - expectation)))
    return DistributionAnalysis(
        item.identifier,
        outcomes,
        probabilities,
        tuple(outcome for outcome, value in zip(outcomes, probabilities) if value > 0.0),
        float(entropy(probabilities)),
        float(np.sum(np.square(probabilities))),
        expectation,
        variance,
    )


def _components(transition: np.ndarray) -> tuple[tuple[int, ...], ...]:
    size = transition.shape[0]
    adjacency = transition > 0.0

    def reachable(start: int) -> set[int]:
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for candidate in np.flatnonzero(adjacency[node]):
                value = int(candidate)
                if value not in seen:
                    seen.add(value)
                    stack.append(value)
        return seen

    reaches = [reachable(index) for index in range(size)]
    remaining = set(range(size))
    groups: list[tuple[int, ...]] = []
    while remaining:
        first = min(remaining)
        group = tuple(index for index in sorted(remaining) if first in reaches[index] and index in reaches[first])
        groups.append(group)
        remaining.difference_update(group)
    return tuple(groups)


def evolve_markov_chain(
    model: ModelIR,
    object_id: str | None = None,
    steps: int = 20,
) -> MarkovEvolution:
    if isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps <= 100000:
        raise ValueError("steps must be an integer between zero and 100000.")
    item = select_object(model, MARKOV_CHAIN_KIND, object_id, label="Markov chain")
    states, initial, transition = _chain(item)
    distributions = np.empty((steps + 1, len(states)), dtype=np.float64)
    distributions[0] = initial
    for index in range(steps):
        distributions[index + 1] = distributions[index] @ transition
    eigenvalues = np.linalg.eigvals(transition.T)
    classes = _components(transition)
    recurrent = tuple(
        group for group in classes
        if not any(transition[index, outside] > 0.0 for index in group for outside in range(len(states)) if outside not in group)
    )
    stationary_rows: list[np.ndarray] = []
    for group in recurrent:
        submatrix = transition[np.ix_(group, group)]
        equations = submatrix.T - np.eye(len(group))
        equations[-1] = 1.0
        target = np.zeros(len(group)); target[-1] = 1.0
        local = np.linalg.lstsq(equations, target, rcond=None)[0]
        local[np.abs(local) < 1e-15] = 0.0
        local = np.maximum(local, 0.0)
        local /= float(np.sum(local))
        embedded = np.zeros(len(states), dtype=np.float64)
        embedded[list(group)] = local
        stationary_rows.append(embedded)
    stationary_family = np.stack(stationary_rows)
    unique = len(stationary_rows) == 1
    stationary = stationary_family[0] if unique else None
    residual = float(np.max(np.abs(stationary_family @ transition - stationary_family)))
    ordered_moduli = sorted((abs(complex(value)) for value in eigenvalues), reverse=True)
    gap = None if not unique or len(ordered_moduli) < 2 else float(max(0.0, 1.0 - ordered_moduli[1]))
    return MarkovEvolution(
        item.identifier,
        states,
        steps,
        distributions,
        np.asarray(entropy(distributions, axis=1), dtype=np.float64),
        stationary,
        stationary_family,
        unique,
        tuple(tuple(states[index] for index in group) for group in recurrent),
        residual,
        tuple(complex(value) for value in eigenvalues),
        gap,
        tuple(tuple(states[index] for index in group) for group in classes),
    )


def simulate_markov_chain(
    model: ModelIR,
    object_id: str | None = None,
    steps: int = 50,
    trajectories: int = 1000,
    seed: int = 0,
) -> MarkovSimulation:
    # Imported lazily: the stochastic persistence layer also imports the experiment
    # compiler, while this module is part of that compiler's installed kind registry.
    from ..stochastic import (
        RNGSpecification,
        SamplingConfiguration,
        StochasticEquivalenceSettings,
        StochasticSampleResult,
        stochastic_reference,
    )
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 10000:
        raise ValueError("steps must be an integer between one and 10000.")
    if isinstance(trajectories, bool) or not isinstance(trajectories, int) or not 1 <= trajectories <= 100000:
        raise ValueError("trajectories must be an integer between one and 100000.")
    if trajectories * (steps + 1) > 2_000_000:
        raise ValueError("The requested Markov simulation exceeds the two-million-state result limit.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**128:
        raise ValueError("seed must be an integer in [0, 2**128).")
    item = select_object(model, MARKOV_CHAIN_KIND, object_id, label="Markov chain")
    states, initial, transition = _chain(item)
    rng = np.random.Generator(np.random.PCG64(seed))
    sampled = np.empty((trajectories, steps + 1), dtype=np.int32)
    sampled[:, 0] = rng.choice(len(states), size=trajectories, p=initial)
    for time in range(steps):
        for state_index in range(len(states)):
            mask = np.flatnonzero(sampled[:, time] == state_index)
            if mask.size:
                sampled[mask, time + 1] = rng.choice(
                    len(states), size=mask.size, p=transition[state_index]
                )
    empirical = np.stack(
        [np.bincount(sampled[:, time], minlength=len(states)) / trajectories for time in range(steps + 1)]
    )
    rng_spec = RNGSpecification(
        seed=seed,
        algorithm="PCG64",
        algorithm_version=np.__version__,
        implementation="numpy.random.Generator",
    )
    sampling = SamplingConfiguration(
        chains=trajectories,
        draws_per_chain=steps + 1,
        settings=(("sampler", "finite-markov-chain"),),
    )
    reference = stochastic_reference(
        StochasticSampleResult("state-index", sampled.astype(np.float64), rng_spec, sampling),
        StochasticEquivalenceSettings(minimum_sample_count=min(200, sampled.size)),
    )
    return MarkovSimulation(
        item.identifier,
        states,
        seed,
        trajectories,
        steps,
        sampled,
        empirical,
        rng_spec.to_dict(),
        reference,
    )


def _distribution_applicable(model: ModelIR) -> tuple[bool, str]:
    return (True, "") if objects_of_kind(model, DISTRIBUTION_KIND) else (False, "a discrete distribution object is required")


def _chain_applicable(model: ModelIR) -> tuple[bool, str]:
    return (True, "") if objects_of_kind(model, MARKOV_CHAIN_KIND) else (False, "a Markov chain object is required")


def _chain_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(
        model, MARKOV_CHAIN_KIND, settings.get("object_id"), label="Markov chain"
    )
    states = len(item.properties["states"])
    return max(1, int(settings.get("steps", 20)) * states * states * int(settings.get("trajectories", 1)))


def _distribution_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(
        model, DISTRIBUTION_KIND, settings.get("object_id"), label="discrete distribution"
    )
    return max(1, len(item.properties["outcomes"]) * 10)


NUMERIC = "org.modellab.comparator.numeric"
STOCHASTIC = "org.modellab.comparator.stochastic"
CAPABILITY_DESCRIPTORS = (
    CapabilityDescriptor(
        "org.modellab.probability.analyse-distribution",
        "1.0",
        PACK_ID,
        "Analyse discrete distribution",
        "Compute support, entropy, concentration and numeric moments where outcome values are supplied.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}}},
        "numpy",
        (ArtifactTypeDescriptor("org.modellab.artifact.discrete-distribution-analysis", "1.0", "Discrete distribution analysis", NUMERIC),),
        _distribution_applicable,
        analyse_distribution,
        _distribution_units,
        ("org.modellab.renderer.plotly-distribution",),
    ),
    CapabilityDescriptor(
        "org.modellab.probability.evolve-markov-chain",
        "1.1",
        PACK_ID,
        "Evolve finite Markov chain",
        "Propagate an exact finite-state distribution and analyse stationarity, spectrum and communicating classes.",
        {
            "type": "object",
            "properties": {
                "object_id": {"type": ["string", "null"]},
                "steps": {"type": "integer", "minimum": 0, "maximum": 100000, "default": 20},
            },
        },
        "numpy",
        (ArtifactTypeDescriptor("org.modellab.artifact.markov-evolution", "1.1", "Markov-chain evolution", NUMERIC),),
        _chain_applicable,
        evolve_markov_chain,
        _chain_units,
        ("org.modellab.renderer.plotly-state-probabilities",),
    ),
    CapabilityDescriptor(
        "org.modellab.probability.simulate-markov-chain",
        "1.0",
        PACK_ID,
        "Simulate finite Markov chain",
        "Generate a bounded PCG64 trajectory ensemble with frozen RNG and statistical-equivalence evidence.",
        {
            "type": "object",
            "properties": {
                "object_id": {"type": ["string", "null"]},
                "steps": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 50},
                "trajectories": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 1000},
                "seed": {"type": "integer", "minimum": 0, "default": 0},
            },
        },
        "numpy",
        (ArtifactTypeDescriptor("org.modellab.artifact.markov-simulation", "1.0", "Markov-chain simulation", STOCHASTIC),),
        _chain_applicable,
        simulate_markov_chain,
        _chain_units,
        ("org.modellab.renderer.plotly-state-probabilities",),
    ),
)


MANIFEST = PackManifest(
    PACK_ID,
    "1.1",
    "Probability, Stochastic Processes and Simulation",
    "Finite probability models, exact Markov evolution, controlled random streams and statistically explicit simulation artifacts.",
    ("org.modellab.pack.multidimensional-mathematics",),
    tuple(item.kind for item in KIND_DESCRIPTORS),
    tuple(item.identifier for item in CAPABILITY_DESCRIPTORS),
)
