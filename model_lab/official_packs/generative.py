"""Generative Models, Inference and Decision Systems official pack."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..model import ModelIR
from ..model_graph import ModelGraphError, ModelObject, ObjectKindDescriptor
from ..protocol import ArtifactTypeDescriptor, CapabilityDescriptor
from .common import (
    PackManifest, entropy, probability_vector, select_object, softmax,
    stochastic_matrix, unique_labels,
)
from .graphs import NETWORK_KIND, _network


PACK_ID = "org.modellab.pack.generative-inference-decision-systems"
HMM_KIND = "org.modellab.generative.hidden-markov-model"
POMDP_KIND = "org.modellab.generative.pomdp"
ACTIVE_INFERENCE_KIND = "org.modellab.generative.active-inference-model"

LABELS = {"type": "array", "minItems": 1, "maxItems": 2048, "items": {"type": "string"}}
PROBABILITIES = {"type": "array", "minItems": 1, "maxItems": 2048, "items": {"type": "number", "minimum": 0, "maximum": 1}}
PROBABILITY_MATRIX = {"type": "array", "minItems": 1, "maxItems": 2048, "items": PROBABILITIES}

HMM_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["states", "observations", "initial", "transition", "emission"],
    "properties": {
        "states": LABELS, "observations": LABELS, "initial": PROBABILITIES,
        "transition": PROBABILITY_MATRIX, "emission": PROBABILITY_MATRIX,
    },
    "additionalProperties": False,
}

POMDP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["states", "observations", "actions", "initial", "transitions", "emissions", "rewards", "discount"],
    "properties": {
        "states": LABELS, "observations": LABELS, "actions": LABELS,
        "initial": PROBABILITIES,
        "transitions": {"type": "array", "minItems": 1, "maxItems": 512, "items": PROBABILITY_MATRIX},
        "emissions": PROBABILITY_MATRIX,
        "rewards": {
            "type": "array", "minItems": 1, "maxItems": 2048,
            "items": {"type": "array", "minItems": 1, "maxItems": 512, "items": {"type": "number"}},
        },
        "discount": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "additionalProperties": False,
}

ACTIVE_INFERENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "states", "observations", "actions", "initial", "transitions", "likelihood",
        "preferences", "policies",
    ],
    "properties": {
        "states": LABELS, "observations": LABELS, "actions": LABELS,
        "initial": PROBABILITIES,
        "transitions": {"type": "array", "minItems": 1, "maxItems": 512, "items": PROBABILITY_MATRIX},
        "likelihood": PROBABILITY_MATRIX,
        "preferences": {"type": "array", "minItems": 1, "maxItems": 2048, "items": {"type": "number"}},
        "policies": {
            "type": "array", "minItems": 1, "maxItems": 4096,
            "items": {
                "type": "object", "required": ["name", "actions"],
                "properties": {"name": {"type": "string"}, "actions": LABELS},
                "additionalProperties": False,
            },
        },
        "policy_prior": PROBABILITIES,
        "policy_precision": {"type": "number", "minimum": 0},
        "likelihood_precision": {"type": "number", "minimum": 0},
        "events": {
            "type": "array", "maxItems": 256,
            "items": {
                "type": "object", "required": ["name", "states"],
                "properties": {"name": {"type": "string"}, "states": LABELS},
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

KIND_DESCRIPTORS = (
    ObjectKindDescriptor(HMM_KIND, "1.0", "Finite hidden Markov model", HMM_SCHEMA, True),
    ObjectKindDescriptor(POMDP_KIND, "1.0", "Finite partially observable Markov decision process", POMDP_SCHEMA, True),
    ObjectKindDescriptor(ACTIVE_INFERENCE_KIND, "1.0", "Finite active-inference generative model", ACTIVE_INFERENCE_SCHEMA, True),
)


def _hmm(item: ModelObject) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    states = unique_labels(item.properties["states"], field=f"{item.identifier}.states")
    observations = unique_labels(item.properties["observations"], field=f"{item.identifier}.observations")
    initial = probability_vector(item.properties["initial"], field=f"{item.identifier}.initial", size=len(states))
    transition = stochastic_matrix(item.properties["transition"], field=f"{item.identifier}.transition", rows=len(states), columns=len(states))
    emission = stochastic_matrix(item.properties["emission"], field=f"{item.identifier}.emission", rows=len(states), columns=len(observations))
    return states, observations, initial, transition, emission


def _pomdp(item: ModelObject) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    states = unique_labels(item.properties["states"], field=f"{item.identifier}.states")
    observations = unique_labels(item.properties["observations"], field=f"{item.identifier}.observations")
    actions = unique_labels(item.properties["actions"], field=f"{item.identifier}.actions")
    initial = probability_vector(item.properties["initial"], field=f"{item.identifier}.initial", size=len(states))
    raw_transitions = item.properties["transitions"]
    if len(raw_transitions) != len(actions):
        raise ModelGraphError(f"{item.identifier}.transitions must contain one matrix per action.")
    transitions = np.stack([
        stochastic_matrix(value, field=f"{item.identifier}.transitions[{index}]", rows=len(states), columns=len(states))
        for index, value in enumerate(raw_transitions)
    ])
    emissions = stochastic_matrix(item.properties["emissions"], field=f"{item.identifier}.emissions", rows=len(states), columns=len(observations))
    rewards = np.asarray(item.properties["rewards"], dtype=np.float64)
    if rewards.shape != (len(states), len(actions)) or not np.all(np.isfinite(rewards)):
        raise ModelGraphError(f"{item.identifier}.rewards must have shape states by actions and be finite.")
    discount = float(item.properties["discount"])
    if not math.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise ModelGraphError(f"{item.identifier}.discount must be between zero and one.")
    return states, observations, actions, initial, transitions, emissions, rewards, discount


def _active(item: ModelObject) -> tuple[
    tuple[str, ...], tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, tuple[tuple[str, tuple[int, ...]], ...], np.ndarray, float, float,
    tuple[tuple[str, tuple[int, ...]], ...],
]:
    p = item.properties
    states = unique_labels(p["states"], field=f"{item.identifier}.states")
    observations = unique_labels(p["observations"], field=f"{item.identifier}.observations")
    actions = unique_labels(p["actions"], field=f"{item.identifier}.actions")
    initial = probability_vector(p["initial"], field=f"{item.identifier}.initial", size=len(states))
    if len(p["transitions"]) != len(actions):
        raise ModelGraphError(f"{item.identifier}.transitions must contain one matrix per action.")
    transitions = np.stack([
        stochastic_matrix(value, field=f"{item.identifier}.transitions[{index}]", rows=len(states), columns=len(states))
        for index, value in enumerate(p["transitions"])
    ])
    likelihood = stochastic_matrix(p["likelihood"], field=f"{item.identifier}.likelihood", rows=len(states), columns=len(observations))
    preferences = np.asarray(p["preferences"], dtype=np.float64)
    if preferences.shape != (len(observations),) or not np.all(np.isfinite(preferences)):
        raise ModelGraphError(f"{item.identifier}.preferences must contain one finite value per observation.")
    action_index = {name: index for index, name in enumerate(actions)}
    policy_names: set[str] = set()
    policies: list[tuple[str, tuple[int, ...]]] = []
    horizon: int | None = None
    for index, policy in enumerate(p["policies"]):
        name = str(policy["name"])
        if not name.strip() or name in policy_names:
            raise ModelGraphError(f"{item.identifier}.policies requires unique non-empty names.")
        policy_names.add(name)
        try:
            sequence = tuple(action_index[str(action)] for action in policy["actions"])
        except KeyError as exc:
            raise ModelGraphError(f"{item.identifier}.policies[{index}] names an unknown action.") from exc
        if horizon is None:
            horizon = len(sequence)
        if not sequence or len(sequence) != horizon:
            raise ModelGraphError(f"{item.identifier}.policies must be non-empty and share one horizon.")
        policies.append((name, sequence))
    prior = probability_vector(p.get("policy_prior", [1.0 / len(policies)] * len(policies)), field=f"{item.identifier}.policy_prior", size=len(policies))
    policy_precision, likelihood_precision = float(p.get("policy_precision", 1.0)), float(p.get("likelihood_precision", 1.0))
    if not all(math.isfinite(value) and value >= 0.0 for value in (policy_precision, likelihood_precision)):
        raise ModelGraphError(f"{item.identifier} precisions must be finite and non-negative.")
    state_index = {name: index for index, name in enumerate(states)}
    event_names: set[str] = set()
    events: list[tuple[str, tuple[int, ...]]] = []
    for index, event in enumerate(p.get("events", [])):
        name = str(event["name"])
        if not name.strip() or name in event_names:
            raise ModelGraphError(f"{item.identifier}.events requires unique non-empty names.")
        event_names.add(name)
        try:
            indices = tuple(sorted({state_index[str(state)] for state in event["states"]}))
        except KeyError as exc:
            raise ModelGraphError(f"{item.identifier}.events[{index}] names an unknown state.") from exc
        events.append((name, indices))
    return states, observations, actions, initial, transitions, likelihood, preferences, tuple(policies), prior, policy_precision, likelihood_precision, tuple(events)


def validate_hmm(item: ModelObject) -> None: _hmm(item)
def validate_pomdp(item: ModelObject) -> None: _pomdp(item)
def validate_active_inference(item: ModelObject) -> None: _active(item)

SEMANTIC_VALIDATORS = {
    HMM_KIND: validate_hmm, POMDP_KIND: validate_pomdp,
    ACTIVE_INFERENCE_KIND: validate_active_inference,
}


@dataclass(frozen=True, slots=True)
class HiddenMarkovInference:
    object_id: str
    states: tuple[str, ...]
    observations: tuple[str, ...]
    observed_sequence: tuple[str, ...]
    filtered_probabilities: np.ndarray
    smoothed_probabilities: np.ndarray
    viterbi_path: tuple[str, ...]
    log_evidence: float


def infer_hidden_markov_model(
    model: ModelIR, object_id: str | None = None, observed_sequence: Sequence[str] = ()
) -> HiddenMarkovInference:
    item = select_object(model, HMM_KIND, object_id, label="hidden Markov model")
    states, observations, initial, transition, emission = _hmm(item)
    if not observed_sequence:
        raise ValueError("observed_sequence must contain at least one observation label.")
    observation_index = {name: index for index, name in enumerate(observations)}
    try:
        sequence = tuple(observation_index[str(value)] for value in observed_sequence)
    except KeyError as exc:
        raise ValueError(f"Unknown observation {exc.args[0]!r}.") from exc
    count, size = len(sequence), len(states)
    filtered = np.empty((count, size), dtype=np.float64)
    scales = np.empty(count, dtype=np.float64)
    current = initial * emission[:, sequence[0]]
    scales[0] = float(np.sum(current))
    if scales[0] <= 0.0:
        raise ValueError("The observed sequence has zero probability under the model.")
    filtered[0] = current / scales[0]
    for time in range(1, count):
        current = (filtered[time - 1] @ transition) * emission[:, sequence[time]]
        scales[time] = float(np.sum(current))
        if scales[time] <= 0.0:
            raise ValueError("The observed sequence has zero probability under the model.")
        filtered[time] = current / scales[time]
    with np.errstate(divide="ignore"):
        log_initial = np.where(initial > 0.0, np.log(initial), -np.inf)
        log_transition = np.where(transition > 0.0, np.log(transition), -np.inf)
        log_emission = np.where(emission > 0.0, np.log(emission), -np.inf)
        log_filtered = np.where(filtered > 0.0, np.log(filtered), -np.inf)
    smoothed = filtered.copy()
    log_backward = np.zeros(size, dtype=np.float64)
    for time in range(count - 2, -1, -1):
        candidates = log_transition + (log_emission[:, sequence[time + 1]] + log_backward)[None, :]
        maxima = np.max(candidates, axis=1)
        log_next = np.full(size, -np.inf, dtype=np.float64)
        finite_rows = np.isfinite(maxima)
        if np.any(finite_rows):
            log_next[finite_rows] = maxima[finite_rows] + np.log(np.sum(
                np.exp(candidates[finite_rows] - maxima[finite_rows, None]), axis=1
            ))
        finite = np.isfinite(log_next)
        if not np.any(finite):
            raise ValueError("Scaled hidden-state smoothing became numerically invalid.")
        # Subtracting a common log scale at every step is the backward analogue of
        # forward scaling and preserves arbitrarily large structural likelihood ratios.
        log_backward = log_next - float(np.max(log_next[finite]))
        log_smoothed = log_filtered[time] + log_backward
        finite_smoothed = np.isfinite(log_smoothed)
        if not np.any(finite_smoothed):
            raise ValueError("Scaled hidden-state smoothing became numerically invalid.")
        maximum = float(np.max(log_smoothed[finite_smoothed]))
        current = np.where(finite_smoothed, np.exp(log_smoothed - maximum), 0.0)
        smoothed[time] = current / float(np.sum(current))
    delta = log_initial + log_emission[:, sequence[0]]
    if not np.any(np.isfinite(delta)):
        raise ValueError("The observed sequence has no structurally possible Viterbi path.")
    backpointers = np.zeros((count, size), dtype=np.int64)
    for time in range(1, count):
        candidates = delta[:, None] + log_transition
        backpointers[time] = np.argmax(candidates, axis=0)
        delta = np.max(candidates, axis=0) + log_emission[:, sequence[time]]
        if not np.any(np.isfinite(delta)):
            raise ValueError("The observed sequence has no structurally possible Viterbi path.")
    path = [int(np.argmax(delta))]
    for time in range(count - 1, 0, -1):
        path.append(int(backpointers[time, path[-1]]))
    path.reverse()
    return HiddenMarkovInference(
        item.identifier, states, observations, tuple(str(value) for value in observed_sequence),
        filtered, smoothed, tuple(states[index] for index in path), float(np.sum(np.log(scales))),
    )


@dataclass(frozen=True, slots=True)
class POMDPDecision:
    object_id: str
    states: tuple[str, ...]
    observations: tuple[str, ...]
    actions: tuple[str, ...]
    horizon: int
    initial_belief: np.ndarray
    initial_action_values: np.ndarray
    recommended_action: str
    optimal_value: float
    evaluated_beliefs: int
    observation_contingencies: tuple[Mapping[str, Any], ...]


def solve_pomdp(
    model: ModelIR, object_id: str | None = None, horizon: int = 5,
) -> POMDPDecision:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 10:
        raise ValueError("horizon must be an integer from 1 to 10.")
    item = select_object(model, POMDP_KIND, object_id, label="POMDP")
    states, observations, actions, initial, transitions, emissions, rewards, discount = _pomdp(item)
    evaluated = 0

    def action_values(belief: np.ndarray, remaining: int) -> np.ndarray:
        nonlocal evaluated
        evaluated += 1
        values = np.empty(len(actions), dtype=np.float64)
        for action_index in range(len(actions)):
            value = float(np.dot(belief, rewards[:, action_index]))
            if remaining > 1:
                predicted = belief @ transitions[action_index]
                observation_probability = predicted @ emissions
                future = 0.0
                for observation_index, probability in enumerate(observation_probability):
                    if probability <= 0.0:
                        continue
                    posterior = predicted * emissions[:, observation_index] / float(probability)
                    future += float(probability) * float(np.max(action_values(posterior, remaining - 1)))
                value += discount * future
            values[action_index] = value
        return values

    initial_action_values = action_values(initial, horizon)
    recommended_index = int(np.argmax(initial_action_values))
    predicted = initial @ transitions[recommended_index]
    observation_probability = predicted @ emissions
    contingencies: list[Mapping[str, Any]] = []
    for observation_index, probability in enumerate(observation_probability):
        if probability <= 0.0:
            continue
        posterior = predicted * emissions[:, observation_index] / float(probability)
        if horizon > 1:
            next_values = action_values(posterior, horizon - 1)
            next_action, next_value = actions[int(np.argmax(next_values))], float(np.max(next_values))
        else:
            next_action, next_value = None, None
        contingencies.append({
            "observation": observations[observation_index],
            "probability": float(probability),
            "posterior_belief": posterior,
            "next_action": next_action,
            "continuation_value": next_value,
        })
    return POMDPDecision(
        item.identifier, states, observations, actions, horizon, initial,
        initial_action_values, actions[recommended_index],
        float(initial_action_values[recommended_index]), evaluated, tuple(contingencies),
    )


@dataclass(frozen=True, slots=True)
class MarkovBlanketAnalysis:
    network_object_id: str
    target: str
    parents: tuple[str, ...]
    children: tuple[str, ...]
    co_parents: tuple[str, ...]
    blanket: tuple[str, ...]
    nodes: tuple[str, ...]
    edges: tuple[Mapping[str, Any], ...]


def analyse_markov_blanket(
    model: ModelIR, network_object_id: str | None = None, target: str | None = None
) -> MarkovBlanketAnalysis:
    item = select_object(model, NETWORK_KIND, network_object_id, label="directed network")
    nodes, edges, directed, _ = _network(item)
    if not directed:
        raise ValueError("Markov-blanket analysis requires a directed network.")
    chosen = nodes[0] if target is None else str(target)
    if chosen not in set(nodes):
        raise ValueError("target must name a network node.")
    parents = {str(edge["source"]) for edge in edges if edge["target"] == chosen}
    children = {str(edge["target"]) for edge in edges if edge["source"] == chosen}
    co_parents = {
        str(edge["source"]) for edge in edges
        if edge["target"] in children and edge["source"] != chosen
    }
    blanket = parents | children | co_parents
    return MarkovBlanketAnalysis(
        item.identifier, chosen, tuple(sorted(parents)), tuple(sorted(children)),
        tuple(sorted(co_parents)), tuple(sorted(blanket)), nodes, edges,
    )


def _precision_rows(matrix: np.ndarray, precision: float) -> np.ndarray:
    if precision == 1.0:
        return matrix.copy()
    result = np.zeros_like(matrix)
    for index, row in enumerate(matrix):
        support = row > 0.0
        if not np.any(support):
            continue
        result[index, support] = softmax(precision * np.log(row[support]))
    return result


@dataclass(frozen=True, slots=True)
class ActiveInferenceEvaluation:
    object_id: str
    states: tuple[str, ...]
    observations: tuple[str, ...]
    actions: tuple[str, ...]
    policy_names: tuple[str, ...]
    expected_free_energy: np.ndarray
    risk: np.ndarray
    ambiguity: np.ndarray
    policy_posterior: np.ndarray
    state_trajectory_by_policy: np.ndarray
    observation_trajectory_by_policy: np.ndarray
    posterior_state_trajectory: np.ndarray
    event_probabilities: tuple[Mapping[str, Any], ...]


def evaluate_active_inference(
    model: ModelIR, object_id: str | None = None,
    policy_precision: float | None = None, likelihood_precision: float | None = None,
) -> ActiveInferenceEvaluation:
    item = select_object(model, ACTIVE_INFERENCE_KIND, object_id, label="active-inference model")
    states, observations, actions, initial, transitions, likelihood, preferences, policies, prior, default_gamma, default_zeta, events = _active(item)
    gamma = default_gamma if policy_precision is None else float(policy_precision)
    zeta = default_zeta if likelihood_precision is None else float(likelihood_precision)
    if not all(math.isfinite(value) and value >= 0.0 for value in (gamma, zeta)):
        raise ValueError("Precisions must be finite and non-negative.")
    likelihood = _precision_rows(likelihood, zeta)
    preferred = softmax(preferences)
    horizon = len(policies[0][1])
    state_paths = np.empty((len(policies), horizon + 1, len(states)), dtype=np.float64)
    observation_paths = np.empty((len(policies), horizon + 1, len(observations)), dtype=np.float64)
    risks = np.zeros(len(policies), dtype=np.float64)
    ambiguities = np.zeros(len(policies), dtype=np.float64)
    likelihood_entropy = np.asarray(entropy(likelihood, axis=1), dtype=np.float64)
    event_rows: list[Mapping[str, Any]] = []
    for policy_index, (name, sequence) in enumerate(policies):
        state_paths[policy_index, 0] = initial
        observation_paths[policy_index, 0] = initial @ likelihood
        surviving = {}
        for event_name, indices in events:
            outside = initial.copy()
            outside[list(indices)] = 0.0
            surviving[event_name] = outside
        for time, action_index in enumerate(sequence, start=1):
            state_paths[policy_index, time] = state_paths[policy_index, time - 1] @ transitions[action_index]
            observation = state_paths[policy_index, time] @ likelihood
            observation_paths[policy_index, time] = observation
            risks[policy_index] += float(np.sum(np.where(observation > 0.0, observation * (np.log(observation) - np.log(np.maximum(preferred, np.finfo(float).tiny))), 0.0)))
            ambiguities[policy_index] += float(np.dot(state_paths[policy_index, time], likelihood_entropy))
            for event_name, indices in events:
                outside = surviving[event_name] @ transitions[action_index]
                outside[list(indices)] = 0.0
                surviving[event_name] = outside
        for event_name, indices in events:
            event_rows.append({
                "policy": name, "event": event_name,
                "terminal_probability": float(np.sum(state_paths[policy_index, -1, list(indices)])),
                "reach_probability": float(1.0 - np.sum(surviving[event_name])),
            })
    free_energy = risks + ambiguities
    posterior = softmax(np.log(np.maximum(prior, np.finfo(float).tiny)) - gamma * free_energy)
    aggregated = np.einsum("p,pts->ts", posterior, state_paths)
    return ActiveInferenceEvaluation(
        item.identifier, states, observations, actions, tuple(name for name, _ in policies),
        free_energy, risks, ambiguities, posterior, state_paths, observation_paths,
        aggregated, tuple(event_rows),
    )


def _kind_applicability(kind: str, label: str):
    def applicable(model: ModelIR) -> tuple[bool, str]:
        found = any(item.kind == kind and not item.opaque for item in model.graph.objects)
        return found, f"an executable {label} object is required"
    return applicable


def _blanket_applicable(model: ModelIR) -> tuple[bool, str]:
    found = any(item.kind == NETWORK_KIND and not item.opaque and item.properties.get("directed") for item in model.graph.objects)
    return found, "an executable directed network object is required"


def _object_units(kind: str, factor: int = 1, *, object_id_key: str = "object_id"):
    def units(settings: Mapping[str, Any], model: ModelIR) -> int:
        item = select_object(
            model, kind, settings.get(object_id_key), label="official generative-system"
        )
        size = len(item.properties.get("states", item.properties.get("nodes", [])))
        return max(1, factor * size**2)
    return units


def _hmm_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(
        model, HMM_KIND, settings.get("object_id"), label="hidden Markov model"
    )
    states = len(item.properties["states"])
    return max(1, len(settings.get("observed_sequence", ())) * states * states)


def _pomdp_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(model, POMDP_KIND, settings.get("object_id"), label="POMDP")
    properties = item.properties
    horizon = int(settings.get("horizon", 5))
    branches = len(properties["actions"]) * len(properties["observations"])
    nodes = horizon if branches == 1 else (branches ** horizon - 1) // (branches - 1)
    return max(1, nodes * len(properties["states"]) * len(properties["actions"]))


NUMERIC = "org.modellab.comparator.numeric"
EXACT = "org.modellab.comparator.exact"

CAPABILITY_DESCRIPTORS = (
    CapabilityDescriptor(
        "org.modellab.generative.infer-hidden-markov-model", "1.1", PACK_ID,
        "Infer hidden-state sequence", "Run scaled filtering, smoothing, evidence and Viterbi inference for a finite hidden Markov model.",
        {"type": "object", "required": ["observed_sequence"], "properties": {"object_id": {"type": ["string", "null"]}, "observed_sequence": {"type": "array", "minItems": 1, "maxItems": 100000, "items": {"type": "string"}}}, "additionalProperties": False},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.hidden-markov-inference", "1.1", "Hidden Markov inference", NUMERIC),),
        _kind_applicability(HMM_KIND, "hidden Markov model"), infer_hidden_markov_model, _hmm_units,
        ("org.modellab.renderer.plotly-hidden-state-posterior",),
    ),
    CapabilityDescriptor(
        "org.modellab.generative.solve-pomdp", "1.0", PACK_ID,
        "Solve finite decision system", "Compute an exact bounded finite-horizon belief policy with observation-contingent continuation decisions.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "horizon": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}}},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.pomdp-decision", "1.0", "POMDP decision analysis", NUMERIC),),
        _kind_applicability(POMDP_KIND, "POMDP"), solve_pomdp, _pomdp_units,
        ("org.modellab.renderer.plotly-decision-values",),
    ),
    CapabilityDescriptor(
        "org.modellab.generative.markov-blanket", "1.0", PACK_ID,
        "Discover Markov blanket", "Identify parents, children and co-parents of a target in a directed dependency network.",
        {"type": "object", "properties": {"network_object_id": {"type": ["string", "null"]}, "target": {"type": ["string", "null"]}}},
        "python", (ArtifactTypeDescriptor("org.modellab.artifact.markov-blanket", "1.0", "Markov blanket", EXACT),),
        _blanket_applicable, analyse_markov_blanket,
        _object_units(NETWORK_KIND, object_id_key="network_object_id"),
        ("org.modellab.renderer.plotly-network",),
    ),
    CapabilityDescriptor(
        "org.modellab.generative.evaluate-active-inference", "1.0", PACK_ID,
        "Evaluate active-inference policies", "Propagate finite policies and compare expected free energy, risk, ambiguity, posterior policy mass and event probabilities.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "policy_precision": {"type": ["number", "null"]}, "likelihood_precision": {"type": ["number", "null"]}}},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.active-inference-evaluation", "1.0", "Active-inference policy evaluation", NUMERIC),),
        _kind_applicability(ACTIVE_INFERENCE_KIND, "active-inference model"), evaluate_active_inference, _object_units(ACTIVE_INFERENCE_KIND, 100),
        ("org.modellab.renderer.plotly-policy-probabilities",),
    ),
)


MANIFEST = PackManifest(
    PACK_ID, "1.1", "Generative Models, Inference and Decision Systems",
    "Broad finite generative systems: hidden-state inference, decision processes, Markov blankets, active inference, predictive-processing primitives and free-energy policy evaluation.",
    (
        "org.modellab.pack.multidimensional-mathematics",
        "org.modellab.pack.probability-stochastic-systems",
        "org.modellab.pack.graphs-networks-discrete",
    ),
    tuple(item.kind for item in KIND_DESCRIPTORS),
    tuple(item.identifier for item in CAPABILITY_DESCRIPTORS),
)
