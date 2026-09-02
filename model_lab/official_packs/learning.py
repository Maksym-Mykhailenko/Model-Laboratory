"""Machine Learning and Computational Intelligence official capability pack.

The pack represents data, trained parameterisations and rule systems as explicit Model
Graph objects.  Algorithms are deterministic under their recorded settings; models never
carry executable Python code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from ..model import ModelIR
from ..model_graph import ModelGraphError, ModelObject, ObjectKindDescriptor
from ..protocol import ArtifactTypeDescriptor, CapabilityDescriptor
from .common import PackManifest, finite_matrix, finite_vector, objects_of_kind, select_object, unique_labels


PACK_ID = "org.modellab.pack.machine-learning-computational-intelligence"
DATASET_KIND = "org.modellab.learning.feature-dataset"
SUPERVISED_KIND = "org.modellab.learning.supervised-study"
NETWORK_KIND = "org.modellab.learning.feedforward-network"
FUZZY_KIND = "org.modellab.intelligence.fuzzy-rule-system"

_ROW = {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "number"}}
_DATA_PROPERTIES = {
    "feature_names": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "string"}},
    "features": {"type": "array", "minItems": 2, "maxItems": 100000, "items": _ROW},
    "sample_ids": {"type": "array", "maxItems": 100000, "items": {"type": "string"}},
    "feature_units": {"type": "array", "maxItems": 256, "items": {"type": "string"}},
}

DATASET_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["feature_names", "features"],
    "properties": _DATA_PROPERTIES, "additionalProperties": False,
}
SUPERVISED_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["feature_names", "features", "target_name", "task", "targets"],
    "properties": {
        **_DATA_PROPERTIES,
        "target_name": {"type": "string", "minLength": 1, "maxLength": 128},
        "task": {"type": "string", "enum": ["regression", "classification"]},
        "targets": {"type": "array", "minItems": 2, "maxItems": 100000, "items": {"type": ["number", "string"]}},
    },
    "additionalProperties": False,
}

NETWORK_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["input_names", "output_names", "weights", "biases", "activations"],
    "properties": {
        "input_names": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "string"}},
        "output_names": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "string"}},
        "weights": {"type": "array", "minItems": 1, "maxItems": 64, "items": {"type": "array", "minItems": 1, "maxItems": 256, "items": _ROW}},
        "biases": {"type": "array", "minItems": 1, "maxItems": 64, "items": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "number"}}},
        "activations": {"type": "array", "minItems": 1, "maxItems": 64, "items": {"type": "string", "enum": ["linear", "relu", "tanh", "sigmoid", "softmax"]}},
    },
    "additionalProperties": False,
}

_FUZZY_INPUT = {
    "type": "object", "required": ["name", "minimum", "maximum", "sets"],
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 128},
        "minimum": {"type": "number"}, "maximum": {"type": "number"},
        "sets": {"type": "object"},
    }, "additionalProperties": False,
}
_FUZZY_RULE = {
    "type": "object", "required": ["antecedents", "consequent"],
    "properties": {
        "antecedents": {"type": "object"},
        "consequent": {"type": "string", "minLength": 1, "maxLength": 128},
        "weight": {"type": "number", "minimum": 0, "maximum": 1},
    }, "additionalProperties": False,
}
FUZZY_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["inputs", "output_name", "output_singletons", "rules"],
    "properties": {
        "inputs": {"type": "array", "minItems": 1, "maxItems": 32, "items": _FUZZY_INPUT},
        "output_name": {"type": "string", "minLength": 1, "maxLength": 128},
        "output_singletons": {"type": "object"},
        "rules": {"type": "array", "minItems": 1, "maxItems": 10000, "items": _FUZZY_RULE},
    }, "additionalProperties": False,
}

KIND_DESCRIPTORS = (
    ObjectKindDescriptor(DATASET_KIND, "1.0", "Finite feature dataset", DATASET_SCHEMA, True),
    ObjectKindDescriptor(SUPERVISED_KIND, "1.0", "Supervised machine-learning study", SUPERVISED_SCHEMA, True),
    ObjectKindDescriptor(NETWORK_KIND, "1.0", "Finite feed-forward neural network", NETWORK_SCHEMA, True),
    ObjectKindDescriptor(FUZZY_KIND, "1.0", "Zero-order Sugeno fuzzy rule system", FUZZY_SCHEMA, True),
)


@dataclass(frozen=True, slots=True)
class _Dataset:
    object_id: str
    feature_names: tuple[str, ...]
    features: np.ndarray
    sample_ids: tuple[str, ...]
    feature_units: tuple[str, ...]
    task: str | None = None
    target_name: str | None = None
    targets: np.ndarray | tuple[str, ...] | None = None
    class_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Network:
    object_id: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    activations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _FuzzySystem:
    object_id: str
    input_names: tuple[str, ...]
    domains: tuple[tuple[float, float], ...]
    sets: tuple[Mapping[str, tuple[float, float, float]], ...]
    output_name: str
    singletons: Mapping[str, float]
    rules: tuple[tuple[tuple[tuple[int, str], ...], str, float], ...]


def _labels(value: object, *, field: str, count: int, prefix: str) -> tuple[str, ...]:
    if value is None:
        return tuple(f"{prefix}{index + 1}" for index in range(count))
    labels = unique_labels(value, field=field)
    if len(labels) != count:
        raise ModelGraphError(f"{field} must align with the number of observations.")
    return labels


def _dataset(item: ModelObject, *, supervised: bool) -> _Dataset:
    names = unique_labels(item.properties["feature_names"], field=f"{item.identifier}.feature_names")
    values = finite_matrix(item.properties["features"], field=f"{item.identifier}.features")
    if values.shape[1] != len(names):
        raise ModelGraphError(f"{item.identifier}.features must align with feature_names.")
    samples = _labels(item.properties.get("sample_ids"), field=f"{item.identifier}.sample_ids", count=values.shape[0], prefix="sample-")
    raw_units = item.properties.get("feature_units")
    if raw_units is None:
        units = tuple("" for _ in names)
    elif not isinstance(raw_units, list) or len(raw_units) != len(names):
        raise ModelGraphError(f"{item.identifier}.feature_units must align with feature_names.")
    else:
        units = tuple(str(value) for value in raw_units)
    if not supervised:
        return _Dataset(item.identifier, names, values, samples, units)
    task = str(item.properties["task"])
    raw_targets = item.properties["targets"]
    if not isinstance(raw_targets, list) or len(raw_targets) != values.shape[0]:
        raise ModelGraphError(f"{item.identifier}.targets must align with features.")
    if task == "regression":
        targets: np.ndarray | tuple[str, ...] = finite_vector(raw_targets, field=f"{item.identifier}.targets")
        classes: tuple[str, ...] = ()
    else:
        labels = tuple(str(value) for value in raw_targets)
        classes = tuple(sorted(set(labels)))
        if not 2 <= len(classes) <= 256 or any(not value for value in labels):
            raise ModelGraphError(
                f"{item.identifier}.targets requires 2 to 256 non-empty classes."
            )
        targets = labels
    return _Dataset(
        item.identifier, names, values, samples, units, task, str(item.properties["target_name"]),
        targets, classes,
    )


def _network(item: ModelObject) -> _Network:
    inputs = unique_labels(item.properties["input_names"], field=f"{item.identifier}.input_names")
    outputs = unique_labels(item.properties["output_names"], field=f"{item.identifier}.output_names")
    raw_weights = item.properties["weights"]
    raw_biases = item.properties["biases"]
    raw_activations = item.properties["activations"]
    if not isinstance(raw_weights, list) or not isinstance(raw_biases, list) or not isinstance(raw_activations, list):
        raise ModelGraphError(f"{item.identifier} network layers must be lists.")
    if not (len(raw_weights) == len(raw_biases) == len(raw_activations)):
        raise ModelGraphError(f"{item.identifier} weights, biases and activations must have equal layer counts.")
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    width = len(inputs)
    total_parameters = 0
    for index, (raw_weight, raw_bias) in enumerate(zip(raw_weights, raw_biases)):
        weight = finite_matrix(raw_weight, field=f"{item.identifier}.weights[{index}]")
        bias = finite_vector(raw_bias, field=f"{item.identifier}.biases[{index}]")
        if weight.shape[1] != width or weight.shape[0] != bias.size:
            raise ModelGraphError(f"{item.identifier} layer {index} dimensions do not compose.")
        width = bias.size
        total_parameters += weight.size + bias.size
        weights.append(weight); biases.append(bias)
    if width != len(outputs) or total_parameters > 1_000_000:
        raise ModelGraphError(f"{item.identifier} output width or parameter budget is invalid.")
    activations = tuple(str(value) for value in raw_activations)
    if "softmax" in activations[:-1]:
        raise ModelGraphError(f"{item.identifier} softmax is permitted only in the final layer.")
    return _Network(item.identifier, inputs, outputs, tuple(weights), tuple(biases), activations)


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ModelGraphError(f"{field} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelGraphError(f"{field} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ModelGraphError(f"{field} must be a finite number.")
    return result


def _fuzzy_system(item: ModelObject) -> _FuzzySystem:
    raw_inputs = item.properties["inputs"]
    if not isinstance(raw_inputs, list):
        raise ModelGraphError(f"{item.identifier}.inputs must be a list.")
    names: list[str] = []
    domains: list[tuple[float, float]] = []
    sets: list[Mapping[str, tuple[float, float, float]]] = []
    for index, raw in enumerate(raw_inputs):
        if not isinstance(raw, Mapping):
            raise ModelGraphError(f"{item.identifier}.inputs[{index}] must be an object.")
        name = str(raw["name"])
        if not name or name in names:
            raise ModelGraphError(f"{item.identifier}.inputs requires unique non-empty names.")
        lower = _finite_number(raw["minimum"], field=f"{item.identifier}.inputs[{index}].minimum")
        upper = _finite_number(raw["maximum"], field=f"{item.identifier}.inputs[{index}].maximum")
        if lower >= upper or not isinstance(raw["sets"], Mapping) or not raw["sets"]:
            raise ModelGraphError(f"{item.identifier}.inputs[{index}] requires an increasing domain and fuzzy sets.")
        parsed: dict[str, tuple[float, float, float]] = {}
        for label, raw_triangle in raw["sets"].items():
            triangle = finite_vector(raw_triangle, field=f"{item.identifier}.inputs[{index}].sets.{label}")
            if triangle.size != 3 or not lower <= triangle[0] <= triangle[1] <= triangle[2] <= upper:
                raise ModelGraphError(f"{item.identifier} fuzzy set '{label}' is not a valid triangular set in its domain.")
            parsed[str(label)] = (float(triangle[0]), float(triangle[1]), float(triangle[2]))
        names.append(name); domains.append((lower, upper)); sets.append(parsed)
    raw_singletons = item.properties["output_singletons"]
    if not isinstance(raw_singletons, Mapping) or not raw_singletons:
        raise ModelGraphError(f"{item.identifier}.output_singletons must be a non-empty mapping.")
    singletons = {str(key): _finite_number(value, field=f"{item.identifier}.output_singletons.{key}") for key, value in raw_singletons.items()}
    raw_rules = item.properties["rules"]
    if not isinstance(raw_rules, list):
        raise ModelGraphError(f"{item.identifier}.rules must be a list.")
    lookup = {name: index for index, name in enumerate(names)}
    rules: list[tuple[tuple[tuple[int, str], ...], str, float]] = []
    for index, raw in enumerate(raw_rules):
        if not isinstance(raw, Mapping) or not isinstance(raw["antecedents"], Mapping) or not raw["antecedents"]:
            raise ModelGraphError(f"{item.identifier}.rules[{index}] requires antecedents.")
        antecedents: list[tuple[int, str]] = []
        for input_name, set_name in raw["antecedents"].items():
            if input_name not in lookup or str(set_name) not in sets[lookup[input_name]]:
                raise ModelGraphError(f"{item.identifier}.rules[{index}] references an unknown input or fuzzy set.")
            antecedents.append((lookup[input_name], str(set_name)))
        consequent = str(raw["consequent"])
        if consequent not in singletons:
            raise ModelGraphError(f"{item.identifier}.rules[{index}] references an unknown consequent.")
        weight = _finite_number(raw.get("weight", 1.0), field=f"{item.identifier}.rules[{index}].weight")
        if not 0.0 <= weight <= 1.0:
            raise ModelGraphError(f"{item.identifier}.rules[{index}].weight must lie in [0, 1].")
        rules.append((tuple(sorted(antecedents)), consequent, weight))
    return _FuzzySystem(item.identifier, tuple(names), tuple(domains), tuple(sets), str(item.properties["output_name"]), singletons, tuple(rules))


SEMANTIC_VALIDATORS = {
    DATASET_KIND: lambda item: _dataset(item, supervised=False),
    SUPERVISED_KIND: lambda item: _dataset(item, supervised=True),
    NETWORK_KIND: _network,
    FUZZY_KIND: _fuzzy_system,
}


@dataclass(frozen=True, slots=True)
class SupervisedLearningFit:
    object_id: str
    task: str
    feature_names: tuple[str, ...]
    target_name: str
    sample_ids: tuple[str, ...]
    class_names: tuple[str, ...]
    coefficients: np.ndarray
    target_values: np.ndarray
    predicted_values: np.ndarray
    predicted_class_indices: np.ndarray
    class_probabilities: np.ndarray
    metrics: Mapping[str, float]
    regularisation: float
    solver_record: Mapping[str, Any]


def fit_supervised_model(
    model: ModelIR, object_id: str | None = None, regularisation: float = 1e-6,
    max_iterations: int = 1000, tolerance: float = 1e-10,
) -> SupervisedLearningFit:
    if not math.isfinite(float(regularisation)) or float(regularisation) <= 0.0:
        raise ValueError("regularisation must be a positive finite number.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 10 <= max_iterations <= 10000:
        raise ValueError("max_iterations must be an integer between 10 and 10000.")
    if not math.isfinite(float(tolerance)) or not 0.0 < float(tolerance) <= 1e-2:
        raise ValueError("tolerance must lie in (0, 1e-2].")
    study = _dataset(select_object(model, SUPERVISED_KIND, object_id, label="supervised-learning study"), supervised=True)
    design = np.column_stack((np.ones(study.features.shape[0]), study.features))
    penalty = np.eye(design.shape[1]); penalty[0, 0] = 0.0
    if study.task == "regression":
        target = np.asarray(study.targets, dtype=np.float64)
        coefficients = np.linalg.solve(design.T @ design + float(regularisation) * penalty, design.T @ target)
        predicted = design @ coefficients
        residual = target - predicted
        mse = float(np.mean(residual * residual))
        total = float(np.sum((target - np.mean(target)) ** 2))
        metrics = {"mean_squared_error": mse, "root_mean_squared_error": math.sqrt(mse), "r_squared": float(1.0 - np.sum(residual * residual) / total) if total > 0 else 1.0}
        return SupervisedLearningFit(
            study.object_id, "regression", study.feature_names, str(study.target_name), study.sample_ids,
            (), coefficients.reshape(1, -1), target, predicted, np.asarray([], dtype=np.int64),
            np.empty((target.size, 0)), metrics, float(regularisation),
            {"scientific": {"converged": True, "algorithm": "closed-form-ridge"}, "presentation": {"iterations": 1}},
        )
    labels = tuple(str(value) for value in study.targets)  # type: ignore[arg-type]
    class_lookup = {name: index for index, name in enumerate(study.class_names)}
    target_indices = np.asarray([class_lookup[value] for value in labels], dtype=np.int64)
    class_count = len(study.class_names)
    reduced_shape = (class_count - 1, design.shape[1])
    # An orthonormal basis for the sum-to-zero class-logit subspace removes the
    # arbitrary reference class.  The likelihood and L2 penalty are consequently
    # invariant to a permutation or renaming of class labels.
    contrasts = np.zeros((class_count, class_count - 1), dtype=np.float64)
    for column in range(class_count - 1):
        scale = math.sqrt((column + 1) * (column + 2))
        contrasts[: column + 1, column] = 1.0 / scale
        contrasts[column + 1, column] = -(column + 1) / scale

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        reduced = flat.reshape(reduced_shape)
        full = contrasts @ reduced
        logits = design @ full.T
        log_probabilities = logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -float(np.sum(log_probabilities[np.arange(target_indices.size), target_indices]))
        loss += 0.5 * float(regularisation) * float(np.sum(full[:, 1:] ** 2))
        probabilities = np.exp(log_probabilities)
        indicator = np.eye(class_count)[target_indices]
        gradient = (probabilities - indicator).T @ design
        gradient[:, 1:] += float(regularisation) * full[:, 1:]
        return loss, (contrasts.T @ gradient).ravel()

    solution = minimize(
        objective, np.zeros(np.prod(reduced_shape), dtype=np.float64), jac=True,
        method="L-BFGS-B", options={"maxiter": max_iterations, "ftol": float(tolerance), "gtol": float(tolerance)},
    )
    if not solution.success:
        raise ValueError(f"Regularised multinomial fitting did not converge: {solution.message}")
    coefficients = contrasts @ solution.x.reshape(reduced_shape)
    logits = design @ coefficients.T
    probabilities = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    predicted_indices = np.argmax(probabilities, axis=1).astype(np.int64)
    cross_entropy = -float(np.mean(np.log(np.maximum(probabilities[np.arange(target_indices.size), target_indices], np.finfo(float).tiny))))
    metrics = {"accuracy": float(np.mean(predicted_indices == target_indices)), "cross_entropy": cross_entropy}
    return SupervisedLearningFit(
        study.object_id, "classification", study.feature_names, str(study.target_name), study.sample_ids,
        study.class_names, coefficients, target_indices, predicted_indices.astype(np.float64),
        predicted_indices, probabilities, metrics, float(regularisation),
        {"scientific": {"converged": True, "algorithm": "sum-to-zero-symmetric-multinomial-logistic"}, "presentation": {"iterations": int(solution.nit), "function_evaluations": int(solution.nfev)}},
    )


@dataclass(frozen=True, slots=True)
class ClusteringResult:
    object_id: str
    feature_names: tuple[str, ...]
    sample_ids: tuple[str, ...]
    feature_units: tuple[str, ...]
    features: np.ndarray
    feature_scaling: str
    coordinate_center: np.ndarray
    coordinate_scale: np.ndarray
    constant_features: tuple[str, ...]
    cluster_count: int
    centres: np.ndarray
    analysis_centres: np.ndarray
    assignments: np.ndarray
    cluster_sizes: np.ndarray
    inertia: float
    seed: int
    iterations: int
    converged: bool


def cluster_kmeans(
    model: ModelIR, object_id: str | None = None, clusters: int = 2, seed: int = 0,
    max_iterations: int = 300, tolerance: float = 1e-8,
    feature_scaling: str = "standardized",
) -> ClusteringResult:
    if isinstance(clusters, bool) or not isinstance(clusters, int) or not 2 <= clusters <= 256:
        raise ValueError("clusters must be an integer between 2 and 256.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 1 <= max_iterations <= 10000:
        raise ValueError("max_iterations must be an integer between 1 and 10000.")
    if not math.isfinite(float(tolerance)) or not 0.0 < float(tolerance) <= 1.0:
        raise ValueError("tolerance must lie in (0, 1].")
    if feature_scaling not in {"raw", "standardized"}:
        raise ValueError("feature_scaling must be raw or standardized.")
    item = select_object(model, (DATASET_KIND, SUPERVISED_KIND), object_id, label="feature dataset")
    data = _dataset(item, supervised=item.kind == SUPERVISED_KIND)
    if clusters > data.features.shape[0]:
        raise ValueError("clusters cannot exceed the number of samples.")
    center = np.mean(data.features, axis=0)
    raw_scale = np.std(data.features, axis=0, ddof=0)
    safe_scale = np.where(raw_scale > 0.0, raw_scale, 1.0)
    if feature_scaling == "standardized":
        analysis_features = (data.features - center) / safe_scale
        coordinate_center = center
        coordinate_scale = safe_scale
    else:
        analysis_features = data.features.copy()
        coordinate_center = np.zeros(data.features.shape[1], dtype=np.float64)
        coordinate_scale = np.ones(data.features.shape[1], dtype=np.float64)
    rng = np.random.default_rng(seed)
    centres = [analysis_features[int(rng.integers(analysis_features.shape[0]))].copy()]
    while len(centres) < clusters:
        distances = np.min(np.sum((analysis_features[:, None, :] - np.asarray(centres)[None, :, :]) ** 2, axis=2), axis=1)
        total = float(np.sum(distances))
        if total <= 0.0:
            candidates = [index for index, row in enumerate(analysis_features) if not any(np.array_equal(row, centre) for centre in centres)]
            if not candidates:
                raise ValueError("The dataset contains fewer distinct points than requested clusters.")
            chosen = candidates[0]
        else:
            chosen = int(rng.choice(analysis_features.shape[0], p=distances / total))
        centres.append(analysis_features[chosen].copy())
    centre_matrix = np.asarray(centres)
    converged = False
    assignments = np.full(analysis_features.shape[0], -1, dtype=np.int64)
    for iteration in range(1, max_iterations + 1):
        squared = np.sum((analysis_features[:, None, :] - centre_matrix[None, :, :]) ** 2, axis=2)
        proposed = np.argmin(squared, axis=1)
        # Lloyd iterations can temporarily lose a cluster. Move the most poorly represented
        # sample from a non-singleton cluster, deterministically, before recomputing centres.
        reserved: set[int] = set()
        for cluster in range(clusters):
            if np.any(proposed == cluster):
                continue
            nearest = squared[np.arange(analysis_features.shape[0]), proposed]
            candidates = np.argsort(-nearest, kind="stable")
            chosen = next(
                (
                    int(index) for index in candidates
                    if int(index) not in reserved and np.count_nonzero(proposed == proposed[index]) > 1
                ),
                None,
            )
            if chosen is None:
                raise ValueError("K-means cannot maintain the requested non-empty clusters.")
            proposed[chosen] = cluster
            reserved.add(chosen)
        updated = centre_matrix.copy()
        for cluster in range(clusters):
            updated[cluster] = np.mean(analysis_features[proposed == cluster], axis=0)
        shift = float(np.max(np.linalg.norm(updated - centre_matrix, axis=1)))
        centre_matrix = updated
        final_squared = np.sum(
            (analysis_features[:, None, :] - centre_matrix[None, :, :]) ** 2, axis=2
        )
        final_assignments = np.argmin(final_squared, axis=1)
        if shift <= float(tolerance) and np.array_equal(final_assignments, proposed):
            assignments = final_assignments
            converged = True
            break
        assignments = proposed
    if not converged:
        raise ValueError("K-means did not converge within max_iterations.")
    order = sorted(range(clusters), key=lambda index: tuple(float(value) for value in centre_matrix[index]))
    inverse = np.empty(clusters, dtype=np.int64)
    for canonical, original in enumerate(order):
        inverse[original] = canonical
    centre_matrix = centre_matrix[order]
    assignments = inverse[assignments]
    residual = analysis_features - centre_matrix[assignments]
    reported_centres = centre_matrix * coordinate_scale + coordinate_center
    return ClusteringResult(
        data.object_id, data.feature_names, data.sample_ids, data.feature_units, data.features,
        feature_scaling, coordinate_center, coordinate_scale,
        tuple(data.feature_names[index] for index in np.flatnonzero(raw_scale == 0.0)),
        clusters, reported_centres, centre_matrix, assignments,
        np.bincount(assignments, minlength=clusters), float(np.sum(residual * residual)), seed,
        iteration, True,
    )


def _activation(name: str, values: np.ndarray) -> np.ndarray:
    if name == "linear": return values
    if name == "relu": return np.maximum(values, 0.0)
    if name == "tanh": return np.tanh(values)
    if name == "sigmoid":
        positive = values >= 0.0
        result = np.empty_like(values)
        result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exponential = np.exp(values[~positive])
        result[~positive] = exponential / (1.0 + exponential)
        return result
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


@dataclass(frozen=True, slots=True)
class NeuralNetworkEvaluation:
    network_object_id: str
    dataset_object_id: str
    sample_ids: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    activations: tuple[str, ...]
    outputs: np.ndarray
    predicted_output_indices: np.ndarray
    layer_widths: tuple[int, ...]


def evaluate_feedforward_network(
    model: ModelIR, network_object_id: str | None = None, dataset_object_id: str | None = None,
) -> NeuralNetworkEvaluation:
    network = _network(select_object(model, NETWORK_KIND, network_object_id, label="feed-forward network"))
    item = select_object(model, (DATASET_KIND, SUPERVISED_KIND), dataset_object_id, label="feature dataset")
    data = _dataset(item, supervised=item.kind == SUPERVISED_KIND)
    if data.feature_names != network.input_names:
        raise ValueError("Network input_names must exactly match dataset feature_names.")
    values = data.features.copy()
    widths = [values.shape[1]]
    for weight, bias, activation in zip(network.weights, network.biases, network.activations):
        values = _activation(activation, values @ weight.T + bias)
        if not np.all(np.isfinite(values)):
            raise ValueError("Network evaluation produced non-finite values.")
        widths.append(values.shape[1])
    predicted = np.argmax(values, axis=1).astype(np.int64) if network.activations[-1] == "softmax" else np.asarray([], dtype=np.int64)
    return NeuralNetworkEvaluation(
        network.object_id, data.object_id, data.sample_ids, network.input_names, network.output_names,
        network.activations, values, predicted, tuple(widths),
    )


def _triangular(value: float, triangle: tuple[float, float, float]) -> float:
    left, centre, right = triangle
    if value < left or value > right:
        return 0.0
    if value == centre:
        return 1.0
    if value < centre:
        return 1.0 if centre == left else (value - left) / (centre - left)
    return 1.0 if right == centre else (right - value) / (right - centre)


@dataclass(frozen=True, slots=True)
class FuzzyInference:
    object_id: str
    input_names: tuple[str, ...]
    output_name: str
    points: np.ndarray
    outputs: np.ndarray
    total_firing_strengths: np.ndarray
    rule_count: int
    conjunction: str
    defuzzification: str


def evaluate_fuzzy_system(
    model: ModelIR, points: list[list[float]], object_id: str | None = None,
) -> FuzzyInference:
    system = _fuzzy_system(select_object(model, FUZZY_KIND, object_id, label="fuzzy-rule system"))
    values = finite_matrix(points, field="points")
    if values.shape[1] != len(system.input_names) or values.shape[0] > 10000:
        raise ValueError("points must contain at most 10000 rows aligned with fuzzy inputs.")
    for axis, (lower, upper) in enumerate(system.domains):
        if np.any(values[:, axis] < lower) or np.any(values[:, axis] > upper):
            raise ValueError(f"points for '{system.input_names[axis]}' must remain inside its declared domain.")
    outputs = np.empty(values.shape[0], dtype=np.float64)
    totals = np.empty(values.shape[0], dtype=np.float64)
    for row_index, row in enumerate(values):
        numerator = 0.0; denominator = 0.0
        for antecedents, consequent, rule_weight in system.rules:
            strength = rule_weight
            for input_index, set_name in antecedents:
                strength *= _triangular(float(row[input_index]), system.sets[input_index][set_name])
            numerator += strength * system.singletons[consequent]
            denominator += strength
        if denominator <= 0.0:
            raise ValueError(f"Fuzzy rule coverage is zero at point {row_index}.")
        outputs[row_index] = numerator / denominator; totals[row_index] = denominator
    return FuzzyInference(system.object_id, system.input_names, system.output_name, values, outputs, totals, len(system.rules), "product", "weighted-singleton-average")


def _kind_applicability(kinds: str | tuple[str, ...], label: str):
    def applicable(model: ModelIR) -> tuple[bool, str]:
        found = bool(objects_of_kind(model, kinds))
        return found, "" if found else f"an executable {label} object is required"
    return applicable


def _supervised_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(
        model, SUPERVISED_KIND, settings.get("object_id"), label="supervised-learning study"
    )
    properties = item.properties
    rows = len(properties["features"])
    columns = len(properties["feature_names"]) + 1
    if properties["task"] == "regression":
        return max(1, rows * columns * columns + columns**3)
    class_count = len({str(value) for value in properties["targets"]})
    return max(1, rows * columns * class_count * int(settings.get("max_iterations", 1000)))


def _clustering_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(
        model, (DATASET_KIND, SUPERVISED_KIND), settings.get("object_id"), label="feature dataset"
    )
    rows = len(item.properties["features"])
    columns = len(item.properties["feature_names"])
    return max(
        1,
        rows
        * columns
        * int(settings.get("clusters", 2))
        * int(settings.get("max_iterations", 300)),
    )


def _network_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    network = select_object(
        model, NETWORK_KIND, settings.get("network_object_id"), label="feed-forward network"
    )
    dataset = select_object(
        model, (DATASET_KIND, SUPERVISED_KIND), settings.get("dataset_object_id"),
        label="feature dataset",
    )
    parameters = sum(len(row) for matrix in network.properties["weights"] for row in matrix)
    parameters += sum(len(layer) for layer in network.properties["biases"])
    return max(1, len(dataset.properties["features"]) * parameters)


def _fuzzy_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    system = select_object(
        model, FUZZY_KIND, settings.get("object_id"), label="fuzzy-rule system"
    )
    return max(1, len(settings.get("points", [])) * len(system.properties["rules"]) * len(system.properties["inputs"]))


NUMERIC = "org.modellab.comparator.numeric"
CAPABILITY_DESCRIPTORS = (
    CapabilityDescriptor(
        "org.modellab.learning.fit-supervised-model", "1.1", PACK_ID,
        "Fit regularised supervised model", "Fit deterministic ridge regression or symmetric sum-to-zero multinomial logistic regression with explicit regularisation and convergence settings.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "regularisation": {"type": "number", "exclusiveMinimum": 0, "default": 1e-6}, "max_iterations": {"type": "integer", "minimum": 10, "maximum": 10000, "default": 1000}, "tolerance": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.01, "default": 1e-10}}, "additionalProperties": False},
        "numpy+scipy", (ArtifactTypeDescriptor("org.modellab.artifact.supervised-learning-fit", "1.1", "Supervised-learning fit", NUMERIC),),
        _kind_applicability(SUPERVISED_KIND, "supervised-learning study"), fit_supervised_model, _supervised_units,
        ("org.modellab.renderer.plotly-supervised-fit",),
    ),
    CapabilityDescriptor(
        "org.modellab.learning.cluster-kmeans", "1.1", PACK_ID,
        "Cluster feature dataset", "Run seeded k-means++ in explicit raw or standardized feature coordinates with deterministic tie-breaking and canonical cluster ordering.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "clusters": {"type": "integer", "minimum": 2, "maximum": 256, "default": 2}, "seed": {"type": "integer", "minimum": 0, "maximum": 4294967295, "default": 0}, "max_iterations": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 300}, "tolerance": {"type": "number", "exclusiveMinimum": 0, "maximum": 1, "default": 1e-8}, "feature_scaling": {"type": "string", "enum": ["raw", "standardized"], "default": "standardized"}}, "additionalProperties": False},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.kmeans-clustering", "1.1", "K-means clustering", NUMERIC),),
        _kind_applicability((DATASET_KIND, SUPERVISED_KIND), "feature dataset"), cluster_kmeans, _clustering_units,
        ("org.modellab.renderer.plotly-clustering",),
    ),
    CapabilityDescriptor(
        "org.modellab.learning.evaluate-feedforward-network", "1.0", PACK_ID,
        "Evaluate feed-forward network", "Evaluate an explicitly parameterised finite feed-forward network against a feature dataset.",
        {"type": "object", "properties": {"network_object_id": {"type": ["string", "null"]}, "dataset_object_id": {"type": ["string", "null"]}}, "additionalProperties": False},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.neural-network-evaluation", "1.0", "Neural-network evaluation", NUMERIC),),
        lambda model: (bool(objects_of_kind(model, NETWORK_KIND)) and bool(objects_of_kind(model, (DATASET_KIND, SUPERVISED_KIND))), "" if objects_of_kind(model, NETWORK_KIND) and objects_of_kind(model, (DATASET_KIND, SUPERVISED_KIND)) else "a network and feature dataset are required"),
        evaluate_feedforward_network, _network_units, ("org.modellab.renderer.plotly-network-output",),
    ),
    CapabilityDescriptor(
        "org.modellab.intelligence.evaluate-fuzzy-system", "1.0", PACK_ID,
        "Evaluate fuzzy rule system", "Evaluate a bounded zero-order Sugeno system using product conjunction and weighted-singleton defuzzification.",
        {"type": "object", "required": ["points"], "properties": {"object_id": {"type": ["string", "null"]}, "points": {"type": "array", "minItems": 1, "maxItems": 10000, "items": {"type": "array", "minItems": 1, "maxItems": 32, "items": {"type": "number"}}}}, "additionalProperties": False},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.fuzzy-inference", "1.0", "Fuzzy inference", NUMERIC),),
        _kind_applicability(FUZZY_KIND, "fuzzy-rule system"), evaluate_fuzzy_system, _fuzzy_units,
        ("org.modellab.renderer.plotly-fuzzy-inference",),
    ),
)

MANIFEST = PackManifest(
    PACK_ID, "1.1", "Machine Learning and Computational Intelligence",
    "Explicit feature studies, regularised supervised learners, seeded clustering, finite neural networks and transparent fuzzy inference.",
    ("org.modellab.pack.multidimensional-mathematics", "org.modellab.pack.statistical-inference-data-modelling", "org.modellab.pack.optimisation-estimation-inverse-problems"),
    tuple(item.kind for item in KIND_DESCRIPTORS), tuple(item.identifier for item in CAPABILITY_DESCRIPTORS),
)
