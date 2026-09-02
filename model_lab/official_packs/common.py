"""Shared validation and selection helpers for official scientific packs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..model import ModelIR
from ..model_graph import ModelGraph, ModelGraphError, ModelObject
from ..protocol import SpectralDecomposition, SubspaceRepresentation


PROBABILITY_TOLERANCE = 1e-12
MAX_INLINE_VALUES = 200_000


def objects_of_kind(model: ModelIR, kinds: str | Iterable[str]) -> tuple[ModelObject, ...]:
    accepted = {kinds} if isinstance(kinds, str) else set(kinds)
    return tuple(item for item in model.graph.objects if item.kind in accepted and not item.opaque)


def select_object(
    model: ModelIR,
    kinds: str | Iterable[str],
    object_id: str | None,
    *,
    label: str,
) -> ModelObject:
    candidates = objects_of_kind(model, kinds)
    if object_id is None:
        if not candidates:
            raise ValueError(f"The model contains no executable {label} object.")
        return candidates[0]
    try:
        selected = model.graph.object(str(object_id))
    except KeyError as exc:
        raise ValueError(f"Unknown Model Graph object '{object_id}'.") from exc
    if selected not in candidates:
        raise ValueError(f"Model Graph object '{object_id}' is not an executable {label}.")
    return selected


def finite_vector(value: object, *, field: str, nonempty: bool = True) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelGraphError(f"{field} must contain real numbers.") from exc
    if result.ndim != 1 or (nonempty and result.size == 0):
        raise ModelGraphError(f"{field} must be a non-empty one-dimensional array.")
    if result.size > MAX_INLINE_VALUES or not np.all(np.isfinite(result)):
        raise ModelGraphError(f"{field} exceeds the finite inline-value contract.")
    result = result.copy()
    result[result == 0.0] = 0.0
    return result


def finite_matrix(value: object, *, field: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelGraphError(f"{field} must be a rectangular real matrix.") from exc
    if result.ndim != 2 or not result.shape[0] or not result.shape[1]:
        raise ModelGraphError(f"{field} must be a non-empty rectangular matrix.")
    if result.size > MAX_INLINE_VALUES or not np.all(np.isfinite(result)):
        raise ModelGraphError(f"{field} exceeds the finite inline-value contract.")
    result = result.copy()
    result[result == 0.0] = 0.0
    return result


def probability_vector(value: object, *, field: str, size: int | None = None) -> np.ndarray:
    result = finite_vector(value, field=field)
    if size is not None and result.size != size:
        raise ModelGraphError(f"{field} must contain exactly {size} probabilities.")
    if np.any(result < 0.0) or not math.isclose(
        float(np.sum(result)), 1.0, rel_tol=0.0, abs_tol=PROBABILITY_TOLERANCE
    ):
        raise ModelGraphError(f"{field} must be non-negative and sum to one.")
    return result


def stochastic_matrix(
    value: object,
    *,
    field: str,
    rows: int | None = None,
    columns: int | None = None,
) -> np.ndarray:
    result = finite_matrix(value, field=field)
    if rows is not None and result.shape[0] != rows:
        raise ModelGraphError(f"{field} must contain {rows} rows.")
    if columns is not None and result.shape[1] != columns:
        raise ModelGraphError(f"{field} must contain {columns} columns.")
    if np.any(result < 0.0) or not np.allclose(
        np.sum(result, axis=1), 1.0, rtol=0.0, atol=PROBABILITY_TOLERANCE
    ):
        raise ModelGraphError(f"Every row of {field} must be non-negative and sum to one.")
    return result


def unique_labels(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ModelGraphError(f"{field} must be a list of labels.")
    labels = tuple(str(item) for item in value)
    if not labels or any(not item.strip() or len(item.encode("utf-8")) > 256 for item in labels):
        raise ModelGraphError(f"{field} contains an empty or oversized label.")
    if len(labels) != len(set(labels)):
        raise ModelGraphError(f"{field} labels must be unique.")
    return labels


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=np.float64) - float(np.max(values))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def _rounded_projector(value: np.ndarray) -> np.ndarray:
    projector = np.round((value + value.T) / 2.0, decimals=13)
    projector[np.abs(projector) < 5e-14] = 0.0
    return projector


def subspace_representation(basis: np.ndarray, *, orientation: str) -> SubspaceRepresentation:
    values = np.asarray(basis, dtype=np.float64).copy()
    if orientation not in {"rows", "columns"}:
        raise ValueError("Subspace basis orientation must be rows or columns.")
    columns = values.T if orientation == "rows" else values
    for column in range(columns.shape[1]):
        pivot = int(np.argmax(np.abs(columns[:, column])))
        if columns[pivot, column] < 0.0:
            columns[:, column] *= -1.0
    projector = _rounded_projector(columns @ columns.T)
    display = columns.T.copy() if orientation == "rows" else columns.copy()
    return SubspaceRepresentation(columns.shape[1], columns.shape[0], projector, display, orientation)


def symmetric_spectral_decomposition(
    matrix: np.ndarray, *, descending: bool, basis_orientation: str,
) -> tuple[np.ndarray, SpectralDecomposition]:
    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=np.float64))
    order = np.argsort(values)[::-1] if descending else np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    for column in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        if vectors[pivot, column] < 0.0:
            vectors[:, column] *= -1.0
    scale = max(1.0, float(np.max(np.abs(values), initial=0.0)))
    tolerance = scale * 1e-12
    groups: list[list[int]] = []
    for index, value in enumerate(values):
        if not groups or abs(float(value - values[groups[-1][0]])) > tolerance:
            groups.append([index])
        else:
            groups[-1].append(index)
    eigenspaces: list[Mapping[str, Any]] = []
    for group in groups:
        subspace = vectors[:, group]
        eigenspaces.append({
            "multiplicity": len(group),
            "eigenvalue_minimum": float(np.min(values[group])),
            "eigenvalue_maximum": float(np.max(values[group])),
            "orthogonal_projector": _rounded_projector(subspace @ subspace.T),
        })
    display = vectors.T.copy() if basis_orientation == "rows" else vectors.copy()
    return values, SpectralDecomposition(values.copy(), tuple(eigenspaces), display, basis_orientation)


def entropy(probabilities: np.ndarray, *, axis: int | None = None) -> np.ndarray | float:
    values = np.asarray(probabilities, dtype=np.float64)
    terms = np.zeros_like(values)
    positive = values > 0.0
    terms[positive] = values[positive] * np.log(values[positive])
    result = -np.sum(terms, axis=axis)
    return float(result) if np.ndim(result) == 0 else result


def validate_all(graph: ModelGraph, validators: Mapping[str, Any]) -> None:
    for item in graph.objects:
        validator = validators.get(item.kind)
        if validator is None or item.opaque:
            continue
        try:
            validator(item)
        except ModelGraphError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise ModelGraphError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class PackManifest:
    identifier: str
    version: str
    title: str
    description: str
    dependencies: tuple[str, ...]
    object_kinds: tuple[str, ...]
    capability_ids: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "version": self.version,
            "title": self.title,
            "description": self.description,
            "dependencies": list(self.dependencies),
            "object_kinds": list(self.object_kinds),
            "capability_ids": list(self.capability_ids),
            "distribution": "official-optional",
            "execution": "local-installed-code-only",
        }
