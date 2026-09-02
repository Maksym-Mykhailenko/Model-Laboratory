"""Multidimensional Mathematics and Scientific Quantities pack."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..model import ModelIR
from ..model_graph import ModelGraphError, ModelObject, ObjectKindDescriptor
from ..protocol import ArtifactTypeDescriptor, CapabilityDescriptor
from .common import PackManifest, finite_vector, objects_of_kind, select_object
from .units import DIMENSIONLESS, add_dimensions, canonical_unit, resolve_unit


PACK_ID = "org.modellab.pack.multidimensional-mathematics"
ARRAY_KIND = "org.modellab.multidimensional.array"
QUANTITY_KIND = "org.modellab.multidimensional.quantity"


ARRAY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["shape", "values"],
    "properties": {
        "shape": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {"type": "integer", "minimum": 1},
        },
        "values": {
            "type": "array",
            "minItems": 1,
            "maxItems": 200000,
            "items": {"type": "number"},
        },
        "axis_labels": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "array", "items": {"type": "string"}},
        },
        "unit": {"type": "string"},
        "dimension_exponents": {
            "type": "array",
            "minItems": 7,
            "maxItems": 7,
            "items": {"type": "integer", "minimum": -32, "maximum": 32},
        },
        "standard_uncertainties": {
            "type": "array",
            "maxItems": 200000,
            "items": {"type": "number", "minimum": 0},
        },
    },
    "additionalProperties": False,
}

QUANTITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["value", "unit", "dimension_exponents", "scale_to_base"],
    "properties": {
        "value": {"type": "number"},
        "unit": {"type": "string"},
        "dimension_exponents": {
            "type": "array",
            "minItems": 7,
            "maxItems": 7,
            "items": {"type": "integer", "minimum": -32, "maximum": 32},
        },
        "scale_to_base": {"type": "number"},
        "offset_to_base": {"type": "number"},
        "standard_uncertainty": {"type": "number", "minimum": 0},
    },
    "additionalProperties": False,
}


KIND_DESCRIPTORS = (
    ObjectKindDescriptor(ARRAY_KIND, "1.1", "Labelled real array or tensor", ARRAY_SCHEMA, True),
    ObjectKindDescriptor(QUANTITY_KIND, "1.1", "Dimensioned scientific quantity", QUANTITY_SCHEMA, True),
)


def _shape(item: ModelObject) -> tuple[int, ...]:
    return tuple(int(value) for value in item.properties["shape"])


def _array(item: ModelObject) -> np.ndarray:
    shape = _shape(item)
    values = finite_vector(item.properties["values"], field=f"{item.identifier}.values")
    if math.prod(shape) != values.size:
        raise ModelGraphError(
            f"{item.identifier}.shape requires {math.prod(shape)} values, but {values.size} were supplied."
        )
    return values.reshape(shape)


def _array_quantity(
    item: ModelObject,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], str | None, str | None]:
    values = _array(item)
    declared_unit = item.properties.get("unit")
    declared_exponents = item.properties.get("dimension_exponents")
    if declared_unit is not None:
        definition = resolve_unit(str(declared_unit), field=f"{item.identifier}.unit")
        dimensions = definition.dimensions
        if declared_exponents is not None and tuple(int(value) for value in declared_exponents) != dimensions:
            raise ModelGraphError(f"{item.identifier}.unit and dimension_exponents disagree.")
        scale = definition.scale_to_si
        unit: str | None = canonical_unit(dimensions)
    else:
        dimensions = DIMENSIONLESS if declared_exponents is None else tuple(int(value) for value in declared_exponents)
        scale = 1.0
        unit = None if dimensions == DIMENSIONLESS else canonical_unit(dimensions)
    raw_uncertainties = item.properties.get("standard_uncertainties")
    uncertainty = np.zeros_like(values) if raw_uncertainties is None else finite_vector(
        raw_uncertainties, field=f"{item.identifier}.standard_uncertainties"
    ).reshape(values.shape)
    return (
        values * scale,
        uncertainty * abs(scale),
        tuple(dimensions),
        unit,
        None if declared_unit is None else str(declared_unit),
    )


def validate_array(item: ModelObject) -> None:
    values, _uncertainties, _dimensions, _unit, _source_unit = _array_quantity(item)
    labels = item.properties.get("axis_labels", [])
    if labels:
        if len(labels) != values.ndim:
            raise ModelGraphError(f"{item.identifier}.axis_labels must contain one list per axis.")
        for index, (axis_labels, size) in enumerate(zip(labels, values.shape)):
            if len(axis_labels) != size or len(set(map(str, axis_labels))) != size:
                raise ModelGraphError(
                    f"{item.identifier}.axis_labels[{index}] must contain {size} unique labels."
                )
    exponents = item.properties.get("dimension_exponents")
    if exponents is not None and len(exponents) != 7:
        raise ModelGraphError(f"{item.identifier}.dimension_exponents must contain seven SI exponents.")
    uncertainties = item.properties.get("standard_uncertainties")
    if uncertainties is not None:
        checked = finite_vector(uncertainties, field=f"{item.identifier}.standard_uncertainties")
        if checked.size != values.size or np.any(checked < 0.0):
            raise ModelGraphError(
                f"{item.identifier}.standard_uncertainties must align with values and be non-negative."
            )


def validate_quantity(item: ModelObject) -> None:
    properties = item.properties
    if len(properties["dimension_exponents"]) != 7:
        raise ModelGraphError(f"{item.identifier}.dimension_exponents must contain seven SI exponents.")
    values = [properties["value"], properties["scale_to_base"], properties.get("offset_to_base", 0.0)]
    if not all(math.isfinite(float(value)) for value in values) or float(properties["scale_to_base"]) == 0.0:
        raise ModelGraphError(f"{item.identifier} contains a non-finite or zero conversion scale.")
    uncertainty = float(properties.get("standard_uncertainty", 0.0))
    if not math.isfinite(uncertainty) or uncertainty < 0.0:
        raise ModelGraphError(f"{item.identifier}.standard_uncertainty must be finite and non-negative.")


SEMANTIC_VALIDATORS = {ARRAY_KIND: validate_array, QUANTITY_KIND: validate_quantity}


@dataclass(frozen=True, slots=True)
class ArrayAnalysis:
    object_id: str
    shape: tuple[int, ...]
    size: int
    rank: int
    unit: str | None
    source_units: Mapping[str, str | None]
    dimension_exponents: tuple[int, ...] | None
    axis_labels: tuple[tuple[str, ...], ...]
    values: np.ndarray
    standard_uncertainties: np.ndarray
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    l1_norm: float
    l2_norm: float
    infinity_norm: float
    matrix_rank: int | None
    determinant: float | None
    singular_values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TensorContraction:
    left_object_id: str
    right_object_id: str
    left_axes: tuple[int, ...]
    right_axes: tuple[int, ...]
    result_shape: tuple[int, ...]
    axis_labels: tuple[tuple[str, ...], ...]
    unit: str
    source_units: Mapping[str, str | None]
    dimension_exponents: tuple[int, ...]
    values: np.ndarray
    standard_uncertainties: np.ndarray
    uncertainty_model: str


@dataclass(frozen=True, slots=True)
class QuantityNormalisation:
    quantities: tuple[Mapping[str, Any], ...]
    compatible_groups: tuple[Mapping[str, Any], ...]


def analyse_array(model: ModelIR, object_id: str | None = None) -> ArrayAnalysis:
    item = select_object(model, ARRAY_KIND, object_id, label="array")
    values, uncertainties, dimensions, unit, source_unit = _array_quantity(item)
    matrix_rank: int | None = None
    determinant: float | None = None
    singular_values: tuple[float, ...] = ()
    if values.ndim == 2:
        matrix_rank = int(np.linalg.matrix_rank(values))
        singular_values = tuple(float(value) for value in np.linalg.svd(values, compute_uv=False))
        if values.shape[0] == values.shape[1]:
            determinant = float(np.linalg.det(values))
    labels = tuple(tuple(str(value) for value in axis) for axis in item.properties.get("axis_labels", []))
    exponents = item.properties.get("dimension_exponents")
    return ArrayAnalysis(
        object_id=item.identifier,
        shape=tuple(values.shape),
        size=int(values.size),
        rank=values.ndim,
        unit=unit,
        source_units={"value": source_unit},
        dimension_exponents=dimensions,
        axis_labels=labels,
        values=values,
        standard_uncertainties=uncertainties,
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        mean=float(np.mean(values)),
        standard_deviation=float(np.std(values)),
        l1_norm=float(np.linalg.norm(values.reshape(-1), ord=1)),
        l2_norm=float(np.linalg.norm(values.reshape(-1), ord=2)),
        infinity_norm=float(np.linalg.norm(values.reshape(-1), ord=np.inf)),
        matrix_rank=matrix_rank,
        determinant=determinant,
        singular_values=singular_values,
    )


def contract_tensors(
    model: ModelIR,
    left_object_id: str | None = None,
    right_object_id: str | None = None,
    left_axes: Sequence[int] = (0,),
    right_axes: Sequence[int] = (0,),
) -> TensorContraction:
    arrays = objects_of_kind(model, ARRAY_KIND)
    left = select_object(model, ARRAY_KIND, left_object_id, label="left array")
    if right_object_id is None:
        right = arrays[1] if len(arrays) > 1 else left
    else:
        right = select_object(model, ARRAY_KIND, right_object_id, label="right array")
    left_values, left_uncertainty, left_dimensions, _left_unit, left_source_unit = _array_quantity(left)
    right_values, right_uncertainty, right_dimensions, _right_unit, right_source_unit = _array_quantity(right)
    lhs = tuple(int(value) for value in left_axes)
    rhs = tuple(int(value) for value in right_axes)
    if not lhs or len(lhs) != len(rhs) or len(set(lhs)) != len(lhs) or len(set(rhs)) != len(rhs):
        raise ValueError("Contraction axes must be non-empty, unique, and paired.")
    if any(value < 0 or value >= left_values.ndim for value in lhs) or any(
        value < 0 or value >= right_values.ndim for value in rhs
    ):
        raise ValueError("A contraction axis is outside the selected tensor rank.")
    if any(left_values.shape[a] != right_values.shape[b] for a, b in zip(lhs, rhs)):
        raise ValueError("Paired contraction axes must have equal dimensions.")
    result = np.tensordot(left_values, right_values, axes=(lhs, rhs))
    if left.identifier == right.identifier:
        propagated_uncertainty = _self_contraction_uncertainty(
            left_values, left_uncertainty, lhs, rhs
        )
        uncertainty_model = "correlated-first-order-shared-input"
    else:
        variance = np.tensordot(
            left_uncertainty**2, right_values**2 + right_uncertainty**2, axes=(lhs, rhs)
        ) + np.tensordot(left_values**2, right_uncertainty**2, axes=(lhs, rhs))
        propagated_uncertainty = np.sqrt(np.maximum(variance, 0.0))
        uncertainty_model = "independent-inputs-product-variance"
    left_labels = tuple(tuple(str(value) for value in axis) for axis in left.properties.get("axis_labels", []))
    right_labels = tuple(tuple(str(value) for value in axis) for axis in right.properties.get("axis_labels", []))
    if not left_labels:
        left_labels = tuple(() for _ in left_values.shape)
    if not right_labels:
        right_labels = tuple(() for _ in right_values.shape)
    result_labels = tuple(left_labels[index] for index in range(left_values.ndim) if index not in lhs) + tuple(
        right_labels[index] for index in range(right_values.ndim) if index not in rhs
    )
    dimensions = add_dimensions(left_dimensions, right_dimensions)
    return TensorContraction(
        left.identifier, right.identifier, lhs, rhs, tuple(result.shape), result_labels,
        canonical_unit(dimensions), {"left": left_source_unit, "right": right_source_unit},
        dimensions, result, propagated_uncertainty, uncertainty_model,
    )


def _self_contraction_uncertainty(
    values: np.ndarray,
    uncertainties: np.ndarray,
    left_axes: tuple[int, ...],
    right_axes: tuple[int, ...],
) -> np.ndarray:
    """First-order propagation when both tensor operands are the same random array."""
    left_free = tuple(axis for axis in range(values.ndim) if axis not in left_axes)
    right_free = tuple(axis for axis in range(values.ndim) if axis not in right_axes)
    contraction_size = math.prod(values.shape[axis] for axis in left_axes)
    left_size = math.prod(values.shape[axis] for axis in left_free)
    right_size = math.prod(values.shape[axis] for axis in right_free)
    indices = np.arange(values.size, dtype=np.int64).reshape(values.shape)
    left_indices = np.transpose(indices, (*left_free, *left_axes)).reshape(left_size, contraction_size)
    right_indices = np.transpose(indices, (*right_axes, *right_free)).reshape(contraction_size, right_size)
    left_values = np.transpose(values, (*left_free, *left_axes)).reshape(left_size, contraction_size)
    right_values = np.transpose(values, (*right_axes, *right_free)).reshape(contraction_size, right_size)
    sigma = uncertainties.reshape(-1)
    variance = np.empty((left_size, right_size), dtype=np.float64)
    for left_index in range(left_size):
        for right_index in range(right_size):
            gradient: dict[int, float] = {}
            for contraction_index in range(contraction_size):
                left_flat = int(left_indices[left_index, contraction_index])
                right_flat = int(right_indices[contraction_index, right_index])
                gradient[left_flat] = gradient.get(left_flat, 0.0) + float(
                    right_values[contraction_index, right_index]
                )
                gradient[right_flat] = gradient.get(right_flat, 0.0) + float(
                    left_values[left_index, contraction_index]
                )
            variance[left_index, right_index] = sum(
                (derivative * float(sigma[flat_index])) ** 2
                for flat_index, derivative in gradient.items()
            )
    result_shape = tuple(values.shape[axis] for axis in (*left_free, *right_free))
    return np.sqrt(np.maximum(variance, 0.0)).reshape(result_shape)


def normalise_quantities(model: ModelIR) -> QuantityNormalisation:
    items = objects_of_kind(model, QUANTITY_KIND)
    rows: list[Mapping[str, Any]] = []
    groups: dict[tuple[int, ...], list[str]] = {}
    for item in items:
        properties = item.properties
        value = float(properties["value"])
        scale = float(properties["scale_to_base"])
        offset = float(properties.get("offset_to_base", 0.0))
        uncertainty = float(properties.get("standard_uncertainty", 0.0))
        dimension = tuple(int(number) for number in properties["dimension_exponents"])
        rows.append(
            {
                "object_id": item.identifier,
                "source_value": value,
                "source_unit": str(properties["unit"]),
                "source_scale_to_base": scale,
                "source_offset_to_base": offset,
                "source_standard_uncertainty": uncertainty,
                "base_value": value * scale + offset,
                "base_standard_uncertainty": abs(scale) * uncertainty,
                "dimension_exponents": dimension,
            }
        )
        groups.setdefault(dimension, []).append(item.identifier)
    return QuantityNormalisation(
        tuple(rows),
        tuple(
            {"dimension_exponents": key, "object_ids": tuple(sorted(value))}
            for key, value in sorted(groups.items())
        ),
    )


def _array_applicable(model: ModelIR) -> tuple[bool, str]:
    return (True, "") if objects_of_kind(model, ARRAY_KIND) else (False, "an official array object is required")


def _quantity_applicable(model: ModelIR) -> tuple[bool, str]:
    return (True, "") if objects_of_kind(model, QUANTITY_KIND) else (False, "an official quantity object is required")


def _array_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(model, ARRAY_KIND, settings.get("object_id"), label="array")
    return max(1, math.prod(_shape(item)))


def _contraction_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    arrays = objects_of_kind(model, ARRAY_KIND)
    left = select_object(
        model, ARRAY_KIND, settings.get("left_object_id"), label="left array"
    )
    right_id = settings.get("right_object_id")
    right = (
        arrays[1] if right_id is None and len(arrays) > 1 else left
    ) if right_id is None else select_object(model, ARRAY_KIND, right_id, label="right array")
    left_shape = _shape(left)
    right_shape = _shape(right)
    left_axes = tuple(int(value) for value in settings.get("left_axes", (0,)))
    right_axes = tuple(int(value) for value in settings.get("right_axes", (0,)))
    if (
        not left_axes
        or len(left_axes) != len(right_axes)
        or any(axis < 0 or axis >= len(left_shape) for axis in left_axes)
        or any(axis < 0 or axis >= len(right_shape) for axis in right_axes)
    ):
        return 1
    contraction_size = math.prod(left_shape[axis] for axis in left_axes)
    result_size = math.prod(
        left_shape[axis] for axis in range(len(left_shape)) if axis not in left_axes
    ) * math.prod(
        right_shape[axis] for axis in range(len(right_shape)) if axis not in right_axes
    )
    return max(1, 2 * contraction_size * result_size)


NUMERIC = "org.modellab.comparator.numeric"
CAPABILITY_DESCRIPTORS = (
    CapabilityDescriptor(
        "org.modellab.multidimensional.analyse-array",
        "1.2",
        PACK_ID,
        "Analyse labelled array or tensor",
        "Inspect shape, labels, units, norms, statistics, rank, determinant and singular values where applicable.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}}},
        "numpy",
        (ArtifactTypeDescriptor("org.modellab.artifact.array-analysis", "1.2", "Array analysis", NUMERIC),),
        _array_applicable,
        analyse_array,
        _array_units,
        ("org.modellab.renderer.plotly-array",),
    ),
    CapabilityDescriptor(
        "org.modellab.multidimensional.contract-tensors",
        "1.2",
        PACK_ID,
        "Contract tensors",
        "Perform a deterministic labelled tensor contraction over explicitly paired axes.",
        {
            "type": "object",
            "properties": {
                "left_object_id": {"type": ["string", "null"]},
                "right_object_id": {"type": ["string", "null"]},
                "left_axes": {"type": "array", "default": [0]},
                "right_axes": {"type": "array", "default": [0]},
            },
        },
        "numpy",
        (ArtifactTypeDescriptor("org.modellab.artifact.tensor-contraction", "1.2", "Tensor contraction", NUMERIC),),
        _array_applicable,
        contract_tensors,
        _contraction_units,
        ("org.modellab.renderer.plotly-array",),
    ),
    CapabilityDescriptor(
        "org.modellab.multidimensional.normalise-quantities",
        "1.1",
        PACK_ID,
        "Normalise scientific quantities",
        "Convert dimensioned values and standard uncertainties to their declared base representation.",
        {"type": "object", "properties": {}},
        "python",
        (ArtifactTypeDescriptor("org.modellab.artifact.quantity-normalisation", "1.1", "Quantity normalisation", NUMERIC),),
        _quantity_applicable,
        normalise_quantities,
        lambda settings, model: max(1, len(objects_of_kind(model, QUANTITY_KIND))),
        ("org.modellab.renderer.plotly-quantities",),
    ),
)


MANIFEST = PackManifest(
    PACK_ID,
    "1.2",
    "Multidimensional Mathematics and Scientific Quantities",
    "Labelled tensors, scientific quantities, dimensional metadata, uncertainty and deterministic multidimensional operations.",
    (),
    tuple(item.kind for item in KIND_DESCRIPTORS),
    tuple(item.identifier for item in CAPABILITY_DESCRIPTORS),
)
