"""Spatial Fields, Continuum Models and PDEs official capability pack."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from ..model import ModelIR
from ..model_graph import ModelGraphError, ModelObject, ObjectKindDescriptor
from ..protocol import ArtifactTypeDescriptor, CapabilityDescriptor
from .common import (
    MAX_INLINE_VALUES,
    PackManifest,
    finite_vector,
    objects_of_kind,
    select_object,
    unique_labels,
)
from .units import LENGTH, add_dimensions, canonical_unit, resolve_unit, subtract_dimensions, to_si


PACK_ID = "org.modellab.pack.spatial-fields-continuum-pdes"
SCALAR_FIELD_KIND = "org.modellab.field.structured-scalar-field"
VECTOR_FIELD_KIND = "org.modellab.field.structured-vector-field"
DIFFUSION_KIND = "org.modellab.pde.diffusion-problem"
POISSON_KIND = "org.modellab.pde.poisson-problem"

_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")

_AXIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["name", "coordinates"],
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 128},
        "coordinates": {
            "type": "array", "minItems": 2, "maxItems": 4096,
            "items": {"type": "number"},
        },
        "unit": {"type": "string", "maxLength": 128},
    },
    "additionalProperties": False,
}

SCALAR_FIELD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["axes", "values"],
    "properties": {
        "axes": {"type": "array", "minItems": 1, "maxItems": 3, "items": _AXIS_SCHEMA},
        "values": {"type": "array", "minItems": 1, "maxItems": 4096, "items": {}},
        "value_name": {"type": "string", "maxLength": 128},
        "value_unit": {"type": "string", "maxLength": 128},
    },
    "additionalProperties": False,
}

VECTOR_FIELD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["axes", "component_names", "values"],
    "properties": {
        "axes": {"type": "array", "minItems": 2, "maxItems": 3, "items": _AXIS_SCHEMA},
        "component_names": {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "string"}},
        "values": {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "array", "minItems": 1, "items": {}}},
        "value_unit": {"type": "string", "maxLength": 128},
    },
    "additionalProperties": False,
}

_BOUNDARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["left", "right"],
    "properties": {"left": {"type": "number"}, "right": {"type": "number"}},
    "additionalProperties": False,
}

DIFFUSION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["coordinate", "initial_values", "diffusivity", "time_span", "dirichlet_boundary"],
    "properties": {
        "coordinate_name": {"type": "string", "maxLength": 128},
        "coordinate": {"type": "array", "minItems": 3, "maxItems": 4096, "items": {"type": "number"}},
        "initial_values": {"type": "array", "minItems": 3, "maxItems": 4096, "items": {"type": "number"}},
        "diffusivity": {"type": "number", "exclusiveMinimum": 0},
        "time_span": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}},
        "dirichlet_boundary": _BOUNDARY_SCHEMA,
        "value_name": {"type": "string", "maxLength": 128},
    },
    "additionalProperties": False,
}

_POISSON_BOUNDARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["left", "right", "bottom", "top"],
    "properties": {
        "left": {"type": "number"}, "right": {"type": "number"},
        "bottom": {"type": "number"}, "top": {"type": "number"},
    },
    "additionalProperties": False,
}

POISSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["x_coordinates", "y_coordinates", "source", "dirichlet_boundary"],
    "properties": {
        "x_coordinates": {"type": "array", "minItems": 3, "maxItems": 1024, "items": {"type": "number"}},
        "y_coordinates": {"type": "array", "minItems": 3, "maxItems": 1024, "items": {"type": "number"}},
        "source": {"type": "array", "minItems": 3, "maxItems": 1024, "items": {"type": "array", "minItems": 3, "maxItems": 1024, "items": {"type": "number"}}},
        "dirichlet_boundary": _POISSON_BOUNDARY_SCHEMA,
        "solution_name": {"type": "string", "maxLength": 128},
    },
    "additionalProperties": False,
}


KIND_DESCRIPTORS = (
    ObjectKindDescriptor(SCALAR_FIELD_KIND, "1.1", "Rectilinear structured scalar field", SCALAR_FIELD_SCHEMA, True),
    ObjectKindDescriptor(VECTOR_FIELD_KIND, "1.1", "Rectilinear structured vector field", VECTOR_FIELD_SCHEMA, True),
    ObjectKindDescriptor(DIFFUSION_KIND, "1.0", "One-dimensional diffusion initial-boundary problem", DIFFUSION_SCHEMA, True),
    ObjectKindDescriptor(POISSON_KIND, "1.0", "Two-dimensional Poisson Dirichlet problem", POISSON_SCHEMA, True),
)


@dataclass(frozen=True, slots=True)
class _Axes:
    names: tuple[str, ...]
    coordinates: tuple[np.ndarray, ...]
    units: tuple[str, ...]
    source_units: tuple[str, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(value) for value in self.coordinates)


def _coordinates(value: object, *, field: str, uniform: bool = False) -> np.ndarray:
    result = finite_vector(value, field=field)
    if result.size < 2 or not np.all(np.diff(result) > 0.0):
        raise ModelGraphError(f"{field} must be strictly increasing.")
    if uniform and not np.allclose(np.diff(result), np.diff(result)[0], rtol=1e-10, atol=1e-12):
        raise ModelGraphError(f"{field} must be uniformly spaced for this finite-difference problem.")
    return result


def _axes(item: ModelObject, *, minimum: int = 1) -> _Axes:
    raw_axes = item.properties["axes"]
    if not isinstance(raw_axes, list) or not minimum <= len(raw_axes) <= 3:
        raise ModelGraphError(f"{item.identifier}.axes must contain {minimum}–3 axes.")
    names: list[str] = []
    coordinates: list[np.ndarray] = []
    units: list[str] = []
    source_units: list[str] = []
    for index, raw in enumerate(raw_axes):
        if not isinstance(raw, Mapping):
            raise ModelGraphError(f"{item.identifier}.axes[{index}] must be an object.")
        name = str(raw["name"])
        if _IDENTIFIER.fullmatch(name) is None or name in names:
            raise ModelGraphError(f"{item.identifier}.axes contains invalid or repeated axis names.")
        names.append(name)
        declared_unit = str(raw.get("unit", "m"))
        coordinate = _coordinates(raw["coordinates"], field=f"{item.identifier}.axes[{index}].coordinates")
        coordinates.append(to_si(coordinate, declared_unit, field=f"{item.identifier}.axes[{index}].unit", expected=LENGTH))
        units.append("m")
        source_units.append(declared_unit)
    return _Axes(tuple(names), tuple(coordinates), tuple(units), tuple(source_units))


def _finite_grid(value: object, shape: tuple[int, ...], *, field: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelGraphError(f"{field} must be a rectangular real grid.") from exc
    if result.shape != shape:
        raise ModelGraphError(f"{field} shape {result.shape} does not match coordinate shape {shape}.")
    if result.size > MAX_INLINE_VALUES or not np.all(np.isfinite(result)):
        raise ModelGraphError(f"{field} exceeds the finite inline-grid contract.")
    result = result.copy()
    result[result == 0.0] = 0.0
    return result


def _scalar_field(item: ModelObject) -> tuple[_Axes, np.ndarray, str, str, str, tuple[int, ...]]:
    axes = _axes(item)
    values = _finite_grid(item.properties["values"], axes.shape, field=f"{item.identifier}.values")
    declared_unit = str(item.properties.get("value_unit", "dimensionless"))
    definition = resolve_unit(declared_unit, field=f"{item.identifier}.value_unit")
    return (
        axes,
        values * definition.scale_to_si,
        str(item.properties.get("value_name", "field")),
        canonical_unit(definition.dimensions),
        declared_unit,
        definition.dimensions,
    )


def _vector_field(item: ModelObject) -> tuple[_Axes, tuple[str, ...], np.ndarray, str, str, tuple[int, ...]]:
    axes = _axes(item, minimum=2)
    names = unique_labels(item.properties["component_names"], field=f"{item.identifier}.component_names")
    if len(names) != len(axes.names):
        raise ModelGraphError(f"{item.identifier}.component_names must contain one component per spatial axis.")
    try:
        values = np.asarray(item.properties["values"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelGraphError(f"{item.identifier}.values must be rectangular component grids.") from exc
    expected = (len(names), *axes.shape)
    if values.shape != expected or values.size > MAX_INLINE_VALUES or not np.all(np.isfinite(values)):
        raise ModelGraphError(f"{item.identifier}.values must have shape {expected} and finite inline data.")
    declared_unit = str(item.properties.get("value_unit", "dimensionless"))
    definition = resolve_unit(declared_unit, field=f"{item.identifier}.value_unit")
    return (
        axes,
        names,
        values.copy() * definition.scale_to_si,
        canonical_unit(definition.dimensions),
        declared_unit,
        definition.dimensions,
    )


def _diffusion(item: ModelObject) -> tuple[str, np.ndarray, np.ndarray, float, tuple[float, float], tuple[float, float], str]:
    coordinate = _coordinates(item.properties["coordinate"], field=f"{item.identifier}.coordinate", uniform=True)
    initial = finite_vector(item.properties["initial_values"], field=f"{item.identifier}.initial_values")
    if initial.size != coordinate.size:
        raise ModelGraphError(f"{item.identifier}.initial_values must align with coordinate.")
    diffusivity = float(item.properties["diffusivity"])
    if not math.isfinite(diffusivity) or diffusivity <= 0.0:
        raise ModelGraphError(f"{item.identifier}.diffusivity must be positive and finite.")
    span = finite_vector(item.properties["time_span"], field=f"{item.identifier}.time_span")
    if span.size != 2 or not span[0] < span[1]:
        raise ModelGraphError(f"{item.identifier}.time_span must contain increasing start/end times.")
    boundary = item.properties["dirichlet_boundary"]
    if not isinstance(boundary, Mapping):
        raise ModelGraphError(f"{item.identifier}.dirichlet_boundary must be an object.")
    left, right = float(boundary["left"]), float(boundary["right"])
    if not all(math.isfinite(value) for value in (left, right)):
        raise ModelGraphError(f"{item.identifier}.dirichlet_boundary values must be finite.")
    if not (math.isclose(initial[0], left, rel_tol=0.0, abs_tol=1e-10) and math.isclose(initial[-1], right, rel_tol=0.0, abs_tol=1e-10)):
        raise ModelGraphError(f"{item.identifier}.initial_values must satisfy the declared Dirichlet boundary.")
    return (
        str(item.properties.get("coordinate_name", "x")), coordinate, initial, diffusivity,
        (float(span[0]), float(span[1])), (left, right), str(item.properties.get("value_name", "u")),
    )


def _poisson(item: ModelObject) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float], str]:
    x = _coordinates(item.properties["x_coordinates"], field=f"{item.identifier}.x_coordinates", uniform=True)
    y = _coordinates(item.properties["y_coordinates"], field=f"{item.identifier}.y_coordinates", uniform=True)
    source = _finite_grid(item.properties["source"], (len(y), len(x)), field=f"{item.identifier}.source")
    boundary = item.properties["dirichlet_boundary"]
    if not isinstance(boundary, Mapping):
        raise ModelGraphError(f"{item.identifier}.dirichlet_boundary must be an object.")
    values = tuple(float(boundary[name]) for name in ("left", "right", "bottom", "top"))
    if not all(math.isfinite(value) for value in values):
        raise ModelGraphError(f"{item.identifier}.dirichlet_boundary values must be finite.")
    return x, y, source, values, str(item.properties.get("solution_name", "u"))


def validate_scalar_field(item: ModelObject) -> None:
    _scalar_field(item)


def validate_vector_field(item: ModelObject) -> None:
    _vector_field(item)


def validate_diffusion(item: ModelObject) -> None:
    _diffusion(item)


def validate_poisson(item: ModelObject) -> None:
    _poisson(item)


SEMANTIC_VALIDATORS = {
    SCALAR_FIELD_KIND: validate_scalar_field,
    VECTOR_FIELD_KIND: validate_vector_field,
    DIFFUSION_KIND: validate_diffusion,
    POISSON_KIND: validate_poisson,
}


def _integral(values: np.ndarray, coordinates: tuple[np.ndarray, ...]) -> float:
    integrated: np.ndarray | float = values
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # NumPy 1.x compatibility
        trapezoid = getattr(np, "trapz")
    for axis in range(values.ndim - 1, -1, -1):
        integrated = trapezoid(integrated, x=coordinates[axis], axis=axis)
    return float(integrated)


def _gradient(values: np.ndarray, coordinates: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    edge_order = 2 if all(len(axis) >= 3 for axis in coordinates) else 1
    result = np.gradient(values, *coordinates, edge_order=edge_order)
    return (np.asarray(result, dtype=np.float64),) if values.ndim == 1 else tuple(np.asarray(item, dtype=np.float64) for item in result)


@dataclass(frozen=True, slots=True)
class ScalarFieldAnalysis:
    object_id: str
    axis_names: tuple[str, ...]
    coordinates: tuple[np.ndarray, ...]
    axis_units: tuple[str, ...]
    value_name: str
    value_unit: str
    source_units: Mapping[str, str]
    gradient_units: tuple[str, ...]
    laplacian_unit: str
    integral_unit: str
    values: np.ndarray
    gradient: tuple[np.ndarray, ...]
    gradient_magnitude: np.ndarray
    laplacian: np.ndarray
    integral: float
    mean: float
    minimum: float
    minimum_coordinates: tuple[float, ...]
    maximum: float
    maximum_coordinates: tuple[float, ...]


def analyse_scalar_field(model: ModelIR, object_id: str | None = None) -> ScalarFieldAnalysis:
    item = select_object(model, SCALAR_FIELD_KIND, object_id, label="structured scalar field")
    axes, values, value_name, value_unit, source_value_unit, value_dimensions = _scalar_field(item)
    gradient = _gradient(values, axes.coordinates)
    laplacian = np.zeros_like(values)
    for dimension, component in enumerate(gradient):
        laplacian += _gradient(component, axes.coordinates)[dimension]
    minimum_index = np.unravel_index(int(np.argmin(values)), values.shape)
    maximum_index = np.unravel_index(int(np.argmax(values)), values.shape)
    return ScalarFieldAnalysis(
        item.identifier, axes.names, axes.coordinates, axes.units, value_name, value_unit,
        {**dict(zip(axes.names, axes.source_units, strict=True)), "value": source_value_unit},
        tuple(canonical_unit(subtract_dimensions(value_dimensions, LENGTH)) for _ in axes.names),
        canonical_unit(subtract_dimensions(subtract_dimensions(value_dimensions, LENGTH), LENGTH)),
        canonical_unit(tuple(value + len(axes.names) * length for value, length in zip(value_dimensions, LENGTH, strict=True))),
        values, gradient, np.sqrt(sum(np.square(component) for component in gradient)),
        laplacian, _integral(values, axes.coordinates), float(np.mean(values)),
        float(values[minimum_index]), tuple(float(axes.coordinates[i][index]) for i, index in enumerate(minimum_index)),
        float(values[maximum_index]), tuple(float(axes.coordinates[i][index]) for i, index in enumerate(maximum_index)),
    )


@dataclass(frozen=True, slots=True)
class VectorFieldAnalysis:
    object_id: str
    axis_names: tuple[str, ...]
    component_names: tuple[str, ...]
    coordinates: tuple[np.ndarray, ...]
    axis_units: tuple[str, ...]
    value_unit: str
    source_units: Mapping[str, str]
    divergence_unit: str
    curl_unit: str
    values: np.ndarray
    magnitude: np.ndarray
    divergence: np.ndarray
    curl: np.ndarray
    curl_kind: str
    mean_divergence: float
    maximum_magnitude: float


def analyse_vector_field(model: ModelIR, object_id: str | None = None) -> VectorFieldAnalysis:
    item = select_object(model, VECTOR_FIELD_KIND, object_id, label="structured vector field")
    axes, names, values, value_unit, source_value_unit, value_dimensions = _vector_field(item)
    component_gradients = [_gradient(component, axes.coordinates) for component in values]
    divergence = sum(component_gradients[index][index] for index in range(len(names)))
    if len(names) == 2:
        curl = component_gradients[1][0] - component_gradients[0][1]
        curl_kind = "scalar-out-of-plane"
    else:
        curl = np.stack((
            component_gradients[2][1] - component_gradients[1][2],
            component_gradients[0][2] - component_gradients[2][0],
            component_gradients[1][0] - component_gradients[0][1],
        ))
        curl_kind = "three-dimensional-vector"
    magnitude = np.sqrt(np.sum(np.square(values), axis=0))
    return VectorFieldAnalysis(
        item.identifier, axes.names, names, axes.coordinates, axes.units, value_unit,
        {**dict(zip(axes.names, axes.source_units, strict=True)), "value": source_value_unit},
        canonical_unit(subtract_dimensions(value_dimensions, LENGTH)),
        canonical_unit(subtract_dimensions(value_dimensions, LENGTH)), values, magnitude,
        np.asarray(divergence, dtype=np.float64), np.asarray(curl, dtype=np.float64),
        curl_kind, float(np.mean(divergence)), float(np.max(magnitude)),
    )


@dataclass(frozen=True, slots=True)
class DiffusionSolution:
    object_id: str
    coordinate_name: str
    coordinate: np.ndarray
    value_name: str
    times: np.ndarray
    values: np.ndarray
    diffusivity: float
    boundary_values: tuple[float, float]
    spatial_step: float
    method: str
    relative_tolerance: float
    absolute_tolerance: float
    spatial_integrals: np.ndarray
    solver_record: Mapping[str, Any]


def solve_diffusion(
    model: ModelIR,
    object_id: str | None = None,
    time_samples: int = 201,
    method: str = "BDF",
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 1e-10,
) -> DiffusionSolution:
    if isinstance(time_samples, bool) or not isinstance(time_samples, int) or not 2 <= time_samples <= 10000:
        raise ValueError("time_samples must be an integer between 2 and 10000.")
    if method not in {"RK45", "BDF", "Radau"}:
        raise ValueError("method must be RK45, BDF, or Radau.")
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in (relative_tolerance, absolute_tolerance)):
        raise ValueError("Diffusion tolerances must be positive finite numbers.")
    item = select_object(model, DIFFUSION_KIND, object_id, label="diffusion problem")
    coordinate_name, coordinate, initial, diffusivity, span, boundary, value_name = _diffusion(item)
    dx = float(coordinate[1] - coordinate[0])
    interior_initial = initial[1:-1]

    def rhs(_time: float, interior: np.ndarray) -> np.ndarray:
        complete = np.concatenate(([boundary[0]], interior, [boundary[1]]))
        return diffusivity * (complete[:-2] - 2.0 * complete[1:-1] + complete[2:]) / (dx * dx)

    times = np.linspace(*span, time_samples, dtype=np.float64)
    solution = solve_ivp(
        rhs, span, interior_initial, method=method, t_eval=times,
        rtol=float(relative_tolerance), atol=float(absolute_tolerance),
    )
    if not solution.success:
        raise ValueError(f"Diffusion solve failed with solver status {solution.status}.")
    values = np.empty((time_samples, coordinate.size), dtype=np.float64)
    values[:, 0], values[:, -1], values[:, 1:-1] = boundary[0], boundary[1], solution.y.T
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # NumPy 1.x compatibility
        trapezoid = getattr(np, "trapz")
    integrals = np.asarray(trapezoid(values, x=coordinate, axis=1), dtype=np.float64)
    return DiffusionSolution(
        item.identifier, coordinate_name, coordinate, value_name, times, values, diffusivity,
        boundary, dx, method, float(relative_tolerance), float(absolute_tolerance), integrals,
        {
            "scientific": {"converged": True},
            "presentation": {
                "function_evaluations": int(solution.nfev),
                "solver_status": int(solution.status),
            },
        },
    )


@dataclass(frozen=True, slots=True)
class PoissonSolution:
    object_id: str
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    solution_name: str
    source: np.ndarray
    solution: np.ndarray
    boundary_values: tuple[float, float, float, float]
    corner_policy: str
    maximum_residual: float
    minimum: float
    maximum: float


def solve_poisson(model: ModelIR, object_id: str | None = None) -> PoissonSolution:
    item = select_object(model, POISSON_KIND, object_id, label="Poisson problem")
    x, y, source, boundary, solution_name = _poisson(item)
    nx, ny = len(x), len(y)
    interior_x, interior_y = nx - 2, ny - 2
    unknowns = interior_x * interior_y
    if unknowns > 200_000:
        raise ValueError("The Poisson problem exceeds the 200,000-interior-node solve limit.")
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    diagonal = 2.0 / dx**2 + 2.0 / dy**2
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    rhs = source[1:-1, 1:-1].reshape(-1).copy()
    left, right, bottom, top = boundary

    def index(j: int, i: int) -> int:
        return j * interior_x + i

    for j in range(interior_y):
        for i in range(interior_x):
            row = index(j, i)
            rows.append(row); columns.append(row); data.append(diagonal)
            if i > 0:
                rows.append(row); columns.append(index(j, i - 1)); data.append(-1.0 / dx**2)
            else:
                rhs[row] += left / dx**2
            if i + 1 < interior_x:
                rows.append(row); columns.append(index(j, i + 1)); data.append(-1.0 / dx**2)
            else:
                rhs[row] += right / dx**2
            if j > 0:
                rows.append(row); columns.append(index(j - 1, i)); data.append(-1.0 / dy**2)
            else:
                rhs[row] += bottom / dy**2
            if j + 1 < interior_y:
                rows.append(row); columns.append(index(j + 1, i)); data.append(-1.0 / dy**2)
            else:
                rhs[row] += top / dy**2
    matrix = csr_matrix((data, (rows, columns)), shape=(unknowns, unknowns))
    interior = np.asarray(spsolve(matrix, rhs), dtype=np.float64)
    if not np.all(np.isfinite(interior)):
        raise ValueError("The Poisson linear system did not produce a finite solution.")
    solution = np.empty((ny, nx), dtype=np.float64)
    solution[:, 0], solution[:, -1], solution[0, :], solution[-1, :] = left, right, bottom, top
    # Corners obey the last (horizontal) boundary assignment; disagreeing declarations are
    # still scientifically visible in the stored boundary tuple.
    solution[1:-1, 1:-1] = interior.reshape(interior_y, interior_x)
    residual = matrix @ interior - rhs
    return PoissonSolution(
        item.identifier, x, y, solution_name, source, solution, boundary,
        "bottom-and-top-edges-own-corner-nodes",
        float(np.max(np.abs(residual), initial=0.0)), float(np.min(solution)), float(np.max(solution)),
    )


def _kind_applicability(kind: str, label: str):
    def applicable(model: ModelIR) -> tuple[bool, str]:
        found = bool(objects_of_kind(model, kind))
        return found, "" if found else f"an executable {label} object is required"
    return applicable


def _field_units(kind: str, factor: int = 1):
    def units(settings: Mapping[str, Any], model: ModelIR) -> int:
        item = select_object(model, kind, settings.get("object_id"), label="structured field")
        try:
            size = int(np.asarray(item.properties["values"]).size)
        except (KeyError, ValueError):
            size = 1
        return max(1, factor * size)
    return units


def _diffusion_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(
        model, DIFFUSION_KIND, settings.get("object_id"), label="diffusion problem"
    )
    return max(1, len(item.properties["coordinate"]) * int(settings.get("time_samples", 201)) * 10)


def _poisson_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(model, POISSON_KIND, settings.get("object_id"), label="Poisson problem")
    nx = len(item.properties["x_coordinates"])
    ny = len(item.properties["y_coordinates"])
    return max(1, nx * ny * 20)


NUMERIC = "org.modellab.comparator.numeric"

CAPABILITY_DESCRIPTORS = (
    CapabilityDescriptor(
        "org.modellab.field.analyse-scalar-field", "1.2", PACK_ID,
        "Analyse structured scalar field", "Compute gradient, Laplacian, integral and extrema on a one- to three-dimensional rectilinear field.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}}},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.scalar-field-analysis", "1.2", "Structured scalar-field analysis", NUMERIC),),
        _kind_applicability(SCALAR_FIELD_KIND, "structured scalar-field"), analyse_scalar_field,
        _field_units(SCALAR_FIELD_KIND, 10), ("org.modellab.renderer.plotly-scalar-field",),
    ),
    CapabilityDescriptor(
        "org.modellab.field.analyse-vector-field", "1.2", PACK_ID,
        "Analyse structured vector field", "Compute magnitude, divergence and dimension-appropriate curl on a two- or three-dimensional rectilinear vector field.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}}},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.vector-field-analysis", "1.2", "Structured vector-field analysis", NUMERIC),),
        _kind_applicability(VECTOR_FIELD_KIND, "structured vector-field"), analyse_vector_field,
        _field_units(VECTOR_FIELD_KIND, 20), ("org.modellab.renderer.plotly-structured-vector-field",),
    ),
    CapabilityDescriptor(
        "org.modellab.pde.solve-diffusion", "1.0", PACK_ID,
        "Solve one-dimensional diffusion", "Solve a one-dimensional constant-diffusivity initial-boundary problem by finite differences and method-of-lines integration.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "time_samples": {"type": "integer", "minimum": 2, "maximum": 10000, "default": 201}, "method": {"type": "string", "enum": ["RK45", "BDF", "Radau"], "default": "BDF"}, "relative_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-8}, "absolute_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-10}}},
        "scipy+numpy", (ArtifactTypeDescriptor("org.modellab.artifact.diffusion-solution", "1.0", "Diffusion solution", NUMERIC),),
        _kind_applicability(DIFFUSION_KIND, "diffusion-problem"), solve_diffusion,
        _diffusion_units, ("org.modellab.renderer.plotly-space-time-field",),
    ),
    CapabilityDescriptor(
        "org.modellab.pde.solve-poisson", "1.0", PACK_ID,
        "Solve two-dimensional Poisson problem", "Solve -Laplacian(u)=source on a uniform rectangular grid with constant Dirichlet edge values and an explicit horizontal-edge corner convention.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}}},
        "scipy+numpy", (ArtifactTypeDescriptor("org.modellab.artifact.poisson-solution", "1.0", "Poisson field solution", NUMERIC),),
        _kind_applicability(POISSON_KIND, "Poisson-problem"), solve_poisson,
        _poisson_units, ("org.modellab.renderer.plotly-poisson-field",),
    ),
)


MANIFEST = PackManifest(
    PACK_ID, "1.2", "Spatial Fields, Continuum Models and PDEs",
    "Structured scalar/vector fields, spatial differential operators and bounded reproducible finite-difference PDE solvers.",
    (
        "org.modellab.pack.multidimensional-mathematics",
        "org.modellab.pack.dynamics-differential-equations-control",
    ),
    tuple(item.kind for item in KIND_DESCRIPTORS),
    tuple(item.identifier for item in CAPABILITY_DESCRIPTORS),
)
