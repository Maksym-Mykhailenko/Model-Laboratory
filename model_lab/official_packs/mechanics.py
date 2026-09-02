"""Mechanics, Structures and Materials official capability pack."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.linalg import eigh

from ..model import ModelIR
from ..model_graph import ModelGraphError, ModelObject, ObjectKindDescriptor
from ..protocol import ArtifactTypeDescriptor, CapabilityDescriptor
from .common import PackManifest, objects_of_kind, select_object
from .units import DENSITY, FORCE, LENGTH, MASS, PRESSURE, THERMAL_EXPANSION, resolve_unit


PACK_ID = "org.modellab.pack.mechanics-structures-materials"
ISOTROPIC_MATERIAL_KIND = "org.modellab.material.isotropic-linear-elastic"
TRUSS_KIND = "org.modellab.mechanics.truss-structure"

ISOTROPIC_MATERIAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["youngs_modulus", "poissons_ratio"],
    "properties": {
        "youngs_modulus": {"type": "number", "exclusiveMinimum": 0},
        "poissons_ratio": {"type": "number", "minimum": -0.999999, "maximum": 0.499999},
        "density": {"type": "number", "exclusiveMinimum": 0},
        "yield_strength": {"type": "number", "exclusiveMinimum": 0},
        "thermal_expansion": {"type": "number"},
        "unit_system": {"type": "string", "maxLength": 128},
        "modulus_unit": {"type": "string", "maxLength": 128},
        "density_unit": {"type": "string", "maxLength": 128},
        "thermal_expansion_unit": {"type": "string", "maxLength": 128},
    },
    "additionalProperties": False,
}

_NODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["id", "coordinates"],
    "properties": {
        "id": {"type": "string", "minLength": 1, "maxLength": 128},
        "coordinates": {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "number"}},
    },
    "additionalProperties": False,
}

_ELEMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["id", "start", "end", "area", "youngs_modulus"],
    "properties": {
        "id": {"type": "string", "minLength": 1, "maxLength": 128},
        "start": {"type": "string", "minLength": 1, "maxLength": 128},
        "end": {"type": "string", "minLength": 1, "maxLength": 128},
        "area": {"type": "number", "exclusiveMinimum": 0},
        "youngs_modulus": {"type": "number", "exclusiveMinimum": 0},
        "density": {"type": "number", "exclusiveMinimum": 0},
    },
    "additionalProperties": False,
}

_SUPPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["node", "fixed"],
    "properties": {
        "node": {"type": "string", "minLength": 1, "maxLength": 128},
        "fixed": {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "boolean"}},
    },
    "additionalProperties": False,
}

_LOAD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["node", "force"],
    "properties": {
        "node": {"type": "string", "minLength": 1, "maxLength": 128},
        "force": {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "number"}},
    },
    "additionalProperties": False,
}

_LOAD_CASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["name", "loads"],
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 128},
        "loads": {"type": "array", "minItems": 1, "maxItems": 100000, "items": _LOAD_SCHEMA},
    },
    "additionalProperties": False,
}

_NODAL_MASS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["node", "mass"],
    "properties": {
        "node": {"type": "string", "minLength": 1, "maxLength": 128},
        "mass": {"type": "number", "exclusiveMinimum": 0},
    },
    "additionalProperties": False,
}

TRUSS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["dimension", "nodes", "elements", "supports", "load_cases"],
    "properties": {
        "dimension": {"type": "integer", "enum": [2, 3]},
        "nodes": {"type": "array", "minItems": 2, "maxItems": 10000, "items": _NODE_SCHEMA},
        "elements": {"type": "array", "minItems": 1, "maxItems": 100000, "items": _ELEMENT_SCHEMA},
        "supports": {"type": "array", "minItems": 1, "maxItems": 10000, "items": _SUPPORT_SCHEMA},
        "load_cases": {"type": "array", "minItems": 1, "maxItems": 1000, "items": _LOAD_CASE_SCHEMA},
        "nodal_masses": {"type": "array", "maxItems": 10000, "items": _NODAL_MASS_SCHEMA},
        "coordinate_unit": {"type": "string", "maxLength": 128},
        "force_unit": {"type": "string", "maxLength": 128},
        "stress_unit": {"type": "string", "maxLength": 128},
        "mass_unit": {"type": "string", "maxLength": 128},
    },
    "additionalProperties": False,
}


KIND_DESCRIPTORS = (
    ObjectKindDescriptor(ISOTROPIC_MATERIAL_KIND, "1.1", "Isotropic linear-elastic material", ISOTROPIC_MATERIAL_SCHEMA, True),
    ObjectKindDescriptor(TRUSS_KIND, "1.1", "Linear pin-jointed truss structure", TRUSS_SCHEMA, True),
)


@dataclass(frozen=True, slots=True)
class _Element:
    identifier: str
    start: int
    end: int
    area: float
    youngs_modulus: float
    density: float | None


@dataclass(frozen=True, slots=True)
class _Truss:
    object_id: str
    dimension: int
    node_ids: tuple[str, ...]
    coordinates: np.ndarray
    elements: tuple[_Element, ...]
    fixed_dofs: tuple[int, ...]
    load_cases: Mapping[str, np.ndarray]
    nodal_masses: np.ndarray
    coordinate_unit: str
    force_unit: str
    stress_unit: str
    source_units: Mapping[str, str]


def _finite_number(value: object, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ModelGraphError(f"{field} must be a finite real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelGraphError(f"{field} must be a finite real number.") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ModelGraphError(f"{field} must be {'positive and ' if positive else ''}finite.")
    return result


def _material(item: ModelObject) -> tuple[float, float, float | None, float | None, float | None, Mapping[str, str]]:
    unit_system = str(item.properties.get("unit_system", "SI"))
    if unit_system != "SI" and not all(key in item.properties for key in ("modulus_unit", "density_unit")):
        raise ModelGraphError(f"{item.identifier}.unit_system requires explicit operational units when it is not SI.")
    modulus_unit = str(item.properties.get("modulus_unit", "Pa"))
    density_unit = str(item.properties.get("density_unit", "kg/m^3"))
    expansion_unit = str(item.properties.get("thermal_expansion_unit", "1/K"))
    modulus_definition = resolve_unit(modulus_unit, field=f"{item.identifier}.modulus_unit", expected=PRESSURE)
    density_definition = resolve_unit(density_unit, field=f"{item.identifier}.density_unit", expected=DENSITY)
    expansion_definition = resolve_unit(expansion_unit, field=f"{item.identifier}.thermal_expansion_unit", expected=THERMAL_EXPANSION)
    E = _finite_number(item.properties["youngs_modulus"], field=f"{item.identifier}.youngs_modulus", positive=True) * modulus_definition.scale_to_si
    nu = _finite_number(item.properties["poissons_ratio"], field=f"{item.identifier}.poissons_ratio")
    if not -1.0 < nu < 0.5:
        raise ModelGraphError(f"{item.identifier}.poissons_ratio must lie strictly between -1 and 0.5.")
    density = None if "density" not in item.properties else _finite_number(item.properties["density"], field=f"{item.identifier}.density", positive=True) * density_definition.scale_to_si
    yield_strength = None if "yield_strength" not in item.properties else _finite_number(item.properties["yield_strength"], field=f"{item.identifier}.yield_strength", positive=True) * modulus_definition.scale_to_si
    expansion = None if "thermal_expansion" not in item.properties else _finite_number(item.properties["thermal_expansion"], field=f"{item.identifier}.thermal_expansion") * expansion_definition.scale_to_si
    return E, nu, density, yield_strength, expansion, {"modulus": modulus_unit, "density": density_unit, "thermal_expansion": expansion_unit, "unit_system": unit_system}


def _truss(item: ModelObject) -> _Truss:
    dimension = int(item.properties["dimension"])
    if dimension not in {2, 3}:
        raise ModelGraphError(f"{item.identifier}.dimension must be 2 or 3.")
    coordinate_unit = str(item.properties.get("coordinate_unit", "m"))
    force_unit = str(item.properties.get("force_unit", "N"))
    stress_unit = str(item.properties.get("stress_unit", "Pa"))
    mass_unit = str(item.properties.get("mass_unit", "kg"))
    coordinate_definition = resolve_unit(coordinate_unit, field=f"{item.identifier}.coordinate_unit", expected=LENGTH)
    force_definition = resolve_unit(force_unit, field=f"{item.identifier}.force_unit", expected=FORCE)
    stress_definition = resolve_unit(stress_unit, field=f"{item.identifier}.stress_unit", expected=PRESSURE)
    mass_definition = resolve_unit(mass_unit, field=f"{item.identifier}.mass_unit", expected=MASS)
    raw_nodes = item.properties["nodes"]
    if not isinstance(raw_nodes, list):
        raise ModelGraphError(f"{item.identifier}.nodes must be a list.")
    node_ids: list[str] = []
    coordinates: list[list[float]] = []
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, Mapping):
            raise ModelGraphError(f"{item.identifier}.nodes[{index}] must be an object.")
        node_id = str(raw["id"])
        if not node_id or node_id in node_ids:
            raise ModelGraphError(f"{item.identifier}.nodes must have unique non-empty IDs.")
        try:
            point = [float(value) for value in raw["coordinates"]]
        except (TypeError, ValueError) as exc:
            raise ModelGraphError(f"{item.identifier}.nodes[{index}].coordinates must be finite.") from exc
        if len(point) != dimension or not all(math.isfinite(value) for value in point):
            raise ModelGraphError(f"{item.identifier}.nodes[{index}].coordinates must match dimension.")
        node_ids.append(node_id); coordinates.append(point)
    lookup = {name: index for index, name in enumerate(node_ids)}
    coordinate_array = np.asarray(coordinates, dtype=np.float64) * coordinate_definition.scale_to_si
    raw_elements = item.properties["elements"]
    if not isinstance(raw_elements, list):
        raise ModelGraphError(f"{item.identifier}.elements must be a list.")
    element_ids: set[str] = set()
    elements: list[_Element] = []
    for index, raw in enumerate(raw_elements):
        if not isinstance(raw, Mapping):
            raise ModelGraphError(f"{item.identifier}.elements[{index}] must be an object.")
        identifier, start_id, end_id = str(raw["id"]), str(raw["start"]), str(raw["end"])
        if not identifier or identifier in element_ids:
            raise ModelGraphError(f"{item.identifier}.elements must have unique non-empty IDs.")
        if start_id not in lookup or end_id not in lookup or start_id == end_id:
            raise ModelGraphError(f"{item.identifier}.elements[{index}] has invalid endpoints.")
        start, end = lookup[start_id], lookup[end_id]
        if np.linalg.norm(coordinate_array[end] - coordinate_array[start]) <= np.finfo(float).eps:
            raise ModelGraphError(f"{item.identifier}.elements[{index}] has zero length.")
        area = _finite_number(raw["area"], field=f"{item.identifier}.elements[{index}].area", positive=True) * coordinate_definition.scale_to_si**2
        E = _finite_number(raw["youngs_modulus"], field=f"{item.identifier}.elements[{index}].youngs_modulus", positive=True) * stress_definition.scale_to_si
        density = None if "density" not in raw else _finite_number(raw["density"], field=f"{item.identifier}.elements[{index}].density", positive=True) * mass_definition.scale_to_si / coordinate_definition.scale_to_si**3
        element_ids.add(identifier); elements.append(_Element(identifier, start, end, area, E, density))
    fixed: set[int] = set()
    for index, raw in enumerate(item.properties["supports"]):
        if not isinstance(raw, Mapping) or str(raw["node"]) not in lookup:
            raise ModelGraphError(f"{item.identifier}.supports[{index}] names an unknown node.")
        flags = raw["fixed"]
        if not isinstance(flags, list) or len(flags) != dimension or not all(isinstance(value, bool) for value in flags):
            raise ModelGraphError(f"{item.identifier}.supports[{index}].fixed must match dimension.")
        node = lookup[str(raw["node"])]
        fixed.update(node * dimension + axis for axis, flag in enumerate(flags) if flag)
    if not fixed:
        raise ModelGraphError(f"{item.identifier}.supports must restrain at least one degree of freedom.")
    load_cases: dict[str, np.ndarray] = {}
    for case_index, raw_case in enumerate(item.properties["load_cases"]):
        if not isinstance(raw_case, Mapping):
            raise ModelGraphError(f"{item.identifier}.load_cases[{case_index}] must be an object.")
        name = str(raw_case["name"])
        if not name or name in load_cases:
            raise ModelGraphError(f"{item.identifier}.load_cases must have unique non-empty names.")
        forces = np.zeros((len(node_ids), dimension), dtype=np.float64)
        for load_index, raw_load in enumerate(raw_case["loads"]):
            if not isinstance(raw_load, Mapping) or str(raw_load["node"]) not in lookup:
                raise ModelGraphError(f"{item.identifier}.load_cases[{case_index}].loads[{load_index}] names an unknown node.")
            try:
                force = np.asarray(raw_load["force"], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ModelGraphError(f"{item.identifier} load force must be finite.") from exc
            if force.shape != (dimension,) or not np.all(np.isfinite(force)):
                raise ModelGraphError(f"{item.identifier} load force must match dimension.")
            forces[lookup[str(raw_load["node"])]] += force * force_definition.scale_to_si
        load_cases[name] = forces
    masses = np.zeros(len(node_ids), dtype=np.float64)
    for index, raw in enumerate(item.properties.get("nodal_masses", [])):
        if not isinstance(raw, Mapping) or str(raw["node"]) not in lookup:
            raise ModelGraphError(f"{item.identifier}.nodal_masses[{index}] names an unknown node.")
        masses[lookup[str(raw["node"])]] += _finite_number(raw["mass"], field=f"{item.identifier}.nodal_masses[{index}].mass", positive=True) * mass_definition.scale_to_si
    return _Truss(
        item.identifier, dimension, tuple(node_ids), coordinate_array, tuple(elements),
        tuple(sorted(fixed)), load_cases, masses, "m", "N", "Pa",
        {"coordinate": coordinate_unit, "force": force_unit, "stress": stress_unit, "mass": mass_unit},
    )


def validate_material(item: ModelObject) -> None:
    _material(item)


def validate_truss(item: ModelObject) -> None:
    _truss(item)


SEMANTIC_VALIDATORS = {ISOTROPIC_MATERIAL_KIND: validate_material, TRUSS_KIND: validate_truss}


@dataclass(frozen=True, slots=True)
class IsotropicMaterialAnalysis:
    object_id: str
    youngs_modulus: float
    poissons_ratio: float
    shear_modulus: float
    bulk_modulus: float
    lame_first_parameter: float
    density: float | None
    yield_strength: float | None
    thermal_expansion: float | None
    unit_system: str
    source_units: Mapping[str, str]
    stiffness_3d_voigt: np.ndarray
    plane_stress_stiffness: np.ndarray
    plane_strain_stiffness: np.ndarray


def analyse_isotropic_material(model: ModelIR, object_id: str | None = None) -> IsotropicMaterialAnalysis:
    item = select_object(model, ISOTROPIC_MATERIAL_KIND, object_id, label="isotropic material")
    E, nu, density, yield_strength, expansion, source_units = _material(item)
    shear = E / (2.0 * (1.0 + nu))
    bulk = E / (3.0 * (1.0 - 2.0 * nu))
    lame = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    stiffness = np.zeros((6, 6), dtype=np.float64)
    stiffness[:3, :3] = lame
    np.fill_diagonal(stiffness[:3, :3], lame + 2.0 * shear)
    np.fill_diagonal(stiffness[3:, 3:], shear)
    plane_stress = E / (1.0 - nu**2) * np.asarray([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]])
    plane_strain = E / ((1.0 + nu) * (1.0 - 2.0 * nu)) * np.asarray([[1.0 - nu, nu, 0.0], [nu, 1.0 - nu, 0.0], [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]])
    return IsotropicMaterialAnalysis(
        item.identifier, E, nu, shear, bulk, lame, density, yield_strength, expansion,
        "SI", source_units, stiffness, plane_stress, plane_strain,
    )


def _assemble(truss: _Truss) -> tuple[np.ndarray, np.ndarray]:
    dofs = len(truss.node_ids) * truss.dimension
    stiffness = np.zeros((dofs, dofs), dtype=np.float64)
    mass = np.repeat(truss.nodal_masses, truss.dimension).astype(np.float64)
    for element in truss.elements:
        vector = truss.coordinates[element.end] - truss.coordinates[element.start]
        length = float(np.linalg.norm(vector))
        direction = vector / length
        projection = np.outer(direction, direction)
        local = element.youngs_modulus * element.area / length * np.block([[projection, -projection], [-projection, projection]])
        indices = np.asarray([
            *(element.start * truss.dimension + axis for axis in range(truss.dimension)),
            *(element.end * truss.dimension + axis for axis in range(truss.dimension)),
        ], dtype=np.int64)
        stiffness[np.ix_(indices, indices)] += local
        if element.density is not None:
            element_mass = element.density * element.area * length
            mass[element.start * truss.dimension:(element.start + 1) * truss.dimension] += element_mass / 2.0
            mass[element.end * truss.dimension:(element.end + 1) * truss.dimension] += element_mass / 2.0
    return stiffness, mass


@dataclass(frozen=True, slots=True)
class TrussElementResult:
    element_id: str
    start_node: str
    end_node: str
    length: float
    axial_displacement: float
    strain: float
    stress: float
    axial_force: float


@dataclass(frozen=True, slots=True)
class TrussStaticResult:
    object_id: str
    dimension: int
    node_ids: tuple[str, ...]
    coordinates: np.ndarray
    load_case: str
    applied_forces: np.ndarray
    displacements: np.ndarray
    reactions: np.ndarray
    elements: tuple[TrussElementResult, ...]
    strain_energy: float
    maximum_displacement: float
    stiffness_condition_number: float
    coordinate_unit: str
    force_unit: str
    stress_unit: str
    source_units: Mapping[str, str]


def solve_truss_static(
    model: ModelIR, object_id: str | None = None, load_case: str | None = None,
) -> TrussStaticResult:
    item = select_object(model, TRUSS_KIND, object_id, label="truss structure")
    truss = _truss(item)
    selected = next(iter(truss.load_cases)) if load_case is None else str(load_case)
    if selected not in truss.load_cases:
        raise ValueError(f"Unknown truss load case '{selected}'.")
    stiffness, _mass = _assemble(truss)
    force = truss.load_cases[selected].reshape(-1)
    all_dofs = np.arange(stiffness.shape[0])
    free = np.setdiff1d(all_dofs, np.asarray(truss.fixed_dofs, dtype=np.int64), assume_unique=True)
    if free.size == 0:
        raise ValueError("The truss has no free degrees of freedom.")
    reduced = stiffness[np.ix_(free, free)]
    condition = float(np.linalg.cond(reduced))
    if not math.isfinite(condition) or condition > 1e15 or np.linalg.matrix_rank(reduced) < len(free):
        raise ValueError("The restrained truss stiffness matrix is singular or numerically unstable.")
    displacement = np.zeros(stiffness.shape[0], dtype=np.float64)
    displacement[free] = np.linalg.solve(reduced, force[free])
    reaction = stiffness @ displacement - force
    displacement_nodes = displacement.reshape(len(truss.node_ids), truss.dimension)
    element_results: list[TrussElementResult] = []
    for element in truss.elements:
        vector = truss.coordinates[element.end] - truss.coordinates[element.start]
        length = float(np.linalg.norm(vector))
        direction = vector / length
        axial_displacement = float(np.dot(direction, displacement_nodes[element.end] - displacement_nodes[element.start]))
        strain = axial_displacement / length
        stress = element.youngs_modulus * strain
        element_results.append(TrussElementResult(
            element.identifier, truss.node_ids[element.start], truss.node_ids[element.end],
            length, axial_displacement, strain, stress, stress * element.area,
        ))
    return TrussStaticResult(
        truss.object_id, truss.dimension, truss.node_ids, truss.coordinates, selected,
        force.reshape(len(truss.node_ids), truss.dimension), displacement_nodes,
        reaction.reshape(len(truss.node_ids), truss.dimension), tuple(element_results),
        float(0.5 * displacement @ stiffness @ displacement),
        float(np.max(np.linalg.norm(displacement_nodes, axis=1))), condition,
        truss.coordinate_unit, truss.force_unit, truss.stress_unit, truss.source_units,
    )


@dataclass(frozen=True, slots=True)
class TrussModalAnalysis:
    object_id: str
    dimension: int
    node_ids: tuple[str, ...]
    coordinates: np.ndarray
    element_ids: tuple[str, ...]
    element_connectivity: tuple[tuple[int, int], ...]
    frequencies_hz: np.ndarray
    angular_frequencies: np.ndarray
    mode_shapes: np.ndarray
    modal_masses: np.ndarray
    mechanism_count: int
    negative_eigenvalue_count: int
    requested_modes: int
    returned_modes: int
    coordinate_unit: str
    source_units: Mapping[str, str]


def analyse_truss_modes(
    model: ModelIR, object_id: str | None = None, modes: int = 6,
) -> TrussModalAnalysis:
    if isinstance(modes, bool) or not isinstance(modes, int) or not 1 <= modes <= 256:
        raise ValueError("modes must be an integer between 1 and 256.")
    item = select_object(model, TRUSS_KIND, object_id, label="truss structure")
    truss = _truss(item)
    stiffness, mass_diagonal = _assemble(truss)
    all_dofs = np.arange(stiffness.shape[0])
    free = np.setdiff1d(all_dofs, np.asarray(truss.fixed_dofs, dtype=np.int64), assume_unique=True)
    if free.size == 0 or np.any(mass_diagonal[free] <= 0.0):
        raise ValueError("Modal analysis requires positive mass on every free degree of freedom.")
    reduced_stiffness = stiffness[np.ix_(free, free)]
    reduced_mass = np.diag(mass_diagonal[free])
    values, vectors = eigh(reduced_stiffness, reduced_mass, check_finite=True)
    threshold = max(1e-12, float(np.max(np.abs(values), initial=0.0)) * 1e-10)
    mechanism_count = int(np.count_nonzero(np.abs(values) <= threshold))
    negative_count = int(np.count_nonzero(values < -threshold))
    positive = np.flatnonzero(values > threshold)
    selected = positive[:modes]
    if selected.size == 0:
        raise ValueError("The truss has no positive elastic vibration modes.")
    angular = np.sqrt(values[selected])
    shapes = np.zeros((len(selected), stiffness.shape[0]), dtype=np.float64)
    shapes[:, free] = vectors[:, selected].T
    modal_mass = np.asarray([
        float(shape @ np.diag(mass_diagonal) @ shape) for shape in shapes
    ], dtype=np.float64)
    return TrussModalAnalysis(
        truss.object_id, truss.dimension, truss.node_ids, truss.coordinates,
        tuple(element.identifier for element in truss.elements),
        tuple((element.start, element.end) for element in truss.elements),
        angular / (2.0 * np.pi), angular, shapes.reshape(len(selected), len(truss.node_ids), truss.dimension),
        modal_mass, mechanism_count, negative_count, modes, len(selected), truss.coordinate_unit, truss.source_units,
    )


def _kind_applicability(kind: str, label: str):
    def applicable(model: ModelIR) -> tuple[bool, str]:
        found = bool(objects_of_kind(model, kind))
        return found, "" if found else f"an executable {label} object is required"
    return applicable


def _material_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    select_object(
        model, ISOTROPIC_MATERIAL_KIND, settings.get("object_id"), label="isotropic material"
    )
    return 100


def _truss_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(model, TRUSS_KIND, settings.get("object_id"), label="truss structure")
    dimension = int(item.properties["dimension"])
    dofs = dimension * len(item.properties["nodes"])
    return max(1, dofs ** 3 + 10 * len(item.properties["elements"]))


NUMERIC = "org.modellab.comparator.numeric"

CAPABILITY_DESCRIPTORS = (
    CapabilityDescriptor(
        "org.modellab.material.analyse-isotropic-elasticity", "1.2", PACK_ID,
        "Analyse isotropic elastic material", "Derive Lamé, shear and bulk moduli plus 3D, plane-stress and plane-strain constitutive matrices.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}}},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.isotropic-material-analysis", "1.2", "Isotropic material analysis", NUMERIC),),
        _kind_applicability(ISOTROPIC_MATERIAL_KIND, "isotropic-material"), analyse_isotropic_material,
        _material_units, ("org.modellab.renderer.plotly-material-matrix",),
    ),
    CapabilityDescriptor(
        "org.modellab.mechanics.solve-truss-static", "1.2", PACK_ID,
        "Solve linear truss load case", "Assemble and solve a two- or three-dimensional pin-jointed linear truss, including reactions and element stress/resultants.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "load_case": {"type": ["string", "null"]}}},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.truss-static-result", "1.2", "Linear truss static result", NUMERIC),),
        _kind_applicability(TRUSS_KIND, "truss-structure"), solve_truss_static,
        _truss_units, ("org.modellab.renderer.plotly-truss-deformation",),
    ),
    CapabilityDescriptor(
        "org.modellab.mechanics.analyse-truss-modes", "1.2", PACK_ID,
        "Analyse truss vibration modes", "Solve the restrained generalised stiffness/mass eigenproblem using declared nodal and element masses.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "modes": {"type": "integer", "minimum": 1, "maximum": 256, "default": 6}}},
        "scipy+numpy", (ArtifactTypeDescriptor("org.modellab.artifact.truss-modal-analysis", "1.2", "Truss modal analysis", NUMERIC),),
        _kind_applicability(TRUSS_KIND, "truss-structure"), analyse_truss_modes,
        _truss_units, ("org.modellab.renderer.plotly-truss-mode",),
    ),
)


MANIFEST = PackManifest(
    PACK_ID, "1.2", "Mechanics, Structures and Materials",
    "First-class constitutive material records and reproducible linear static/modal analysis for two- and three-dimensional truss structures.",
    (
        "org.modellab.pack.multidimensional-mathematics",
        "org.modellab.pack.dynamics-differential-equations-control",
        "org.modellab.pack.geometry-meshes-spatial-computation",
    ),
    tuple(item.kind for item in KIND_DESCRIPTORS),
    tuple(item.identifier for item in CAPABILITY_DESCRIPTORS),
)
