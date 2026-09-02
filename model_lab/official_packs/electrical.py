"""Electrical, Electronic and Electromagnetic Systems official capability pack."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from ..model import ModelIR
from ..model_graph import ModelGraphError, ModelObject, ObjectKindDescriptor
from ..protocol import ArtifactTypeDescriptor, CapabilityDescriptor
from .common import PackManifest, objects_of_kind, select_object, unique_labels
from .units import (
    CHARGE, CURRENT, LENGTH, PERMITTIVITY, VOLTAGE, resolve_unit, scalar_to_si, to_si,
)


PACK_ID = "org.modellab.pack.electrical-electronic-electromagnetic-systems"
CIRCUIT_KIND = "org.modellab.electrical.linear-circuit"
DIODE_KIND = "org.modellab.electronics.shockley-diode"
POINT_CHARGE_KIND = "org.modellab.electromagnetics.point-charge-system"
VACUUM_PERMITTIVITY = 8.8541878128e-12

_ELEMENT_SCHEMA = {
    "type": "object", "required": ["id", "type", "from", "to", "value"],
    "properties": {
        "id": {"type": "string", "minLength": 1, "maxLength": 128},
        "type": {"type": "string", "enum": ["resistor", "capacitor", "inductor", "current-source", "voltage-source"]},
        "from": {"type": "string", "minLength": 1, "maxLength": 128},
        "to": {"type": "string", "minLength": 1, "maxLength": 128},
        "value": {"type": "number"},
        "phase_degrees": {"type": "number"},
    }, "additionalProperties": False,
}

CIRCUIT_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["nodes", "ground", "elements"],
    "properties": {
        "nodes": {"type": "array", "minItems": 2, "maxItems": 4096, "items": {"type": "string"}},
        "ground": {"type": "string", "minLength": 1, "maxLength": 128},
        "elements": {"type": "array", "minItems": 1, "maxItems": 100000, "items": _ELEMENT_SCHEMA},
        "frequencies_hz": {"type": "array", "minItems": 1, "maxItems": 20000, "items": {"type": "number"}},
        "voltage_unit": {"type": "string", "maxLength": 128},
        "current_unit": {"type": "string", "maxLength": 128},
    }, "additionalProperties": False,
}

DIODE_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["saturation_current", "ideality_factor", "temperature_kelvin", "voltage_span"],
    "properties": {
        "saturation_current": {"type": "number", "exclusiveMinimum": 0},
        "ideality_factor": {"type": "number", "exclusiveMinimum": 0},
        "temperature_kelvin": {"type": "number", "exclusiveMinimum": 0},
        "voltage_span": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}},
        "area_scale": {"type": "number", "exclusiveMinimum": 0},
    }, "additionalProperties": False,
}

_CHARGE_SCHEMA = {
    "type": "object", "required": ["id", "charge", "position"],
    "properties": {
        "id": {"type": "string", "minLength": 1, "maxLength": 128},
        "charge": {"type": "number"},
        "position": {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "number"}},
    }, "additionalProperties": False,
}

POINT_CHARGE_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["dimension", "charges", "evaluation_points"],
    "properties": {
        "dimension": {"type": "integer", "enum": [2, 3]},
        "charges": {"type": "array", "minItems": 1, "maxItems": 10000, "items": _CHARGE_SCHEMA},
        "evaluation_points": {"type": "array", "minItems": 1, "maxItems": 100000, "items": {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "number"}}},
        "permittivity": {"type": "number", "exclusiveMinimum": 0},
        "coordinate_unit": {"type": "string", "maxLength": 128},
        "charge_unit": {"type": "string", "maxLength": 128},
        "permittivity_unit": {"type": "string", "maxLength": 128},
    }, "additionalProperties": False,
}

KIND_DESCRIPTORS = (
    ObjectKindDescriptor(CIRCUIT_KIND, "1.1", "Linear lumped electrical circuit", CIRCUIT_SCHEMA, True),
    ObjectKindDescriptor(DIODE_KIND, "1.0", "Ideal Shockley diode model", DIODE_SCHEMA, True),
    ObjectKindDescriptor(POINT_CHARGE_KIND, "1.1", "Electrostatic point-charge system", POINT_CHARGE_SCHEMA, True),
)


@dataclass(frozen=True, slots=True)
class _Element:
    identifier: str
    kind: str
    source: int
    target: int
    value: float
    phase_degrees: float


@dataclass(frozen=True, slots=True)
class _Circuit:
    object_id: str
    nodes: tuple[str, ...]
    ground: int
    elements: tuple[_Element, ...]
    frequencies_hz: np.ndarray
    voltage_unit: str
    current_unit: str
    declared_voltage_unit: str
    declared_current_unit: str


def _finite(value: object, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ModelGraphError(f"{field} must be a finite real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelGraphError(f"{field} must be a finite real number.") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ModelGraphError(f"{field} must be {'positive and ' if positive else ''}finite.")
    return result


def _circuit(item: ModelObject) -> _Circuit:
    nodes = unique_labels(item.properties["nodes"], field=f"{item.identifier}.nodes")
    ground_name = str(item.properties["ground"])
    if ground_name not in nodes:
        raise ModelGraphError(f"{item.identifier}.ground must name a declared node.")
    lookup = {name: index for index, name in enumerate(nodes)}
    declared_voltage = str(item.properties.get("voltage_unit", "V"))
    declared_current = str(item.properties.get("current_unit", "A"))
    voltage_definition = resolve_unit(declared_voltage, field=f"{item.identifier}.voltage_unit", expected=VOLTAGE)
    current_definition = resolve_unit(declared_current, field=f"{item.identifier}.current_unit", expected=CURRENT)
    identifiers: set[str] = set(); elements: list[_Element] = []
    for index, raw in enumerate(item.properties["elements"]):
        if not isinstance(raw, Mapping):
            raise ModelGraphError(f"{item.identifier}.elements[{index}] must be an object.")
        identifier = str(raw["id"]); kind = str(raw["type"])
        source_name = str(raw["from"]); target_name = str(raw["to"])
        if not identifier.strip() or identifier in identifiers:
            raise ModelGraphError(f"{item.identifier}.elements must have unique non-empty IDs.")
        if source_name not in lookup or target_name not in lookup or source_name == target_name:
            raise ModelGraphError(f"{item.identifier}.elements[{index}] must connect two distinct declared nodes.")
        value = _finite(raw["value"], field=f"{item.identifier}.elements[{index}].value")
        if kind in {"resistor", "capacitor", "inductor"} and value <= 0.0:
            raise ModelGraphError(f"{item.identifier}.elements[{index}].value must be positive for passive elements.")
        phase = _finite(raw.get("phase_degrees", 0.0), field=f"{item.identifier}.elements[{index}].phase_degrees")
        coherent_scale = {
            "resistor": voltage_definition.scale_to_si / current_definition.scale_to_si,
            "capacitor": current_definition.scale_to_si / voltage_definition.scale_to_si,
            "inductor": voltage_definition.scale_to_si / current_definition.scale_to_si,
            "current-source": current_definition.scale_to_si,
            "voltage-source": voltage_definition.scale_to_si,
        }[kind]
        value *= coherent_scale
        identifiers.add(identifier)
        elements.append(_Element(identifier, kind, lookup[source_name], lookup[target_name], value, phase))
    raw_frequencies = item.properties.get("frequencies_hz", [])
    try:
        frequencies = np.asarray(raw_frequencies, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelGraphError(f"{item.identifier}.frequencies_hz must be finite and positive.") from exc
    if frequencies.ndim != 1 or frequencies.size > 20000 or (frequencies.size and (not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0) or np.any(np.diff(frequencies) <= 0.0))):
        raise ModelGraphError(f"{item.identifier}.frequencies_hz must be strictly increasing positive finite values.")
    return _Circuit(
        item.identifier, nodes, nodes.index(ground_name), tuple(elements), frequencies,
        "V", "A", declared_voltage, declared_current,
    )


@dataclass(frozen=True, slots=True)
class _Diode:
    object_id: str
    saturation_current: float
    ideality_factor: float
    temperature_kelvin: float
    voltage_span: tuple[float, float]
    area_scale: float


def _diode(item: ModelObject) -> _Diode:
    current = _finite(item.properties["saturation_current"], field=f"{item.identifier}.saturation_current", positive=True)
    factor = _finite(item.properties["ideality_factor"], field=f"{item.identifier}.ideality_factor", positive=True)
    temperature = _finite(item.properties["temperature_kelvin"], field=f"{item.identifier}.temperature_kelvin", positive=True)
    raw_span = item.properties["voltage_span"]
    if not isinstance(raw_span, list) or len(raw_span) != 2:
        raise ModelGraphError(f"{item.identifier}.voltage_span must contain two values.")
    lower = _finite(raw_span[0], field=f"{item.identifier}.voltage_span[0]")
    upper = _finite(raw_span[1], field=f"{item.identifier}.voltage_span[1]")
    if lower >= upper:
        raise ModelGraphError(f"{item.identifier}.voltage_span requires lower < upper.")
    area = _finite(item.properties.get("area_scale", 1.0), field=f"{item.identifier}.area_scale", positive=True)
    thermal = 1.380649e-23 * temperature / 1.602176634e-19
    if upper / (factor * thermal) > 700.0:
        raise ModelGraphError(f"{item.identifier}.voltage_span exceeds the finite Shockley evaluation range.")
    return _Diode(item.identifier, current, factor, temperature, (lower, upper), area)


@dataclass(frozen=True, slots=True)
class _ChargeSystem:
    object_id: str
    dimension: int
    identifiers: tuple[str, ...]
    charges: np.ndarray
    positions: np.ndarray
    evaluation_points: np.ndarray
    permittivity: float
    coordinate_unit: str
    charge_unit: str
    declared_coordinate_unit: str
    declared_charge_unit: str
    declared_permittivity_unit: str


def _charge_system(item: ModelObject) -> _ChargeSystem:
    dimension = int(item.properties["dimension"])
    coordinate_unit = str(item.properties.get("coordinate_unit", "m"))
    charge_unit = str(item.properties.get("charge_unit", "C"))
    permittivity_unit = str(item.properties.get("permittivity_unit", "F/m"))
    coordinate_definition = resolve_unit(coordinate_unit, field=f"{item.identifier}.coordinate_unit", expected=LENGTH)
    charge_definition = resolve_unit(charge_unit, field=f"{item.identifier}.charge_unit", expected=CHARGE)
    identifiers: list[str] = []; charges: list[float] = []; positions: list[list[float]] = []
    for index, raw in enumerate(item.properties["charges"]):
        if not isinstance(raw, Mapping):
            raise ModelGraphError(f"{item.identifier}.charges[{index}] must be an object.")
        identifier = str(raw["id"])
        if not identifier.strip() or identifier in identifiers:
            raise ModelGraphError(f"{item.identifier}.charges must have unique non-empty IDs.")
        charge = _finite(raw["charge"], field=f"{item.identifier}.charges[{index}].charge")
        try:
            position = [float(value) for value in raw["position"]]
        except (TypeError, ValueError) as exc:
            raise ModelGraphError(f"{item.identifier}.charges[{index}].position must be finite.") from exc
        if len(position) != dimension or not all(math.isfinite(value) for value in position):
            raise ModelGraphError(f"{item.identifier}.charges[{index}].position must match dimension.")
        identifiers.append(identifier); charges.append(charge * charge_definition.scale_to_si); positions.append(position)
    position_array = np.asarray(positions, dtype=np.float64) * coordinate_definition.scale_to_si
    if len({tuple(row) for row in position_array.tolist()}) != len(position_array):
        raise ModelGraphError(f"{item.identifier}.charges cannot occupy identical positions.")
    try:
        points = np.asarray(item.properties["evaluation_points"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelGraphError(f"{item.identifier}.evaluation_points must be a finite matrix.") from exc
    if points.ndim != 2 or points.shape[1] != dimension or not np.all(np.isfinite(points)):
        raise ModelGraphError(f"{item.identifier}.evaluation_points must match dimension.")
    points *= coordinate_definition.scale_to_si
    for point in points:
        if np.any(np.all(point == position_array, axis=1)):
            raise ModelGraphError(f"{item.identifier}.evaluation_points cannot coincide with a point charge.")
    permittivity = scalar_to_si(
        _finite(item.properties.get("permittivity", VACUUM_PERMITTIVITY), field=f"{item.identifier}.permittivity", positive=True),
        permittivity_unit, field=f"{item.identifier}.permittivity_unit", expected=PERMITTIVITY,
    )
    return _ChargeSystem(
        item.identifier, dimension, tuple(identifiers), np.asarray(charges), position_array, points,
        permittivity, "m", "C", coordinate_unit, charge_unit, permittivity_unit,
    )


SEMANTIC_VALIDATORS = {
    CIRCUIT_KIND: lambda item: _circuit(item),
    DIODE_KIND: lambda item: _diode(item),
    POINT_CHARGE_KIND: lambda item: _charge_system(item),
}


def _phasor(element: _Element, *, alternating: bool) -> complex:
    if not alternating:
        return complex(element.value)
    return element.value * np.exp(1j * np.deg2rad(element.phase_degrees))


def _solve_mna(circuit: _Circuit, frequency: float | None) -> tuple[np.ndarray, np.ndarray, float]:
    alternating = frequency is not None
    omega = 0.0 if frequency is None else 2.0 * math.pi * frequency
    active_nodes = [index for index in range(len(circuit.nodes)) if index != circuit.ground]
    node_index = {node: index for index, node in enumerate(active_nodes)}
    branch_elements = [element for element in circuit.elements if element.kind == "voltage-source" or (not alternating and element.kind == "inductor")]
    size = len(active_nodes) + len(branch_elements)
    matrix = np.zeros((size, size), dtype=np.complex128)
    right = np.zeros(size, dtype=np.complex128)

    def stamp_admittance(source: int, target: int, admittance: complex) -> None:
        if source != circuit.ground:
            left = node_index[source]; matrix[left, left] += admittance
        if target != circuit.ground:
            right_index = node_index[target]; matrix[right_index, right_index] += admittance
        if source != circuit.ground and target != circuit.ground:
            left = node_index[source]; right_index = node_index[target]
            matrix[left, right_index] -= admittance; matrix[right_index, left] -= admittance

    for element in circuit.elements:
        if element.kind == "resistor":
            stamp_admittance(element.source, element.target, 1.0 / element.value)
        elif element.kind == "capacitor" and alternating:
            stamp_admittance(element.source, element.target, 1j * omega * element.value)
        elif element.kind == "inductor" and alternating:
            stamp_admittance(element.source, element.target, 1.0 / (1j * omega * element.value))
        elif element.kind == "current-source":
            current = _phasor(element, alternating=alternating)
            if element.source != circuit.ground:
                right[node_index[element.source]] -= current
            if element.target != circuit.ground:
                right[node_index[element.target]] += current
    for branch, element in enumerate(branch_elements):
        index = len(active_nodes) + branch
        if element.source != circuit.ground:
            node = node_index[element.source]; matrix[node, index] += 1.0; matrix[index, node] += 1.0
        if element.target != circuit.ground:
            node = node_index[element.target]; matrix[node, index] -= 1.0; matrix[index, node] -= 1.0
        right[index] = 0.0 if element.kind == "inductor" else _phasor(element, alternating=alternating)
    try:
        solution = np.linalg.solve(matrix, right)
    except np.linalg.LinAlgError as exc:
        mode = "AC" if alternating else "DC"
        raise ValueError(f"The linear circuit has a singular {mode} modified-nodal system.") from exc
    node_voltages = np.zeros(len(circuit.nodes), dtype=np.complex128)
    node_voltages[active_nodes] = solution[:len(active_nodes)]
    branch_lookup = {element.identifier: solution[len(active_nodes) + index] for index, element in enumerate(branch_elements)}
    currents = np.zeros(len(circuit.elements), dtype=np.complex128)
    for index, element in enumerate(circuit.elements):
        voltage = node_voltages[element.source] - node_voltages[element.target]
        if element.kind == "resistor": currents[index] = voltage / element.value
        elif element.kind == "capacitor": currents[index] = 1j * omega * element.value * voltage if alternating else 0.0
        elif element.kind == "inductor": currents[index] = voltage / (1j * omega * element.value) if alternating else branch_lookup[element.identifier]
        elif element.kind == "current-source": currents[index] = _phasor(element, alternating=alternating)
        else: currents[index] = branch_lookup[element.identifier]
    residual = float(np.max(np.abs(matrix @ solution - right), initial=0.0))
    return node_voltages, currents, residual


@dataclass(frozen=True, slots=True)
class CircuitDCAnalysis:
    object_id: str
    node_names: tuple[str, ...]
    ground_node: str
    element_ids: tuple[str, ...]
    element_types: tuple[str, ...]
    node_voltages: np.ndarray
    element_currents: np.ndarray
    element_voltage_drops: np.ndarray
    element_powers: np.ndarray
    maximum_linear_residual: float
    total_element_power: float
    voltage_unit: str
    current_unit: str
    source_voltage_unit: str
    source_current_unit: str


def analyse_circuit_dc(model: ModelIR, object_id: str | None = None) -> CircuitDCAnalysis:
    circuit = _circuit(select_object(model, CIRCUIT_KIND, object_id, label="linear circuit"))
    voltages, currents, residual = _solve_mna(circuit, None)
    drops = np.asarray([voltages[item.source] - voltages[item.target] for item in circuit.elements])
    powers = np.real(drops * np.conjugate(currents))
    return CircuitDCAnalysis(
        circuit.object_id, circuit.nodes, circuit.nodes[circuit.ground],
        tuple(item.identifier for item in circuit.elements), tuple(item.kind for item in circuit.elements),
        np.real(voltages), np.real(currents), np.real(drops), powers, residual,
        float(np.sum(powers)), circuit.voltage_unit, circuit.current_unit,
        circuit.declared_voltage_unit, circuit.declared_current_unit,
    )


@dataclass(frozen=True, slots=True)
class CircuitACAnalysis:
    object_id: str
    node_names: tuple[str, ...]
    ground_node: str
    element_ids: tuple[str, ...]
    frequencies_hz: np.ndarray
    node_voltage_phasors: np.ndarray
    node_voltage_magnitudes: np.ndarray
    node_voltage_phases_degrees: np.ndarray
    element_current_phasors: np.ndarray
    element_current_magnitudes: np.ndarray
    maximum_linear_residual: float
    voltage_unit: str
    current_unit: str
    source_voltage_unit: str
    source_current_unit: str


def analyse_circuit_ac(
    model: ModelIR, object_id: str | None = None, frequencies_hz: list[float] | None = None,
) -> CircuitACAnalysis:
    circuit = _circuit(select_object(model, CIRCUIT_KIND, object_id, label="linear circuit"))
    frequencies = circuit.frequencies_hz if frequencies_hz is None else np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or not 1 <= frequencies.size <= 20000 or not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0) or np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("frequencies_hz must contain 1–20000 strictly increasing positive finite values.")
    voltages: list[np.ndarray] = []; currents: list[np.ndarray] = []; residual = 0.0
    for frequency in frequencies:
        voltage, current, local_residual = _solve_mna(circuit, float(frequency))
        voltages.append(voltage); currents.append(current); residual = max(residual, local_residual)
    voltage_array = np.asarray(voltages); current_array = np.asarray(currents)
    return CircuitACAnalysis(
        circuit.object_id, circuit.nodes, circuit.nodes[circuit.ground],
        tuple(item.identifier for item in circuit.elements), frequencies.copy(), voltage_array,
        np.abs(voltage_array), np.rad2deg(np.angle(voltage_array)), current_array,
        np.abs(current_array), residual, circuit.voltage_unit, circuit.current_unit,
        circuit.declared_voltage_unit, circuit.declared_current_unit,
    )


@dataclass(frozen=True, slots=True)
class DiodeCharacteristic:
    object_id: str
    voltages: np.ndarray
    currents: np.ndarray
    differential_conductance: np.ndarray
    dynamic_resistance: np.ndarray
    thermal_voltage: float
    saturation_current: float
    ideality_factor: float
    temperature_kelvin: float
    area_scale: float


def evaluate_diode(model: ModelIR, object_id: str | None = None, samples: int = 501) -> DiodeCharacteristic:
    if isinstance(samples, bool) or not isinstance(samples, int) or not 2 <= samples <= 200000:
        raise ValueError("samples must be an integer between 2 and 200000.")
    diode = _diode(select_object(model, DIODE_KIND, object_id, label="Shockley-diode"))
    thermal = 1.380649e-23 * diode.temperature_kelvin / 1.602176634e-19
    voltages = np.linspace(*diode.voltage_span, samples)
    exponent = voltages / (diode.ideality_factor * thermal)
    scale = diode.saturation_current * diode.area_scale
    currents = scale * np.expm1(exponent)
    conductance = scale * np.exp(exponent) / (diode.ideality_factor * thermal)
    resistance = np.divide(1.0, conductance, out=np.full_like(conductance, np.inf), where=conductance > 0.0)
    return DiodeCharacteristic(
        diode.object_id, voltages, currents, conductance, resistance, thermal,
        diode.saturation_current, diode.ideality_factor, diode.temperature_kelvin, diode.area_scale,
    )


@dataclass(frozen=True, slots=True)
class ElectrostaticAnalysis:
    object_id: str
    dimension: int
    charge_ids: tuple[str, ...]
    charges: np.ndarray
    charge_positions: np.ndarray
    evaluation_points: np.ndarray
    potential: np.ndarray
    electric_field: np.ndarray
    field_magnitude: np.ndarray
    net_charge: float
    dipole_moment: np.ndarray
    pair_potential_energy: float
    permittivity: float
    coordinate_unit: str
    charge_unit: str
    potential_unit: str
    electric_field_unit: str
    dipole_moment_unit: str
    energy_unit: str
    source_units: Mapping[str, str]


def analyse_electrostatics(model: ModelIR, object_id: str | None = None) -> ElectrostaticAnalysis:
    system = _charge_system(select_object(model, POINT_CHARGE_KIND, object_id, label="point-charge system"))
    factor = 1.0 / (4.0 * math.pi * system.permittivity)
    displacement = system.evaluation_points[:, None, :] - system.positions[None, :, :]
    distance = np.linalg.norm(displacement, axis=2)
    potential = factor * np.sum(system.charges[None, :] / distance, axis=1)
    field = factor * np.sum(system.charges[None, :, None] * displacement / distance[:, :, None] ** 3, axis=1)
    energy = 0.0
    for left in range(len(system.charges)):
        for right in range(left + 1, len(system.charges)):
            separation = float(np.linalg.norm(system.positions[left] - system.positions[right]))
            energy += factor * system.charges[left] * system.charges[right] / separation
    return ElectrostaticAnalysis(
        system.object_id, system.dimension, system.identifiers, system.charges, system.positions,
        system.evaluation_points, potential, field, np.linalg.norm(field, axis=1),
        float(np.sum(system.charges)), np.sum(system.charges[:, None] * system.positions, axis=0),
        float(energy), system.permittivity, system.coordinate_unit, system.charge_unit,
        "V", "V/m", "C*m", "J", {
            "coordinate": system.declared_coordinate_unit,
            "charge": system.declared_charge_unit,
            "permittivity": system.declared_permittivity_unit,
        },
    )


def _kind_applicability(kind: str, label: str):
    def applicable(model: ModelIR) -> tuple[bool, str]:
        found = bool(objects_of_kind(model, kind))
        return found, "" if found else f"an executable {label} object is required"
    return applicable


def _circuit_units(settings: Mapping[str, Any], model: ModelIR, alternating: bool) -> int:
    item = select_object(model, CIRCUIT_KIND, settings.get("object_id"), label="linear circuit")
    nodes = len(item.properties["nodes"]); elements = len(item.properties["elements"])
    frequency_count = len(settings.get("frequencies_hz") or item.properties.get("frequencies_hz", [])) if alternating else 1
    return max(1, frequency_count * (nodes ** 3 + elements))


def _simple_units(settings: Mapping[str, Any], model: ModelIR, kind: str) -> int:
    item = select_object(model, kind, settings.get("object_id"), label="electrical-system object")
    return max(1, int(settings.get("samples", len(item.properties.get("evaluation_points", [])) or 501)))


NUMERIC = "org.modellab.comparator.numeric"
CAPABILITY_DESCRIPTORS = (
    CapabilityDescriptor(
        "org.modellab.electrical.analyse-dc-circuit", "1.2", PACK_ID,
        "Analyse DC circuit", "Solve a linear RLC/source circuit by modified nodal analysis; capacitors are open and inductors ideal shorts at DC.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}}},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.dc-circuit-analysis", "1.2", "DC circuit analysis", NUMERIC),),
        _kind_applicability(CIRCUIT_KIND, "linear-circuit"), analyse_circuit_dc,
        lambda settings, model: _circuit_units(settings, model, False), ("org.modellab.renderer.plotly-dc-circuit",),
    ),
    CapabilityDescriptor(
        "org.modellab.electrical.analyse-ac-circuit", "1.2", PACK_ID,
        "Analyse AC circuit", "Solve complex linear RLC/source phasors over explicit positive frequencies by modified nodal analysis.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "frequencies_hz": {"type": ["array", "null"], "minItems": 1, "maxItems": 20000, "items": {"type": "number"}}}},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.ac-circuit-analysis", "1.2", "AC circuit analysis", NUMERIC),),
        _kind_applicability(CIRCUIT_KIND, "linear-circuit"), analyse_circuit_ac,
        lambda settings, model: _circuit_units(settings, model, True), ("org.modellab.renderer.plotly-ac-response",),
    ),
    CapabilityDescriptor(
        "org.modellab.electronics.evaluate-shockley-diode", "1.0", PACK_ID,
        "Evaluate Shockley diode", "Evaluate ideal-diode current, differential conductance and dynamic resistance over an explicit voltage span.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "samples": {"type": "integer", "minimum": 2, "maximum": 200000, "default": 501}}},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.diode-characteristic", "1.0", "Diode characteristic", NUMERIC),),
        _kind_applicability(DIODE_KIND, "Shockley-diode"), evaluate_diode,
        lambda settings, model: _simple_units(settings, model, DIODE_KIND), ("org.modellab.renderer.plotly-diode-characteristic",),
    ),
    CapabilityDescriptor(
        "org.modellab.electromagnetics.analyse-point-charges", "1.2", PACK_ID,
        "Analyse electrostatic point charges", "Compute potential and electric field at declared points plus net charge, dipole moment and pair energy.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}}},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.electrostatic-analysis", "1.2", "Electrostatic point-charge analysis", NUMERIC),),
        _kind_applicability(POINT_CHARGE_KIND, "point-charge-system"), analyse_electrostatics,
        lambda settings, model: _simple_units(settings, model, POINT_CHARGE_KIND), ("org.modellab.renderer.plotly-electrostatic-field",),
    ),
)

MANIFEST = PackManifest(
    PACK_ID, "1.2", "Electrical, Electronic and Electromagnetic Systems",
    "Linear DC/AC lumped circuits, ideal semiconductor characteristics and electrostatic point-charge fields with explicit physical semantics.",
    ("org.modellab.pack.multidimensional-mathematics",),
    tuple(item.kind for item in KIND_DESCRIPTORS), tuple(item.identifier for item in CAPABILITY_DESCRIPTORS),
)
