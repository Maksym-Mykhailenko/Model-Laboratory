"""Shared operational SI unit registry for official scientific packs.

Every accepted unit token has a dimension vector and an exact decimal scale to canonical
SI.  Official numerical runners either convert through this module or reject the token;
they never retain a unit as a merely reassuring label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..model_graph import ModelGraphError


Dimensions = tuple[int, int, int, int, int, int, int]  # L, M, T, I, temperature, amount, luminous
DIMENSIONLESS: Dimensions = (0, 0, 0, 0, 0, 0, 0)
LENGTH: Dimensions = (1, 0, 0, 0, 0, 0, 0)
MASS: Dimensions = (0, 1, 0, 0, 0, 0, 0)
TIME: Dimensions = (0, 0, 1, 0, 0, 0, 0)
CURRENT: Dimensions = (0, 0, 0, 1, 0, 0, 0)
TEMPERATURE: Dimensions = (0, 0, 0, 0, 1, 0, 0)
AMOUNT: Dimensions = (0, 0, 0, 0, 0, 1, 0)
FORCE: Dimensions = (1, 1, -2, 0, 0, 0, 0)
PRESSURE: Dimensions = (-1, 1, -2, 0, 0, 0, 0)
CHARGE: Dimensions = (0, 0, 1, 1, 0, 0, 0)
VOLTAGE: Dimensions = (2, 1, -3, -1, 0, 0, 0)
RESISTANCE: Dimensions = (2, 1, -3, -2, 0, 0, 0)
CAPACITANCE: Dimensions = (-2, -1, 4, 2, 0, 0, 0)
INDUCTANCE: Dimensions = (2, 1, -2, -2, 0, 0, 0)
PERMITTIVITY: Dimensions = (-3, -1, 4, 2, 0, 0, 0)
CONCENTRATION: Dimensions = (-3, 0, 0, 0, 0, 1, 0)
VELOCITY: Dimensions = (1, 0, -1, 0, 0, 0, 0)
DENSITY: Dimensions = (-3, 1, 0, 0, 0, 0, 0)
FREQUENCY: Dimensions = (0, 0, -1, 0, 0, 0, 0)
THERMAL_EXPANSION: Dimensions = (0, 0, 0, 0, -1, 0, 0)


@dataclass(frozen=True, slots=True)
class UnitDefinition:
    symbol: str
    scale_to_si: float
    dimensions: Dimensions


def _unit(symbol: str, scale: float, dimensions: Dimensions) -> UnitDefinition:
    return UnitDefinition(symbol, float(scale), dimensions)


_DEFINITIONS = (
    _unit("1", 1, DIMENSIONLESS), _unit("dimensionless", 1, DIMENSIONLESS),
    _unit("scaled-density", 1, DIMENSIONLESS), _unit("count", 1, DIMENSIONLESS),
    _unit("m", 1, LENGTH), _unit("cm", 1e-2, LENGTH), _unit("mm", 1e-3, LENGTH),
    _unit("um", 1e-6, LENGTH), _unit("µm", 1e-6, LENGTH), _unit("nm", 1e-9, LENGTH), _unit("km", 1e3, LENGTH),
    _unit("kg", 1, MASS), _unit("g", 1e-3, MASS), _unit("mg", 1e-6, MASS), _unit("ug", 1e-9, MASS), _unit("µg", 1e-9, MASS),
    _unit("s", 1, TIME), _unit("ms", 1e-3, TIME), _unit("min", 60, TIME), _unit("h", 3600, TIME), _unit("day", 86400, TIME),
    _unit("A", 1, CURRENT), _unit("mA", 1e-3, CURRENT), _unit("uA", 1e-6, CURRENT), _unit("µA", 1e-6, CURRENT),
    _unit("K", 1, TEMPERATURE),
    _unit("1/K", 1, THERMAL_EXPANSION), _unit("K^-1", 1, THERMAL_EXPANSION),
    _unit("mol", 1, AMOUNT), _unit("mmol", 1e-3, AMOUNT), _unit("umol", 1e-6, AMOUNT), _unit("µmol", 1e-6, AMOUNT),
    _unit("N", 1, FORCE), _unit("kN", 1e3, FORCE), _unit("mN", 1e-3, FORCE),
    _unit("Pa", 1, PRESSURE), _unit("kPa", 1e3, PRESSURE), _unit("MPa", 1e6, PRESSURE), _unit("GPa", 1e9, PRESSURE),
    _unit("C", 1, CHARGE), _unit("mC", 1e-3, CHARGE), _unit("uC", 1e-6, CHARGE), _unit("µC", 1e-6, CHARGE), _unit("nC", 1e-9, CHARGE), _unit("pC", 1e-12, CHARGE),
    _unit("V", 1, VOLTAGE), _unit("mV", 1e-3, VOLTAGE), _unit("kV", 1e3, VOLTAGE),
    _unit("ohm", 1, RESISTANCE), _unit("Ω", 1, RESISTANCE), _unit("kohm", 1e3, RESISTANCE), _unit("kΩ", 1e3, RESISTANCE), _unit("Mohm", 1e6, RESISTANCE), _unit("MΩ", 1e6, RESISTANCE),
    _unit("F", 1, CAPACITANCE), _unit("mF", 1e-3, CAPACITANCE), _unit("uF", 1e-6, CAPACITANCE), _unit("µF", 1e-6, CAPACITANCE), _unit("nF", 1e-9, CAPACITANCE), _unit("pF", 1e-12, CAPACITANCE),
    _unit("H", 1, INDUCTANCE), _unit("mH", 1e-3, INDUCTANCE), _unit("uH", 1e-6, INDUCTANCE), _unit("µH", 1e-6, INDUCTANCE),
    _unit("F/m", 1, PERMITTIVITY), _unit("pF/m", 1e-12, PERMITTIVITY),
    _unit("mol/m^3", 1, CONCENTRATION), _unit("mol/L", 1e3, CONCENTRATION), _unit("M", 1e3, CONCENTRATION), _unit("mmol/L", 1, CONCENTRATION), _unit("mM", 1, CONCENTRATION), _unit("umol/L", 1e-3, CONCENTRATION), _unit("µmol/L", 1e-3, CONCENTRATION), _unit("uM", 1e-3, CONCENTRATION), _unit("µM", 1e-3, CONCENTRATION),
    _unit("m/s", 1, VELOCITY), _unit("cm/s", 1e-2, VELOCITY),
    _unit("Hz", 1, FREQUENCY), _unit("s^-1", 1, FREQUENCY),
    _unit("kg/m^3", 1, DENSITY), _unit("g/cm^3", 1e3, DENSITY),
)
UNIT_REGISTRY = {definition.symbol: definition for definition in _DEFINITIONS}

_CANONICAL = {
    DIMENSIONLESS: "1", LENGTH: "m", MASS: "kg", TIME: "s", CURRENT: "A",
    TEMPERATURE: "K", AMOUNT: "mol", FORCE: "N", PRESSURE: "Pa", CHARGE: "C",
    VOLTAGE: "V", RESISTANCE: "ohm", CAPACITANCE: "F", INDUCTANCE: "H",
    PERMITTIVITY: "F/m", CONCENTRATION: "mol/m^3", VELOCITY: "m/s", DENSITY: "kg/m^3", FREQUENCY: "s^-1",
    THERMAL_EXPANSION: "1/K",
}


def resolve_unit(symbol: str, *, field: str, expected: Dimensions | None = None) -> UnitDefinition:
    token = str(symbol).strip()
    try:
        definition = UNIT_REGISTRY[token]
    except KeyError as exc:
        raise ModelGraphError(
            f"{field} declares unsupported unit '{token}'. Official numerical packs accept "
            "only operational units with a registered SI conversion."
        ) from exc
    if expected is not None and definition.dimensions != expected:
        raise ModelGraphError(f"{field} unit '{token}' has incompatible physical dimensions.")
    return definition


def to_si(value: Any, symbol: str, *, field: str, expected: Dimensions | None = None) -> np.ndarray:
    definition = resolve_unit(symbol, field=field, expected=expected)
    result = np.asarray(value, dtype=np.float64) * definition.scale_to_si
    if not np.all(np.isfinite(result)):
        raise ModelGraphError(f"{field} conversion produced non-finite SI values.")
    return result


def scalar_to_si(value: float, symbol: str, *, field: str, expected: Dimensions | None = None) -> float:
    return float(to_si(value, symbol, field=field, expected=expected))


def canonical_unit(dimensions: Sequence[int]) -> str:
    key = tuple(int(value) for value in dimensions)
    if len(key) != 7:
        raise ModelGraphError("A physical dimension vector must contain seven SI exponents.")
    return _CANONICAL.get(key, "SI[" + ",".join(str(value) for value in key) + "]")


def add_dimensions(left: Sequence[int], right: Sequence[int]) -> Dimensions:
    return tuple(int(a) + int(b) for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def subtract_dimensions(left: Sequence[int], right: Sequence[int]) -> Dimensions:
    return tuple(int(a) - int(b) for a, b in zip(left, right, strict=True))  # type: ignore[return-value]
