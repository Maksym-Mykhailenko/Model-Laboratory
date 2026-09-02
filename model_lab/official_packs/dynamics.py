"""Dynamical Systems, Differential Equations and Control official pack.

The pack deliberately separates a nonlinear ODE model from a linear state-space
control model.  Both are first-class Model Graph objects; neither is encoded as an
anonymous collection of scalar functions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import root

from ..expression import (
    ExpressionError,
    expression_dependencies,
    expression_to_sympy,
    parse_expression,
)
from ..model import ModelIR
from ..model_graph import ModelGraphError, ModelObject, ObjectKindDescriptor
from ..protocol import ArtifactTypeDescriptor, CapabilityDescriptor
from .common import (
    PackManifest,
    finite_matrix,
    finite_vector,
    objects_of_kind,
    select_object,
    unique_labels,
)


PACK_ID = "org.modellab.pack.dynamics-differential-equations-control"
ODE_KIND = "org.modellab.dynamics.ode-system"
STATE_SPACE_KIND = "org.modellab.control.state-space-system"

_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")

_PARAMETER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["name", "value"],
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 128},
        "value": {"type": "number"},
    },
    "additionalProperties": False,
}

ODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["states", "initial_state", "time_span", "equations"],
    "properties": {
        "states": {
            "type": "array", "minItems": 1, "maxItems": 128,
            "items": {"type": "string"},
        },
        "initial_state": {
            "type": "array", "minItems": 1, "maxItems": 128,
            "items": {"type": "number"},
        },
        "time_span": {
            "type": "array", "minItems": 2, "maxItems": 2,
            "items": {"type": "number"},
        },
        "time_symbol": {"type": "string", "minLength": 1, "maxLength": 128},
        "equations": {
            "type": "array", "minItems": 1, "maxItems": 128,
            "items": {"type": "string", "minLength": 1, "maxLength": 8192},
        },
        "parameters": {
            "type": "array", "maxItems": 256, "items": _PARAMETER_SCHEMA,
        },
    },
    "additionalProperties": False,
}

STATE_SPACE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "states", "inputs", "outputs", "A", "B", "C", "D", "initial_state",
    ],
    "properties": {
        "states": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "string"}},
        "inputs": {"type": "array", "minItems": 1, "maxItems": 128, "items": {"type": "string"}},
        "outputs": {"type": "array", "minItems": 1, "maxItems": 128, "items": {"type": "string"}},
        "A": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "number"}}},
        "B": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "array", "minItems": 1, "maxItems": 128, "items": {"type": "number"}}},
        "C": {"type": "array", "minItems": 1, "maxItems": 128, "items": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "number"}}},
        "D": {"type": "array", "minItems": 1, "maxItems": 128, "items": {"type": "array", "minItems": 1, "maxItems": 128, "items": {"type": "number"}}},
        "initial_state": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "number"}},
    },
    "additionalProperties": False,
}


KIND_DESCRIPTORS = (
    ObjectKindDescriptor(ODE_KIND, "1.0", "Ordinary differential-equation system", ODE_SCHEMA, True),
    ObjectKindDescriptor(STATE_SPACE_KIND, "1.0", "Continuous linear state-space system", STATE_SPACE_SCHEMA, True),
)


@dataclass(frozen=True, slots=True)
class _ODEDefinition:
    object_id: str
    states: tuple[str, ...]
    initial_state: np.ndarray
    time_span: tuple[float, float]
    time_symbol: str
    equations: tuple[sp.Expr, ...]
    equation_sources: tuple[str, ...]
    parameters: Mapping[str, float]
    autonomous: bool


@dataclass(frozen=True, slots=True)
class _StateSpaceDefinition:
    object_id: str
    states: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    initial_state: np.ndarray


def _symbol_names(value: object, *, field: str) -> tuple[str, ...]:
    names = unique_labels(value, field=field)
    if any(_IDENTIFIER.fullmatch(name) is None for name in names):
        raise ModelGraphError(f"{field} must contain expression-language identifiers.")
    return names


def _ode(item: ModelObject) -> _ODEDefinition:
    states = _symbol_names(item.properties["states"], field=f"{item.identifier}.states")
    initial = finite_vector(item.properties["initial_state"], field=f"{item.identifier}.initial_state")
    if initial.size != len(states):
        raise ModelGraphError(f"{item.identifier}.initial_state must align with states.")
    span = finite_vector(item.properties["time_span"], field=f"{item.identifier}.time_span")
    if span.size != 2 or not span[0] < span[1]:
        raise ModelGraphError(f"{item.identifier}.time_span must contain increasing start/end times.")
    time_symbol = str(item.properties.get("time_symbol", "t"))
    if _IDENTIFIER.fullmatch(time_symbol) is None or time_symbol in states:
        raise ModelGraphError(f"{item.identifier}.time_symbol must be a distinct expression identifier.")
    raw_parameters = item.properties.get("parameters", [])
    if not isinstance(raw_parameters, list):
        raise ModelGraphError(f"{item.identifier}.parameters must be a list.")
    parameters: dict[str, float] = {}
    for index, raw in enumerate(raw_parameters):
        if not isinstance(raw, Mapping):
            raise ModelGraphError(f"{item.identifier}.parameters[{index}] must be an object.")
        name = str(raw["name"])
        value = float(raw["value"])
        if _IDENTIFIER.fullmatch(name) is None or name in states or name == time_symbol:
            raise ModelGraphError(f"{item.identifier}.parameters contains an invalid or colliding name '{name}'.")
        if name in parameters or not math.isfinite(value):
            raise ModelGraphError(f"{item.identifier}.parameters must have unique finite values.")
        parameters[name] = value
    raw_equations = item.properties["equations"]
    if not isinstance(raw_equations, list) or len(raw_equations) != len(states):
        raise ModelGraphError(f"{item.identifier}.equations must contain one expression per state.")
    declared = set(states) | set(parameters) | {time_symbol}
    symbols = {name: sp.Symbol(name, real=True) for name in declared}
    equations: list[sp.Expr] = []
    autonomous = True
    try:
        for raw in raw_equations:
            source = str(raw).strip()
            node = parse_expression(source, declared)
            dependencies = expression_dependencies(node)
            autonomous = autonomous and time_symbol not in dependencies
            equations.append(sp.simplify(expression_to_sympy(node, symbols)))
    except ExpressionError as exc:
        raise ModelGraphError(f"{item.identifier}.equations: {exc}") from exc
    return _ODEDefinition(
        item.identifier, states, initial, (float(span[0]), float(span[1])), time_symbol,
        tuple(equations), tuple(str(value).strip() for value in raw_equations), parameters,
        autonomous,
    )


def _state_space(item: ModelObject) -> _StateSpaceDefinition:
    states = _symbol_names(item.properties["states"], field=f"{item.identifier}.states")
    inputs = _symbol_names(item.properties["inputs"], field=f"{item.identifier}.inputs")
    outputs = _symbol_names(item.properties["outputs"], field=f"{item.identifier}.outputs")
    A = finite_matrix(item.properties["A"], field=f"{item.identifier}.A")
    B = finite_matrix(item.properties["B"], field=f"{item.identifier}.B")
    C = finite_matrix(item.properties["C"], field=f"{item.identifier}.C")
    D = finite_matrix(item.properties["D"], field=f"{item.identifier}.D")
    initial = finite_vector(item.properties["initial_state"], field=f"{item.identifier}.initial_state")
    n, p, q = len(states), len(inputs), len(outputs)
    if A.shape != (n, n) or B.shape != (n, p) or C.shape != (q, n) or D.shape != (q, p):
        raise ModelGraphError(f"{item.identifier} state-space matrices do not match declared labels.")
    if initial.size != n:
        raise ModelGraphError(f"{item.identifier}.initial_state must align with states.")
    return _StateSpaceDefinition(item.identifier, states, inputs, outputs, A, B, C, D, initial)


def validate_ode(item: ModelObject) -> None:
    _ode(item)


def validate_state_space(item: ModelObject) -> None:
    _state_space(item)


SEMANTIC_VALIDATORS = {ODE_KIND: validate_ode, STATE_SPACE_KIND: validate_state_space}


def _ode_callable(definition: _ODEDefinition):
    state_symbols = [sp.Symbol(name, real=True) for name in definition.states]
    time_symbol = sp.Symbol(definition.time_symbol, real=True)
    parameter_symbols = [sp.Symbol(name, real=True) for name in definition.parameters]
    compiled = sp.lambdify(
        [time_symbol, *state_symbols, *parameter_symbols],
        sp.Matrix(definition.equations), modules="numpy",
    )
    parameter_values = tuple(definition.parameters.values())

    def evaluate(time: float, state: np.ndarray) -> np.ndarray:
        values = np.asarray(compiled(time, *state, *parameter_values), dtype=np.float64).reshape(-1)
        if values.size != len(definition.states) or not np.all(np.isfinite(values)):
            raise ValueError("ODE evaluation produced non-finite or incorrectly shaped derivatives.")
        return values

    return evaluate


@dataclass(frozen=True, slots=True)
class ODETrajectory:
    object_id: str
    state_names: tuple[str, ...]
    times: np.ndarray
    states: np.ndarray
    equation_sources: tuple[str, ...]
    parameters: Mapping[str, float]
    method: str
    relative_tolerance: float
    absolute_tolerance: float
    solver_record: Mapping[str, Any]


def integrate_ode(
    model: ModelIR,
    object_id: str | None = None,
    samples: int = 501,
    method: str = "DOP853",
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-12,
) -> ODETrajectory:
    if isinstance(samples, bool) or not isinstance(samples, int) or not 2 <= samples <= 20000:
        raise ValueError("samples must be an integer between 2 and 20000.")
    if method not in {"RK45", "DOP853", "Radau", "BDF"}:
        raise ValueError("method must be RK45, DOP853, Radau, or BDF.")
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in (relative_tolerance, absolute_tolerance)):
        raise ValueError("ODE tolerances must be positive finite numbers.")
    item = select_object(model, ODE_KIND, object_id, label="ODE system")
    definition = _ode(item)
    times = np.linspace(*definition.time_span, samples, dtype=np.float64)
    solution = solve_ivp(
        _ode_callable(definition), definition.time_span, definition.initial_state,
        method=method, t_eval=times, rtol=float(relative_tolerance), atol=float(absolute_tolerance),
    )
    if not solution.success or solution.y.shape != (len(definition.states), samples):
        raise ValueError(f"ODE integration failed with solver status {solution.status}.")
    return ODETrajectory(
        definition.object_id, definition.states, times, solution.y.T.copy(),
        definition.equation_sources, dict(definition.parameters), method,
        float(relative_tolerance), float(absolute_tolerance),
        {
            "scientific": {"converged": True},
            "presentation": {
                "function_evaluations": int(solution.nfev),
                "solver_status": int(solution.status),
            },
        },
    )


@dataclass(frozen=True, slots=True)
class EquilibriumPoint:
    coordinates: np.ndarray
    residual_norm: float
    jacobian: np.ndarray
    eigenvalues: tuple[complex, ...]
    classification: str


@dataclass(frozen=True, slots=True)
class ODEEquilibriumAnalysis:
    object_id: str
    state_names: tuple[str, ...]
    autonomous: bool
    points: tuple[EquilibriumPoint, ...]
    attempted_guesses: int
    root_tolerance: float
    stability_tolerance: float


def _stability(eigenvalues: np.ndarray, tolerance: float) -> str:
    real = np.real(eigenvalues)
    if np.all(real < -tolerance):
        return "asymptotically stable"
    if np.any(real > tolerance) and np.any(real < -tolerance):
        return "saddle"
    if np.any(real > tolerance):
        return "unstable"
    return "non-hyperbolic"


def analyse_ode_equilibria(
    model: ModelIR,
    object_id: str | None = None,
    guesses: list[list[float]] | None = None,
    root_tolerance: float = 1e-10,
    stability_tolerance: float = 1e-8,
) -> ODEEquilibriumAnalysis:
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in (root_tolerance, stability_tolerance)):
        raise ValueError("Equilibrium tolerances must be positive finite numbers.")
    item = select_object(model, ODE_KIND, object_id, label="ODE system")
    definition = _ode(item)
    if not definition.autonomous:
        raise ValueError("Equilibria are defined here only for autonomous ODE systems.")
    raw_guesses = guesses if guesses is not None else [np.zeros(len(definition.states)).tolist(), definition.initial_state.tolist()]
    try:
        initial_guesses = np.asarray(raw_guesses, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("guesses must be a rectangular array of finite state vectors.") from exc
    if initial_guesses.ndim != 2 or initial_guesses.shape[1] != len(definition.states) or not 1 <= initial_guesses.shape[0] <= 4096 or not np.all(np.isfinite(initial_guesses)):
        raise ValueError("guesses must contain 1–4096 finite vectors aligned with the ODE states.")
    evaluate = _ode_callable(definition)
    state_symbols = [sp.Symbol(name, real=True) for name in definition.states]
    parameter_symbols = [sp.Symbol(name, real=True) for name in definition.parameters]
    substituted = sp.Matrix(definition.equations).subs(
        {symbol: value for symbol, value in zip(parameter_symbols, definition.parameters.values())}
    )
    jacobian_expression = substituted.jacobian(state_symbols)
    jacobian_function = sp.lambdify(state_symbols, jacobian_expression, modules="numpy")
    accepted: list[np.ndarray] = []
    for guess in initial_guesses:
        solved = root(lambda state: evaluate(definition.time_span[0], state), guess, method="hybr", tol=float(root_tolerance))
        candidate = np.asarray(solved.x, dtype=np.float64)
        residual = float(np.linalg.norm(evaluate(definition.time_span[0], candidate), ord=np.inf))
        if solved.success and np.all(np.isfinite(candidate)) and residual <= max(1e-8, root_tolerance * 100):
            if not any(np.linalg.norm(candidate - known, ord=np.inf) <= max(1e-7, root_tolerance * 1000) for known in accepted):
                accepted.append(candidate)
    accepted.sort(key=lambda row: tuple(float(value) for value in row))
    points: list[EquilibriumPoint] = []
    for coordinates in accepted:
        jacobian = np.asarray(jacobian_function(*coordinates), dtype=np.float64)
        eigenvalues = np.linalg.eigvals(jacobian)
        points.append(EquilibriumPoint(
            coordinates.copy(),
            float(np.linalg.norm(evaluate(definition.time_span[0], coordinates), ord=np.inf)),
            jacobian,
            tuple(complex(value) for value in eigenvalues),
            _stability(eigenvalues, float(stability_tolerance)),
        ))
    return ODEEquilibriumAnalysis(
        definition.object_id, definition.states, True, tuple(points), len(initial_guesses),
        float(root_tolerance), float(stability_tolerance),
    )


@dataclass(frozen=True, slots=True)
class StateSpaceAnalysis:
    object_id: str
    state_names: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    poles: tuple[complex, ...]
    spectral_abscissa: float
    stability: str
    controllability_rank: int
    controllable: bool
    observability_rank: int
    observable: bool
    controllability_matrix: np.ndarray
    observability_matrix: np.ndarray
    direct_feedthrough: bool


def analyse_state_space(
    model: ModelIR, object_id: str | None = None, stability_tolerance: float = 1e-9,
) -> StateSpaceAnalysis:
    if not math.isfinite(float(stability_tolerance)) or float(stability_tolerance) <= 0.0:
        raise ValueError("stability_tolerance must be a positive finite number.")
    item = select_object(model, STATE_SPACE_KIND, object_id, label="state-space system")
    system = _state_space(item)
    n = len(system.states)
    controllability = np.concatenate(
        [np.linalg.matrix_power(system.A, power) @ system.B for power in range(n)], axis=1,
    )
    observability = np.concatenate(
        [system.C @ np.linalg.matrix_power(system.A, power) for power in range(n)], axis=0,
    )
    poles = np.linalg.eigvals(system.A)
    abscissa = float(np.max(np.real(poles)))
    tolerance = float(stability_tolerance)
    stability = "asymptotically stable" if abscissa < -tolerance else "unstable" if abscissa > tolerance else "marginal or non-hyperbolic"
    controllability_rank = int(np.linalg.matrix_rank(controllability))
    observability_rank = int(np.linalg.matrix_rank(observability))
    return StateSpaceAnalysis(
        system.object_id, system.states, system.inputs, system.outputs,
        tuple(complex(value) for value in poles), abscissa, stability,
        controllability_rank, controllability_rank == n,
        observability_rank, observability_rank == n,
        controllability, observability, bool(np.any(system.D != 0.0)),
    )


@dataclass(frozen=True, slots=True)
class StateSpaceTrajectory:
    object_id: str
    state_names: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    times: np.ndarray
    states: np.ndarray
    outputs: np.ndarray
    constant_input: np.ndarray
    method: str
    relative_tolerance: float
    absolute_tolerance: float
    solver_record: Mapping[str, Any]


def simulate_state_space(
    model: ModelIR,
    object_id: str | None = None,
    duration: float = 10.0,
    samples: int = 501,
    constant_input: list[float] | None = None,
    method: str = "DOP853",
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
) -> StateSpaceTrajectory:
    if not math.isfinite(float(duration)) or float(duration) <= 0.0:
        raise ValueError("duration must be a positive finite number.")
    if isinstance(samples, bool) or not isinstance(samples, int) or not 2 <= samples <= 20000:
        raise ValueError("samples must be an integer between 2 and 20000.")
    if method not in {"RK45", "DOP853", "Radau", "BDF"}:
        raise ValueError("method must be RK45, DOP853, Radau, or BDF.")
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in (relative_tolerance, absolute_tolerance)):
        raise ValueError("State-space tolerances must be positive finite numbers.")
    item = select_object(model, STATE_SPACE_KIND, object_id, label="state-space system")
    system = _state_space(item)
    raw_input = np.zeros(len(system.inputs)) if constant_input is None else np.asarray(constant_input, dtype=np.float64)
    if raw_input.ndim != 1 or raw_input.size != len(system.inputs) or not np.all(np.isfinite(raw_input)):
        raise ValueError("constant_input must be a finite vector aligned with inputs.")
    times = np.linspace(0.0, float(duration), samples, dtype=np.float64)
    solution = solve_ivp(
        lambda _time, state: system.A @ state + system.B @ raw_input,
        (0.0, float(duration)), system.initial_state, method=method, t_eval=times,
        rtol=float(relative_tolerance), atol=float(absolute_tolerance),
    )
    if not solution.success:
        raise ValueError(f"State-space integration failed with solver status {solution.status}.")
    states = solution.y.T.copy()
    outputs = states @ system.C.T + raw_input @ system.D.T
    return StateSpaceTrajectory(
        system.object_id, system.states, system.inputs, system.outputs, times, states,
        np.asarray(outputs, dtype=np.float64), raw_input.copy(), method,
        float(relative_tolerance), float(absolute_tolerance),
        {
            "scientific": {"converged": True},
            "presentation": {
                "function_evaluations": int(solution.nfev),
                "solver_status": int(solution.status),
            },
        },
    )


def _kind_applicability(kind: str, label: str):
    def applicable(model: ModelIR) -> tuple[bool, str]:
        found = bool(objects_of_kind(model, kind))
        return found, "" if found else f"an executable {label} object is required"
    return applicable


def _ode_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(model, ODE_KIND, settings.get("object_id"), label="ODE system")
    samples = int(settings.get("samples", 501))
    return max(1, samples * len(item.properties["states"]) * 20)


def _equilibrium_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(model, ODE_KIND, settings.get("object_id"), label="ODE system")
    guesses = settings.get("guesses")
    count = len(guesses) if isinstance(guesses, list) else 2
    dimension = len(item.properties["states"])
    return max(1, count * dimension * dimension * 100)


def _state_space_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(
        model, STATE_SPACE_KIND, settings.get("object_id"), label="state-space system"
    )
    n = len(item.properties["states"])
    return max(1, n ** 3)


def _simulation_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(
        model, STATE_SPACE_KIND, settings.get("object_id"), label="state-space system"
    )
    return max(1, int(settings.get("samples", 501)) * len(item.properties["states"]) * 10)


NUMERIC = "org.modellab.comparator.numeric"

CAPABILITY_DESCRIPTORS = (
    CapabilityDescriptor(
        "org.modellab.dynamics.integrate-ode", "1.0", PACK_ID,
        "Integrate ODE system", "Integrate a finite-dimensional ODE on its declared interval with an explicit SciPy method and tolerances.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "samples": {"type": "integer", "minimum": 2, "maximum": 20000, "default": 501}, "method": {"type": "string", "enum": ["RK45", "DOP853", "Radau", "BDF"], "default": "DOP853"}, "relative_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-9}, "absolute_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-12}}},
        "scipy+numpy+sympy", (ArtifactTypeDescriptor("org.modellab.artifact.ode-trajectory", "1.0", "ODE state trajectory", NUMERIC),),
        _kind_applicability(ODE_KIND, "ODE-system"), integrate_ode, _ode_units,
        ("org.modellab.renderer.plotly-state-trajectory",),
    ),
    CapabilityDescriptor(
        "org.modellab.dynamics.analyse-equilibria", "1.0", PACK_ID,
        "Analyse ODE equilibria", "Solve an autonomous ODE from bounded user-declared guesses and classify local stability from Jacobian eigenvalues.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "guesses": {"type": ["array", "null"], "maxItems": 4096, "items": {"type": "array", "items": {"type": "number"}}}, "root_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-10}, "stability_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-8}}},
        "scipy+numpy+sympy", (ArtifactTypeDescriptor("org.modellab.artifact.ode-equilibria", "1.0", "ODE equilibrium analysis", NUMERIC),),
        _kind_applicability(ODE_KIND, "ODE-system"), analyse_ode_equilibria, _equilibrium_units,
        ("org.modellab.renderer.plotly-equilibrium-spectrum",),
    ),
    CapabilityDescriptor(
        "org.modellab.control.analyse-state-space", "1.0", PACK_ID,
        "Analyse linear state-space system", "Compute poles, stability, controllability and observability for a continuous linear state-space model.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "stability_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-9}}},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.state-space-analysis", "1.0", "State-space structural analysis", NUMERIC),),
        _kind_applicability(STATE_SPACE_KIND, "state-space"), analyse_state_space, _state_space_units,
        ("org.modellab.renderer.plotly-pole-map",),
    ),
    CapabilityDescriptor(
        "org.modellab.control.simulate-state-space", "1.0", PACK_ID,
        "Simulate linear state-space system", "Simulate continuous state and output trajectories under a declared constant input.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "duration": {"type": "number", "exclusiveMinimum": 0, "default": 10}, "samples": {"type": "integer", "minimum": 2, "maximum": 20000, "default": 501}, "constant_input": {"type": ["array", "null"], "items": {"type": "number"}}, "method": {"type": "string", "enum": ["RK45", "DOP853", "Radau", "BDF"], "default": "DOP853"}, "relative_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-10}, "absolute_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-12}}},
        "scipy+numpy", (ArtifactTypeDescriptor("org.modellab.artifact.state-space-trajectory", "1.0", "State-space trajectory", NUMERIC),),
        _kind_applicability(STATE_SPACE_KIND, "state-space"), simulate_state_space, _simulation_units,
        ("org.modellab.renderer.plotly-state-space-trajectory",),
    ),
)


MANIFEST = PackManifest(
    PACK_ID, "1.0", "Dynamical Systems, Differential Equations and Control",
    "Finite-dimensional nonlinear ODE trajectories and equilibria plus continuous linear state-space structure and response.",
    ("org.modellab.pack.multidimensional-mathematics",),
    tuple(item.kind for item in KIND_DESCRIPTORS),
    tuple(item.identifier for item in CAPABILITY_DESCRIPTORS),
)
