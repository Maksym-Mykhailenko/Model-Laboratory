"""Optimisation, Estimation and Inverse Problems official capability pack."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

import numpy as np
import sympy as sp
from scipy.optimize import least_squares, minimize

from ..expression import ExpressionError, expression_to_sympy, parse_expression
from ..model import ModelIR
from ..model_graph import ModelGraphError, ModelObject, ObjectKindDescriptor
from ..protocol import ArtifactTypeDescriptor, CapabilityDescriptor, SubspaceRepresentation
from .common import PackManifest, finite_matrix, finite_vector, objects_of_kind, select_object, subspace_representation, unique_labels


PACK_ID = "org.modellab.pack.optimisation-estimation-inverse-problems"
OPTIMISATION_KIND = "org.modellab.optimisation.nonlinear-problem"
NONLINEAR_LEAST_SQUARES_KIND = "org.modellab.estimation.nonlinear-least-squares"
LINEAR_INVERSE_KIND = "org.modellab.inverse.linear-problem"
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")

_BOUNDS_SCHEMA = {
    "type": "array", "minItems": 1, "maxItems": 256,
    "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}},
}
_PARAMETER_SCHEMA = {
    "type": "object", "required": ["name", "value"],
    "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
    "additionalProperties": False,
}
_CONSTRAINT_SCHEMA = {
    "type": "object", "required": ["name", "relation", "expression"],
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 128},
        "relation": {"type": "string", "enum": ["equality", "greater-equal", "less-equal"]},
        "expression": {"type": "string", "minLength": 1, "maxLength": 8192},
    },
    "additionalProperties": False,
}

OPTIMISATION_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["variables", "initial", "bounds", "objective"],
    "properties": {
        "variables": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "string"}},
        "initial": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "number"}},
        "bounds": _BOUNDS_SCHEMA,
        "objective": {"type": "string", "minLength": 1, "maxLength": 8192},
        "constraints": {"type": "array", "maxItems": 512, "items": _CONSTRAINT_SCHEMA},
        "parameters": {"type": "array", "maxItems": 256, "items": _PARAMETER_SCHEMA},
        "objective_sense": {"type": "string", "enum": ["minimise", "maximise"]},
    }, "additionalProperties": False,
}

NONLINEAR_LEAST_SQUARES_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["variables", "initial", "bounds", "residuals"],
    "properties": {
        "variables": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "string"}},
        "initial": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "number"}},
        "bounds": _BOUNDS_SCHEMA,
        "residuals": {"type": "array", "minItems": 1, "maxItems": 100000, "items": {"type": "string", "minLength": 1, "maxLength": 8192}},
        "parameters": {"type": "array", "maxItems": 256, "items": _PARAMETER_SCHEMA},
        "residual_names": {"type": "array", "maxItems": 100000, "items": {"type": "string"}},
    }, "additionalProperties": False,
}

_MATRIX_ROW = {"type": "array", "minItems": 1, "maxItems": 512, "items": {"type": "number"}}
LINEAR_INVERSE_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["parameter_names", "observation_names", "design_matrix", "observations"],
    "properties": {
        "parameter_names": {"type": "array", "minItems": 1, "maxItems": 512, "items": {"type": "string"}},
        "observation_names": {"type": "array", "minItems": 1, "maxItems": 100000, "items": {"type": "string"}},
        "design_matrix": {"type": "array", "minItems": 1, "maxItems": 100000, "items": _MATRIX_ROW},
        "observations": {"type": "array", "minItems": 1, "maxItems": 100000, "items": {"type": "number"}},
        "standard_deviations": {"type": "array", "maxItems": 100000, "items": {"type": "number"}},
        "reference_parameters": {"type": "array", "maxItems": 512, "items": {"type": "number"}},
    }, "additionalProperties": False,
}

KIND_DESCRIPTORS = (
    ObjectKindDescriptor(OPTIMISATION_KIND, "1.0", "Bounded nonlinear optimisation problem", OPTIMISATION_SCHEMA, True),
    ObjectKindDescriptor(NONLINEAR_LEAST_SQUARES_KIND, "1.0", "Bounded nonlinear least-squares estimation study", NONLINEAR_LEAST_SQUARES_SCHEMA, True),
    ObjectKindDescriptor(LINEAR_INVERSE_KIND, "1.0", "Weighted linear inverse problem", LINEAR_INVERSE_SCHEMA, True),
)


@dataclass(frozen=True, slots=True)
class _ExpressionSystem:
    object_id: str
    variables: tuple[str, ...]
    initial: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    expressions: tuple[sp.Expr, ...]
    sources: tuple[str, ...]
    parameters: Mapping[str, float]


def _symbols(value: object, *, field: str) -> tuple[str, ...]:
    names = unique_labels(value, field=field)
    if any(_IDENTIFIER.fullmatch(name) is None for name in names):
        raise ModelGraphError(f"{field} must contain expression-language identifiers.")
    return names


def _expression_system(
    item: ModelObject, sources: list[object], *, source_field: str,
) -> _ExpressionSystem:
    variables = _symbols(item.properties["variables"], field=f"{item.identifier}.variables")
    initial = finite_vector(item.properties["initial"], field=f"{item.identifier}.initial")
    bounds = finite_matrix(item.properties["bounds"], field=f"{item.identifier}.bounds")
    if initial.size != len(variables) or bounds.shape != (len(variables), 2):
        raise ModelGraphError(f"{item.identifier} initial values and bounds must align with variables.")
    if np.any(bounds[:, 0] >= bounds[:, 1]) or np.any(initial < bounds[:, 0]) or np.any(initial > bounds[:, 1]):
        raise ModelGraphError(f"{item.identifier} requires lower < upper bounds containing every initial value.")
    parameters: dict[str, float] = {}
    for index, raw in enumerate(item.properties.get("parameters", [])):
        if not isinstance(raw, Mapping):
            raise ModelGraphError(f"{item.identifier}.parameters[{index}] must be an object.")
        name = str(raw["name"]); value = float(raw["value"])
        if _IDENTIFIER.fullmatch(name) is None or name in variables or name in parameters or not math.isfinite(value):
            raise ModelGraphError(f"{item.identifier}.parameters contains an invalid, colliding or non-finite entry.")
        parameters[name] = value
    declared = set(variables) | set(parameters)
    symbols = {name: sp.Symbol(name, real=True) for name in declared}
    expressions: list[sp.Expr] = []
    clean_sources: list[str] = []
    try:
        for raw in sources:
            source = str(raw).strip()
            expressions.append(sp.simplify(expression_to_sympy(parse_expression(source, declared), symbols)))
            clean_sources.append(source)
    except ExpressionError as exc:
        raise ModelGraphError(f"{item.identifier}.{source_field}: {exc}") from exc
    return _ExpressionSystem(
        item.identifier, variables, initial, bounds[:, 0].copy(), bounds[:, 1].copy(),
        tuple(expressions), tuple(clean_sources), parameters,
    )


@dataclass(frozen=True, slots=True)
class _Constraint:
    name: str
    relation: str
    expression: sp.Expr
    source: str


def _optimisation(item: ModelObject) -> tuple[_ExpressionSystem, tuple[_Constraint, ...], str]:
    raw_constraints = item.properties.get("constraints", [])
    if not isinstance(raw_constraints, list):
        raise ModelGraphError(f"{item.identifier}.constraints must be a list.")
    sources = [item.properties["objective"], *(raw["expression"] for raw in raw_constraints)]
    system = _expression_system(item, sources, source_field="objective/constraints")
    names: set[str] = set()
    constraints: list[_Constraint] = []
    for index, raw in enumerate(raw_constraints):
        if not isinstance(raw, Mapping):
            raise ModelGraphError(f"{item.identifier}.constraints[{index}] must be an object.")
        name = str(raw["name"]); relation = str(raw["relation"])
        if not name.strip() or name in names:
            raise ModelGraphError(f"{item.identifier}.constraints must have unique non-empty names.")
        names.add(name)
        constraints.append(_Constraint(name, relation, system.expressions[index + 1], system.sources[index + 1]))
    sense = str(item.properties.get("objective_sense", "minimise"))
    return _ExpressionSystem(
        system.object_id, system.variables, system.initial, system.lower, system.upper,
        (system.expressions[0],), (system.sources[0],), system.parameters,
    ), tuple(constraints), sense


def _least_squares(item: ModelObject) -> tuple[_ExpressionSystem, tuple[str, ...]]:
    raw = item.properties["residuals"]
    if not isinstance(raw, list):
        raise ModelGraphError(f"{item.identifier}.residuals must be a list.")
    system = _expression_system(item, raw, source_field="residuals")
    names = _optional_names(item.properties.get("residual_names"), len(raw), f"{item.identifier}.residual_names", "residual-")
    return system, names


def _optional_names(value: object, count: int, field: str, prefix: str) -> tuple[str, ...]:
    if value is None:
        return tuple(f"{prefix}{index + 1}" for index in range(count))
    names = unique_labels(value, field=field)
    if len(names) != count:
        raise ModelGraphError(f"{field} must contain exactly {count} labels.")
    return names


@dataclass(frozen=True, slots=True)
class _LinearInverse:
    object_id: str
    parameter_names: tuple[str, ...]
    observation_names: tuple[str, ...]
    design: np.ndarray
    observations: np.ndarray
    standard_deviations: np.ndarray
    reference: np.ndarray


def _linear_inverse(item: ModelObject) -> _LinearInverse:
    parameters = unique_labels(item.properties["parameter_names"], field=f"{item.identifier}.parameter_names")
    observations = unique_labels(item.properties["observation_names"], field=f"{item.identifier}.observation_names")
    design = finite_matrix(item.properties["design_matrix"], field=f"{item.identifier}.design_matrix")
    values = finite_vector(item.properties["observations"], field=f"{item.identifier}.observations")
    if design.shape != (len(observations), len(parameters)) or values.size != len(observations):
        raise ModelGraphError(f"{item.identifier} design and observations must align with declared labels.")
    raw_sigma = item.properties.get("standard_deviations")
    sigma = np.ones(values.size, dtype=np.float64) if raw_sigma is None else finite_vector(raw_sigma, field=f"{item.identifier}.standard_deviations")
    if sigma.size != values.size or np.any(sigma <= 0.0):
        raise ModelGraphError(f"{item.identifier}.standard_deviations must be positive and align with observations.")
    raw_reference = item.properties.get("reference_parameters")
    reference = np.zeros(len(parameters), dtype=np.float64) if raw_reference is None else finite_vector(raw_reference, field=f"{item.identifier}.reference_parameters")
    if reference.size != len(parameters):
        raise ModelGraphError(f"{item.identifier}.reference_parameters must align with parameter_names.")
    return _LinearInverse(item.identifier, parameters, observations, design, values, sigma, reference)


SEMANTIC_VALIDATORS = {
    OPTIMISATION_KIND: lambda item: _optimisation(item),
    NONLINEAR_LEAST_SQUARES_KIND: lambda item: _least_squares(item),
    LINEAR_INVERSE_KIND: lambda item: _linear_inverse(item),
}


def _compiled(system: _ExpressionSystem):
    variables = [sp.Symbol(name, real=True) for name in system.variables]
    parameters = [sp.Symbol(name, real=True) for name in system.parameters]
    values = tuple(system.parameters.values())
    function = sp.lambdify([*variables, *parameters], sp.Matrix(system.expressions), modules="numpy")
    jacobian_expression = sp.Matrix(system.expressions).jacobian(variables)
    jacobian = sp.lambdify([*variables, *parameters], jacobian_expression, modules="numpy")

    def evaluate(point: np.ndarray) -> np.ndarray:
        result = np.asarray(function(*point, *values), dtype=np.float64).reshape(-1)
        if result.size != len(system.expressions) or not np.all(np.isfinite(result)):
            raise ValueError("Expression evaluation produced non-finite or incorrectly shaped values.")
        return result

    def derivative(point: np.ndarray) -> np.ndarray:
        result = np.asarray(jacobian(*point, *values), dtype=np.float64).reshape(len(system.expressions), len(system.variables))
        if not np.all(np.isfinite(result)):
            raise ValueError("Expression derivative produced non-finite values.")
        return result

    return evaluate, derivative, variables, values


def _parameter_null_space(matrix: np.ndarray, rank: int) -> SubspaceRepresentation:
    """Return sign-canonical parameter directions invisible to the observations."""
    _, _, right = np.linalg.svd(matrix, full_matrices=True)
    basis = right[rank:].T.copy()
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    return subspace_representation(basis, orientation="columns")


@dataclass(frozen=True, slots=True)
class OptimisationResult:
    object_id: str
    variable_names: tuple[str, ...]
    optimum: np.ndarray
    objective_value: float
    objective_sense: str
    objective_source: str
    constraint_names: tuple[str, ...]
    constraint_relations: tuple[str, ...]
    constraint_sources: tuple[str, ...]
    constraint_values: np.ndarray
    feasible: bool
    maximum_constraint_violation: float
    objective_gradient: np.ndarray
    objective_hessian: np.ndarray
    objective_hessian_eigenvalues: np.ndarray
    method: str
    feasibility_tolerance: float
    solver_record: Mapping[str, Any]


def solve_nonlinear_problem(
    model: ModelIR, object_id: str | None = None, method: str = "SLSQP",
    maximum_iterations: int = 1000, tolerance: float = 1e-10,
    feasibility_tolerance: float = 1e-8,
) -> OptimisationResult:
    if method != "SLSQP":
        raise ValueError("method must be SLSQP for capability version 1.0.")
    if isinstance(maximum_iterations, bool) or not isinstance(maximum_iterations, int) or not 1 <= maximum_iterations <= 100000:
        raise ValueError("maximum_iterations must be an integer between 1 and 100000.")
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in (tolerance, feasibility_tolerance)):
        raise ValueError("Optimisation tolerances must be positive finite numbers.")
    problem, constraints, sense = _optimisation(select_object(model, OPTIMISATION_KIND, object_id, label="nonlinear optimisation problem"))
    objective_eval, objective_jac, variable_symbols, parameter_values = _compiled(problem)
    sign = -1.0 if sense == "maximise" else 1.0
    scipy_constraints: list[dict[str, Any]] = []
    constraint_functions: list[Any] = []
    constraint_jacobians: list[Any] = []
    for constraint in constraints:
        subsystem = _ExpressionSystem(problem.object_id, problem.variables, problem.initial, problem.lower, problem.upper, (constraint.expression,), (constraint.source,), problem.parameters)
        evaluate, derivative, _, _ = _compiled(subsystem)
        multiplier = -1.0 if constraint.relation == "less-equal" else 1.0
        kind = "eq" if constraint.relation == "equality" else "ineq"
        scipy_constraints.append({
            "type": kind,
            "fun": lambda point, fn=evaluate, factor=multiplier: factor * float(fn(point)[0]),
            "jac": lambda point, fn=derivative, factor=multiplier: factor * fn(point)[0],
        })
        constraint_functions.append(evaluate); constraint_jacobians.append(derivative)
    result = minimize(
        lambda point: sign * float(objective_eval(point)[0]), problem.initial,
        jac=lambda point: sign * objective_jac(point)[0], bounds=list(zip(problem.lower, problem.upper)),
        constraints=scipy_constraints, method=method,
        options={"maxiter": maximum_iterations, "ftol": float(tolerance), "disp": False},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"Nonlinear optimisation failed with solver status {result.status}.")
    point = np.asarray(result.x, dtype=np.float64)
    values = np.asarray([fn(point)[0] for fn in constraint_functions], dtype=np.float64)
    violations = []
    for value, constraint in zip(values, constraints):
        if constraint.relation == "equality":
            violations.append(abs(float(value)))
        elif constraint.relation == "greater-equal":
            violations.append(max(0.0, -float(value)))
        else:
            violations.append(max(0.0, float(value)))
    bound_violation = max(float(np.max(problem.lower - point)), float(np.max(point - problem.upper)), 0.0)
    maximum_violation = max([bound_violation, *violations], default=bound_violation)
    if maximum_violation > feasibility_tolerance:
        raise ValueError("The nonlinear solver returned a point outside the declared feasibility tolerance.")
    objective_expression = problem.expressions[0]
    hessian_expression = sp.hessian(objective_expression, variable_symbols)
    hessian_function = sp.lambdify([*variable_symbols, *[sp.Symbol(name, real=True) for name in problem.parameters]], hessian_expression, modules="numpy")
    hessian = np.asarray(hessian_function(*point, *parameter_values), dtype=np.float64).reshape(len(point), len(point))
    return OptimisationResult(
        problem.object_id, problem.variables, point, float(objective_eval(point)[0]), sense,
        problem.sources[0], tuple(value.name for value in constraints), tuple(value.relation for value in constraints),
        tuple(value.source for value in constraints), values, maximum_violation <= feasibility_tolerance,
        maximum_violation, objective_jac(point)[0], hessian, np.linalg.eigvalsh((hessian + hessian.T) / 2.0),
        method, float(feasibility_tolerance),
        {"scientific": {"converged": True}, "presentation": {"iterations": int(result.nit), "function_evaluations": int(result.nfev), "solver_status": int(result.status), "message": str(result.message)}},
    )


@dataclass(frozen=True, slots=True)
class NonlinearLeastSquaresFit:
    object_id: str
    variable_names: tuple[str, ...]
    residual_names: tuple[str, ...]
    estimate: np.ndarray
    residuals: np.ndarray
    jacobian: np.ndarray
    covariance: np.ndarray | None
    standard_errors: np.ndarray | None
    residual_sum_squares: float
    degrees_of_freedom: int
    jacobian_rank: int
    parameter_count: int
    identifiable: bool
    identifiability_status: str
    unidentifiable_directions: SubspaceRepresentation
    condition_number: float
    method: str
    solver_record: Mapping[str, Any]


def fit_nonlinear_least_squares(
    model: ModelIR, object_id: str | None = None, method: str = "trf",
    maximum_evaluations: int = 10000, relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12, gradient_tolerance: float = 1e-10,
) -> NonlinearLeastSquaresFit:
    if method not in {"trf", "dogbox"}:
        raise ValueError("method must be trf or dogbox.")
    if isinstance(maximum_evaluations, bool) or not isinstance(maximum_evaluations, int) or not 1 <= maximum_evaluations <= 1000000:
        raise ValueError("maximum_evaluations must be an integer between 1 and 1000000.")
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in (relative_tolerance, absolute_tolerance, gradient_tolerance)):
        raise ValueError("Least-squares tolerances must be positive finite numbers.")
    system, names = _least_squares(select_object(model, NONLINEAR_LEAST_SQUARES_KIND, object_id, label="nonlinear least-squares study"))
    evaluate, derivative, _, _ = _compiled(system)
    result = least_squares(
        evaluate, system.initial, jac=derivative, bounds=(system.lower, system.upper), method=method,
        max_nfev=maximum_evaluations, ftol=float(relative_tolerance), xtol=float(absolute_tolerance),
        gtol=float(gradient_tolerance),
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"Nonlinear least squares failed with solver status {result.status}.")
    residuals = np.asarray(result.fun, dtype=np.float64)
    jacobian = np.asarray(result.jac, dtype=np.float64)
    rank = int(np.linalg.matrix_rank(jacobian))
    dof = int(residuals.size - rank)
    rss = float(residuals @ residuals)
    scale = rss / dof if dof > 0 else 0.0
    identifiable = rank == len(system.variables)
    covariance = None if dof <= 0 or not identifiable else scale * np.linalg.inv(jacobian.T @ jacobian)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(np.inf if not identifiable or singular.size == 0 or singular[-1] == 0.0 else singular[0] / singular[-1])
    status = "data-identifiable" if identifiable else "rank-deficient-unidentifiable"
    return NonlinearLeastSquaresFit(
        system.object_id, system.variables, names, np.asarray(result.x), residuals, jacobian,
        covariance, None if covariance is None else np.sqrt(np.maximum(np.diag(covariance), 0.0)), rss, dof,
        rank, len(system.variables), identifiable, status, _parameter_null_space(jacobian, rank), condition, method,
        {"scientific": {"converged": True}, "presentation": {"function_evaluations": int(result.nfev), "jacobian_evaluations": None if result.njev is None else int(result.njev), "solver_status": int(result.status), "message": str(result.message)}},
    )


@dataclass(frozen=True, slots=True)
class LinearInverseSolution:
    object_id: str
    parameter_names: tuple[str, ...]
    observation_names: tuple[str, ...]
    estimate: np.ndarray
    reference_parameters: np.ndarray
    predicted_observations: np.ndarray
    residuals: np.ndarray
    standardised_residuals: np.ndarray
    weighted_residual_sum_squares: float
    reduced_chi_squared: float | None
    singular_values: np.ndarray
    design_rank: int
    parameter_count: int
    data_identifiable: bool
    identifiability_status: str
    unidentifiable_directions: SubspaceRepresentation
    regularised_solution_unique: bool
    condition_number: float
    regularisation: float
    covariance: np.ndarray | None
    standard_errors: np.ndarray | None
    resolution_matrix: np.ndarray


def solve_linear_inverse(
    model: ModelIR, object_id: str | None = None, regularisation: float = 0.0,
) -> LinearInverseSolution:
    if not math.isfinite(float(regularisation)) or float(regularisation) < 0.0:
        raise ValueError("regularisation must be a finite non-negative number.")
    problem = _linear_inverse(select_object(model, LINEAR_INVERSE_KIND, object_id, label="linear inverse problem"))
    weighted_design = problem.design / problem.standard_deviations[:, None]
    weighted_observations = problem.observations / problem.standard_deviations
    alpha = float(regularisation)
    normal = weighted_design.T @ weighted_design + alpha * np.eye(weighted_design.shape[1])
    right = weighted_design.T @ weighted_observations + alpha * problem.reference
    estimate = np.linalg.solve(normal, right) if alpha > 0.0 else np.linalg.pinv(weighted_design) @ weighted_observations
    predicted = problem.design @ estimate
    residuals = problem.observations - predicted
    standardised = residuals / problem.standard_deviations
    chi_squared = float(standardised @ standardised)
    singular = np.linalg.svd(weighted_design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(weighted_design))
    dof = len(problem.observations) - rank
    inverse_normal = np.linalg.pinv(normal, hermitian=True)
    identifiable = rank == weighted_design.shape[1]
    covariance = None if not identifiable else inverse_normal @ (weighted_design.T @ weighted_design) @ inverse_normal
    condition = float(np.inf if not identifiable or singular.size == 0 or singular[-1] == 0.0 else singular[0] / singular[-1])
    return LinearInverseSolution(
        problem.object_id, problem.parameter_names, problem.observation_names, np.asarray(estimate),
        problem.reference, predicted, residuals, standardised, chi_squared,
        None if dof <= 0 else float(chi_squared / dof), singular, rank, len(problem.parameter_names),
        identifiable, "data-identifiable" if identifiable else "rank-deficient-unidentifiable",
        _parameter_null_space(weighted_design, rank), bool(alpha > 0.0 or identifiable), condition, alpha,
        covariance, None if covariance is None else np.sqrt(np.maximum(np.diag(covariance), 0.0)),
        inverse_normal @ (weighted_design.T @ weighted_design),
    )


def _kind_applicability(kind: str, label: str):
    def applicable(model: ModelIR) -> tuple[bool, str]:
        found = bool(objects_of_kind(model, kind))
        return found, "" if found else f"an executable {label} object is required"
    return applicable


def _expression_units(settings: Mapping[str, Any], model: ModelIR, kind: str) -> int:
    item = select_object(
        model, kind, settings.get("object_id"), label="optimisation or estimation problem"
    )
    variables = len(item.properties["variables"])
    evaluations = int(settings.get("maximum_evaluations", settings.get("maximum_iterations", 1000)))
    return max(1, evaluations * variables * variables)


def _inverse_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(
        model, LINEAR_INVERSE_KIND, settings.get("object_id"), label="linear inverse problem"
    )
    rows = len(item.properties["observations"]); columns = len(item.properties["parameter_names"])
    return max(1, rows * columns * columns)


NUMERIC = "org.modellab.comparator.numeric"
CAPABILITY_DESCRIPTORS = (
    CapabilityDescriptor(
        "org.modellab.optimisation.solve-nonlinear-problem", "1.0", PACK_ID,
        "Solve constrained nonlinear problem", "Solve a bounded smooth objective with declared equality and inequality constraints using exact symbolic derivatives.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "method": {"type": "string", "enum": ["SLSQP"], "default": "SLSQP"}, "maximum_iterations": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 1000}, "tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-10}, "feasibility_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-8}}},
        "scipy+numpy+sympy", (ArtifactTypeDescriptor("org.modellab.artifact.nonlinear-optimisation", "1.0", "Nonlinear optimisation result", NUMERIC),),
        _kind_applicability(OPTIMISATION_KIND, "nonlinear-optimisation-problem"), solve_nonlinear_problem,
        lambda settings, model: _expression_units(settings, model, OPTIMISATION_KIND),
        ("org.modellab.renderer.plotly-optimisation",),
    ),
    CapabilityDescriptor(
        "org.modellab.estimation.fit-nonlinear-least-squares", "1.1", PACK_ID,
        "Fit nonlinear least squares", "Estimate bounded parameters from an explicit residual system using exact symbolic Jacobians and covariance diagnostics.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "method": {"type": "string", "enum": ["trf", "dogbox"], "default": "trf"}, "maximum_evaluations": {"type": "integer", "minimum": 1, "maximum": 1000000, "default": 10000}, "relative_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-10}, "absolute_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-12}, "gradient_tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-10}}},
        "scipy+numpy+sympy", (ArtifactTypeDescriptor("org.modellab.artifact.nonlinear-least-squares", "1.1", "Nonlinear least-squares fit", NUMERIC),),
        _kind_applicability(NONLINEAR_LEAST_SQUARES_KIND, "nonlinear-least-squares-study"), fit_nonlinear_least_squares,
        lambda settings, model: _expression_units(settings, model, NONLINEAR_LEAST_SQUARES_KIND),
        ("org.modellab.renderer.plotly-estimation-residuals",),
    ),
    CapabilityDescriptor(
        "org.modellab.inverse.solve-linear-problem", "1.1", PACK_ID,
        "Solve weighted linear inverse problem", "Solve a weighted linear inverse problem with optional zero-order Tikhonov regularisation and resolution diagnostics.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "regularisation": {"type": "number", "minimum": 0, "default": 0}}},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.linear-inverse-solution", "1.1", "Linear inverse solution", NUMERIC),),
        _kind_applicability(LINEAR_INVERSE_KIND, "linear-inverse-problem"), solve_linear_inverse, _inverse_units,
        ("org.modellab.renderer.plotly-inverse-fit",),
    ),
)

MANIFEST = PackManifest(
    PACK_ID, "1.1", "Optimisation, Estimation and Inverse Problems",
    "Bounded constrained optimisation, nonlinear residual estimation and weighted regularised linear inverse problems.",
    ("org.modellab.pack.multidimensional-mathematics",),
    tuple(item.kind for item in KIND_DESCRIPTORS), tuple(item.identifier for item in CAPABILITY_DESCRIPTORS),
)
