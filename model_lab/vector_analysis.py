"""Vector, matrix, root, stability, and constrained-optimisation analyses.

These routines operate on first-class vector and matrix functions.  They are independent
of the desktop renderer and return deterministic scientific result objects suitable for
conversion into typed artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.stats import qmc
import sympy as sp

from .issues import NumericalDiagnostic, NumericalStatus
from .model import ConstraintRelation, MatrixFunction, ModelIR, VectorFunction
from .provenance import Provenance, model_ref


class VectorAnalysisError(ValueError):
    """Raised when a vector/matrix analysis request is invalid."""


@dataclass(frozen=True, slots=True)
class VectorDifferentialAnalysis:
    function_name: str
    input_names: tuple[str, ...]
    component_names: tuple[str, ...]
    jacobian: sp.Matrix
    divergence: sp.Expr | None
    curl: sp.Expr | tuple[sp.Expr, sp.Expr, sp.Expr] | None
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class SystemRoot:
    coordinates: tuple[float, ...]
    residual: tuple[float, ...]
    residual_norm: float
    jacobian: tuple[tuple[float, ...], ...]
    eigenvalues: tuple[complex, ...]
    stability: str | None


@dataclass(frozen=True, slots=True)
class SystemRootResult:
    function_name: str
    input_names: tuple[str, ...]
    component_names: tuple[str, ...]
    roots: tuple[SystemRoot, ...]
    attempted_seeds: int
    converged_seeds: int
    numerical_status: NumericalStatus
    diagnostics: tuple[NumericalDiagnostic, ...]
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class VectorFieldEvaluation:
    function_name: str
    x_name: str
    y_name: str
    component_names: tuple[str, str]
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    speed: np.ndarray
    equilibria: tuple[SystemRoot, ...]
    provenance: Provenance
    numerical_status: NumericalStatus = NumericalStatus.COMPLETE
    diagnostics: tuple[NumericalDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class MatrixAnalysisResult:
    function_name: str
    input_names: tuple[str, ...]
    point: tuple[float, ...]
    symbolic_matrix: sp.Matrix
    numerical_matrix: tuple[tuple[float | complex, ...], ...]
    determinant: complex | None
    symbolic_determinant: sp.Expr | None
    rank: int
    singular_values: tuple[float, ...]
    condition_number: float
    eigenvalues: tuple[complex, ...]
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class ConstrainedOptimum:
    coordinates: tuple[float, ...]
    value: float
    maximum_constraint_violation: float


@dataclass(frozen=True, slots=True)
class ConstrainedOptimizationResult:
    function_name: str
    objective: Literal["minimize", "maximize"]
    input_names: tuple[str, ...]
    optimum: ConstrainedOptimum | None
    attempted_seeds: int
    converged_seeds: int
    numerical_status: NumericalStatus
    diagnostics: tuple[NumericalDiagnostic, ...]
    provenance: Provenance


def _selected_vector_function(model: ModelIR, name: str | None) -> VectorFunction:
    if not model.vector_functions:
        raise VectorAnalysisError("The model does not declare a vector function.")
    if name is None:
        if len(model.vector_functions) != 1:
            raise VectorAnalysisError("Select one vector function for this analysis.")
        return model.vector_functions[0]
    try:
        return model.vector_function(name)
    except KeyError as exc:
        raise VectorAnalysisError(f"Unknown vector function '{name}'.") from exc


def _selected_matrix_function(model: ModelIR, name: str | None) -> MatrixFunction:
    if not model.matrix_functions:
        raise VectorAnalysisError("The model does not declare a matrix function.")
    if name is None:
        if len(model.matrix_functions) != 1:
            raise VectorAnalysisError("Select one matrix function for this analysis.")
        return model.matrix_functions[0]
    try:
        return model.matrix_function(name)
    except KeyError as exc:
        raise VectorAnalysisError(f"Unknown matrix function '{name}'.") from exc


def _parameter_values(model: ModelIR, overrides: dict[str, float] | None) -> dict[str, float]:
    values = model.parameter_defaults()
    if overrides:
        unknown = set(overrides) - set(values)
        if unknown:
            raise VectorAnalysisError(
                "Unknown parameter override(s): " + ", ".join(sorted(unknown)) + "."
            )
        values.update({name: float(value) for name, value in overrides.items()})
    for parameter in model.parameters:
        value = values[parameter.name]
        if not math.isfinite(value) or not parameter.domain.contains(value):
            raise VectorAnalysisError(
                f"Parameter '{parameter.name}' is non-finite or outside its declared domain."
            )
    return values


def _symbols_and_fixed(
    model: ModelIR, parameter_values: dict[str, float] | None
) -> tuple[tuple[sp.Symbol, ...], tuple[sp.Symbol, ...], tuple[float, ...]]:
    variable_symbols = tuple(sp.Symbol(item.name, real=True) for item in model.variables)
    fixed_symbols = tuple(
        sp.Symbol(item.name, real=True) for item in (*model.parameters, *model.constants)
    )
    parameters = _parameter_values(model, parameter_values)
    fixed_values = tuple(
        [parameters[item.name] for item in model.parameters]
        + [item.value for item in model.constants]
    )
    return variable_symbols, fixed_symbols, fixed_values


def _source_refs_for_vector(model: ModelIR, function: VectorFunction) -> tuple[str, ...]:
    dependencies = sorted(
        {dependency for component in function.components for dependency in component.dependencies}
    )
    return (
        model_ref("vector_function", function.name),
        *(model_ref("derived_quantity", item) for item in model.transitive_derived_dependencies(tuple(dependencies))),
        *(model_ref("variable", item.name) for item in model.variables),
        *(model_ref("parameter", item.name) for item in model.parameters),
        *(model_ref("constant", item.name) for item in model.constants),
    )


def analyse_vector_function(
    model: ModelIR, function_name: str | None = None
) -> VectorDifferentialAnalysis:
    """Return the general m-by-n Jacobian and applicable field operators."""
    function = _selected_vector_function(model, function_name)
    variables = tuple(sp.Symbol(item.name, real=True) for item in model.variables)
    expressions = sp.Matrix([item.expression for item in function.components])
    jacobian = expressions.jacobian(variables).applyfunc(sp.simplify)
    divergence: sp.Expr | None = None
    curl: sp.Expr | tuple[sp.Expr, sp.Expr, sp.Expr] | None = None
    if len(function.components) == len(variables):
        divergence = sp.simplify(sum(jacobian[index, index] for index in range(len(variables))))
    if len(variables) == len(function.components) == 2:
        curl = sp.simplify(jacobian[1, 0] - jacobian[0, 1])
    elif len(variables) == len(function.components) == 3:
        curl = (
            sp.simplify(jacobian[2, 1] - jacobian[1, 2]),
            sp.simplify(jacobian[0, 2] - jacobian[2, 0]),
            sp.simplify(jacobian[1, 0] - jacobian[0, 1]),
        )
    return VectorDifferentialAnalysis(
        function_name=function.name,
        input_names=tuple(item.name for item in model.variables),
        component_names=tuple(item.name for item in function.components),
        jacobian=jacobian,
        divergence=divergence,
        curl=curl,
        provenance=Provenance.derived(
            _source_refs_for_vector(model, function),
            f"symbolic Jacobian and applicable field operators of {function.name}",
        ),
    )


def _vector_numerical_functions(
    model: ModelIR,
    function: VectorFunction,
    parameter_values: dict[str, float] | None,
) -> tuple[Callable[..., object], Callable[..., object], tuple[float, ...]]:
    variable_symbols, fixed_symbols, fixed_values = _symbols_and_fixed(model, parameter_values)
    expressions = sp.Matrix([item.expression for item in function.components])
    jacobian = expressions.jacobian(variable_symbols)
    arguments = (*variable_symbols, *fixed_symbols)
    return (
        sp.lambdify(arguments, expressions, modules="numpy"),
        sp.lambdify(arguments, jacobian, modules="numpy"),
        fixed_values,
    )


def classify_local_stability(eigenvalues: tuple[complex, ...], *, tolerance: float = 1e-9) -> str:
    """Classify a square autonomous vector-field equilibrium from Jacobian eigenvalues."""
    real_parts = np.asarray([value.real for value in eigenvalues], dtype=float)
    if np.all(real_parts < -tolerance):
        return "asymptotically stable"
    if np.any(real_parts > tolerance) and np.any(real_parts < -tolerance):
        return "saddle / unstable"
    if np.any(real_parts > tolerance):
        return "unstable"
    if np.all(np.abs(real_parts) <= tolerance):
        return "centre or non-hyperbolic / inconclusive"
    return "non-hyperbolic / inconclusive"


def _deterministic_seeds(model: ModelIR, count: int) -> np.ndarray:
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 4096:
        raise VectorAnalysisError("Seed count must be an integer between 1 and 4096.")
    dimension = len(model.variables)
    if dimension == 0:
        raise VectorAnalysisError("Root solving requires at least one variable.")
    lower = np.asarray([item.domain.lower for item in model.variables], dtype=float)
    upper = np.asarray([item.domain.upper for item in model.variables], dtype=float)
    candidates: list[np.ndarray] = []
    if all(item.initial_value is not None for item in model.variables):
        candidates.append(np.asarray([item.initial_value for item in model.variables], dtype=float))
    candidates.append((lower + upper) / 2.0)
    remaining = max(0, count - len(candidates))
    if remaining:
        samples = qmc.Halton(d=dimension, scramble=False).random(remaining + 1)[1:]
        candidates.extend(qmc.scale(samples, lower, upper))
    return np.asarray(candidates[:count], dtype=float)


def solve_vector_system(
    model: ModelIR,
    parameter_values: dict[str, float] | None = None,
    *,
    function_name: str | None = None,
    seeds: int = 64,
    residual_tolerance: float = 1e-9,
    coordinate_tolerance: float = 1e-8,
    max_nfev: int = 2000,
) -> SystemRootResult:
    """Solve F(x)=0 using bounded deterministic seeds and mathematical de-duplication."""
    if model.blocking_ambiguities:
        raise VectorAnalysisError("Resolve blocking ambiguities before root solving.")
    if residual_tolerance <= 0 or coordinate_tolerance <= 0:
        raise VectorAnalysisError("Root tolerances must be positive.")
    function = _selected_vector_function(model, function_name)
    numerical, numerical_jacobian, fixed = _vector_numerical_functions(
        model, function, parameter_values
    )
    lower = np.asarray([item.domain.lower for item in model.variables], dtype=float)
    upper = np.asarray([item.domain.upper for item in model.variables], dtype=float)
    candidates = _deterministic_seeds(model, seeds)

    def residual(point: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            value = np.asarray(numerical(*point, *fixed), dtype=float).reshape(-1)
        if value.size != len(function.components) or not np.all(np.isfinite(value)):
            return np.full(len(function.components), 1e100, dtype=float)
        return value

    def jacobian(point: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            value = np.asarray(numerical_jacobian(*point, *fixed), dtype=float)
        if value.shape != (len(function.components), len(model.variables)) or not np.all(
            np.isfinite(value)
        ):
            raise ValueError("non-finite Jacobian")
        return value

    roots: list[SystemRoot] = []
    converged = 0
    for seed in candidates:
        try:
            solution = least_squares(
                residual,
                seed,
                jac=jacobian,
                bounds=(lower, upper),
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=max_nfev,
            )
        except Exception:
            continue
        point = np.asarray(solution.x, dtype=float)
        values = residual(point)
        norm = float(np.linalg.norm(values, ord=2))
        if not solution.success or not np.isfinite(norm) or norm > residual_tolerance:
            continue
        converged += 1
        scale = np.maximum(1.0, np.abs(point))
        if any(
            np.all(np.abs(point - np.asarray(item.coordinates)) <= coordinate_tolerance * scale)
            for item in roots
        ):
            continue
        jacobian_value = jacobian(point)
        eigenvalues: tuple[complex, ...] = ()
        stability: str | None = None
        if jacobian_value.shape[0] == jacobian_value.shape[1]:
            eigenvalues = tuple(complex(item) for item in np.linalg.eigvals(jacobian_value))
            stability = classify_local_stability(eigenvalues)
        roots.append(
            SystemRoot(
                coordinates=tuple(float(item) for item in point),
                residual=tuple(float(item) for item in values),
                residual_norm=norm,
                jacobian=tuple(tuple(float(item) for item in row) for row in jacobian_value),
                eigenvalues=eigenvalues,
                stability=stability,
            )
        )
    roots.sort(key=lambda item: item.coordinates)
    diagnostics: list[NumericalDiagnostic] = []
    status = NumericalStatus.COMPLETE
    if not roots:
        status = NumericalStatus.PARTIAL
        diagnostics.append(
            NumericalDiagnostic.warning(
                "no_system_root_located",
                "No system root was located within the declared domains and deterministic seed budget.",
                attempted_seeds=len(candidates),
            )
        )
    return SystemRootResult(
        function_name=function.name,
        input_names=tuple(item.name for item in model.variables),
        component_names=tuple(item.name for item in function.components),
        roots=tuple(roots),
        attempted_seeds=len(candidates),
        converged_seeds=converged,
        numerical_status=status,
        diagnostics=tuple(diagnostics),
        provenance=Provenance.derived(
            _source_refs_for_vector(model, function),
            f"bounded deterministic least-squares solution of {function.name}(x)=0",
        ),
    )


def evaluate_vector_field_2d(
    model: ModelIR,
    parameter_values: dict[str, float] | None = None,
    *,
    function_name: str | None = None,
    points_per_axis: int = 25,
    root_seeds: int = 64,
) -> VectorFieldEvaluation:
    """Evaluate a two-dimensional vector field for arrows, streamlines, and nullclines."""
    if len(model.variables) != 2:
        raise VectorAnalysisError("Vector-field visualisation requires exactly two variables.")
    function = _selected_vector_function(model, function_name)
    if len(function.components) != 2:
        raise VectorAnalysisError("Vector-field visualisation requires exactly two components.")
    if not isinstance(points_per_axis, int) or not 3 <= points_per_axis <= 250:
        raise VectorAnalysisError("points_per_axis must be an integer between 3 and 250.")
    numerical, _, fixed = _vector_numerical_functions(model, function, parameter_values)
    x_values = np.linspace(
        model.variables[0].domain.lower, model.variables[0].domain.upper, points_per_axis
    )
    y_values = np.linspace(
        model.variables[1].domain.lower, model.variables[1].domain.upper, points_per_axis
    )
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
    with np.errstate(all="ignore"):
        raw = np.asarray(numerical(x_grid, y_grid, *fixed), dtype=float)
    raw = np.squeeze(raw)
    if raw.shape == (2, *x_grid.shape):
        u, v = raw[0], raw[1]
    elif raw.shape == (*x_grid.shape, 2):
        u, v = raw[..., 0], raw[..., 1]
    else:
        raise VectorAnalysisError("The vector field did not return two values per grid point.")
    speed = np.hypot(u, v)
    finite = np.isfinite(u) & np.isfinite(v)
    diagnostics: list[NumericalDiagnostic] = []
    status = NumericalStatus.COMPLETE
    if not np.any(finite):
        raise VectorAnalysisError("The vector field has no finite samples in its declared domain.")
    if not np.all(finite):
        status = NumericalStatus.PARTIAL
        diagnostics.append(
            NumericalDiagnostic.warning(
                "non_finite_vector_field_samples",
                "Some vector-field samples were non-finite.",
                count=int(np.size(finite) - np.count_nonzero(finite)),
                total=int(np.size(finite)),
            )
        )
    roots = solve_vector_system(
        model,
        parameter_values,
        function_name=function.name,
        seeds=root_seeds,
    )
    return VectorFieldEvaluation(
        function_name=function.name,
        x_name=model.variables[0].name,
        y_name=model.variables[1].name,
        component_names=(function.components[0].name, function.components[1].name),
        x=x_grid,
        y=y_grid,
        u=np.where(finite, u, np.nan),
        v=np.where(finite, v, np.nan),
        speed=np.where(finite, speed, np.nan),
        equilibria=roots.roots,
        provenance=Provenance.derived(
            _source_refs_for_vector(model, function),
            f"two-dimensional grid evaluation of vector field {function.name}",
        ),
        numerical_status=status,
        diagnostics=tuple(diagnostics),
    )


def analyse_matrix_function(
    model: ModelIR,
    point: tuple[float, ...],
    parameter_values: dict[str, float] | None = None,
    *,
    function_name: str | None = None,
) -> MatrixAnalysisResult:
    """Evaluate and analyse a rectangular matrix-valued function at one point."""
    function = _selected_matrix_function(model, function_name)
    if len(point) != len(model.variables) or not all(math.isfinite(float(item)) for item in point):
        raise VectorAnalysisError("Matrix-analysis point must supply one finite value per variable.")
    for variable, value in zip(model.variables, point):
        if not variable.domain.contains(float(value)):
            raise VectorAnalysisError(f"Point coordinate '{variable.name}' lies outside its domain.")
    variable_symbols, fixed_symbols, fixed_values = _symbols_and_fixed(model, parameter_values)
    symbolic = sp.Matrix([[item.expression for item in row] for row in function.entries])
    numerical_callable = sp.lambdify((*variable_symbols, *fixed_symbols), symbolic, modules="numpy")
    with np.errstate(all="ignore"):
        numeric = np.asarray(numerical_callable(*point, *fixed_values), dtype=complex)
    if numeric.shape != function.shape or not np.all(np.isfinite(numeric)):
        raise VectorAnalysisError("Matrix function produced a non-finite or incorrectly shaped value.")
    singular_values = tuple(float(item) for item in np.linalg.svd(numeric, compute_uv=False))
    rank = int(np.linalg.matrix_rank(numeric))
    condition = float(np.linalg.cond(numeric))
    determinant: complex | None = None
    symbolic_determinant: sp.Expr | None = None
    eigenvalues: tuple[complex, ...] = ()
    if numeric.shape[0] == numeric.shape[1]:
        determinant = complex(np.linalg.det(numeric))
        symbolic_determinant = sp.simplify(symbolic.det())
        eigenvalues = tuple(complex(item) for item in np.linalg.eigvals(numeric))
    real_if_close = np.real_if_close(numeric)
    return MatrixAnalysisResult(
        function_name=function.name,
        input_names=tuple(item.name for item in model.variables),
        point=tuple(float(item) for item in point),
        symbolic_matrix=symbolic,
        numerical_matrix=tuple(
            tuple(float(item) if np.isrealobj(real_if_close) else complex(item) for item in row)
            for row in real_if_close
        ),  # type: ignore[arg-type]
        determinant=determinant,
        symbolic_determinant=symbolic_determinant,
        rank=rank,
        singular_values=singular_values,
        condition_number=condition,
        eigenvalues=eigenvalues,
        provenance=Provenance.derived(
            (model_ref("matrix_function", function.name),),
            f"matrix evaluation and spectral analysis of {function.name}",
        ),
    )


def optimize_scalar_with_constraints(
    model: ModelIR,
    parameter_values: dict[str, float] | None = None,
    *,
    function_name: str | None = None,
    objective: Literal["minimize", "maximize"] = "minimize",
    seeds: int = 32,
    strict_margin: float = 1e-10,
    feasibility_tolerance: float = 1e-8,
) -> ConstrainedOptimizationResult:
    """Optimise a scalar function subject to variable bounds and declared constraints."""
    if not model.functions:
        raise VectorAnalysisError("Constrained optimisation requires a scalar function.")
    if function_name is None:
        if len(model.functions) != 1:
            raise VectorAnalysisError("Select one scalar objective function.")
        function = model.functions[0]
    else:
        try:
            function = model.function(function_name)
        except KeyError as exc:
            raise VectorAnalysisError(f"Unknown scalar function '{function_name}'.") from exc
    if objective not in {"minimize", "maximize"}:
        raise VectorAnalysisError("objective must be 'minimize' or 'maximize'.")
    variable_symbols, fixed_symbols, fixed_values = _symbols_and_fixed(model, parameter_values)
    arguments = (*variable_symbols, *fixed_symbols)
    objective_callable = sp.lambdify(arguments, function.expression, modules="numpy")
    sign = 1.0 if objective == "minimize" else -1.0

    def objective_fn(point: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            value = float(objective_callable(*point, *fixed_values))
        return sign * value if math.isfinite(value) else 1e100

    scipy_constraints: list[dict[str, object]] = []
    feasibility_functions: list[tuple[str, Callable[[np.ndarray], float]]] = []
    for constraint in model.constraints:
        difference = constraint.left - constraint.right
        callable_difference = sp.lambdify(arguments, difference, modules="numpy")

        def raw(point: np.ndarray, fn: Callable[..., object] = callable_difference) -> float:
            with np.errstate(all="ignore"):
                return float(fn(*point, *fixed_values))

        if constraint.relation == ConstraintRelation.EQUAL:
            scipy_constraints.append({"type": "eq", "fun": raw})
            feasibility_functions.append(("eq", raw))
        elif constraint.relation in {ConstraintRelation.LESS_EQUAL, ConstraintRelation.LESS_THAN}:
            margin = strict_margin if constraint.relation == ConstraintRelation.LESS_THAN else 0.0

            def le(point: np.ndarray, fn: Callable[[np.ndarray], float] = raw, m: float = margin) -> float:
                return -fn(point) - m

            scipy_constraints.append({"type": "ineq", "fun": le})
            feasibility_functions.append(("ineq", le))
        else:
            margin = strict_margin if constraint.relation == ConstraintRelation.GREATER_THAN else 0.0

            def ge(point: np.ndarray, fn: Callable[[np.ndarray], float] = raw, m: float = margin) -> float:
                return fn(point) - m

            scipy_constraints.append({"type": "ineq", "fun": ge})
            feasibility_functions.append(("ineq", ge))

    def violation(point: np.ndarray) -> float:
        values: list[float] = []
        for kind, function_value in feasibility_functions:
            try:
                value = function_value(point)
            except Exception:
                return math.inf
            values.append(abs(value) if kind == "eq" else max(0.0, -value))
        return max(values, default=0.0)

    bounds = [(item.domain.lower, item.domain.upper) for item in model.variables]
    solutions: list[ConstrainedOptimum] = []
    converged = 0
    candidates = _deterministic_seeds(model, seeds)
    for seed in candidates:
        try:
            result = minimize(
                objective_fn,
                seed,
                method="SLSQP",
                bounds=bounds,
                constraints=scipy_constraints,
                options={"maxiter": 2000, "ftol": 1e-12, "disp": False},
            )
        except Exception:
            continue
        point = np.asarray(result.x, dtype=float)
        maximum_violation = violation(point)
        value = sign * objective_fn(point)
        if not result.success or not math.isfinite(value) or maximum_violation > feasibility_tolerance:
            continue
        converged += 1
        solutions.append(
            ConstrainedOptimum(
                coordinates=tuple(float(item) for item in point),
                value=float(value),
                maximum_constraint_violation=float(maximum_violation),
            )
        )
    solutions.sort(key=lambda item: (item.value if objective == "minimize" else -item.value, item.coordinates))
    optimum = solutions[0] if solutions else None
    diagnostics: list[NumericalDiagnostic] = []
    status = NumericalStatus.COMPLETE
    if optimum is None:
        status = NumericalStatus.PARTIAL
        diagnostics.append(
            NumericalDiagnostic.warning(
                "no_feasible_optimum_located",
                "No feasible constrained optimum was located within the deterministic seed budget.",
                attempted_seeds=len(candidates),
            )
        )
    return ConstrainedOptimizationResult(
        function_name=function.name,
        objective=objective,
        input_names=tuple(item.name for item in model.variables),
        optimum=optimum,
        attempted_seeds=len(candidates),
        converged_seeds=converged,
        numerical_status=status,
        diagnostics=tuple(diagnostics),
        provenance=Provenance.derived(
            (
                model_ref("function", function.name),
                *(model_ref("constraint", item.name) for item in model.constraints),
                *(model_ref("variable", item.name) for item in model.variables),
            ),
            f"deterministic bounded {objective} optimisation with declared constraints",
        ),
    )
