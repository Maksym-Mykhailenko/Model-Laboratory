"""Numerically robust stationary-point analysis for validated scalar-function models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Callable

import numpy as np
import sympy as sp
from scipy.optimize import brentq, least_squares, minimize_scalar, root

from .issues import (
    NumericalDiagnostic,
    NumericalStatistics,
    NumericalStatus,
)
from .model import ModelIR
from .symbolic import analyse_one_variable_function, analyse_two_variable_function


class CriticalPointAnalysisError(ValueError):
    """Raised when stationary-point analysis cannot be configured safely."""


@dataclass(frozen=True, slots=True)
class OneDimensionalCriticalPoint:
    """One stationary point of a one-variable scalar function."""

    x: float
    value: float
    second_derivative: float
    classification: str


@dataclass(frozen=True, slots=True)
class TwoDimensionalCriticalPoint:
    """One stationary point of a two-variable scalar function."""

    x: float
    y: float
    value: float
    eigenvalues: tuple[float, float]
    classification: str


CriticalPoint = OneDimensionalCriticalPoint | TwoDimensionalCriticalPoint


@dataclass(frozen=True, slots=True)
class CriticalPointSearchResult:
    """Stationary-point search output with explicit numerical quality information."""

    points: tuple[CriticalPoint, ...]
    status: NumericalStatus
    diagnostics: tuple[NumericalDiagnostic, ...] = ()
    statistics: NumericalStatistics = NumericalStatistics()


def _parameter_substitutions(
    model: ModelIR, parameter_values: dict[str, float] | None
) -> dict[sp.Symbol, float]:
    values = model.parameter_defaults()
    if parameter_values:
        unknown = set(parameter_values) - {parameter.name for parameter in model.parameters}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise CriticalPointAnalysisError(f"Unknown parameter value(s): {names}.")
        values.update(parameter_values)

    substitutions: dict[sp.Symbol, float] = {}
    for parameter in model.parameters:
        value = float(values[parameter.name])
        if not math.isfinite(value):
            raise CriticalPointAnalysisError(
                f"Parameter '{parameter.name}' must have a finite numerical value."
            )
        if not parameter.domain.contains(value):
            raise CriticalPointAnalysisError(
                f"Parameter '{parameter.name}' = {value} lies outside its domain "
                f"[{parameter.domain.lower}, {parameter.domain.upper}]."
            )
        substitutions[sp.Symbol(parameter.name, real=True)] = value

    for constant in model.constants:
        substitutions[sp.Symbol(constant.name, real=True)] = float(constant.value)

    return substitutions


def _classification_tolerance(values: np.ndarray | list[float] | tuple[float, ...]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    scale = float(np.max(np.abs(finite))) if finite.size else 1.0
    return max(1e-9, 1e-7 * max(1.0, scale))


def _classify_one_dimensional(second_derivative: float) -> str:
    tolerance = _classification_tolerance([second_derivative])
    if second_derivative > tolerance:
        return "local minimum"
    if second_derivative < -tolerance:
        return "local maximum"
    return "degenerate / inconclusive"


def _classify_two_dimensional(eigenvalues: np.ndarray) -> str:
    tolerance = _classification_tolerance(eigenvalues)
    if np.all(eigenvalues > tolerance):
        return "local minimum"
    if np.all(eigenvalues < -tolerance):
        return "local maximum"
    if np.any(eigenvalues > tolerance) and np.any(eigenvalues < -tolerance):
        return "saddle"
    return "degenerate / inconclusive"


def _deduplicate_scalars(
    values: list[float],
    tolerance: float,
    *,
    residual: Callable[[float], float] | None = None,
) -> list[float]:
    """Cluster nearby scalar candidates and keep the best-residual representative."""
    if not values:
        return []
    clusters: list[list[float]] = []
    for value in sorted(values):
        if not clusters or abs(value - clusters[-1][-1]) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)

    unique: list[float] = []
    for cluster in clusters:
        if residual is None or len(cluster) == 1:
            unique.append(cluster[0])
            continue
        scored: list[tuple[float, float]] = []
        for value in cluster:
            try:
                score = float(residual(value))
            except Exception:
                score = math.inf
            scored.append((score if math.isfinite(score) else math.inf, value))
        unique.append(min(scored)[1])
    return unique


def _coerce_real_array(raw: object, shape: tuple[int, ...], *, label: str) -> np.ndarray:
    array = np.asarray(raw)
    if np.iscomplexobj(array):
        imaginary = np.abs(np.imag(array))
        finite_imaginary = imaginary[np.isfinite(imaginary)]
        max_imaginary = float(np.max(finite_imaginary)) if finite_imaginary.size else math.inf
        if max_imaginary > 1e-10:
            raise CriticalPointAnalysisError(
                f"{label} produced non-real numerical values inside the declared domain."
            )
        array = np.real(array)

    try:
        array = np.asarray(array, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CriticalPointAnalysisError(f"{label} could not be converted to real numbers.") from exc

    if array.ndim == 0:
        return np.full(shape, float(array), dtype=float)
    try:
        return np.broadcast_to(array, shape).astype(float, copy=True)
    except ValueError as exc:
        raise CriticalPointAnalysisError(
            f"{label} did not produce the expected numerical shape."
        ) from exc


def _real_scalar_value(function: Callable[..., object], *arguments: float) -> float | None:
    """Evaluate a scalar numerical function without discarding a meaningful imaginary part."""
    try:
        with np.errstate(all="ignore"):
            raw = function(*arguments)
        array = np.asarray(raw)
        if array.size != 1:
            return None
        if np.iscomplexobj(array):
            value = complex(array.reshape(-1)[0])
            if abs(value.imag) > 1e-10:
                return None
            numeric = float(value.real)
        else:
            numeric = float(array.reshape(-1)[0])
        return numeric if math.isfinite(numeric) else None
    except Exception:
        return None


def _is_numerically_differentiable_1d(
    function: Callable[..., object],
    x_value: float,
    lower: float,
    upper: float,
    derivative_value: float,
) -> bool | None:
    """Check two-sided local slopes at an interior candidate.

    ``None`` means that the check could not be performed reliably. Boundary candidates are
    left to the existing derivative test because a two-sided check is unavailable there.
    """
    width = max(upper - lower, 1.0)
    distance_to_boundary = min(x_value - lower, upper - x_value)
    if distance_to_boundary <= width * 1e-7:
        return None

    f0 = _real_scalar_value(function, x_value)
    if f0 is None:
        return None

    attempted = 0
    failures = 0
    for factor in (1e-4, 3e-5, 1e-5):
        h = min(width * factor, distance_to_boundary * 0.25)
        if h <= 1e-10 * width:
            continue
        fm = _real_scalar_value(function, x_value - h)
        fp = _real_scalar_value(function, x_value + h)
        if fm is None or fp is None:
            continue
        attempted += 1
        left_slope = (f0 - fm) / h
        right_slope = (fp - f0) / h
        scale = max(1.0, abs(left_slope), abs(right_slope), abs(derivative_value))
        if abs(left_slope - right_slope) > max(1e-5, 1e-3 * scale):
            failures += 1

    if attempted == 0:
        return None
    # A smooth curved function can show a small finite-difference mismatch at the
    # coarsest probe. Requiring agreement at the finer probes avoids rejecting ordinary
    # curvature while still rejecting cusps such as |x|, which fail at every scale.
    return failures < attempted


def _is_numerically_differentiable_2d(
    function: Callable[..., object],
    x_value: float,
    y_value: float,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> bool | None:
    """Check coordinate-wise two-sided slopes at an interior two-dimensional candidate."""
    f0 = _real_scalar_value(function, x_value, y_value)
    if f0 is None:
        return None

    checks = 0
    failures = 0
    for axis, value, bounds in (
        (0, x_value, x_bounds),
        (1, y_value, y_bounds),
    ):
        lower, upper = bounds
        width = max(upper - lower, 1.0)
        distance = min(value - lower, upper - value)
        if distance <= width * 1e-7:
            continue
        h = min(width * 1e-5, distance * 0.25)
        if h <= 1e-10 * width:
            continue
        if axis == 0:
            fm = _real_scalar_value(function, x_value - h, y_value)
            fp = _real_scalar_value(function, x_value + h, y_value)
        else:
            fm = _real_scalar_value(function, x_value, y_value - h)
            fp = _real_scalar_value(function, x_value, y_value + h)
        if fm is None or fp is None:
            continue
        checks += 1
        left_slope = (f0 - fm) / h
        right_slope = (fp - f0) / h
        scale = max(1.0, abs(left_slope), abs(right_slope))
        if abs(left_slope - right_slope) > max(1e-5, 1e-3 * scale):
            failures += 1

    if checks == 0:
        return None
    return failures == 0


def search_one_variable_stationary_points(
    model: ModelIR,
    parameter_values: dict[str, float] | None = None,
    samples: int = 2001,
    root_tolerance: float = 1e-9,
) -> CriticalPointSearchResult:
    """Find and classify stationary points of a one-variable scalar function.

    Sign-change bracketing is complemented by bounded minimisation of ``|f'|`` so roots
    with even multiplicity are not dependent on landing exactly on a sampling point.
    """
    if len(model.variables) != 1 or len(model.functions) != 1:
        raise CriticalPointAnalysisError(
            "Stationary-point analysis requires exactly one continuous variable and one scalar function."
        )
    if not isinstance(samples, int) or samples < 10 or samples > 200_001:
        raise CriticalPointAnalysisError("Samples must be an integer between 10 and 200001.")
    if not math.isfinite(root_tolerance) or root_tolerance <= 0:
        raise CriticalPointAnalysisError("Root tolerance must be a positive finite number.")

    variable = model.variables[0]
    function = model.functions[0]
    symbol = sp.Symbol(variable.name, real=True)
    substitutions = _parameter_substitutions(model, parameter_values)
    symbolic = analyse_one_variable_function(model)

    derivative_expr = sp.simplify(symbolic.first_derivative.subs(substitutions))
    second_expr = sp.simplify(symbolic.second_derivative.subs(substitutions))
    function_expr = sp.simplify(function.expression.subs(substitutions))
    requires_differentiability_check = bool(
        function_expr.has(sp.Abs, sp.sign, sp.DiracDelta, sp.Heaviside, sp.Piecewise)
    )

    if derivative_expr.is_zero is True:
        diagnostic = NumericalDiagnostic.warning(
            "stationary_continuum",
            "The first derivative is identically zero at the current parameter values; every point in the domain is stationary.",
        )
        return CriticalPointSearchResult(
            points=(),
            status=NumericalStatus.INDETERMINATE,
            diagnostics=(diagnostic,),
            statistics=NumericalStatistics((("samples", samples),)),
        )

    derivative_fn = sp.lambdify(symbol, derivative_expr, modules="numpy")
    second_fn = sp.lambdify(symbol, second_expr, modules="numpy")
    function_fn = sp.lambdify(symbol, function_expr, modules="numpy")

    grid = np.linspace(variable.domain.lower, variable.domain.upper, samples, dtype=float)
    try:
        with np.errstate(all="ignore"):
            derivative_values = _coerce_real_array(
                derivative_fn(grid), grid.shape, label="Derivative evaluation"
            )
    except CriticalPointAnalysisError as exc:
        return CriticalPointSearchResult(
            points=(),
            status=NumericalStatus.FAILED,
            diagnostics=(NumericalDiagnostic.error("derivative_evaluation_failed", str(exc)),),
            statistics=NumericalStatistics((("samples", samples),)),
        )
    except Exception as exc:
        return CriticalPointSearchResult(
            points=(),
            status=NumericalStatus.FAILED,
            diagnostics=(
                NumericalDiagnostic.error(
                    "derivative_evaluation_failed", f"Derivative evaluation failed: {exc}"
                ),
            ),
            statistics=NumericalStatistics((("samples", samples),)),
        )

    finite = np.isfinite(derivative_values)
    finite_count = int(np.count_nonzero(finite))
    diagnostics: list[NumericalDiagnostic] = []
    status = NumericalStatus.COMPLETE

    if finite_count == 0:
        return CriticalPointSearchResult(
            points=(),
            status=NumericalStatus.FAILED,
            diagnostics=(
                NumericalDiagnostic.error(
                    "no_finite_derivative_samples",
                    "The derivative has no finite numerical values on the sampled domain.",
                ),
            ),
            statistics=NumericalStatistics(
                (("samples", samples), ("finite_derivative_samples", finite_count))
            ),
        )

    nonfinite_count = samples - finite_count
    if nonfinite_count:
        status = NumericalStatus.PARTIAL
        diagnostics.append(
            NumericalDiagnostic.warning(
                "non_finite_derivative_samples",
                f"The derivative was non-finite at {nonfinite_count} of {samples} sampled points.",
                count=nonfinite_count,
                total=samples,
            )
        )

    residual_tolerance = max(root_tolerance * 100.0, 1e-8)
    near_zero_threshold = max(root_tolerance * 1000.0, 1e-7)
    candidates: list[float] = []
    bracket_attempts = 0
    bracket_failures = 0
    tangent_attempts = 0
    tangent_acceptances = 0

    near_indices = np.where(finite & (np.abs(derivative_values) <= near_zero_threshold))[0]

    # A flat root can put several neighbouring grid samples below the near-zero
    # threshold.  Treat each contiguous run as one numerical candidate region and
    # refine that region before adding a candidate.  Adding every near-zero grid
    # sample separately creates false duplicate stationary points around high-order
    # roots.
    near_zero_runs: list[tuple[int, int]] = []
    if near_indices.size:
        run_start = int(near_indices[0])
        run_end = run_start
        for raw_index in near_indices[1:]:
            index = int(raw_index)
            if index == run_end + 1:
                run_end = index
            else:
                near_zero_runs.append((run_start, run_end))
                run_start = run_end = index
        near_zero_runs.append((run_start, run_end))

    near_zero_refinement_attempts = 0
    near_zero_refinement_acceptances = 0

    def derivative_residual(value: float) -> float:
        try:
            with np.errstate(all="ignore"):
                raw = derivative_fn(value)
            array = np.asarray(raw)
            if array.size != 1:
                return math.inf
            if np.iscomplexobj(array):
                scalar = complex(array.reshape(-1)[0])
                if abs(scalar.imag) > 1e-10:
                    return math.inf
                numeric = float(scalar.real)
            else:
                numeric = float(array.reshape(-1)[0])
            return abs(numeric) if math.isfinite(numeric) else math.inf
        except Exception:
            return math.inf

    for run_start, run_end in near_zero_runs:
        best_index = min(
            range(run_start, run_end + 1),
            key=lambda index: abs(float(derivative_values[index])),
        )
        best_sample = float(grid[best_index])
        left_index = max(0, run_start - 1)
        right_index = min(len(grid) - 1, run_end + 1)
        left = float(grid[left_index])
        right = float(grid[right_index])

        refined: float | None = None
        if right > left:
            near_zero_refinement_attempts += 1
            try:
                minimised = minimize_scalar(
                    derivative_residual,
                    bounds=(left, right),
                    method="bounded",
                    options={"xatol": root_tolerance, "maxiter": 200},
                )
            except Exception:
                minimised = None
            if (
                minimised is not None
                and minimised.success
                and math.isfinite(float(minimised.fun))
                and float(minimised.fun) <= residual_tolerance
            ):
                refined = float(minimised.x)
                near_zero_refinement_acceptances += 1

        if refined is not None:
            candidates.append(refined)
        elif derivative_residual(best_sample) <= residual_tolerance:
            # Boundary runs and unusually shaped residuals may not refine cleanly.
            # Retain only the best sample from the run, never the whole run.
            candidates.append(best_sample)

    # Standard sign-change roots.
    for index in range(len(grid) - 1):
        if not (finite[index] and finite[index + 1]):
            continue
        left_value = derivative_values[index]
        right_value = derivative_values[index + 1]
        if left_value == 0 or right_value == 0 or np.signbit(left_value) == np.signbit(right_value):
            continue
        bracket_attempts += 1
        try:
            with np.errstate(all="ignore"):
                candidate = brentq(
                    lambda value: float(derivative_fn(value)),
                    float(grid[index]),
                    float(grid[index + 1]),
                    xtol=root_tolerance,
                    rtol=max(4 * np.finfo(float).eps, root_tolerance),
                    maxiter=200,
                )
            candidates.append(float(candidate))
        except (ValueError, RuntimeError, OverflowError, FloatingPointError):
            bracket_failures += 1

    # Roots where the derivative touches zero without changing sign.  We minimise the
    # absolute derivative around local sampled minima and accept only a verified residual.
    abs_derivative = np.abs(derivative_values)
    local_minima = []
    for index in range(1, len(grid) - 1):
        if not (finite[index - 1] and finite[index] and finite[index + 1]):
            continue
        if (
            abs_derivative[index] <= abs_derivative[index - 1]
            and abs_derivative[index] <= abs_derivative[index + 1]
            and (
                abs_derivative[index] < abs_derivative[index - 1]
                or abs_derivative[index] < abs_derivative[index + 1]
            )
        ):
            local_minima.append(index)

    # Bound work on highly oscillatory inputs while recording that the refinement was partial.
    maximum_tangent_refinements = 1000
    if len(local_minima) > maximum_tangent_refinements:
        status = NumericalStatus.PARTIAL
        diagnostics.append(
            NumericalDiagnostic.warning(
                "tangent_refinement_limit",
                f"Tangent-root refinement was limited to {maximum_tangent_refinements} candidate intervals.",
                candidates=len(local_minima),
            )
        )
        local_minima = local_minima[:maximum_tangent_refinements]

    for index in local_minima:
        tangent_attempts += 1
        left = float(grid[index - 1])
        right = float(grid[index + 1])

        def objective(value: float) -> float:
            try:
                with np.errstate(all="ignore"):
                    raw = derivative_fn(value)
                scalar = np.asarray(raw)
                if np.iscomplexobj(scalar):
                    if abs(complex(scalar).imag) > 1e-10:
                        return math.inf
                    scalar = np.real(scalar)
                numeric = float(scalar)
                return abs(numeric) if math.isfinite(numeric) else math.inf
            except Exception:
                return math.inf

        try:
            minimised = minimize_scalar(
                objective,
                bounds=(left, right),
                method="bounded",
                options={"xatol": root_tolerance, "maxiter": 200},
            )
        except Exception:
            continue
        if minimised.success and math.isfinite(float(minimised.fun)) and float(minimised.fun) <= residual_tolerance:
            candidates.append(float(minimised.x))
            tangent_acceptances += 1

    grid_step = variable.domain.width / max(samples - 1, 1)
    dedup_tolerance = max(
        grid_step * 1e-3,
        variable.domain.width * 1e-8,
        root_tolerance * 20.0,
        1e-9,
    )

    roots = _deduplicate_scalars(
        candidates, dedup_tolerance, residual=derivative_residual
    )

    points: list[OneDimensionalCriticalPoint] = []
    rejected_candidates = 0
    classification_warnings = 0
    nonsmooth_candidates = 0
    for root_value in roots:
        if not (
            variable.domain.lower - dedup_tolerance
            <= root_value
            <= variable.domain.upper + dedup_tolerance
        ):
            rejected_candidates += 1
            continue
        try:
            with np.errstate(all="ignore"):
                derivative_raw = derivative_fn(root_value)
                function_raw = function_fn(root_value)
            derivative_at_root = float(np.real_if_close(derivative_raw))
            function_value = float(np.real_if_close(function_raw))
        except Exception:
            rejected_candidates += 1
            continue

        if not math.isfinite(derivative_at_root) or abs(derivative_at_root) > residual_tolerance:
            rejected_candidates += 1
            continue
        if not math.isfinite(function_value):
            rejected_candidates += 1
            continue

        if requires_differentiability_check:
            differentiable = _is_numerically_differentiable_1d(
                function_fn,
                root_value,
                variable.domain.lower,
                variable.domain.upper,
                derivative_at_root,
            )
            if differentiable is False:
                nonsmooth_candidates += 1
                rejected_candidates += 1
                continue

        try:
            with np.errstate(all="ignore"):
                second_raw = second_fn(root_value)
            second_value = float(np.real_if_close(second_raw))
        except Exception:
            second_value = math.nan

        if math.isfinite(second_value):
            classification = _classify_one_dimensional(second_value)
        else:
            classification = "degenerate / inconclusive"
            classification_warnings += 1

        points.append(
            OneDimensionalCriticalPoint(
                x=float(root_value),
                value=function_value,
                second_derivative=second_value,
                classification=classification,
            )
        )

    if classification_warnings:
        status = NumericalStatus.PARTIAL
        diagnostics.append(
            NumericalDiagnostic.warning(
                "non_finite_second_derivative",
                f"The second derivative could not be evaluated finitely at {classification_warnings} stationary point(s).",
                count=classification_warnings,
            )
        )
    if bracket_failures and bracket_failures >= max(3, bracket_attempts // 4):
        status = NumericalStatus.PARTIAL
        diagnostics.append(
            NumericalDiagnostic.warning(
                "root_bracketing_failures",
                f"{bracket_failures} bracketed root searches did not converge.",
                failures=bracket_failures,
                attempts=bracket_attempts,
            )
        )
    if nonsmooth_candidates:
        status = NumericalStatus.PARTIAL
        diagnostics.append(
            NumericalDiagnostic.warning(
                "non_differentiable_candidates",
                f"{nonsmooth_candidates} symbolic derivative zero(s) were excluded because the function was not numerically differentiable there.",
                count=nonsmooth_candidates,
            )
        )

    return CriticalPointSearchResult(
        points=tuple(points),
        status=status,
        diagnostics=tuple(diagnostics),
        statistics=NumericalStatistics(
            (
                ("samples", samples),
                ("finite_derivative_samples", finite_count),
                ("bracket_attempts", bracket_attempts),
                ("bracket_failures", bracket_failures),
                ("near_zero_regions", len(near_zero_runs)),
                ("near_zero_refinement_attempts", near_zero_refinement_attempts),
                ("near_zero_refinement_acceptances", near_zero_refinement_acceptances),
                ("tangent_refinement_attempts", tangent_attempts),
                ("tangent_refinement_acceptances", tangent_acceptances),
                ("candidate_roots", len(roots)),
                ("rejected_candidates", rejected_candidates),
                ("stationary_points", len(points)),
            )
        ),
    )


def search_two_variable_stationary_points(
    model: ModelIR,
    parameter_values: dict[str, float] | None = None,
    seeds_per_axis: int = 11,
    root_tolerance: float = 1e-9,
) -> CriticalPointSearchResult:
    """Find and classify stationary points in a two-variable scalar function.

    A bounded nonlinear least-squares solver is seeded on a deterministic grid.  Every
    candidate is independently verified by its gradient residual before it is accepted.
    """
    if len(model.variables) != 2 or len(model.functions) != 1:
        raise CriticalPointAnalysisError(
            "Stationary-point analysis requires exactly two continuous variables and one scalar function."
        )
    if not isinstance(seeds_per_axis, int) or seeds_per_axis < 3 or seeds_per_axis > 51:
        raise CriticalPointAnalysisError(
            "Root-search seeds per axis must be an integer between 3 and 51."
        )
    if not math.isfinite(root_tolerance) or root_tolerance <= 0:
        raise CriticalPointAnalysisError("Root tolerance must be a positive finite number.")

    x_variable, y_variable = model.variables
    function = model.functions[0]
    x_symbol = sp.Symbol(x_variable.name, real=True)
    y_symbol = sp.Symbol(y_variable.name, real=True)
    substitutions = _parameter_substitutions(model, parameter_values)
    symbolic = analyse_two_variable_function(model)

    gradient_expr = tuple(
        sp.simplify(component.subs(substitutions)) for component in symbolic.gradient
    )
    hessian_expr = symbolic.hessian.subs(substitutions).applyfunc(sp.simplify)
    function_expr = sp.simplify(function.expression.subs(substitutions))
    requires_differentiability_check = bool(
        function_expr.has(sp.Abs, sp.sign, sp.DiracDelta, sp.Heaviside, sp.Piecewise)
    )

    zero_gradient_components = tuple(
        index for index, component in enumerate(gradient_expr) if component.is_zero is True
    )
    if len(zero_gradient_components) == 2:
        diagnostic = NumericalDiagnostic.warning(
            "stationary_continuum",
            "The gradient is identically zero at the current parameter values; every point in the domain is stationary.",
        )
        return CriticalPointSearchResult(
            points=(),
            status=NumericalStatus.INDETERMINATE,
            diagnostics=(diagnostic,),
            statistics=NumericalStatistics((("seeds", seeds_per_axis**2),)),
        )

    if len(zero_gradient_components) == 1:
        remaining_component = gradient_expr[1 - zero_gradient_components[0]]
        # If SymPy can prove that the remaining component is never zero, then the
        # gradient cannot vanish anywhere and there are no stationary points.
        if remaining_component.is_nonzero is True:
            return CriticalPointSearchResult(
                points=(),
                status=NumericalStatus.COMPLETE,
                diagnostics=(),
                statistics=NumericalStatistics((("seeds", seeds_per_axis**2),)),
            )

        # Otherwise any root of the remaining component extends along the direction
        # whose gradient component vanishes identically.  The current analyser returns
        # isolated stationary points, so representing a finite seed-dependent sample
        # would be misleading.
        diagnostic = NumericalDiagnostic.warning(
            "non_isolated_stationary_structure",
            "A gradient component is identically zero, so the stationary set may be non-isolated. The current stationary-point analyser cannot represent that set as a finite collection of points.",
            zero_component=zero_gradient_components[0],
        )
        return CriticalPointSearchResult(
            points=(),
            status=NumericalStatus.INDETERMINATE,
            diagnostics=(diagnostic,),
            statistics=NumericalStatistics((("seeds", seeds_per_axis**2),)),
        )

    # For polynomial gradients, inspect the algebraic dimension before launching a
    # finite-point root search.  A positive-dimensional gradient-zero set represents
    # a curve or region of stationary points, not a finite collection.
    try:
        if all(component.is_polynomial(x_symbol, y_symbol) for component in gradient_expr):
            basis = sp.groebner(list(gradient_expr), x_symbol, y_symbol)
            basis_expressions = tuple(sp.expand(item) for item in basis)
            if len(basis_expressions) == 1 and basis_expressions[0] == 1:
                return CriticalPointSearchResult(
                    points=(),
                    status=NumericalStatus.COMPLETE,
                    diagnostics=(),
                    statistics=NumericalStatistics((("seeds", seeds_per_axis**2),)),
                )
            if not basis.is_zero_dimensional:
                diagnostic = NumericalDiagnostic.warning(
                    "non_isolated_stationary_structure",
                    "The gradient-zero set is non-isolated; the stationary set cannot be represented as a finite collection of points.",
                )
                return CriticalPointSearchResult(
                    points=(),
                    status=NumericalStatus.INDETERMINATE,
                    diagnostics=(diagnostic,),
                    statistics=NumericalStatistics((("seeds", seeds_per_axis**2),)),
                )
    except (sp.PolynomialError, ValueError, TypeError):
        # Non-polynomial systems proceed to the numerical isolated-root search below.
        pass

    gradient_fn = sp.lambdify((x_symbol, y_symbol), gradient_expr, modules="numpy")
    hessian_fn = sp.lambdify((x_symbol, y_symbol), hessian_expr, modules="numpy")
    function_fn = sp.lambdify((x_symbol, y_symbol), function_expr, modules="numpy")

    def numerical_gradient(values: np.ndarray) -> np.ndarray:
        try:
            with np.errstate(all="ignore"):
                raw = gradient_fn(float(values[0]), float(values[1]))
        except Exception:
            return np.array([np.nan, np.nan], dtype=float)
        array = np.asarray(raw)
        if np.iscomplexobj(array):
            if np.any(np.abs(np.imag(array)) > 1e-10):
                return np.array([np.nan, np.nan], dtype=float)
            array = np.real(array)
        try:
            result = np.asarray(array, dtype=float).reshape(2)
        except (TypeError, ValueError, OverflowError):
            return np.array([np.nan, np.nan], dtype=float)
        return result

    x_seeds = np.linspace(x_variable.domain.lower, x_variable.domain.upper, seeds_per_axis)
    y_seeds = np.linspace(y_variable.domain.lower, y_variable.domain.upper, seeds_per_axis)
    lower_bounds = np.array([x_variable.domain.lower, y_variable.domain.lower], dtype=float)
    upper_bounds = np.array([x_variable.domain.upper, y_variable.domain.upper], dtype=float)
    residual_tolerance = max(root_tolerance * 100.0, 1e-7)

    candidates: list[tuple[float, float]] = []
    total_seeds = seeds_per_axis**2
    finite_seed_count = 0
    solver_failures = 0
    residual_rejections = 0

    for x_seed in x_seeds:
        for y_seed in y_seeds:
            initial = np.array([x_seed, y_seed], dtype=float)
            initial_gradient = numerical_gradient(initial)
            if not np.all(np.isfinite(initial_gradient)):
                continue
            finite_seed_count += 1
            try:
                solution = least_squares(
                    numerical_gradient,
                    initial,
                    bounds=(lower_bounds, upper_bounds),
                    xtol=root_tolerance,
                    ftol=root_tolerance,
                    gtol=root_tolerance,
                    max_nfev=1000,
                )
            except Exception:
                solver_failures += 1
                continue
            if not solution.success or solution.x.shape != (2,):
                solver_failures += 1
                continue

            candidate_vector = np.asarray(solution.x, dtype=float)
            # Polish a bounded least-squares candidate with an unconstrained root solve.
            # The polished point is accepted only if it remains inside the declared domain
            # and improves or preserves the verified residual.
            try:
                polished = root(
                    numerical_gradient,
                    candidate_vector,
                    method="hybr",
                    options={"xtol": root_tolerance, "maxfev": 300},
                )
                if polished.success and polished.x.shape == (2,):
                    polished_vector = np.asarray(polished.x, dtype=float)
                    if np.all(polished_vector >= lower_bounds - 1e-10) and np.all(
                        polished_vector <= upper_bounds + 1e-10
                    ):
                        initial_residual = np.linalg.norm(numerical_gradient(candidate_vector))
                        polished_residual = np.linalg.norm(numerical_gradient(polished_vector))
                        if np.isfinite(polished_residual) and polished_residual <= initial_residual:
                            candidate_vector = polished_vector
            except Exception:
                pass

            x_value, y_value = float(candidate_vector[0]), float(candidate_vector[1])
            gradient_value = numerical_gradient(np.array([x_value, y_value], dtype=float))
            if not np.all(np.isfinite(gradient_value)):
                residual_rejections += 1
                continue
            if float(np.linalg.norm(gradient_value, ord=2)) > residual_tolerance:
                residual_rejections += 1
                continue
            candidates.append((x_value, y_value))

    diagnostics: list[NumericalDiagnostic] = []
    status = NumericalStatus.COMPLETE

    if finite_seed_count == 0:
        return CriticalPointSearchResult(
            points=(),
            status=NumericalStatus.FAILED,
            diagnostics=(
                NumericalDiagnostic.error(
                    "no_finite_gradient_seeds",
                    "The gradient was non-finite at every root-search seed in the declared domain.",
                ),
            ),
            statistics=NumericalStatistics(
                (("seeds", total_seeds), ("finite_seeds", finite_seed_count))
            ),
        )

    nonfinite_seeds = total_seeds - finite_seed_count
    if nonfinite_seeds:
        diagnostics.append(
            NumericalDiagnostic.warning(
                "non_finite_gradient_seeds",
                f"The gradient was non-finite at {nonfinite_seeds} of {total_seeds} root-search seeds.",
                count=nonfinite_seeds,
                total=total_seeds,
            )
        )
        if nonfinite_seeds / total_seeds >= 0.05:
            status = NumericalStatus.PARTIAL

    if solver_failures:
        diagnostics.append(
            NumericalDiagnostic.warning(
                "solver_failures",
                f"The numerical root solver did not converge from {solver_failures} seed(s).",
                failures=solver_failures,
                finite_seeds=finite_seed_count,
            )
        )
        if solver_failures / finite_seed_count >= 0.10:
            status = NumericalStatus.PARTIAL

    x_scale = max(x_variable.domain.width, 1.0)
    y_scale = max(y_variable.domain.width, 1.0)
    dedup_tolerance = max(1e-7, root_tolerance * 100.0)
    unique: list[tuple[float, float]] = []
    for candidate in sorted(candidates):
        if all(
            np.hypot(
                (candidate[0] - previous[0]) / x_scale,
                (candidate[1] - previous[1]) / y_scale,
            )
            > dedup_tolerance
            for previous in unique
        ):
            unique.append(candidate)

    points: list[TwoDimensionalCriticalPoint] = []
    classification_warnings = 0
    rejected_candidates = 0
    nonsmooth_candidates = 0
    for x_value, y_value in unique:
        try:
            with np.errstate(all="ignore"):
                function_raw = function_fn(x_value, y_value)
            function_array = np.asarray(function_raw)
            if np.iscomplexobj(function_array):
                if abs(complex(function_array).imag) > 1e-10:
                    rejected_candidates += 1
                    continue
                function_array = np.real(function_array)
            function_value = float(function_array)
        except Exception:
            rejected_candidates += 1
            continue
        if not math.isfinite(function_value):
            rejected_candidates += 1
            continue

        if requires_differentiability_check:
            differentiable = _is_numerically_differentiable_2d(
                function_fn,
                x_value,
                y_value,
                (x_variable.domain.lower, x_variable.domain.upper),
                (y_variable.domain.lower, y_variable.domain.upper),
            )
            if differentiable is False:
                nonsmooth_candidates += 1
                rejected_candidates += 1
                continue

        try:
            with np.errstate(all="ignore"):
                hessian_raw = hessian_fn(x_value, y_value)
            hessian_array = np.asarray(hessian_raw)
            if np.iscomplexobj(hessian_array):
                if np.any(np.abs(np.imag(hessian_array)) > 1e-10):
                    raise ValueError
                hessian_array = np.real(hessian_array)
            hessian_value = np.asarray(hessian_array, dtype=float).reshape(2, 2)
        except Exception:
            hessian_value = np.full((2, 2), np.nan)

        if np.all(np.isfinite(hessian_value)):
            eigenvalues = np.linalg.eigvalsh(hessian_value)
            classification = _classify_two_dimensional(eigenvalues)
        else:
            eigenvalues = np.array([np.nan, np.nan], dtype=float)
            classification = "degenerate / inconclusive"
            classification_warnings += 1

        points.append(
            TwoDimensionalCriticalPoint(
                x=x_value,
                y=y_value,
                value=function_value,
                eigenvalues=(float(eigenvalues[0]), float(eigenvalues[1])),
                classification=classification,
            )
        )

    if classification_warnings:
        status = NumericalStatus.PARTIAL
        diagnostics.append(
            NumericalDiagnostic.warning(
                "non_finite_hessian",
                f"The Hessian could not be evaluated finitely at {classification_warnings} stationary point(s).",
                count=classification_warnings,
            )
        )
    if nonsmooth_candidates:
        status = NumericalStatus.PARTIAL
        diagnostics.append(
            NumericalDiagnostic.warning(
                "non_differentiable_candidates",
                f"{nonsmooth_candidates} gradient zero(s) were excluded because the function was not numerically differentiable there.",
                count=nonsmooth_candidates,
            )
        )

    return CriticalPointSearchResult(
        points=tuple(points),
        status=status,
        diagnostics=tuple(diagnostics),
        statistics=NumericalStatistics(
            (
                ("seeds", total_seeds),
                ("finite_seeds", finite_seed_count),
                ("solver_failures", solver_failures),
                ("residual_rejections", residual_rejections),
                ("candidate_roots", len(unique)),
                ("rejected_candidates", rejected_candidates),
                ("stationary_points", len(points)),
            )
        ),
    )


def find_one_variable_stationary_points(
    model: ModelIR,
    parameter_values: dict[str, float] | None = None,
    samples: int = 2001,
    root_tolerance: float = 1e-9,
) -> tuple[OneDimensionalCriticalPoint, ...]:
    """Compatibility wrapper returning only accepted one-dimensional points."""
    result = search_one_variable_stationary_points(
        model,
        parameter_values=parameter_values,
        samples=samples,
        root_tolerance=root_tolerance,
    )
    return tuple(point for point in result.points if isinstance(point, OneDimensionalCriticalPoint))


def find_two_variable_stationary_points(
    model: ModelIR,
    parameter_values: dict[str, float] | None = None,
    seeds_per_axis: int = 11,
    root_tolerance: float = 1e-9,
) -> tuple[TwoDimensionalCriticalPoint, ...]:
    """Compatibility wrapper returning only accepted two-dimensional points."""
    result = search_two_variable_stationary_points(
        model,
        parameter_values=parameter_values,
        seeds_per_axis=seeds_per_axis,
        root_tolerance=root_tolerance,
    )
    return tuple(point for point in result.points if isinstance(point, TwoDimensionalCriticalPoint))
