"""Numerical evaluation of validated Model IR objects."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import sympy as sp

from .issues import NumericalDiagnostic, NumericalStatus
from .model import ModelIR
from .provenance import Provenance, model_ref


class EvaluationError(ValueError):
    """Raised when a validated model cannot be evaluated with the requested settings."""


@dataclass(frozen=True, slots=True)
class FunctionEvaluation:
    x_name: str
    y_name: str
    x: np.ndarray
    y: np.ndarray
    provenance: Provenance
    numerical_status: NumericalStatus = NumericalStatus.COMPLETE
    diagnostics: tuple[NumericalDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class SurfaceEvaluation:
    x_name: str
    y_name: str
    z_name: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    provenance: Provenance
    numerical_status: NumericalStatus = NumericalStatus.COMPLETE
    diagnostics: tuple[NumericalDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class ScalarPointEvaluation:
    """Evaluation of a scalar function at explicit points in arbitrary dimension."""

    input_names: tuple[str, ...]
    output_name: str
    points: np.ndarray
    values: np.ndarray
    provenance: Provenance
    numerical_status: NumericalStatus = NumericalStatus.COMPLETE
    diagnostics: tuple[NumericalDiagnostic, ...] = ()


def _resolved_parameters(
    model: ModelIR, overrides: dict[str, float] | None
) -> dict[str, float]:
    values = model.parameter_defaults()

    if overrides:
        unknown = set(overrides) - {parameter.name for parameter in model.parameters}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise EvaluationError(f"Unknown parameter override(s): {names}.")
        try:
            values.update({name: float(value) for name, value in overrides.items()})
        except (TypeError, ValueError, OverflowError) as exc:
            raise EvaluationError("Parameter overrides must be finite real numbers.") from exc

    for name, value in values.items():
        parameter = model.parameter(name)
        if not math.isfinite(float(value)):
            raise EvaluationError(f"Parameter '{name}' must have a finite numerical value.")
        if not parameter.domain.contains(float(value)):
            raise EvaluationError(
                f"Parameter '{name}' = {value} lies outside its domain "
                f"[{parameter.domain.lower}, {parameter.domain.upper}]."
            )

    return values


def _numerical_function(model: ModelIR):
    function = model.functions[0]
    symbol_names = [
        *[variable.name for variable in model.variables],
        *[parameter.name for parameter in model.parameters],
        *[constant.name for constant in model.constants],
    ]
    symbols = [sp.Symbol(name, real=True) for name in symbol_names]
    return sp.lambdify(symbols, function.expression, modules="numpy")


def evaluate_scalar_at_points(
    model: ModelIR,
    points: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
    parameter_values: dict[str, float] | None = None,
    *,
    function_name: str | None = None,
) -> ScalarPointEvaluation:
    """Evaluate a selected scalar function at explicit R^n points.

    This is the dimension-independent numerical primitive. Grid and surface renderers
    remain intentionally restricted to dimensions where they have a clear visual meaning.
    """
    if model.blocking_ambiguities:
        names = ", ".join(item.name for item in model.blocking_ambiguities)
        raise EvaluationError(f"Resolve blocking ambiguity/ambiguities before evaluation: {names}.")
    if not model.variables or not model.functions:
        raise EvaluationError("Point evaluation requires variables and a scalar function.")
    if function_name is None:
        if len(model.functions) != 1:
            raise EvaluationError("Select one scalar function for point evaluation.")
        function = model.functions[0]
    else:
        try:
            function = model.function(function_name)
        except KeyError as exc:
            raise EvaluationError(f"Unknown scalar function '{function_name}'.") from exc
    try:
        point_array = np.asarray(points, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluationError("Evaluation points must be finite real numbers.") from exc
    if point_array.ndim == 1:
        point_array = point_array.reshape(1, -1)
    if point_array.ndim != 2 or point_array.shape[1] != len(model.variables):
        raise EvaluationError("Evaluation points must have one column per model variable.")
    if not 1 <= point_array.shape[0] <= 1_000_000 or not np.all(np.isfinite(point_array)):
        raise EvaluationError("Evaluation requires between 1 and 1000000 finite points.")
    for index, variable in enumerate(model.variables):
        values = point_array[:, index]
        if np.any(values < variable.domain.lower) or np.any(values > variable.domain.upper):
            raise EvaluationError(f"Evaluation point for '{variable.name}' lies outside its domain.")
    parameters = _resolved_parameters(model, parameter_values)
    names = [
        *[item.name for item in model.variables],
        *[item.name for item in model.parameters],
        *[item.name for item in model.constants],
    ]
    symbols = [sp.Symbol(name, real=True) for name in names]
    numerical = sp.lambdify(symbols, function.expression, modules="numpy")
    arguments = [
        *[point_array[:, index] for index in range(point_array.shape[1])],
        *[parameters[item.name] for item in model.parameters],
        *[item.value for item in model.constants],
    ]
    try:
        with np.errstate(all="ignore"):
            raw = numerical(*arguments)
        values = _coerce_real_output(raw, (point_array.shape[0],), output_name=function.name)
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"Numerical point evaluation failed: {exc}") from exc
    status, diagnostics = _nonfinite_diagnostics(
        values, total_count=point_array.shape[0], output_name=function.name
    )
    values[~np.isfinite(values)] = np.nan
    return ScalarPointEvaluation(
        input_names=tuple(item.name for item in model.variables),
        output_name=function.name,
        points=point_array,
        values=values,
        provenance=Provenance.derived(
            (
                model_ref("function", function.name),
                *(model_ref("variable", item.name) for item in model.variables),
                *(model_ref("parameter", item.name) for item in model.parameters),
                *(model_ref("constant", item.name) for item in model.constants),
            ),
            f"explicit-point numerical evaluation of {function.name}",
        ),
        numerical_status=status,
        diagnostics=diagnostics,
    )


def _coerce_real_output(
    raw: object,
    target_shape: tuple[int, ...],
    *,
    output_name: str,
) -> np.ndarray:
    array = np.asarray(raw)
    if np.iscomplexobj(array):
        imaginary = np.abs(np.imag(array))
        finite_imaginary = imaginary[np.isfinite(imaginary)]
        maximum_imaginary = (
            float(np.max(finite_imaginary)) if finite_imaginary.size else math.inf
        )
        if maximum_imaginary > 1e-10:
            raise EvaluationError(
                f"Function '{output_name}' produced non-real values on the declared domain."
            )
        array = np.real(array)

    try:
        array = np.asarray(array, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluationError(
            f"Function '{output_name}' could not be converted to real numerical values."
        ) from exc

    if array.ndim == 0:
        return np.full(target_shape, float(array), dtype=float)
    try:
        return np.broadcast_to(array, target_shape).astype(float, copy=True)
    except ValueError as exc:
        raise EvaluationError(
            f"Function '{output_name}' did not evaluate to one scalar value per sampled point."
        ) from exc


def _nonfinite_diagnostics(
    values: np.ndarray,
    *,
    total_count: int,
    output_name: str,
) -> tuple[NumericalStatus, tuple[NumericalDiagnostic, ...]]:
    finite_count = int(np.count_nonzero(np.isfinite(values)))
    if finite_count == 0:
        raise EvaluationError(
            f"Function '{output_name}' has no finite real values on the sampled domain."
        )
    nonfinite_count = total_count - finite_count
    if nonfinite_count == 0:
        return NumericalStatus.COMPLETE, ()
    diagnostic = NumericalDiagnostic.warning(
        "non_finite_evaluation_samples",
        f"Function '{output_name}' was non-finite at {nonfinite_count} of {total_count} sampled points.",
        count=nonfinite_count,
        total=total_count,
    )
    return NumericalStatus.PARTIAL, (diagnostic,)


def evaluate_single_variable_function(
    model: ModelIR,
    parameter_values: dict[str, float] | None = None,
    points: int = 1000,
) -> FunctionEvaluation:
    """Evaluate a one-variable, one-function Model IR on a regular grid."""
    if model.blocking_ambiguities:
        names = ", ".join(item.name for item in model.blocking_ambiguities)
        raise EvaluationError(f"Resolve blocking ambiguity/ambiguities before evaluation: {names}.")
    if len(model.variables) != 1 or len(model.functions) != 1:
        raise EvaluationError(
            "This evaluator requires exactly one continuous variable and one scalar function."
        )
    if not isinstance(points, int) or points < 2 or points > 200_000:
        raise EvaluationError("Evaluation points must be an integer between 2 and 200000.")

    variable = model.variables[0]
    function = model.functions[0]
    parameters = _resolved_parameters(model, parameter_values)

    x = np.linspace(variable.domain.lower, variable.domain.upper, points, dtype=float)
    numerical_function = _numerical_function(model)
    arguments = [
        x,
        *[parameters[parameter.name] for parameter in model.parameters],
        *[constant.value for constant in model.constants],
    ]

    try:
        with np.errstate(all="ignore"):
            raw_y = numerical_function(*arguments)
        y = _coerce_real_output(raw_y, x.shape, output_name=function.name)
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"Numerical evaluation failed: {exc}") from exc

    status, diagnostics = _nonfinite_diagnostics(
        y, total_count=points, output_name=function.name
    )
    y[~np.isfinite(y)] = np.nan

    source_refs = [model_ref("function", function.name), model_ref("variable", variable.name)]
    source_refs.extend(
        model_ref("derived_quantity", name)
        for name in model.transitive_derived_dependencies(function.dependencies)
    )
    source_refs.extend(model_ref("parameter", parameter.name) for parameter in model.parameters)
    source_refs.extend(model_ref("constant", constant.name) for constant in model.constants)

    return FunctionEvaluation(
        x_name=variable.name,
        y_name=function.name,
        x=x,
        y=y,
        provenance=Provenance.derived(
            tuple(source_refs),
            f"numerical evaluation of {function.name} over the declared {variable.name} domain",
        ),
        numerical_status=status,
        diagnostics=diagnostics,
    )


def evaluate_two_variable_function(
    model: ModelIR,
    parameter_values: dict[str, float] | None = None,
    points_per_axis: int = 150,
) -> SurfaceEvaluation:
    """Evaluate a two-variable, one-function Model IR on a rectangular grid."""
    if model.blocking_ambiguities:
        names = ", ".join(item.name for item in model.blocking_ambiguities)
        raise EvaluationError(f"Resolve blocking ambiguity/ambiguities before evaluation: {names}.")
    if len(model.variables) != 2 or len(model.functions) != 1:
        raise EvaluationError(
            "This evaluator requires exactly two continuous variables and one scalar function."
        )
    if not isinstance(points_per_axis, int) or points_per_axis < 2 or points_per_axis > 1000:
        raise EvaluationError(
            "Evaluation points per axis must be an integer between 2 and 1000."
        )

    x_variable, y_variable = model.variables
    function = model.functions[0]
    parameters = _resolved_parameters(model, parameter_values)

    x_values = np.linspace(
        x_variable.domain.lower, x_variable.domain.upper, points_per_axis, dtype=float
    )
    y_values = np.linspace(
        y_variable.domain.lower, y_variable.domain.upper, points_per_axis, dtype=float
    )
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")

    numerical_function = _numerical_function(model)
    arguments = [
        x_grid,
        y_grid,
        *[parameters[parameter.name] for parameter in model.parameters],
        *[constant.value for constant in model.constants],
    ]

    try:
        with np.errstate(all="ignore"):
            raw_z = numerical_function(*arguments)
        z = _coerce_real_output(raw_z, x_grid.shape, output_name=function.name)
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"Numerical evaluation failed: {exc}") from exc

    total_count = points_per_axis * points_per_axis
    status, diagnostics = _nonfinite_diagnostics(
        z, total_count=total_count, output_name=function.name
    )
    z[~np.isfinite(z)] = np.nan

    source_refs = [
        model_ref("function", function.name),
        model_ref("variable", x_variable.name),
        model_ref("variable", y_variable.name),
    ]
    source_refs.extend(
        model_ref("derived_quantity", name)
        for name in model.transitive_derived_dependencies(function.dependencies)
    )
    source_refs.extend(model_ref("parameter", parameter.name) for parameter in model.parameters)
    source_refs.extend(model_ref("constant", constant.name) for constant in model.constants)

    return SurfaceEvaluation(
        x_name=x_variable.name,
        y_name=y_variable.name,
        z_name=function.name,
        x=x_grid,
        y=y_grid,
        z=z,
        provenance=Provenance.derived(
            tuple(source_refs),
            f"numerical grid evaluation of {function.name} over the declared variable domains",
        ),
        numerical_status=status,
        diagnostics=diagnostics,
    )
