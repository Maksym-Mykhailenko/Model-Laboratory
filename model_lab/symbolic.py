"""Symbolic analysis of validated Model IR objects."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .model import ModelIR
from .provenance import Provenance, analysis_ref, model_ref


class SymbolicAnalysisError(ValueError):
    """Raised when a symbolic analysis is not defined for the supplied model."""


@dataclass(frozen=True, slots=True)
class OneVariableSymbolicAnalysis:
    """First- and second-order symbolic derivatives of a scalar function."""

    function_name: str
    variable_name: str
    first_derivative: sp.Expr
    second_derivative: sp.Expr
    first_derivative_provenance: Provenance
    second_derivative_provenance: Provenance


@dataclass(frozen=True, slots=True)
class TwoVariableSymbolicAnalysis:
    """Gradient and Hessian of a scalar function of two variables."""

    function_name: str
    variable_names: tuple[str, str]
    gradient: tuple[sp.Expr, sp.Expr]
    hessian: sp.Matrix
    gradient_provenance: Provenance
    hessian_provenance: Provenance


@dataclass(frozen=True, slots=True)
class ScalarDifferentialAnalysis:
    """General gradient and Hessian of a scalar function of n variables."""

    function_name: str
    variable_names: tuple[str, ...]
    gradient: tuple[sp.Expr, ...]
    hessian: sp.Matrix
    gradient_provenance: Provenance
    hessian_provenance: Provenance


def analyse_scalar_function(
    model: ModelIR, function_name: str | None = None
) -> ScalarDifferentialAnalysis:
    """Derive a gradient and Hessian for an arbitrary finite input dimension."""
    if not model.variables or not model.functions:
        raise SymbolicAnalysisError(
            "Scalar differential analysis requires variables and a scalar function."
        )
    if function_name is None:
        if len(model.functions) != 1:
            raise SymbolicAnalysisError("Select one scalar function for symbolic analysis.")
        function = model.functions[0]
    else:
        try:
            function = model.function(function_name)
        except KeyError as exc:
            raise SymbolicAnalysisError(f"Unknown scalar function '{function_name}'.") from exc
    symbols = tuple(sp.Symbol(item.name, real=True) for item in model.variables)
    gradient_matrix = sp.Matrix([function.expression]).jacobian(symbols)
    hessian = sp.hessian(function.expression, symbols).applyfunc(sp.simplify)
    gradient = tuple(sp.simplify(gradient_matrix[0, index]) for index in range(len(symbols)))
    source_refs = (
        model_ref("function", function.name),
        *(
            model_ref("derived_quantity", name)
            for name in model.transitive_derived_dependencies(function.dependencies)
        ),
        *(model_ref("variable", item.name) for item in model.variables),
    )
    gradient_ref = analysis_ref("gradient", function.name)
    return ScalarDifferentialAnalysis(
        function_name=function.name,
        variable_names=tuple(item.name for item in model.variables),
        gradient=gradient,
        hessian=hessian,
        gradient_provenance=Provenance.derived(
            source_refs, f"symbolic gradient of {function.name}"
        ),
        hessian_provenance=Provenance.derived(
            (gradient_ref,), f"symbolic Hessian of {function.name}"
        ),
    )


def analyse_one_variable_function(model: ModelIR) -> OneVariableSymbolicAnalysis:
    """Derive the first and second derivatives of a one-variable scalar function."""
    if len(model.variables) != 1 or len(model.functions) != 1:
        raise SymbolicAnalysisError(
            "This symbolic analysis requires exactly one continuous variable and one scalar function."
        )

    variable = model.variables[0]
    function = model.functions[0]
    general = analyse_scalar_function(model, function.name)

    first_ref = analysis_ref("first_derivative", function.name, variable.name)
    return OneVariableSymbolicAnalysis(
        function_name=function.name,
        variable_name=variable.name,
        first_derivative=general.gradient[0],
        second_derivative=general.hessian[0, 0],
        first_derivative_provenance=Provenance.derived(
            (
                model_ref("function", function.name),
                *(
                    model_ref("derived_quantity", name)
                    for name in model.transitive_derived_dependencies(function.dependencies)
                ),
                model_ref("variable", variable.name),
            ),
            f"symbolic differentiation of {function.name} with respect to {variable.name}",
        ),
        second_derivative_provenance=Provenance.derived(
            (first_ref, model_ref("variable", variable.name)),
            f"symbolic differentiation of the first derivative with respect to {variable.name}",
        ),
    )


def analyse_two_variable_function(model: ModelIR) -> TwoVariableSymbolicAnalysis:
    """Derive the gradient and Hessian of a two-variable scalar function."""
    if len(model.variables) != 2 or len(model.functions) != 1:
        raise SymbolicAnalysisError(
            "This symbolic analysis requires exactly two continuous variables and one scalar function."
        )

    first_variable, second_variable = model.variables
    function = model.functions[0]
    general = analyse_scalar_function(model, function.name)

    return TwoVariableSymbolicAnalysis(
        function_name=function.name,
        variable_names=(first_variable.name, second_variable.name),
        gradient=(
            general.gradient[0],
            general.gradient[1],
        ),
        hessian=general.hessian,
        gradient_provenance=general.gradient_provenance,
        hessian_provenance=general.hessian_provenance,
    )
