"""Deterministic analysis orchestration for Model Laboratory.

This module returns immutable result objects. It contains no plotting or user-interface
code; visualisation modules consume the results afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .critical_points import (
    CriticalPointAnalysisError,
    OneDimensionalCriticalPoint,
    TwoDimensionalCriticalPoint,
    search_one_variable_stationary_points,
    search_two_variable_stationary_points,
)
from .issues import NumericalDiagnostic, NumericalStatistics, NumericalStatus
from .model import ModelIR
from .provenance import Provenance, analysis_ref, model_ref
from .workload import WorkloadBudgetError, enforce_parameter_sweep_budget


CriticalPoint = OneDimensionalCriticalPoint | TwoDimensionalCriticalPoint


class AnalysisError(ValueError):
    """Raised when a requested deterministic analysis is invalid for the model/settings."""


@dataclass(frozen=True, slots=True)
class StationaryPointAnalysisResult:
    """Stationary points computed for one parameter state."""

    parameter_values: tuple[tuple[str, float], ...]
    points: tuple[CriticalPoint, ...]
    provenance: Provenance
    numerical_status: NumericalStatus = NumericalStatus.COMPLETE
    diagnostics: tuple[NumericalDiagnostic, ...] = ()
    statistics: NumericalStatistics = NumericalStatistics()


@dataclass(frozen=True, slots=True)
class ParameterSweepStep:
    """Stationary-point result at one value of a swept parameter."""

    parameter_value: float
    points: tuple[CriticalPoint, ...]
    error: str | None = None
    numerical_status: NumericalStatus = NumericalStatus.COMPLETE
    diagnostics: tuple[NumericalDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class ParameterSweepResult:
    """A deterministic sweep of one parameter while all other parameters remain fixed."""

    parameter_name: str
    function_name: str
    variable_names: tuple[str, ...]
    start: float
    end: float
    step_count: int
    fixed_parameters: tuple[tuple[str, float], ...]
    steps: tuple[ParameterSweepStep, ...]
    provenance: Provenance

    @property
    def failed_step_count(self) -> int:
        return sum(
            step.numerical_status in (NumericalStatus.FAILED, NumericalStatus.INDETERMINATE)
            for step in self.steps
        )

    @property
    def partial_step_count(self) -> int:
        return sum(step.numerical_status is NumericalStatus.PARTIAL for step in self.steps)

    @property
    def successful_step_count(self) -> int:
        return self.step_count - self.failed_step_count


def _validate_parameter_values(
    model: ModelIR,
    parameter_values: dict[str, float] | None,
) -> dict[str, float]:
    values = model.parameter_defaults()
    if parameter_values:
        unknown = set(parameter_values) - {parameter.name for parameter in model.parameters}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise AnalysisError(f"Unknown parameter value(s): {names}.")
        try:
            values.update({name: float(value) for name, value in parameter_values.items()})
        except (TypeError, ValueError, OverflowError) as exc:
            raise AnalysisError("Parameter values must be finite real numbers.") from exc

    for parameter in model.parameters:
        value = float(values[parameter.name])
        if not math.isfinite(value):
            raise AnalysisError(f"Parameter '{parameter.name}' must have a finite numerical value.")
        if not parameter.domain.contains(value):
            raise AnalysisError(
                f"Parameter '{parameter.name}' = {value} lies outside its domain "
                f"[{parameter.domain.lower}, {parameter.domain.upper}]."
            )
    return values


def run_stationary_point_analysis(
    model: ModelIR,
    parameter_values: dict[str, float] | None = None,
    *,
    samples_1d: int = 2001,
    seeds_per_axis_2d: int = 11,
    root_tolerance: float = 1e-9,
) -> StationaryPointAnalysisResult:
    """Compute stationary points without making assumptions about rendering."""
    if model.blocking_ambiguities:
        names = ", ".join(item.name for item in model.blocking_ambiguities)
        raise AnalysisError(f"Resolve blocking ambiguity/ambiguities before analysis: {names}.")
    if model.constraints:
        raise AnalysisError(
            "Stationary-point analysis does not currently apply declared constraints."
        )
    values = _validate_parameter_values(model, parameter_values)

    try:
        if len(model.variables) == 1 and len(model.functions) == 1:
            search = search_one_variable_stationary_points(
                model,
                values,
                samples=samples_1d,
                root_tolerance=root_tolerance,
            )
        elif len(model.variables) == 2 and len(model.functions) == 1:
            search = search_two_variable_stationary_points(
                model,
                values,
                seeds_per_axis=seeds_per_axis_2d,
                root_tolerance=root_tolerance,
            )
        else:
            raise AnalysisError(
                "Stationary-point analysis currently requires one or two continuous "
                "variables and exactly one scalar function."
            )
    except CriticalPointAnalysisError as exc:
        raise AnalysisError(str(exc)) from exc

    function = model.functions[0]
    source_refs = [model_ref("function", function.name)]
    source_refs.extend(
        model_ref("derived_quantity", name)
        for name in model.transitive_derived_dependencies(function.dependencies)
    )
    source_refs.extend(model_ref("variable", variable.name) for variable in model.variables)
    source_refs.extend(model_ref("parameter", parameter.name) for parameter in model.parameters)
    source_refs.extend(model_ref("constant", constant.name) for constant in model.constants)

    return StationaryPointAnalysisResult(
        parameter_values=tuple(
            (parameter.name, values[parameter.name]) for parameter in model.parameters
        ),
        points=search.points,
        provenance=Provenance.derived(
            tuple(source_refs),
            "numerical stationary-point search and local classification",
        ),
        numerical_status=search.status,
        diagnostics=search.diagnostics,
        statistics=search.statistics,
    )


def _step_error_message(result: StationaryPointAnalysisResult) -> str | None:
    if result.numerical_status not in (NumericalStatus.FAILED, NumericalStatus.INDETERMINATE):
        return None
    if result.diagnostics:
        return result.diagnostics[0].message
    return f"Stationary-point analysis was {result.numerical_status.value}."


def run_parameter_sweep(
    model: ModelIR,
    parameter_name: str,
    *,
    start: float | None = None,
    end: float | None = None,
    step_count: int = 21,
    fixed_parameter_values: dict[str, float] | None = None,
    samples_1d: int = 2001,
    seeds_per_axis_2d: int = 11,
    root_tolerance: float = 1e-9,
) -> ParameterSweepResult:
    """Sweep one declared parameter and repeat stationary-point analysis at each step."""
    if model.blocking_ambiguities:
        names = ", ".join(item.name for item in model.blocking_ambiguities)
        raise AnalysisError(f"Resolve blocking ambiguity/ambiguities before analysis: {names}.")
    if model.constraints:
        raise AnalysisError("Parameter sweep does not currently apply declared constraints.")
    if not model.parameters:
        raise AnalysisError("Parameter sweep requires at least one adjustable parameter.")

    try:
        parameter = model.parameter(parameter_name)
    except KeyError as exc:
        raise AnalysisError(f"Unknown sweep parameter: {parameter_name}.") from exc

    if len(model.variables) not in (1, 2) or len(model.functions) != 1:
        raise AnalysisError(
            "Parameter sweep currently requires one or two continuous variables and exactly "
            "one scalar function."
        )

    if not isinstance(step_count, int) or step_count < 2:
        raise AnalysisError("A parameter sweep requires at least two steps.")
    if step_count > 201:
        raise AnalysisError("A parameter sweep is limited to 201 steps in this version.")

    try:
        enforce_parameter_sweep_budget(
            model,
            step_count=step_count,
            samples_1d=samples_1d,
            seeds_per_axis_2d=seeds_per_axis_2d,
        )
    except WorkloadBudgetError as exc:
        raise AnalysisError(f"Parameter sweep exceeds the computational budget: {exc}") from exc

    lower = parameter.domain.lower if start is None else float(start)
    upper = parameter.domain.upper if end is None else float(end)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise AnalysisError("Sweep bounds must be finite numbers.")

    if not parameter.domain.contains(lower) or not parameter.domain.contains(upper):
        raise AnalysisError(
            f"Sweep bounds must lie inside parameter '{parameter_name}' domain "
            f"[{parameter.domain.lower}, {parameter.domain.upper}]."
        )
    if lower >= upper:
        raise AnalysisError("The sweep start must be smaller than the sweep end.")

    fixed_values = _validate_parameter_values(model, fixed_parameter_values)
    fixed_values.pop(parameter_name, None)

    sweep_values = np.linspace(lower, upper, step_count, dtype=float)
    sweep_steps: list[ParameterSweepStep] = []

    for parameter_value in sweep_values:
        current_values = dict(fixed_values)
        current_values[parameter_name] = float(parameter_value)
        try:
            result = run_stationary_point_analysis(
                model,
                current_values,
                samples_1d=samples_1d,
                seeds_per_axis_2d=seeds_per_axis_2d,
                root_tolerance=root_tolerance,
            )
            sweep_steps.append(
                ParameterSweepStep(
                    parameter_value=float(parameter_value),
                    points=result.points,
                    error=_step_error_message(result),
                    numerical_status=result.numerical_status,
                    diagnostics=result.diagnostics,
                )
            )
        except AnalysisError as exc:
            sweep_steps.append(
                ParameterSweepStep(
                    parameter_value=float(parameter_value),
                    points=(),
                    error=str(exc),
                    numerical_status=NumericalStatus.FAILED,
                    diagnostics=(
                        NumericalDiagnostic.error("analysis_failed", str(exc)),
                    ),
                )
            )

    return ParameterSweepResult(
        parameter_name=parameter_name,
        function_name=model.functions[0].name,
        variable_names=tuple(variable.name for variable in model.variables),
        start=lower,
        end=upper,
        step_count=step_count,
        fixed_parameters=tuple(
            (parameter_item.name, fixed_values[parameter_item.name])
            for parameter_item in model.parameters
            if parameter_item.name != parameter_name
        ),
        steps=tuple(sweep_steps),
        provenance=Provenance.derived(
            (
                analysis_ref("stationary_points", model.functions[0].name),
                model_ref("parameter", parameter_name),
            ),
            f"parameter sweep of stationary-point analysis over {parameter_name}",
        ),
    )
