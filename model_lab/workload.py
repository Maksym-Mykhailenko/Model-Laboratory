"""Deterministic computational-workload estimation and budget enforcement.

Experiment settings are validated both individually and in combination.  This module
prevents otherwise legal settings from composing into an unexpectedly large workload,
particularly two-dimensional stationary-point sweeps where the number of nonlinear
solver starts grows as ``seeds_per_axis**2 * analysis_count``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .model import ModelIR

if TYPE_CHECKING:  # pragma: no cover - imported only for static typing
    from .experiment import EvaluationSettings, StationaryPointSettings, SweepConfiguration


MAX_DIRECT_EVALUATION_SAMPLES = 1_000_000
MAX_TOTAL_1D_SEARCH_SAMPLES = 5_000_000
MAX_TOTAL_2D_SOLVER_STARTS = 50_000


class WorkloadBudgetError(ValueError):
    """Raised when a requested experiment exceeds the deterministic workload budget."""


@dataclass(frozen=True, slots=True)
class WorkloadEstimate:
    """Estimated deterministic workload for one experiment configuration."""

    variable_count: int
    evaluation_samples: int
    stationary_analysis_count: int
    one_dimensional_search_samples: int
    two_dimensional_solver_starts: int
    sweep_steps: int
    violations: tuple[str, ...] = ()

    @property
    def within_budget(self) -> bool:
        return not self.violations

    def summary_rows(self) -> tuple[tuple[str, str], ...]:
        rows: list[tuple[str, str]] = [
            ("Direct evaluation samples", f"{self.evaluation_samples:,}"),
            ("Stationary analyses", f"{self.stationary_analysis_count:,}"),
        ]
        if self.variable_count == 1:
            rows.append(("Total 1D stationary-search samples", f"{self.one_dimensional_search_samples:,}"))
        elif self.variable_count == 2:
            rows.append(("Total 2D nonlinear solver starts", f"{self.two_dimensional_solver_starts:,}"))
        if self.sweep_steps:
            rows.append(("Parameter-sweep steps", f"{self.sweep_steps:,}"))
        return tuple(rows)


def estimate_experiment_workload(
    model: ModelIR,
    evaluation_settings: "EvaluationSettings",
    stationary_settings: "StationaryPointSettings",
    sweep: "SweepConfiguration | None" = None,
) -> WorkloadEstimate:
    """Estimate combined evaluation and stationary-analysis work.

    The stationary analysis performed for the current parameter state counts once.  A
    saved parameter sweep adds one stationary analysis per sweep step.  This deliberately
    counts the combined workload rather than validating each setting in isolation.
    """

    variable_count = len(model.variables)
    sweep_steps = 0 if sweep is None else int(sweep.step_count)
    stationary_analysis_count = 1 + sweep_steps

    if variable_count == 1:
        evaluation_samples = int(evaluation_settings.points_1d)
        one_dimensional_search_samples = (
            int(stationary_settings.samples_1d) * stationary_analysis_count
        )
        two_dimensional_solver_starts = 0
    elif variable_count == 2:
        evaluation_samples = int(evaluation_settings.points_per_axis_2d) ** 2
        one_dimensional_search_samples = 0
        two_dimensional_solver_starts = (
            int(stationary_settings.seeds_per_axis_2d) ** 2 * stationary_analysis_count
        )
    else:
        # The current experiment subsystem only reproduces visualisations supported by
        # the current one-/two-variable laboratory, but retain a safe neutral estimate.
        evaluation_samples = 0
        one_dimensional_search_samples = 0
        two_dimensional_solver_starts = 0

    violations: list[str] = []
    if evaluation_samples > MAX_DIRECT_EVALUATION_SAMPLES:
        violations.append(
            "direct evaluation requests "
            f"{evaluation_samples:,} samples; limit is {MAX_DIRECT_EVALUATION_SAMPLES:,}"
        )
    if one_dimensional_search_samples > MAX_TOTAL_1D_SEARCH_SAMPLES:
        violations.append(
            "combined 1D stationary searches request "
            f"{one_dimensional_search_samples:,} samples; limit is {MAX_TOTAL_1D_SEARCH_SAMPLES:,}"
        )
    if two_dimensional_solver_starts > MAX_TOTAL_2D_SOLVER_STARTS:
        violations.append(
            "combined 2D stationary searches request "
            f"{two_dimensional_solver_starts:,} nonlinear solver starts; limit is "
            f"{MAX_TOTAL_2D_SOLVER_STARTS:,}"
        )

    return WorkloadEstimate(
        variable_count=variable_count,
        evaluation_samples=evaluation_samples,
        stationary_analysis_count=stationary_analysis_count,
        one_dimensional_search_samples=one_dimensional_search_samples,
        two_dimensional_solver_starts=two_dimensional_solver_starts,
        sweep_steps=sweep_steps,
        violations=tuple(violations),
    )



def estimate_parameter_sweep_workload(
    model: ModelIR,
    *,
    step_count: int,
    samples_1d: int,
    seeds_per_axis_2d: int,
) -> WorkloadEstimate:
    """Estimate the stationary-search work performed by one parameter sweep."""
    variable_count = len(model.variables)
    if variable_count == 1:
        one_dimensional_search_samples = int(samples_1d) * int(step_count)
        two_dimensional_solver_starts = 0
    elif variable_count == 2:
        one_dimensional_search_samples = 0
        two_dimensional_solver_starts = int(seeds_per_axis_2d) ** 2 * int(step_count)
    else:
        one_dimensional_search_samples = 0
        two_dimensional_solver_starts = 0

    violations: list[str] = []
    if one_dimensional_search_samples > MAX_TOTAL_1D_SEARCH_SAMPLES:
        violations.append(
            "parameter sweep requests "
            f"{one_dimensional_search_samples:,} one-dimensional stationary-search samples; "
            f"limit is {MAX_TOTAL_1D_SEARCH_SAMPLES:,}"
        )
    if two_dimensional_solver_starts > MAX_TOTAL_2D_SOLVER_STARTS:
        violations.append(
            "parameter sweep requests "
            f"{two_dimensional_solver_starts:,} nonlinear solver starts; limit is "
            f"{MAX_TOTAL_2D_SOLVER_STARTS:,}"
        )

    return WorkloadEstimate(
        variable_count=variable_count,
        evaluation_samples=0,
        stationary_analysis_count=int(step_count),
        one_dimensional_search_samples=one_dimensional_search_samples,
        two_dimensional_solver_starts=two_dimensional_solver_starts,
        sweep_steps=int(step_count),
        violations=tuple(violations),
    )


def enforce_parameter_sweep_budget(
    model: ModelIR,
    *,
    step_count: int,
    samples_1d: int,
    seeds_per_axis_2d: int,
) -> WorkloadEstimate:
    """Reject a parameter sweep whose combined stationary-search workload is excessive."""
    estimate = estimate_parameter_sweep_workload(
        model,
        step_count=step_count,
        samples_1d=samples_1d,
        seeds_per_axis_2d=seeds_per_axis_2d,
    )
    if not estimate.within_budget:
        raise WorkloadBudgetError("; ".join(estimate.violations) + ".")
    return estimate

def enforce_experiment_workload_budget(
    model: ModelIR,
    evaluation_settings: "EvaluationSettings",
    stationary_settings: "StationaryPointSettings",
    sweep: "SweepConfiguration | None" = None,
) -> WorkloadEstimate:
    """Return the workload estimate or raise when its combined budget is exceeded."""

    estimate = estimate_experiment_workload(
        model,
        evaluation_settings,
        stationary_settings,
        sweep,
    )
    if not estimate.within_budget:
        raise WorkloadBudgetError("; ".join(estimate.violations) + ".")
    return estimate
