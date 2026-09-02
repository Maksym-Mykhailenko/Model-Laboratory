from __future__ import annotations

import pytest

from model_lab.analysis import AnalysisError, run_parameter_sweep
from model_lab.experiment import (
    EvaluationSettings,
    ExperimentStateError,
    StationaryPointSettings,
    SweepConfiguration,
    validate_experiment_state_for_model,
)
from model_lab.parser import parse_model_text
from model_lab.validator import validate_model
from model_lab.workload import (
    MAX_TOTAL_2D_SOLVER_STARTS,
    estimate_experiment_workload,
)


def _surface_model():
    return validate_model(
        parse_model_text(
            """
name: Workload surface
variables:
  x: {domain: [-2, 2]}
  y: {domain: [-2, 2]}
parameters:
  a: {default: 1, domain: [0, 2]}
functions:
  z: a*(x**2 + y**2)
"""
        )
    )


def test_combined_2d_sweep_workload_is_estimated_from_cross_product() -> None:
    model = _surface_model()
    estimate = estimate_experiment_workload(
        model,
        EvaluationSettings(points_per_axis_2d=1000),
        StationaryPointSettings(seeds_per_axis_2d=101),
        SweepConfiguration("a", 0, 2, 201),
    )

    assert estimate.evaluation_samples == 1_000_000
    assert estimate.stationary_analysis_count == 202
    assert estimate.two_dimensional_solver_starts == 101 * 101 * 202
    assert estimate.two_dimensional_solver_starts > MAX_TOTAL_2D_SOLVER_STARTS
    assert not estimate.within_budget


def test_ordinary_default_workload_remains_within_budget() -> None:
    model = _surface_model()
    estimate = estimate_experiment_workload(
        model,
        EvaluationSettings(),
        StationaryPointSettings(),
        SweepConfiguration("a", 0, 2, 21),
    )

    assert estimate.within_budget


def test_parameter_sweep_core_rejects_excessive_cross_setting_workload() -> None:
    model = _surface_model()
    with pytest.raises(AnalysisError, match="computational budget"):
        run_parameter_sweep(
            model,
            "a",
            start=0,
            end=2,
            step_count=201,
            seeds_per_axis_2d=101,
        )
