from __future__ import annotations

import pytest

from model_lab.analysis import AnalysisError, run_parameter_sweep, run_stationary_point_analysis
from model_lab.parser import parse_model_text
from model_lab.validator import validate_model


def _quadratic_family():
    return validate_model(
        parse_model_text(
            """
name: Quadratic family
variables:
  x:
    domain: [-3, 3]
parameters:
  a:
    default: 0
    domain: [-2, 2]
functions:
  y: x**2 + a*x
"""
        )
    )


def test_stationary_point_analysis_returns_result_object() -> None:
    result = run_stationary_point_analysis(_quadratic_family(), {"a": 2.0})

    assert dict(result.parameter_values) == {"a": 2.0}
    assert len(result.points) == 1
    assert result.points[0].x == pytest.approx(-1.0, abs=1e-6)
    assert result.points[0].classification == "local minimum"


def test_parameter_sweep_tracks_stationary_position_without_modifying_model() -> None:
    model = _quadratic_family()
    original_default = model.parameter("a").default

    result = run_parameter_sweep(model, "a", start=-2, end=2, step_count=3)

    assert [step.parameter_value for step in result.steps] == pytest.approx([-2.0, 0.0, 2.0])
    assert [step.points[0].x for step in result.steps] == pytest.approx([1.0, 0.0, -1.0], abs=1e-6)
    assert all(step.points[0].classification == "local minimum" for step in result.steps)
    assert model.parameter("a").default == original_default


def test_parameter_sweep_records_a_failed_step_and_continues() -> None:
    model = validate_model(
        parse_model_text(
            """
name: Degenerate sweep
variables:
  x:
    domain: [-2, 2]
parameters:
  a:
    default: 1
    domain: [-1, 1]
functions:
  y: a*x
"""
        )
    )

    result = run_parameter_sweep(model, "a", start=-1, end=1, step_count=3)

    assert result.failed_step_count == 1
    assert result.steps[1].parameter_value == pytest.approx(0.0)
    assert "every point in the domain is stationary" in result.steps[1].error
    assert result.steps[0].error is None
    assert result.steps[2].error is None


def test_parameter_sweep_rejects_invalid_range() -> None:
    with pytest.raises(AnalysisError, match="start must be smaller"):
        run_parameter_sweep(_quadratic_family(), "a", start=1, end=-1, step_count=5)
