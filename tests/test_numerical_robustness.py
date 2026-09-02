from __future__ import annotations

import numpy as np
import pytest

from model_lab.analysis import run_parameter_sweep, run_stationary_point_analysis
from model_lab.critical_points import (
    search_one_variable_stationary_points,
    search_two_variable_stationary_points,
)
from model_lab.evaluator import EvaluationError, evaluate_single_variable_function
from model_lab.issues import NumericalStatus
from model_lab.parser import parse_model_text
from model_lab.validator import validate_model


def _compile(text: str):
    return validate_model(parse_model_text(text))


def test_one_dimensional_search_finds_tangent_root_between_grid_points() -> None:
    model = _compile(
        """
name: Tangent derivative root
variables:
  x:
    domain: [-1, 1]
functions:
  y: (x - 0.12345)**3
"""
    )

    result = search_one_variable_stationary_points(model, samples=501)

    assert result.status is NumericalStatus.COMPLETE
    assert len(result.points) == 1
    assert result.points[0].x == pytest.approx(0.12345, abs=1e-7)
    assert result.points[0].classification == "degenerate / inconclusive"
    assert result.statistics.as_dict()["tangent_refinement_acceptances"] >= 1


def test_one_dimensional_search_reports_partial_nonfinite_derivative_domain() -> None:
    model = _compile(
        """
name: Logarithm
variables:
  x:
    domain: [-1, 1]
functions:
  y: log(x)
"""
    )

    result = search_one_variable_stationary_points(model, samples=101)

    assert result.status is NumericalStatus.PARTIAL
    assert result.points == ()
    assert any(item.code == "non_finite_derivative_samples" for item in result.diagnostics)


def test_two_dimensional_search_converges_to_off_grid_minimum() -> None:
    model = _compile(
        """
name: Off-grid minimum
variables:
  x:
    domain: [-1, 1]
  y:
    domain: [-1, 1]
functions:
  z: (x - 0.137)**2 + (y + 0.221)**2
"""
    )

    result = search_two_variable_stationary_points(model, seeds_per_axis=7)

    assert result.status is NumericalStatus.COMPLETE
    assert len(result.points) == 1
    point = result.points[0]
    assert point.x == pytest.approx(0.137, abs=1e-7)
    assert point.y == pytest.approx(-0.221, abs=1e-7)
    assert point.classification == "local minimum"


def test_identically_zero_derivative_is_structured_as_indeterminate() -> None:
    model = _compile(
        """
name: Flat function
variables:
  x:
    domain: [-2, 2]
functions:
  y: "1"
"""
    )

    result = run_stationary_point_analysis(model)

    assert result.numerical_status is NumericalStatus.INDETERMINATE
    assert result.points == ()
    assert result.diagnostics[0].code == "stationary_continuum"


def test_parameter_sweep_preserves_indeterminate_step_as_failed_step() -> None:
    model = _compile(
        """
name: Flat sweep step
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

    result = run_parameter_sweep(model, "a", start=-1, end=1, step_count=3)

    assert result.failed_step_count == 1
    assert result.steps[1].numerical_status is NumericalStatus.INDETERMINATE
    assert "every point in the domain is stationary" in result.steps[1].error


def test_partial_function_evaluation_records_nonfinite_samples() -> None:
    model = _compile(
        """
name: Partial logarithm
variables:
  x:
    domain: [-1, 1]
functions:
  y: log(x)
"""
    )

    result = evaluate_single_variable_function(model, points=101)

    assert result.numerical_status is NumericalStatus.PARTIAL
    assert np.count_nonzero(np.isnan(result.y)) == 51
    assert result.diagnostics[0].code == "non_finite_evaluation_samples"


def test_evaluation_rejects_domain_with_no_finite_real_values() -> None:
    model = _compile(
        """
name: No real values
variables:
  x:
    domain: [-1, 1]
functions:
  y: sqrt(-(x**2) - 1)
"""
    )

    with pytest.raises(EvaluationError, match="no finite real values"):
        evaluate_single_variable_function(model, points=101)


def test_evaluation_rejects_non_real_output_instead_of_dropping_imaginary_part() -> None:
    model = _compile(
        """
name: Complex output
variables:
  x:
    domain: [-1, 1]
functions:
  y: (-1)**0.5 + x
"""
    )

    with pytest.raises(EvaluationError, match="non-real values"):
        evaluate_single_variable_function(model, points=11)


def test_numerical_parameter_override_must_be_finite() -> None:
    model = _compile(
        """
name: Finite parameter
variables:
  x:
    domain: [-1, 1]
parameters:
  a:
    default: 1
    domain: [0, 2]
functions:
  y: a*x
"""
    )

    with pytest.raises(EvaluationError, match="finite numerical value"):
        evaluate_single_variable_function(model, parameter_values={"a": float("nan")})


def test_nondifferentiable_absolute_value_cusp_is_not_reported_as_stationary() -> None:
    model = _compile(
        """
name: Absolute-value cusp
variables:
  x:
    domain: [-1, 1]
functions:
  y: abs(x)
"""
    )

    result = search_one_variable_stationary_points(model, samples=101)

    assert result.status is NumericalStatus.PARTIAL
    assert result.points == ()
    assert any(item.code == "non_differentiable_candidates" for item in result.diagnostics)


def test_two_dimensional_nondifferentiable_cusp_is_excluded_without_crashing() -> None:
    model = _compile(
        """
name: Two-dimensional cusp
variables:
  x:
    domain: [-1, 1]
  y:
    domain: [-1, 1]
functions:
  z: abs(x) + y**2
"""
    )

    result = search_two_variable_stationary_points(model, seeds_per_axis=7)

    assert result.status is NumericalStatus.PARTIAL
    assert result.points == ()
    assert any(item.code == "non_differentiable_candidates" for item in result.diagnostics)


def test_flat_high_order_stationary_point_is_not_duplicated() -> None:
    model = _compile(
        """
name: Flat quartic minimum
variables:
  x:
    domain: [-2, 2]
functions:
  y: (x - 0.3333)**4
"""
    )

    result = search_one_variable_stationary_points(model, samples=2001)

    assert result.status is NumericalStatus.COMPLETE
    assert len(result.points) == 1
    point = result.points[0]
    assert point.x == pytest.approx(0.3333, abs=1e-7)
    assert point.classification == "degenerate / inconclusive"


def test_two_dimensional_non_isolated_stationary_line_is_not_sampled_as_finite_points() -> None:
    model = _compile(
        """
name: Stationary line
variables:
  x:
    domain: [-1, 1]
  y:
    domain: [-1, 1]
functions:
  z: x**2
"""
    )

    result = search_two_variable_stationary_points(model, seeds_per_axis=7)

    assert result.status is NumericalStatus.INDETERMINATE
    assert result.points == ()
    assert any(
        item.code == "non_isolated_stationary_structure" for item in result.diagnostics
    )


def test_two_dimensional_identically_zero_component_with_provably_nonzero_other_component_has_no_stationary_points() -> None:
    model = _compile(
        """
name: No stationary points
variables:
  x:
    domain: [-1, 1]
  y:
    domain: [-1, 1]
functions:
  z: x
"""
    )

    result = search_two_variable_stationary_points(model, seeds_per_axis=7)

    assert result.status is NumericalStatus.COMPLETE
    assert result.points == ()
    assert result.diagnostics == ()


def test_two_dimensional_dependent_polynomial_gradient_is_reported_as_non_isolated() -> None:
    model = _compile(
        """
name: Dependent gradient
variables:
  x:
    domain: [-2, 2]
  y:
    domain: [-2, 2]
functions:
  z: (x + y)**2
"""
    )

    result = search_two_variable_stationary_points(model, seeds_per_axis=11)

    assert result.status is NumericalStatus.INDETERMINATE
    assert result.points == ()
    assert any(
        item.code == "non_isolated_stationary_structure" for item in result.diagnostics
    )
