from __future__ import annotations

import numpy as np

from model_lab.evaluator import evaluate_single_variable_function
from model_lab.parser import parse_model_text
from model_lab.validator import validate_model


def _quadratic_model():
    spec = parse_model_text(
        """
name: Quadratic
variables:
  x:
    domain: [-2, 2]
parameters:
  a:
    default: 1
    domain: [-5, 5]
  b:
    default: 0
    domain: [-5, 5]
  c:
    default: 0
    domain: [-5, 5]
functions:
  y: a*x**2 + b*x + c
"""
    )
    return validate_model(spec)


def test_evaluator_uses_parameter_values() -> None:
    model = _quadratic_model()
    result = evaluate_single_variable_function(
        model,
        parameter_values={"a": 2, "b": 3, "c": 1},
        points=5,
    )

    expected = 2 * result.x**2 + 3 * result.x + 1
    np.testing.assert_allclose(result.y, expected)


def test_evaluator_uses_requested_number_of_points() -> None:
    model = _quadratic_model()
    result = evaluate_single_variable_function(model, points=17)

    assert len(result.x) == 17
    assert len(result.y) == 17


def test_two_variable_evaluator_builds_expected_grid() -> None:
    from model_lab.evaluator import evaluate_two_variable_function

    spec = parse_model_text(
        """
name: Plane
variables:
  x:
    domain: [-1, 1]
  y:
    domain: [-2, 2]
parameters:
  a:
    default: 2
    domain: [0, 5]
functions:
  z: a*x + y
"""
    )
    model = validate_model(spec)
    result = evaluate_two_variable_function(model, points_per_axis=7)

    assert result.x.shape == (7, 7)
    assert result.y.shape == (7, 7)
    assert result.z.shape == (7, 7)
    np.testing.assert_allclose(result.z, 2 * result.x + result.y)
