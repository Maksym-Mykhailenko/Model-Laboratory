from __future__ import annotations

from model_lab.critical_points import (
    find_one_variable_stationary_points,
    find_two_variable_stationary_points,
)
from model_lab.parser import parse_model_text
from model_lab.validator import validate_model


def _model(text: str):
    return validate_model(parse_model_text(text))


def test_one_variable_minimum() -> None:
    model = _model(
        """
name: Minimum
variables:
  x:
    domain: [-3, 3]
parameters: {}
functions:
  y: x**2
"""
    )
    points = find_one_variable_stationary_points(model)
    assert len(points) == 1
    assert abs(points[0].x) < 1e-8
    assert points[0].classification == "local minimum"


def test_one_variable_maximum() -> None:
    model = _model(
        """
name: Maximum
variables:
  x:
    domain: [-3, 3]
parameters: {}
functions:
  y: -x**2
"""
    )
    points = find_one_variable_stationary_points(model)
    assert len(points) == 1
    assert abs(points[0].x) < 1e-8
    assert points[0].classification == "local maximum"


def test_two_variable_minimum() -> None:
    model = _model(
        """
name: Bowl
variables:
  x:
    domain: [-3, 3]
  y:
    domain: [-3, 3]
parameters: {}
functions:
  z: x**2 + y**2
"""
    )
    points = find_two_variable_stationary_points(model)
    assert len(points) == 1
    point = points[0]
    assert abs(point.x) < 1e-7
    assert abs(point.y) < 1e-7
    assert point.classification == "local minimum"


def test_two_variable_saddle() -> None:
    model = _model(
        """
name: Saddle
variables:
  x:
    domain: [-3, 3]
  y:
    domain: [-3, 3]
parameters: {}
functions:
  z: x**2 - y**2
"""
    )
    points = find_two_variable_stationary_points(model)
    assert len(points) == 1
    point = points[0]
    assert abs(point.x) < 1e-7
    assert abs(point.y) < 1e-7
    assert point.classification == "saddle"
