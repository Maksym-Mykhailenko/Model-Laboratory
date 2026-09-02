from __future__ import annotations

import sympy as sp

from model_lab.parser import parse_model_text
from model_lab.symbolic import analyse_one_variable_function, analyse_two_variable_function
from model_lab.validator import validate_model


def test_one_variable_symbolic_derivatives() -> None:
    spec = parse_model_text(
        """
name: Polynomial
variables:
  x:
    domain: [-3, 3]
parameters:
  a:
    default: 2
    domain: [-5, 5]
functions:
  y: a*x**3 + x
"""
    )
    model = validate_model(spec)
    analysis = analyse_one_variable_function(model)
    x, a = sp.symbols("x a", real=True)

    assert sp.simplify(analysis.first_derivative - (3 * a * x**2 + 1)) == 0
    assert sp.simplify(analysis.second_derivative - 6 * a * x) == 0


def test_two_variable_symbolic_gradient_and_hessian() -> None:
    spec = parse_model_text(
        """
name: Surface
variables:
  x:
    domain: [-3, 3]
  y:
    domain: [-3, 3]
parameters:
  a:
    default: 2
    domain: [-5, 5]
functions:
  z: a*x**2 + x*y + y**2
"""
    )
    model = validate_model(spec)
    analysis = analyse_two_variable_function(model)
    x, y, a = sp.symbols("x y a", real=True)

    assert sp.simplify(analysis.gradient[0] - (2 * a * x + y)) == 0
    assert sp.simplify(analysis.gradient[1] - (x + 2 * y)) == 0
    assert analysis.hessian == sp.Matrix([[2 * a, 1], [1, 2]])
