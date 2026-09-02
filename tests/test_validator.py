from __future__ import annotations

import sympy as sp
import pytest

from model_lab.parser import parse_model_text
from model_lab.validator import ModelValidationError, validate_model


def test_validator_compiles_expression() -> None:
    spec = parse_model_text(
        """
name: Quadratic
variables:
  x:
    domain: [-10, 10]
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

    model = validate_model(spec)
    x, a, b, c = sp.symbols("x a b c", real=True)

    assert sp.expand(model.functions[0].expression - (a * x**2 + b * x + c)) == 0


def test_validator_rejects_undefined_symbol() -> None:
    spec = parse_model_text(
        """
name: Invalid expression
variables:
  x:
    domain: [-10, 10]
parameters:
  a:
    default: 1
    domain: [-5, 5]
functions:
  y: a*x**2 + q
"""
    )

    with pytest.raises(ModelValidationError, match="Undefined symbol: q"):
        validate_model(spec)
