from __future__ import annotations

import pytest

from model_lab.parser import ModelParseError, parse_model_text


VALID_MODEL = """
name: Test model
variables:
  x:
    domain: [-2, 2]
parameters:
  a:
    default: 1
    domain: [-5, 5]
functions:
  y: a*x**2
"""


def test_parser_reads_valid_model() -> None:
    model = parse_model_text(VALID_MODEL)

    assert model.name == "Test model"
    assert model.variables["x"].domain == (-2.0, 2.0)
    assert model.parameters["a"].default == 1.0
    assert model.functions["y"] == "a*x**2"


def test_parser_rejects_parameter_default_outside_domain() -> None:
    invalid = """
name: Invalid model
variables:
  x:
    domain: [-2, 2]
parameters:
  a:
    default: 10
    domain: [-5, 5]
functions:
  y: a*x
"""

    with pytest.raises(ModelParseError):
        parse_model_text(invalid)
