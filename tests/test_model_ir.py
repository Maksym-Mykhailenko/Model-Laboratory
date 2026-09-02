from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from model_lab.evaluator import evaluate_single_variable_function
from model_lab.model import ConstraintRelation, ProvenanceKind
from model_lab.parser import ModelParseError, parse_model_text
from model_lab.validator import ModelValidationError, validate_model


def _compile(text: str):
    return validate_model(parse_model_text(text))


def test_rich_model_elements_are_compiled_into_ir() -> None:
    model = _compile(
        """
name: Rich scalar model
metadata:
  description: A structured scalar model.
  tags: [example, structured]
variables:
  x:
    domain: [-2, 2]
    initial: 0.5
    label: Position
    unit: m
parameters:
  a:
    default: 2
    domain: [0, 4]
    label: Gain
constants:
  c:
    value: 3
    unit: s
functions:
  y:
    expression: a*x + c
    label: Response
    description: Main scalar output.
"""
    )

    assert model.metadata.description == "A structured scalar model."
    assert model.metadata.tags == ("example", "structured")
    assert model.variable("x").initial_value == 0.5
    assert model.variable("x").metadata.label == "Position"
    assert model.variable("x").metadata.unit == "m"
    assert model.parameter("a").metadata.label == "Gain"
    assert model.constant("c").value == 3
    assert model.constant("c").metadata.unit == "s"
    assert model.function("y").metadata.label == "Response"
    assert model.function("y").dependencies == ("a", "c", "x")
    assert model.function("y").provenance.kind == ProvenanceKind.SUPPLIED


def test_constants_participate_in_numerical_evaluation() -> None:
    model = _compile(
        """
name: Constant evaluation
variables:
  x:
    domain: [0, 2]
constants:
  c:
    value: 3
functions:
  y: c*x
"""
    )

    result = evaluate_single_variable_function(model, points=3)

    assert np.allclose(result.x, [0, 1, 2])
    assert np.allclose(result.y, [0, 3, 6])


def test_derived_quantities_can_depend_on_other_derived_quantities() -> None:
    model = _compile(
        """
name: Derived chain
variables:
  x:
    domain: [-2, 2]
parameters:
  a:
    default: 2
    domain: [0, 4]
derived_quantities:
  q:
    expression: a*x
  r:
    expression: q**2 + 1
functions:
  y: r + q
"""
    )

    x, a = sp.symbols("x a", real=True)
    q = model.derived_quantity("q")
    r = model.derived_quantity("r")
    y = model.function("y")

    assert q.dependencies == ("a", "x")
    assert r.dependencies == ("q",)
    assert sp.simplify(q.expression - a * x) == 0
    assert sp.simplify(r.expression - ((a * x) ** 2 + 1)) == 0
    assert sp.simplify(y.expression - (((a * x) ** 2 + 1) + a * x)) == 0


def test_cyclic_derived_quantity_dependency_is_rejected() -> None:
    specification = parse_model_text(
        """
name: Derived cycle
variables:
  x:
    domain: [-2, 2]
derived_quantities:
  q:
    expression: r + x
  r:
    expression: q + 1
functions:
  y: x
"""
    )

    with pytest.raises(ModelValidationError, match="Cyclic derived-quantity dependency"):
        validate_model(specification)


def test_constraints_are_compiled_and_may_reference_function_output() -> None:
    model = _compile(
        """
name: Constrained scalar model
variables:
  x:
    domain: [-2, 2]
parameters:
  a:
    default: 1
    domain: [0, 3]
functions:
  y: a*x**2
constraints:
  nonnegative_output:
    left: y
    relation: ">="
    right: 0
    description: Output is constrained to be non-negative.
"""
    )

    constraint = model.constraint("nonnegative_output")
    x, a = sp.symbols("x a", real=True)

    assert constraint.relation == ConstraintRelation.GREATER_EQUAL
    assert constraint.dependencies == ("y",)
    assert sp.simplify(constraint.left - a * x**2) == 0
    assert constraint.right == 0
    assert constraint.metadata.description == "Output is constrained to be non-negative."


def test_equal_sign_constraint_is_normalised() -> None:
    model = _compile(
        """
name: Equality constraint
variables:
  x:
    domain: [-2, 2]
functions:
  y: x
constraints:
  origin:
    left: x
    relation: "="
    right: 0
"""
    )

    assert model.constraint("origin").relation == ConstraintRelation.EQUAL
    assert model.constraint("origin").relation.value == "=="


def test_variable_initial_value_must_lie_inside_domain() -> None:
    invalid = """
name: Invalid initial value
variables:
  x:
    domain: [-1, 1]
    initial: 2
functions:
  y: x
"""

    with pytest.raises(ModelParseError, match="outside the variable domain"):
        parse_model_text(invalid)


def test_identifiers_are_unique_across_expanded_ir_categories() -> None:
    invalid = """
name: Duplicate identifier
variables:
  x:
    domain: [-1, 1]
constants:
  x:
    value: 2
functions:
  y: x
"""

    with pytest.raises(ModelParseError, match="Model identifiers must be unique"):
        parse_model_text(invalid)


def test_declared_names_cover_all_named_ir_elements_in_order() -> None:
    model = _compile(
        """
name: Name inventory
variables:
  x:
    domain: [-1, 1]
parameters:
  a:
    default: 1
    domain: [0, 2]
constants:
  c:
    value: 2
derived_quantities:
  q:
    expression: a*x
functions:
  y: q + c
constraints:
  bounded:
    left: y
    relation: "<="
    right: 10
"""
    )

    assert model.declared_names() == ("x", "a", "c", "q", "y", "bounded")
    assert model.constant_values() == {"c": 2.0}


def test_unknown_symbol_in_derived_quantity_is_rejected_without_guessing() -> None:
    specification = parse_model_text(
        """
name: Undefined derived symbol
variables:
  x:
    domain: [-1, 1]
derived_quantities:
  q:
    expression: x + missing
functions:
  y: x
"""
    )

    with pytest.raises(ModelValidationError, match="Derived quantity 'q': Undefined symbol: missing"):
        validate_model(specification)
