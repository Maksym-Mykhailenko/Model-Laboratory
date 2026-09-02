from __future__ import annotations

import json
from decimal import getcontext

import pytest
import sympy as sp

from model_lab.canonical import (
    CANONICAL_MODEL_IR_SCHEMA_VERSION,
    canonical_model_ir_payload,
)
from model_lab.expression import (
    EXPRESSION_AST_SCHEMA_VERSION,
    BinaryOperation,
    ExpressionError,
    FunctionCall,
    IntegerLiteral,
    RealLiteral,
    SymbolReference,
    expression_ast_from_payload,
    expression_ast_payload,
    expression_dependencies,
    expression_to_sympy,
    parse_expression,
)
from model_lab.parser import parse_model_text
from model_lab.validator import validate_model


def test_expression_ast_round_trip_is_library_independent_and_versioned() -> None:
    node = parse_expression("a*x**2 + sin(x) - 0.25", {"a", "x"})
    payload = expression_ast_payload(node)

    assert payload["schema_version"] == EXPRESSION_AST_SCHEMA_VERSION
    assert expression_ast_from_payload(json.loads(json.dumps(payload))) == node
    assert expression_dependencies(node) == ("a", "x")


def test_expression_ast_preserves_execution_semantics_through_sympy_adapter() -> None:
    node = parse_expression("a*x**2 + log(x, 10)", {"a", "x"})
    symbols = {name: sp.Symbol(name, real=True) for name in ("a", "x")}
    expression = expression_to_sympy(node, symbols)

    assert sp.simplify(expression - (symbols["a"] * symbols["x"] ** 2 + sp.log(symbols["x"], 10))) == 0


def test_hyperbolic_functions_are_owned_by_the_expression_ast_and_sympy_adapter() -> None:
    node = parse_expression("sinh(x) + cosh(x) - tanh(x)", {"x"})
    symbols = {"x": sp.Symbol("x", real=True)}
    expression = expression_to_sympy(node, symbols)

    assert expression_dependencies(node) == ("x",)
    assert sp.simplify(
        expression
        - (sp.sinh(symbols["x"]) + sp.cosh(symbols["x"]) - sp.tanh(symbols["x"]))
    ) == 0
    payload = expression_ast_payload(node)
    assert expression_ast_from_payload(json.loads(json.dumps(payload))) == node


@pytest.mark.parametrize("function_name", ["sinh", "cosh", "tanh"])
def test_hyperbolic_function_names_are_reserved(function_name: str) -> None:
    with pytest.raises(Exception, match="reserved mathematical names"):
        validate_model(
            parse_model_text(
                f"""
name: Reserved hyperbolic function
variables:
  {function_name}:
    domain: [-1, 1]
functions:
  y: {function_name}
"""
            )
        )


def test_canonical_ir_uses_model_laboratory_ast_and_not_sympy_srepr(monkeypatch) -> None:
    model = validate_model(
        parse_model_text(
            """
name: AST canonical example
variables:
  x:
    domain: [-2, 2]
parameters:
  a:
    default: 1
    domain: [-3, 3]
functions:
  y: a*x**2 + 0.5
"""
        )
    )

    def forbidden(*args, **kwargs):  # pragma: no cover - regression tripwire
        raise AssertionError("SymPy srepr must not participate in canonical Model IR")

    monkeypatch.setattr(sp, "srepr", forbidden)
    payload = canonical_model_ir_payload(model)

    assert payload["schema_version"] == CANONICAL_MODEL_IR_SCHEMA_VERSION == "3.0"
    expression = payload["functions"][0]["expression"]
    assert expression["schema_version"] == EXPRESSION_AST_SCHEMA_VERSION
    assert "sympy_srepr" not in json.dumps(payload)


def test_expression_ast_distinguishes_integer_and_real_execution_literals() -> None:
    integer = parse_expression("2", set())
    real = parse_expression("2.0", set())
    assert isinstance(integer, IntegerLiteral)
    assert isinstance(real, RealLiteral)
    assert (real.coefficient, real.exponent) == ("2", 0)
    assert expression_ast_payload(integer) != expression_ast_payload(real)


def test_expression_ast_rejects_invalid_or_unknown_payloads() -> None:
    with pytest.raises(ExpressionError, match="schema version"):
        expression_ast_from_payload(
            {
                "schema": "model-laboratory-expression-ast",
                "schema_version": "99.0",
                "root": {"type": "integer", "value": 1},
            }
        )


def test_ast_shape_for_simple_polynomial_is_explicit() -> None:
    node = parse_expression("a*x**2 + b", {"a", "x", "b"})
    assert isinstance(node, BinaryOperation)
    assert node.operator == "add"
    assert isinstance(node.left, BinaryOperation)
    assert node.left.operator == "multiply"
    assert isinstance(node.left.left, SymbolReference)
    assert isinstance(node.left.right, BinaryOperation)
    assert node.left.right.operator == "power"


def test_canonical_model_identity_ignores_runtime_sympy_tree_shape() -> None:
    from dataclasses import replace
    from model_lab.canonical import canonical_model_ir_sha256

    model = validate_model(
        parse_model_text(
            """
name: Runtime backend independence
variables:
  x:
    domain: [-2, 2]
parameters:
  a:
    default: 1
    domain: [-3, 3]
functions:
  y: a*x + x
"""
        )
    )
    function = model.functions[0]
    x = sp.Symbol("x", real=True)
    a = sp.Symbol("a", real=True)
    deliberately_different_runtime_tree = sp.Add(a * x, x, evaluate=False)
    variant = replace(
        model,
        functions=(replace(function, expression=deliberately_different_runtime_tree),),
    )

    assert sp.srepr(function.expression) != sp.srepr(deliberately_different_runtime_tree)
    assert canonical_model_ir_sha256(model) == canonical_model_ir_sha256(variant)


def test_real_literal_canonicalisation_is_independent_of_decimal_context() -> None:
    source = "1.23456789012345678901234567890123456789"
    original = getcontext().prec
    try:
        payloads = []
        for precision in (10, 28, 50):
            getcontext().prec = precision
            payloads.append(expression_ast_payload(parse_expression(source, set())))
    finally:
        getcontext().prec = original

    assert payloads[0] == payloads[1] == payloads[2]
    root = payloads[0]["root"]
    assert root == {
        "type": "real",
        "coefficient": "123456789012345678901234567890123456789",
        "exponent": -38,
    }


def test_extreme_decimal_exponent_remains_compact() -> None:
    payload = expression_ast_payload(parse_expression("1e-999999", set()))
    assert payload["root"] == {"type": "real", "coefficient": "1", "exponent": -999999}
    assert len(json.dumps(payload)) < 300


def test_equivalent_real_spellings_have_one_canonical_real_representation() -> None:
    payloads = [
        expression_ast_payload(parse_expression(source, set()))
        for source in ("1.2300", "1.23", "1.23000e0")
    ]
    assert payloads[0] == payloads[1] == payloads[2]
    assert payloads[0]["root"] == {"type": "real", "coefficient": "123", "exponent": -2}


def test_integer_and_real_literal_types_remain_distinct() -> None:
    integer = expression_ast_payload(parse_expression("2", set()))
    real = expression_ast_payload(parse_expression("2.0", set()))
    assert integer["root"] == {"type": "integer", "value": 2}
    assert real["root"] == {"type": "real", "coefficient": "2", "exponent": 0}
    assert integer != real


def test_negative_real_zero_has_one_canonical_representation() -> None:
    negative = expression_ast_payload(parse_expression("-0.0", set()))
    positive = expression_ast_payload(parse_expression("0.0", set()))
    assert negative == positive
    assert positive["root"] == {"type": "real", "coefficient": "0", "exponent": 0}


def test_expression_ast_1_0_real_literal_migrates_to_compact_1_1_form() -> None:
    legacy = {
        "schema": "model-laboratory-expression-ast",
        "schema_version": "1.0",
        "root": {"type": "real", "value": "0.0000012300"},
    }
    node = expression_ast_from_payload(legacy)
    assert isinstance(node, RealLiteral)
    assert (node.coefficient, node.exponent) == ("123", -8)
    assert expression_ast_payload(node)["root"] == {
        "type": "real",
        "coefficient": "123",
        "exponent": -8,
    }


def test_canonical_model_ir_hash_is_independent_of_decimal_context() -> None:
    from model_lab.canonical import canonical_model_ir_sha256

    source = """
name: Decimal-context independence
variables:
  x:
    domain: [-1, 1]
functions:
  y: x + 1.23456789012345678901234567890123456789
"""
    original = getcontext().prec
    try:
        digests = []
        for precision in (10, 28, 50):
            getcontext().prec = precision
            model = validate_model(parse_model_text(source))
            digests.append(canonical_model_ir_sha256(model))
    finally:
        getcontext().prec = original
    assert digests[0] == digests[1] == digests[2]
