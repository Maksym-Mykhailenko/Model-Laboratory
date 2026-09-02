"""Model Laboratory-owned mathematical expression AST.

The persisted Model IR must not depend on SymPy's internal representation.  Model source
is therefore parsed into a small, versioned expression tree owned by Model Laboratory.
SymPy remains an execution backend: the validated AST is adapted to SymPy only when
symbolic or numerical computation is required.
"""

from __future__ import annotations

import ast as py_ast
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
from typing import Any, Mapping, TypeAlias

import sympy as sp


EXPRESSION_AST_SCHEMA = "model-laboratory-expression-ast"
EXPRESSION_AST_SCHEMA_VERSION = "1.2"
PREVIOUS_EXPRESSION_AST_SCHEMA_VERSION = "1.1"
LEGACY_EXPRESSION_AST_SCHEMA_VERSION = "1.0"


class ExpressionError(ValueError):
    """Raised when a Model Laboratory expression is invalid or unsupported."""


@dataclass(frozen=True, slots=True)
class IntegerLiteral:
    value: int


@dataclass(frozen=True, slots=True)
class RealLiteral:
    """Finite decimal literal in compact, context-independent base-10 form.

    ``coefficient`` is a canonical signed integer string with no leading zeroes and no
    trailing zeroes unless it is exactly ``"0"``. ``exponent`` is the power of ten by
    which that coefficient is multiplied.  The representation is derived directly from
    the lexical decimal token and never uses the mutable :mod:`decimal` context.
    """

    coefficient: str
    exponent: int


@dataclass(frozen=True, slots=True)
class SymbolReference:
    name: str


@dataclass(frozen=True, slots=True)
class NamedConstant:
    name: str


@dataclass(frozen=True, slots=True)
class UnaryOperation:
    operator: str
    operand: "ExpressionNode"


@dataclass(frozen=True, slots=True)
class BinaryOperation:
    operator: str
    left: "ExpressionNode"
    right: "ExpressionNode"


@dataclass(frozen=True, slots=True)
class FunctionCall:
    name: str
    arguments: tuple["ExpressionNode", ...]


ExpressionNode: TypeAlias = (
    IntegerLiteral
    | RealLiteral
    | SymbolReference
    | NamedConstant
    | UnaryOperation
    | BinaryOperation
    | FunctionCall
)


NAMED_CONSTANTS = frozenset({"pi", "E"})
FUNCTION_ARITIES: dict[str, tuple[int, int]] = {
    "sin": (1, 1),
    "cos": (1, 1),
    "tan": (1, 1),
    "sinh": (1, 1),
    "cosh": (1, 1),
    "tanh": (1, 1),
    "exp": (1, 1),
    "log": (1, 2),
    "sqrt": (1, 1),
    "abs": (1, 1),
}
# Formats written before Model Laboratory 1.8 did not define hyperbolic functions.  This
# registry is part of their historical language contract: sinh/cosh/tanh were ordinary
# identifiers there and must remain so when an archived bundle is reconstructed.
LEGACY_FUNCTION_ARITIES: dict[str, tuple[int, int]] = {
    name: arity
    for name, arity in FUNCTION_ARITIES.items()
    if name not in {"sinh", "cosh", "tanh"}
}

_BINARY_OPERATORS: dict[type[py_ast.operator], str] = {
    py_ast.Add: "add",
    py_ast.Sub: "subtract",
    py_ast.Mult: "multiply",
    py_ast.Div: "divide",
    py_ast.Pow: "power",
}
_UNARY_OPERATORS: dict[type[py_ast.unaryop], str] = {
    py_ast.UAdd: "positive",
    py_ast.USub: "negative",
}


def _canonical_decimal_components(source: str) -> tuple[str, int]:
    """Return canonical ``(coefficient, exponent)`` for one finite decimal token.

    Construction of :class:`~decimal.Decimal` from a string is exact and
    context-independent.  We inspect ``as_tuple()`` directly and perform all
    canonicalisation with integer/string operations, so changing
    ``decimal.getcontext().prec`` cannot alter the persisted AST.

    Examples::

        1.2300      -> ("123", -2)
        1.23        -> ("123", -2)
        1e-999999   -> ("1", -999999)
        2.0         -> ("2", 0)
        -0.0        -> ("0", 0)
    """
    cleaned = source.replace("_", "")
    try:
        value = Decimal(cleaned)
    except InvalidOperation as exc:
        raise ExpressionError(f"Invalid numerical literal: {source}") from exc
    if not value.is_finite():
        raise ExpressionError("Expression constants must be finite numbers.")

    sign, digits, raw_exponent = value.as_tuple()
    if not isinstance(raw_exponent, int):  # defensive: finite Decimal always uses int
        raise ExpressionError("Expression decimal exponent is invalid.")

    coefficient = "".join(str(digit) for digit in digits).lstrip("0")
    if not coefficient:
        return "0", 0

    exponent = int(raw_exponent)
    trailing = len(coefficient) - len(coefficient.rstrip("0"))
    if trailing:
        coefficient = coefficient[:-trailing]
        exponent += trailing

    if sign:
        coefficient = "-" + coefficient
    return coefficient, exponent


def _canonical_real_components(coefficient: object, exponent: object) -> tuple[str, int]:
    """Validate a persisted compact real-literal representation."""
    if not isinstance(coefficient, str) or not isinstance(exponent, int) or isinstance(exponent, bool):
        raise ExpressionError("Expression AST real literal is malformed.")
    if coefficient == "0":
        if exponent != 0:
            raise ExpressionError("Expression AST zero real literal must use exponent 0.")
        return "0", 0
    negative = coefficient.startswith("-")
    digits = coefficient[1:] if negative else coefficient
    if not digits or not digits.isdigit():
        raise ExpressionError("Expression AST real coefficient is invalid.")
    if digits.startswith("0") or digits.endswith("0"):
        raise ExpressionError("Expression AST real coefficient is not canonical.")
    return coefficient, exponent


def _real_literal_from_legacy_value(value: object) -> RealLiteral:
    """Migrate a version-1.0 fixed-decimal real literal into compact representation."""
    if not isinstance(value, str):
        raise ExpressionError("Legacy expression AST real value is invalid.")
    coefficient, exponent = _canonical_decimal_components(value)
    return RealLiteral(coefficient, exponent)


class _Parser(py_ast.NodeVisitor):
    def __init__(
        self,
        source: str,
        declared_symbols: set[str],
        function_arities: Mapping[str, tuple[int, int]],
    ) -> None:
        self.source = source
        self.declared_symbols = declared_symbols
        self.function_arities = function_arities

    def parse(self) -> ExpressionNode:
        try:
            tree = py_ast.parse(self.source, mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(f"Invalid expression syntax: {self.source}") from exc
        return self.visit(tree.body)

    def visit_Constant(self, node: py_ast.Constant) -> ExpressionNode:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError("Only numerical constants are permitted in expressions.")
        if isinstance(node.value, int):
            return IntegerLiteral(node.value)
        if not math.isfinite(node.value):
            raise ExpressionError("Expression constants must be finite numbers.")
        source = py_ast.get_source_segment(self.source, node) or repr(node.value)
        coefficient, exponent = _canonical_decimal_components(source)
        return RealLiteral(coefficient, exponent)

    def visit_Name(self, node: py_ast.Name) -> ExpressionNode:
        if node.id in self.declared_symbols:
            return SymbolReference(node.id)
        if node.id in NAMED_CONSTANTS:
            return NamedConstant(node.id)
        if node.id in self.function_arities:
            raise ExpressionError(
                f"Mathematical function '{node.id}' must be called with parentheses."
            )
        raise ExpressionError(f"Undefined symbol: {node.id}")

    def visit_BinOp(self, node: py_ast.BinOp) -> ExpressionNode:
        operator = _BINARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ExpressionError(
                f"Operator '{type(node.op).__name__}' is not permitted in expressions."
            )
        return BinaryOperation(operator, self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: py_ast.UnaryOp) -> ExpressionNode:
        operator = _UNARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ExpressionError(
                f"Unary operator '{type(node.op).__name__}' is not permitted in expressions."
            )
        operand = self.visit(node.operand)
        # Signed zero has no distinct mathematical value in the Model Laboratory AST.
        # Collapse it here so ``-0.0`` and ``0.0`` have one canonical representation.
        if operator == "negative" and (
            (isinstance(operand, IntegerLiteral) and operand.value == 0)
            or (isinstance(operand, RealLiteral) and operand.coefficient == "0")
        ):
            return operand
        return UnaryOperation(operator, operand)

    def visit_Call(self, node: py_ast.Call) -> ExpressionNode:
        if not isinstance(node.func, py_ast.Name):
            raise ExpressionError("Only named mathematical functions may be called.")
        if node.keywords:
            raise ExpressionError("Keyword arguments are not permitted in expressions.")
        name = node.func.id
        arity = self.function_arities.get(name)
        if arity is None:
            raise ExpressionError(f"Unsupported mathematical function: {name}")
        arguments = tuple(self.visit(argument) for argument in node.args)
        minimum, maximum = arity
        if not (minimum <= len(arguments) <= maximum):
            if minimum == maximum:
                expected = f"exactly {minimum} argument" + ("" if minimum == 1 else "s")
            else:
                expected = f"between {minimum} and {maximum} arguments"
            raise ExpressionError(f"Mathematical function '{name}' requires {expected}.")
        return FunctionCall(name, arguments)

    def generic_visit(self, node: py_ast.AST) -> ExpressionNode:
        raise ExpressionError(
            f"Expression element '{type(node).__name__}' is not permitted."
        )


def parse_expression(
    source: str,
    declared_symbols: set[str] | frozenset[str],
    *,
    function_arities: Mapping[str, tuple[int, int]] | None = None,
) -> ExpressionNode:
    """Parse source into the stable Model Laboratory expression AST."""
    return _Parser(
        source,
        set(declared_symbols),
        FUNCTION_ARITIES if function_arities is None else function_arities,
    ).parse()


def expression_dependencies(node: ExpressionNode) -> tuple[str, ...]:
    """Return direct symbol dependencies in deterministic lexical order."""
    names: set[str] = set()

    def visit(item: ExpressionNode) -> None:
        if isinstance(item, SymbolReference):
            names.add(item.name)
        elif isinstance(item, UnaryOperation):
            visit(item.operand)
        elif isinstance(item, BinaryOperation):
            visit(item.left)
            visit(item.right)
        elif isinstance(item, FunctionCall):
            for argument in item.arguments:
                visit(argument)

    visit(node)
    return tuple(sorted(names))


def expression_to_sympy(
    node: ExpressionNode,
    symbols: Mapping[str, sp.Symbol],
) -> sp.Expr:
    """Adapt a validated Model Laboratory AST to SymPy for execution."""
    if isinstance(node, IntegerLiteral):
        return sp.Integer(node.value)
    if isinstance(node, RealLiteral):
        # Preserve the historical execution semantics of YAML source: a decimal real
        # literal becomes a binary64 value before it enters SymPy.  Scientific notation
        # keeps extreme exponents compact instead of expanding them into enormous fixed
        # decimal strings.
        token = f"{node.coefficient}e{node.exponent}"
        numeric = float(token)
        if not math.isfinite(numeric):
            raise ExpressionError("Real literal lies outside the supported binary64 execution range.")
        return sp.Float(numeric)
    if isinstance(node, SymbolReference):
        try:
            return symbols[node.name]
        except KeyError as exc:
            raise ExpressionError(
                f"No execution symbol is available for '{node.name}'."
            ) from exc
    if isinstance(node, NamedConstant):
        if node.name == "pi":
            return sp.pi
        if node.name == "E":
            return sp.E
        raise ExpressionError(f"Unsupported named constant: {node.name}")
    if isinstance(node, UnaryOperation):
        operand = expression_to_sympy(node.operand, symbols)
        if node.operator == "positive":
            return operand
        if node.operator == "negative":
            return -operand
        raise ExpressionError(f"Unsupported unary operator: {node.operator}")
    if isinstance(node, BinaryOperation):
        left = expression_to_sympy(node.left, symbols)
        right = expression_to_sympy(node.right, symbols)
        if node.operator == "add":
            return left + right
        if node.operator == "subtract":
            return left - right
        if node.operator == "multiply":
            return left * right
        if node.operator == "divide":
            return left / right
        if node.operator == "power":
            return left**right
        raise ExpressionError(f"Unsupported binary operator: {node.operator}")
    if isinstance(node, FunctionCall):
        functions: dict[str, object] = {
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "sinh": sp.sinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "abs": sp.Abs,
        }
        function = functions.get(node.name)
        if function is None:
            raise ExpressionError(f"Unsupported mathematical function: {node.name}")
        arguments = [expression_to_sympy(argument, symbols) for argument in node.arguments]
        try:
            return sp.sympify(function(*arguments))
        except (TypeError, ValueError) as exc:
            raise ExpressionError(
                f"Invalid arguments for mathematical function '{node.name}'."
            ) from exc
    raise TypeError(f"Unsupported expression node: {type(node).__name__}")


def _node_payload(node: ExpressionNode) -> dict[str, Any]:
    if isinstance(node, IntegerLiteral):
        return {"type": "integer", "value": node.value}
    if isinstance(node, RealLiteral):
        return {"type": "real", "coefficient": node.coefficient, "exponent": node.exponent}
    if isinstance(node, SymbolReference):
        return {"type": "symbol", "name": node.name}
    if isinstance(node, NamedConstant):
        return {"type": "constant", "name": node.name}
    if isinstance(node, UnaryOperation):
        return {
            "type": "unary",
            "operator": node.operator,
            "operand": _node_payload(node.operand),
        }
    if isinstance(node, BinaryOperation):
        return {
            "type": "binary",
            "operator": node.operator,
            "left": _node_payload(node.left),
            "right": _node_payload(node.right),
        }
    if isinstance(node, FunctionCall):
        return {
            "type": "call",
            "name": node.name,
            "arguments": [_node_payload(argument) for argument in node.arguments],
        }
    raise TypeError(f"Unsupported expression node: {type(node).__name__}")


def expression_ast_payload(
    node: ExpressionNode,
    *,
    schema_version: str = EXPRESSION_AST_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Return the formally versioned JSON representation of ``node``."""
    if schema_version not in {
        EXPRESSION_AST_SCHEMA_VERSION,
        PREVIOUS_EXPRESSION_AST_SCHEMA_VERSION,
    }:
        raise ExpressionError("Unsupported expression AST output schema version.")
    return {
        "schema": EXPRESSION_AST_SCHEMA,
        "schema_version": schema_version,
        "root": _node_payload(node),
    }


def _node_from_payload(value: object) -> ExpressionNode:
    if not isinstance(value, dict):
        raise ExpressionError("Expression AST node must be an object.")
    node_type = value.get("type")

    if node_type == "integer" and set(value) == {"type", "value"}:
        raw = value["value"]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ExpressionError("Expression AST integer value is invalid.")
        return IntegerLiteral(raw)

    if node_type == "real" and set(value) == {"type", "coefficient", "exponent"}:
        coefficient, exponent = _canonical_real_components(
            value["coefficient"], value["exponent"]
        )
        return RealLiteral(coefficient, exponent)

    if node_type == "symbol" and set(value) == {"type", "name"}:
        name = value["name"]
        if not isinstance(name, str) or not name:
            raise ExpressionError("Expression AST symbol name is invalid.")
        return SymbolReference(name)

    if node_type == "constant" and set(value) == {"type", "name"}:
        name = value["name"]
        if not isinstance(name, str) or name not in NAMED_CONSTANTS:
            raise ExpressionError("Expression AST named constant is invalid.")
        return NamedConstant(name)

    if node_type == "unary" and set(value) == {"type", "operator", "operand"}:
        operator = value["operator"]
        if operator not in {"positive", "negative"}:
            raise ExpressionError("Expression AST unary operator is invalid.")
        return UnaryOperation(str(operator), _node_from_payload(value["operand"]))

    if node_type == "binary" and set(value) == {"type", "operator", "left", "right"}:
        operator = value["operator"]
        if operator not in {"add", "subtract", "multiply", "divide", "power"}:
            raise ExpressionError("Expression AST binary operator is invalid.")
        return BinaryOperation(
            str(operator),
            _node_from_payload(value["left"]),
            _node_from_payload(value["right"]),
        )

    if node_type == "call" and set(value) == {"type", "name", "arguments"}:
        name = value["name"]
        arguments = value["arguments"]
        if not isinstance(name, str) or name not in FUNCTION_ARITIES:
            raise ExpressionError("Expression AST function name is invalid.")
        if not isinstance(arguments, list):
            raise ExpressionError("Expression AST function arguments are invalid.")
        result = FunctionCall(name, tuple(_node_from_payload(item) for item in arguments))
        minimum, maximum = FUNCTION_ARITIES[name]
        if not (minimum <= len(result.arguments) <= maximum):
            raise ExpressionError("Expression AST function-call arity is invalid.")
        return result

    raise ExpressionError("Expression AST node has an unsupported shape.")


def _legacy_node_from_payload(value: object) -> ExpressionNode:
    """Validate and migrate one expression-AST 1.0 node."""
    if not isinstance(value, dict):
        raise ExpressionError("Legacy expression AST node must be an object.")
    node_type = value.get("type")
    if node_type == "real" and set(value) == {"type", "value"}:
        return _real_literal_from_legacy_value(value["value"])
    if node_type in {"integer", "symbol", "constant"}:
        return _node_from_payload(value)
    if node_type == "unary" and set(value) == {"type", "operator", "operand"}:
        operator = value["operator"]
        if operator not in {"positive", "negative"}:
            raise ExpressionError("Legacy expression AST unary operator is invalid.")
        return UnaryOperation(str(operator), _legacy_node_from_payload(value["operand"]))
    if node_type == "binary" and set(value) == {"type", "operator", "left", "right"}:
        operator = value["operator"]
        if operator not in {"add", "subtract", "multiply", "divide", "power"}:
            raise ExpressionError("Legacy expression AST binary operator is invalid.")
        return BinaryOperation(
            str(operator),
            _legacy_node_from_payload(value["left"]),
            _legacy_node_from_payload(value["right"]),
        )
    if node_type == "call" and set(value) == {"type", "name", "arguments"}:
        name = value["name"]
        arguments = value["arguments"]
        if not isinstance(name, str) or name not in LEGACY_FUNCTION_ARITIES or not isinstance(arguments, list):
            raise ExpressionError("Legacy expression AST function call is invalid.")
        result = FunctionCall(name, tuple(_legacy_node_from_payload(item) for item in arguments))
        minimum, maximum = LEGACY_FUNCTION_ARITIES[name]
        if not (minimum <= len(result.arguments) <= maximum):
            raise ExpressionError("Legacy expression AST function-call arity is invalid.")
        return result
    raise ExpressionError("Legacy expression AST node has an unsupported shape.")


def expression_ast_from_payload(value: object) -> ExpressionNode:
    """Validate and reconstruct a versioned Model Laboratory expression AST payload.

    Versions 1.0 and 1.1 are accepted only as migration inputs. New serialisation emits
    version 1.2, which uniquely identifies the language containing hyperbolic functions.
    """
    if not isinstance(value, dict):
        raise ExpressionError("Expression AST payload must be an object.")
    if set(value) != {"schema", "schema_version", "root"}:
        raise ExpressionError("Expression AST payload has an unsupported shape.")
    if value.get("schema") != EXPRESSION_AST_SCHEMA:
        raise ExpressionError("Unsupported expression AST schema.")
    version = value.get("schema_version")
    if version == EXPRESSION_AST_SCHEMA_VERSION:
        return _node_from_payload(value["root"])
    if version == PREVIOUS_EXPRESSION_AST_SCHEMA_VERSION:
        # AST 1.1 was emitted both before and during v1.8. Accept the complete v1.8
        # node vocabulary when migrating; bundle-format-specific parsing separately
        # restores pre-v1.8 identifier semantics for model.yaml.
        return _node_from_payload(value["root"])
    if version == LEGACY_EXPRESSION_AST_SCHEMA_VERSION:
        return _legacy_node_from_payload(value["root"])
    raise ExpressionError("Unsupported expression AST schema version.")
