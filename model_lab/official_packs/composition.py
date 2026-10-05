"""Declarative, bounded composition of reusable scientific analysis primitives.

An analysis recipe is data in the Model Graph, never executable source code.  The
installed runner interprets a small, explicit operation vocabulary and records a
content-addressed value graph.  This gives model authors room to define new derived
quantities and figures without adding one Python capability for every scientific model.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import keyword
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import linalg

from ..canonical import canonical_json_sha256
from ..evaluator import evaluate_scalar_at_points
from ..model import ModelIR
from ..model_graph import ModelGraphError, ModelObject, ObjectKindDescriptor
from ..protocol import (
    ArtifactTypeDescriptor,
    CapabilityDescriptor,
    ComparisonOutcome,
    ScientificArtifact,
    comparator_registry,
    compare_numeric_data,
    portable_value,
)
from ..vector_analysis import analyse_matrix_function
from .common import MAX_INLINE_VALUES, PackManifest, objects_of_kind, select_object


PACK_ID = "org.modellab.pack.analysis-composition"
RECIPE_KIND = "org.modellab.composition.analysis-recipe"

MAX_RECIPE_STEPS = 128
MAX_RECIPE_OUTPUTS = 64
MAX_RECIPE_VIEWS = 1
MAX_EXPRESSION_LENGTH = 4096
MAX_EXPRESSION_NODES = 512
MAX_VALUE_ELEMENTS = MAX_INLINE_VALUES
MAX_RECIPE_STORED_ELEMENTS = 1_000_000
MAX_FORMULA_LINEAR_ALGEBRA_OPERATIONS = 5_000_000

_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_OPERATIONS = {
    "array.literal",
    "array.linspace",
    "array.logspace",
    "array.expression",
    "matrix.evaluate",
    "matrix.generalized-eigenvalues",
    "scalar.evaluate-points",
    "series.first-crossing",
}


_STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["id", "operation"],
    "properties": {
        "id": {"type": "string"},
        "operation": {"type": "string", "enum": sorted(_OPERATIONS)},
        "inputs": {"type": "object"},
        "settings": {"type": "object"},
    },
    "additionalProperties": False,
}

_SERIES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["value", "label"],
    "properties": {
        "value": {"type": "string"},
        "label": {"type": "string"},
        "line_dash": {
            "type": "string",
            "enum": ["solid", "dash", "dot", "dashdot"],
        },
    },
    "additionalProperties": False,
}

_PANEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["x", "series"],
    "properties": {
        "title": {"type": "string"},
        "x": {"type": "string"},
        "x_label": {"type": "string"},
        "y_label": {"type": "string"},
        "x_scale": {"type": "string", "enum": ["linear", "log"]},
        "series": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": _SERIES_SCHEMA,
        },
    },
    "additionalProperties": False,
}

_VIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["title", "panels"],
    "properties": {
        "title": {"type": "string"},
        "panels": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": _PANEL_SCHEMA,
        },
    },
    "additionalProperties": False,
}

RECIPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["steps", "outputs"],
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_RECIPE_STEPS,
            "items": _STEP_SCHEMA,
        },
        "outputs": {"type": "object"},
        "views": {
            "type": "array",
            "maxItems": MAX_RECIPE_VIEWS,
            "items": _VIEW_SCHEMA,
        },
    },
    "additionalProperties": False,
}

KIND_DESCRIPTORS = (
    ObjectKindDescriptor(
        RECIPE_KIND,
        "1.0",
        "Declarative analysis recipe",
        RECIPE_SCHEMA,
        True,
    ),
)


class AnalysisRecipeError(ValueError):
    """Raised when a declarative recipe is invalid or cannot be evaluated safely."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelGraphError(f"{field} must be an object.")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelGraphError(f"{field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ModelGraphError(f"{field} must be a finite number.")
    return result


def _strict_keys(
    value: Mapping[str, Any], *, required: set[str], optional: set[str], field: str
) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise ModelGraphError(f"{field} is missing: {', '.join(sorted(missing))}.")
    if unknown:
        raise ModelGraphError(f"{field} contains unknown fields: {', '.join(sorted(unknown))}.")


def _numeric_literal(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ModelGraphError(f"{field} must contain real numbers, not booleans.")
    if isinstance(value, (int, float)):
        _finite_number(value, field=field)
        return 1
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        count = 0
        for index, item in enumerate(value):
            count += _numeric_literal(item, field=f"{field}[{index}]")
            if count > MAX_VALUE_ELEMENTS:
                raise ModelGraphError(f"{field} exceeds the inline-value limit.")
        return count
    raise ModelGraphError(f"{field} must be a scalar or a rectangular array of real numbers.")


def _numeric_vector(value: object, *, field: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ModelGraphError(f"{field} must be a non-empty vector of finite real numbers.")
    if not value:
        raise ModelGraphError(f"{field} must be a non-empty vector of finite real numbers.")
    for index, item in enumerate(value):
        _finite_number(item, field=f"{field}[{index}]")


def _bounded_text(value: object, *, field: str, maximum_bytes: int = 256) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > maximum_bytes
    ):
        raise ModelGraphError(
            f"{field} must be non-empty and at most {maximum_bytes} UTF-8 bytes."
        )
    return value


def _validated_expression(expression: object, *, field: str) -> ast.Expression:
    if not isinstance(expression, str) or not expression.strip():
        raise ModelGraphError(f"{field} must be a non-empty expression string.")
    if len(expression.encode("utf-8")) > MAX_EXPRESSION_LENGTH:
        raise ModelGraphError(f"{field} exceeds the expression-length limit.")
    try:
        parsed = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ModelGraphError(f"{field} is not a valid expression: {exc}.") from exc
    nodes = tuple(ast.walk(parsed))
    if len(nodes) > MAX_EXPRESSION_NODES:
        raise ModelGraphError(f"{field} exceeds the expression-complexity limit.")
    allowed = (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.MatMult,
        ast.UAdd,
        ast.USub,
    )
    for node in nodes:
        if not isinstance(node, allowed):
            raise ModelGraphError(
                f"{field} contains unsupported syntax '{type(node).__name__}'."
            )
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool) or not isinstance(node.value, (int, float))
        ):
            raise ModelGraphError(f"{field} may contain only real numeric literals.")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.keywords
        ):
            raise ModelGraphError(f"{field} permits only named functions and positional arguments.")
    return parsed


def _validate_step(
    step: Mapping[str, Any], *, recipe_id: str, earlier: set[str]
) -> str:
    step_id = str(step["id"])
    if _IDENTIFIER.fullmatch(step_id) is None:
        raise ModelGraphError(
            f"{recipe_id}.steps id '{step_id}' must be a short identifier beginning with a letter."
        )
    if step_id in earlier:
        raise ModelGraphError(f"{recipe_id}.steps contains duplicate id '{step_id}'.")
    operation = str(step["operation"])
    if operation not in _OPERATIONS:
        raise ModelGraphError(f"{recipe_id}.{step_id} uses unsupported operation '{operation}'.")
    inputs = _mapping(step.get("inputs", {}), field=f"{recipe_id}.{step_id}.inputs")
    settings = _mapping(step.get("settings", {}), field=f"{recipe_id}.{step_id}.settings")
    for local_name, reference in inputs.items():
        if (
            _IDENTIFIER.fullmatch(str(local_name)) is None
            or keyword.iskeyword(str(local_name))
        ):
            raise ModelGraphError(f"{recipe_id}.{step_id} has invalid input name '{local_name}'.")
        if str(local_name) in {*_SAFE_FUNCTIONS, "pi", "e"}:
            raise ModelGraphError(
                f"{recipe_id}.{step_id} input name '{local_name}' is reserved by the expression engine."
            )
        if not isinstance(reference, str) or reference not in earlier:
            raise ModelGraphError(
                f"{recipe_id}.{step_id} input '{local_name}' must reference an earlier step."
            )

    field = f"{recipe_id}.{step_id}.settings"
    if operation == "array.literal":
        if inputs:
            raise ModelGraphError(f"{recipe_id}.{step_id} does not accept inputs.")
        _strict_keys(settings, required={"value"}, optional=set(), field=field)
        _numeric_literal(settings["value"], field=f"{field}.value")
        try:
            np.asarray(settings["value"], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelGraphError(f"{field}.value must be a rectangular numerical array.") from exc
    elif operation == "array.linspace":
        if inputs:
            raise ModelGraphError(f"{recipe_id}.{step_id} does not accept inputs.")
        _strict_keys(settings, required={"start", "stop", "count"}, optional=set(), field=field)
        start = _finite_number(settings["start"], field=f"{field}.start")
        stop = _finite_number(settings["stop"], field=f"{field}.stop")
        count = settings["count"]
        if start >= stop:
            raise ModelGraphError(f"{field} requires start < stop.")
        if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= MAX_VALUE_ELEMENTS:
            raise ModelGraphError(f"{field}.count must be between 2 and {MAX_VALUE_ELEMENTS}.")
    elif operation == "array.logspace":
        if inputs:
            raise ModelGraphError(f"{recipe_id}.{step_id} does not accept inputs.")
        _strict_keys(
            settings,
            required={"start_exponent", "stop_exponent", "count"},
            optional={"base"},
            field=field,
        )
        start = _finite_number(settings["start_exponent"], field=f"{field}.start_exponent")
        stop = _finite_number(settings["stop_exponent"], field=f"{field}.stop_exponent")
        base = _finite_number(settings.get("base", 10.0), field=f"{field}.base")
        count = settings["count"]
        if start >= stop:
            raise ModelGraphError(f"{field} requires start_exponent < stop_exponent.")
        if base <= 1.0:
            raise ModelGraphError(f"{field}.base must be greater than one.")
        if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= MAX_VALUE_ELEMENTS:
            raise ModelGraphError(f"{field}.count must be between 2 and {MAX_VALUE_ELEMENTS}.")
    elif operation == "array.expression":
        _strict_keys(settings, required={"expression"}, optional=set(), field=field)
        parsed = _validated_expression(settings["expression"], field=f"{field}.expression")
        permitted_names = set(inputs) | set(_SAFE_FUNCTIONS) | {"pi", "e"}
        unknown = {
            node.id for node in ast.walk(parsed)
            if isinstance(node, ast.Name) and node.id not in permitted_names
        }
        if unknown:
            raise ModelGraphError(
                f"{field}.expression contains unknown names: {', '.join(sorted(unknown))}."
            )
    elif operation == "matrix.evaluate":
        if inputs:
            raise ModelGraphError(f"{recipe_id}.{step_id} does not accept inputs.")
        _strict_keys(
            settings,
            required={"matrix", "point"},
            optional={"parameter_values"},
            field=field,
        )
        if not isinstance(settings["matrix"], str) or _IDENTIFIER.fullmatch(settings["matrix"]) is None:
            raise ModelGraphError(f"{field}.matrix must name a matrix function.")
        _numeric_vector(settings["point"], field=f"{field}.point")
        parameters = _mapping(settings.get("parameter_values", {}), field=f"{field}.parameter_values")
        for name, value in parameters.items():
            if _IDENTIFIER.fullmatch(str(name)) is None:
                raise ModelGraphError(f"{field}.parameter_values contains invalid name '{name}'.")
            _finite_number(value, field=f"{field}.parameter_values.{name}")
    elif operation == "matrix.generalized-eigenvalues":
        _strict_keys(inputs, required={"matrix"}, optional={"metric"}, field=f"{recipe_id}.{step_id}.inputs")
        _strict_keys(
            settings,
            required=set(),
            optional={"require_positive", "symmetry_tolerance", "positivity_tolerance"},
            field=field,
        )
        if "require_positive" in settings and not isinstance(settings["require_positive"], bool):
            raise ModelGraphError(f"{field}.require_positive must be boolean.")
        for name in ("symmetry_tolerance", "positivity_tolerance"):
            if name in settings and _finite_number(settings[name], field=f"{field}.{name}") <= 0.0:
                raise ModelGraphError(f"{field}.{name} must be positive.")
    elif operation == "scalar.evaluate-points":
        if inputs:
            raise ModelGraphError(f"{recipe_id}.{step_id} does not accept inputs.")
        _strict_keys(
            settings,
            required={"function", "points"},
            optional={"parameter_values"},
            field=field,
        )
        if not isinstance(settings["function"], str) or _IDENTIFIER.fullmatch(settings["function"]) is None:
            raise ModelGraphError(f"{field}.function must name a scalar function.")
        _numeric_literal(settings["points"], field=f"{field}.points")
        parameters = _mapping(settings.get("parameter_values", {}), field=f"{field}.parameter_values")
        for name, value in parameters.items():
            if _IDENTIFIER.fullmatch(str(name)) is None:
                raise ModelGraphError(f"{field}.parameter_values contains invalid name '{name}'.")
            _finite_number(value, field=f"{field}.parameter_values.{name}")
    elif operation == "series.first-crossing":
        _strict_keys(inputs, required={"x", "y"}, optional=set(), field=f"{recipe_id}.{step_id}.inputs")
        _strict_keys(settings, required={"threshold"}, optional={"direction"}, field=field)
        _finite_number(settings["threshold"], field=f"{field}.threshold")
        if settings.get("direction", "below") not in {"below", "above"}:
            raise ModelGraphError(f"{field}.direction must be 'below' or 'above'.")
    return step_id


def validate_recipe(item: ModelObject) -> None:
    properties = item.properties
    steps = properties["steps"]
    outputs = _mapping(properties["outputs"], field=f"{item.identifier}.outputs")
    if not 1 <= len(outputs) <= MAX_RECIPE_OUTPUTS:
        raise ModelGraphError(
            f"{item.identifier}.outputs must contain between 1 and {MAX_RECIPE_OUTPUTS} entries."
        )
    earlier: set[str] = set()
    for raw_step in steps:
        step = _mapping(raw_step, field=f"{item.identifier}.steps")
        earlier.add(_validate_step(step, recipe_id=item.identifier, earlier=earlier))
    for name, reference in outputs.items():
        _bounded_text(name, field=f"{item.identifier}.outputs label", maximum_bytes=128)
        if not isinstance(reference, str) or reference not in earlier:
            raise ModelGraphError(
                f"{item.identifier}.outputs.{name} must reference a declared step."
            )
    for view_index, view in enumerate(properties.get("views", [])):
        _bounded_text(
            view["title"], field=f"{item.identifier}.views[{view_index}].title"
        )
        for panel_index, panel in enumerate(view["panels"]):
            for label_field in ("title", "x_label", "y_label"):
                if label_field in panel:
                    _bounded_text(
                        panel[label_field],
                        field=(
                            f"{item.identifier}.views[{view_index}].panels"
                            f"[{panel_index}].{label_field}"
                        ),
                    )
            x_reference = panel["x"]
            if x_reference not in earlier:
                raise ModelGraphError(
                    f"{item.identifier}.views[{view_index}].panels[{panel_index}].x references an unknown step."
                )
            for series_index, series in enumerate(panel["series"]):
                if series["value"] not in earlier:
                    raise ModelGraphError(
                        f"{item.identifier}.views[{view_index}].panels[{panel_index}].series[{series_index}] references an unknown step."
                    )
                _bounded_text(
                    series["label"],
                    field=(
                        f"{item.identifier}.views[{view_index}].panels[{panel_index}]"
                        f".series[{series_index}].label"
                    ),
                    maximum_bytes=128,
                )


SEMANTIC_VALIDATORS = {RECIPE_KIND: validate_recipe}


def _value_array(value: object, *, field: str) -> np.ndarray:
    if isinstance(value, bool):
        raise AnalysisRecipeError(f"{field} produced a boolean rather than numerical data.")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisRecipeError(f"{field} did not produce real numerical data.") from exc
    if result.size == 0:
        raise AnalysisRecipeError(f"{field} produced an empty numerical value.")
    if result.size > MAX_VALUE_ELEMENTS:
        raise AnalysisRecipeError(f"{field} exceeds the {MAX_VALUE_ELEMENTS}-value limit.")
    if not np.all(np.isfinite(result)):
        raise AnalysisRecipeError(f"{field} produced non-finite numerical data.")
    result = result.copy()
    result[result == 0.0] = 0.0
    return result


def _normalised_value(value: object, *, field: str) -> float | np.ndarray:
    result = _value_array(value, field=field)
    return float(result) if result.ndim == 0 else result


def _bounded_element_count(shape: Sequence[int], *, field: str) -> int:
    count = math.prod(int(item) for item in shape) if shape else 1
    if count > MAX_VALUE_ELEMENTS:
        raise AnalysisRecipeError(
            f"{field} would exceed the bounded {MAX_VALUE_ELEMENTS}-value intermediate limit."
        )
    return count


def _bounded_linear_algebra_work(operations: int, *, field: str) -> None:
    if operations > MAX_FORMULA_LINEAR_ALGEBRA_OPERATIONS:
        raise AnalysisRecipeError(
            f"{field} exceeds the bounded formula linear-algebra work limit."
        )


def _elementwise_operands(
    left: object, right: object, *, field: str
) -> tuple[np.ndarray, np.ndarray]:
    left_value = _value_array(left, field=f"{field}.left")
    right_value = _value_array(right, field=f"{field}.right")
    try:
        result_shape = np.broadcast_shapes(left_value.shape, right_value.shape)
    except ValueError as exc:
        raise AnalysisRecipeError(f"{field} operands cannot be broadcast together.") from exc
    _bounded_element_count(result_shape, field=field)
    return left_value, right_value


def _matmul_operands(left: object, right: object) -> tuple[np.ndarray, np.ndarray]:
    left_value = _value_array(left, field="matrix multiplication left")
    right_value = _value_array(right, field="matrix multiplication right")
    if left_value.ndim == 0 or right_value.ndim == 0:
        raise AnalysisRecipeError("Matrix multiplication requires arrays with at least one dimension.")
    left_shape = (1, *left_value.shape) if left_value.ndim == 1 else left_value.shape
    right_shape = (*right_value.shape, 1) if right_value.ndim == 1 else right_value.shape
    if left_shape[-1] != right_shape[-2]:
        raise AnalysisRecipeError("Matrix multiplication inner dimensions do not match.")
    try:
        batch_shape = np.broadcast_shapes(left_shape[:-2], right_shape[:-2])
    except ValueError as exc:
        raise AnalysisRecipeError(
            "Matrix multiplication batch dimensions cannot be broadcast together."
        ) from exc
    result_shape = [*batch_shape, left_shape[-2], right_shape[-1]]
    if left_value.ndim == 1:
        result_shape.pop(-2)
    if right_value.ndim == 1:
        result_shape.pop(-1)
    result_count = _bounded_element_count(result_shape, field="matrix multiplication")
    _bounded_linear_algebra_work(
        result_count * int(left_shape[-1]), field="matrix multiplication"
    )
    return left_value, right_value


def _axis(args: tuple[object, ...], *, function: str) -> int | None:
    if len(args) == 1:
        return None
    if len(args) != 2 or isinstance(args[1], bool) or not isinstance(args[1], (int, np.integer)):
        raise AnalysisRecipeError(f"{function} accepts a value and an optional integer axis.")
    return int(args[1])


def _reduce(function: str, args: tuple[object, ...]) -> object:
    axis = _axis(args, function=function)
    value = _value_array(args[0], field=function)
    if function == "sum":
        return np.sum(value, axis=axis)
    if function == "mean":
        return np.mean(value, axis=axis)
    if function == "std":
        return np.std(value, axis=axis)
    if function == "min":
        return np.min(value, axis=axis)
    if function == "max":
        return np.max(value, axis=axis)
    if function == "prod":
        return np.prod(value, axis=axis)
    raise AnalysisRecipeError(f"Unsupported reduction '{function}'.")


def _unary_numpy(function: str, args: tuple[object, ...]) -> object:
    if len(args) != 1:
        raise AnalysisRecipeError(f"{function} accepts exactly one argument.")
    value = _value_array(args[0], field=function)
    implementation = {
        "abs": np.abs,
        "sqrt": np.sqrt,
        "exp": np.exp,
        "log": np.log,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
    }[function]
    return implementation(value)


def _outer(args: tuple[object, ...]) -> np.ndarray:
    if len(args) != 2:
        raise AnalysisRecipeError("outer accepts exactly two arguments.")
    left = _value_array(args[0], field="outer.left").reshape(-1)
    right = _value_array(args[1], field="outer.right").reshape(-1)
    if left.size * right.size > MAX_VALUE_ELEMENTS:
        raise AnalysisRecipeError("outer would exceed the bounded intermediate-value limit.")
    return np.outer(left, right)


def _dot(args: tuple[object, ...]) -> object:
    if len(args) != 2:
        raise AnalysisRecipeError("dot accepts exactly two arguments.")
    left = _value_array(args[0], field="dot.left")
    right = _value_array(args[1], field="dot.right")
    if left.ndim not in {1, 2} or right.ndim not in {1, 2}:
        raise AnalysisRecipeError("dot accepts vectors and matrices only.")
    if left.shape[-1] != right.shape[0]:
        raise AnalysisRecipeError("dot inner dimensions do not match.")
    if left.ndim == 1 and right.ndim == 1:
        result_shape: tuple[int, ...] = ()
    elif left.ndim == 2 and right.ndim == 1:
        result_shape = (left.shape[0],)
    elif left.ndim == 1 and right.ndim == 2:
        result_shape = (right.shape[1],)
    else:
        result_shape = (left.shape[0], right.shape[1])
    result_count = _bounded_element_count(result_shape, field="dot")
    _bounded_linear_algebra_work(
        result_count * int(left.shape[-1]), field="dot"
    )
    return np.dot(left, right)


def _transpose(args: tuple[object, ...]) -> np.ndarray:
    if len(args) != 1:
        raise AnalysisRecipeError("transpose accepts exactly one argument.")
    return np.transpose(_value_array(args[0], field="transpose"))


def _trace(args: tuple[object, ...]) -> float:
    if len(args) != 1:
        raise AnalysisRecipeError("trace accepts exactly one argument.")
    value = _value_array(args[0], field="trace")
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise AnalysisRecipeError("trace requires a square matrix.")
    return float(np.trace(value))


def _determinant(args: tuple[object, ...]) -> float:
    if len(args) != 1:
        raise AnalysisRecipeError("det accepts exactly one argument.")
    value = _value_array(args[0], field="det")
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise AnalysisRecipeError("det requires a square matrix.")
    _bounded_linear_algebra_work(int(value.shape[0]) ** 3, field="det")
    return float(np.linalg.det(value))


_SAFE_FUNCTIONS = {
    **{name: (lambda args, name=name: _unary_numpy(name, args)) for name in (
        "abs", "sqrt", "exp", "log", "sin", "cos", "tan", "sinh", "cosh", "tanh",
    )},
    **{name: (lambda args, name=name: _reduce(name, args)) for name in (
        "sum", "mean", "std", "min", "max", "prod",
    )},
    "outer": _outer,
    "dot": _dot,
    "transpose": _transpose,
    "trace": _trace,
    "det": _determinant,
}


def _evaluate_expression_node(node: ast.AST, names: Mapping[str, object]) -> object:
    if isinstance(node, ast.Expression):
        return _evaluate_expression_node(node.body, names)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        if node.id == "e":
            return math.e
        if node.id not in names:
            raise AnalysisRecipeError(f"Unknown expression input '{node.id}'.")
        return names[node.id]
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_expression_node(node.operand, names)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate_expression_node(node.left, names)
        right = _evaluate_expression_node(node.right, names)
        with np.errstate(all="ignore"):
            if isinstance(node.op, ast.Add):
                left, right = _elementwise_operands(left, right, field="addition")
                result = left + right
            elif isinstance(node.op, ast.Sub):
                left, right = _elementwise_operands(left, right, field="subtraction")
                result = left - right
            elif isinstance(node.op, ast.Mult):
                left, right = _elementwise_operands(left, right, field="multiplication")
                result = left * right
            elif isinstance(node.op, ast.Div):
                left, right = _elementwise_operands(left, right, field="division")
                result = left / right
            elif isinstance(node.op, ast.MatMult):
                left, right = _matmul_operands(left, right)
                result = np.matmul(left, right)
            elif isinstance(node.op, ast.Pow):
                exponent = _normalised_value(right, field="power exponent")
                if isinstance(exponent, np.ndarray) or abs(exponent) > 64:
                    raise AnalysisRecipeError("Power exponents must be scalar and no larger than 64 in magnitude.")
                left, exponent_array = _elementwise_operands(
                    left, exponent, field="power"
                )
                result = np.power(left, exponent_array)
            else:  # pragma: no cover - AST validation makes this unreachable
                raise AnalysisRecipeError("Unsupported binary operator.")
        return _normalised_value(result, field="expression intermediate")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = _SAFE_FUNCTIONS.get(node.func.id)
        if function is None:
            raise AnalysisRecipeError(f"Expression function '{node.func.id}' is not installed.")
        arguments = tuple(_evaluate_expression_node(item, names) for item in node.args)
        with np.errstate(all="ignore"):
            return _normalised_value(function(arguments), field=f"function {node.func.id}")
    raise AnalysisRecipeError(f"Unsupported expression syntax '{type(node).__name__}'.")


def evaluate_safe_expression(expression: str, inputs: Mapping[str, object]) -> float | np.ndarray:
    """Evaluate a whitelisted numerical expression without Python ``eval`` or attribute access."""
    try:
        parsed = _validated_expression(expression, field="analysis expression")
    except ModelGraphError as exc:
        raise AnalysisRecipeError(str(exc)) from exc
    unknown = {
        node.id for node in ast.walk(parsed)
        if isinstance(node, ast.Name)
        and node.id not in set(inputs)
        and node.id not in _SAFE_FUNCTIONS
        and node.id not in {"pi", "e"}
    }
    if unknown:
        raise AnalysisRecipeError(
            "Expression contains unknown names: " + ", ".join(sorted(unknown)) + "."
        )
    return _normalised_value(
        _evaluate_expression_node(parsed, dict(inputs)), field="analysis expression"
    )


@dataclass(frozen=True, slots=True)
class ComposedStepResult:
    step_id: str
    operation: str
    inputs: Mapping[str, str]
    settings: Mapping[str, Any]
    value: float | np.ndarray
    value_sha256: str


@dataclass(frozen=True, slots=True)
class ComposedOutput:
    step_id: str
    value_sha256: str
    value: float | np.ndarray


@dataclass(frozen=True, slots=True)
class ComposedAnalysisResult:
    recipe_id: str
    steps: tuple[ComposedStepResult, ...]
    outputs: Mapping[str, ComposedOutput]
    views: tuple[Mapping[str, Any], ...]

    def value(self, step_id: str) -> float | np.ndarray:
        for step in self.steps:
            if step.step_id == step_id:
                return step.value
        raise KeyError(step_id)


def _step_digest(
    *, operation: str, inputs: Mapping[str, str], settings: Mapping[str, Any],
    input_hashes: Mapping[str, str], value: object,
) -> str:
    return canonical_json_sha256({
        "operation": operation,
        "inputs": dict(inputs),
        "input_hashes": dict(input_hashes),
        "settings": portable_value(settings),
        "value": portable_value(value),
    })


def _matrix_value(model: ModelIR, settings: Mapping[str, Any]) -> np.ndarray:
    result = analyse_matrix_function(
        model,
        point=tuple(float(item) for item in settings["point"]),
        function_name=str(settings["matrix"]),
        parameter_values={
            str(name): float(value)
            for name, value in dict(settings.get("parameter_values", {})).items()
        },
    )
    try:
        matrix = np.asarray(result.numerical_matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AnalysisRecipeError("matrix.evaluate currently requires a real matrix value.") from exc
    return _value_array(matrix, field="matrix.evaluate")


def _generalized_eigenvalues(
    matrix: object, metric: object | None, settings: Mapping[str, Any]
) -> np.ndarray:
    hessian = _value_array(matrix, field="generalized eigenvalue matrix")
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise AnalysisRecipeError("Generalized eigenvalues require a square matrix.")
    if hessian.shape[0] > 512:
        raise AnalysisRecipeError("Generalized eigenvalue matrices may not exceed 512 by 512.")
    geometry = np.eye(hessian.shape[0], dtype=np.float64) if metric is None else _value_array(
        metric, field="generalized eigenvalue metric"
    )
    if geometry.shape != hessian.shape:
        raise AnalysisRecipeError("The matrix and metric must have the same square shape.")
    symmetry_tolerance = float(settings.get("symmetry_tolerance", 1e-10))
    scale = max(
        1.0,
        float(np.max(np.abs(hessian), initial=0.0)),
        float(np.max(np.abs(geometry), initial=0.0)),
    )
    absolute_tolerance = symmetry_tolerance * scale
    if not np.allclose(hessian, hessian.T, rtol=0.0, atol=absolute_tolerance):
        raise AnalysisRecipeError("The generalized-eigenvalue matrix is not symmetric.")
    if not np.allclose(geometry, geometry.T, rtol=0.0, atol=absolute_tolerance):
        raise AnalysisRecipeError("The generalized-eigenvalue metric is not symmetric.")
    try:
        linalg.cholesky((geometry + geometry.T) / 2.0, lower=True, check_finite=True)
        eigenvalues = linalg.eigvalsh(
            (hessian + hessian.T) / 2.0,
            (geometry + geometry.T) / 2.0,
            check_finite=True,
        )
    except linalg.LinAlgError as exc:
        raise AnalysisRecipeError("The generalized-eigenvalue metric must be positive definite.") from exc
    tolerance = float(settings.get("positivity_tolerance", 1e-12)) * scale
    eigenvalues[np.abs(eigenvalues) <= tolerance] = 0.0
    if bool(settings.get("require_positive", False)) and np.any(eigenvalues <= tolerance):
        raise AnalysisRecipeError("The generalized spectrum is not strictly positive.")
    return _value_array(eigenvalues, field="generalized eigenvalues")


def _first_crossing(
    x_value: object, y_value: object, settings: Mapping[str, Any]
) -> float:
    x = _value_array(x_value, field="first-crossing x").reshape(-1)
    y = _value_array(y_value, field="first-crossing y").reshape(-1)
    if x.size != y.size or x.size < 2:
        raise AnalysisRecipeError("first-crossing requires aligned x and y vectors with at least two points.")
    if np.any(np.diff(x) <= 0.0):
        raise AnalysisRecipeError("first-crossing x values must be strictly increasing.")
    threshold = float(settings["threshold"])
    direction = str(settings.get("direction", "below"))
    selected = y <= threshold if direction == "below" else y >= threshold
    indices = np.flatnonzero(selected)
    if not indices.size:
        raise AnalysisRecipeError("The requested threshold is not crossed on the supplied series.")
    index = int(indices[0])
    if index == 0:
        return float(x[0])
    x0, x1 = float(x[index - 1]), float(x[index])
    y0, y1 = float(y[index - 1]), float(y[index])
    if y1 == y0:
        return x1
    fraction = (threshold - y0) / (y1 - y0)
    return float(x0 + min(1.0, max(0.0, fraction)) * (x1 - x0))


def _execute_step(
    model: ModelIR,
    operation: str,
    inputs: Mapping[str, object],
    settings: Mapping[str, Any],
) -> float | np.ndarray:
    if operation == "array.literal":
        return _normalised_value(settings["value"], field="array.literal")
    if operation == "array.linspace":
        return np.linspace(
            float(settings["start"]),
            float(settings["stop"]),
            int(settings["count"]),
            dtype=np.float64,
        )
    if operation == "array.logspace":
        return np.logspace(
            float(settings["start_exponent"]),
            float(settings["stop_exponent"]),
            int(settings["count"]),
            base=float(settings.get("base", 10.0)),
            dtype=np.float64,
        )
    if operation == "array.expression":
        return evaluate_safe_expression(str(settings["expression"]), inputs)
    if operation == "matrix.evaluate":
        return _matrix_value(model, settings)
    if operation == "matrix.generalized-eigenvalues":
        return _generalized_eigenvalues(inputs["matrix"], inputs.get("metric"), settings)
    if operation == "scalar.evaluate-points":
        result = evaluate_scalar_at_points(
            model,
            settings["points"],
            function_name=str(settings["function"]),
            parameter_values={
                str(name): float(value)
                for name, value in dict(settings.get("parameter_values", {})).items()
            },
        )
        if not np.all(np.isfinite(result.values)):
            raise AnalysisRecipeError("scalar.evaluate-points produced non-finite values.")
        return _value_array(result.values, field="scalar.evaluate-points")
    if operation == "series.first-crossing":
        return _first_crossing(inputs["x"], inputs["y"], settings)
    raise AnalysisRecipeError(f"Unsupported recipe operation '{operation}'.")


def run_analysis_recipe(
    model: ModelIR, object_id: str | None = None
) -> ComposedAnalysisResult:
    """Execute an ordered, declarative value graph using only installed primitives."""
    item = select_object(model, RECIPE_KIND, object_id, label="analysis recipe")
    values: dict[str, float | np.ndarray] = {}
    hashes: dict[str, str] = {}
    step_results: list[ComposedStepResult] = []
    stored_elements = 0
    for raw_step in item.properties["steps"]:
        step = dict(raw_step)
        step_id = str(step["id"])
        operation = str(step["operation"])
        input_references = {
            str(name): str(reference)
            for name, reference in dict(step.get("inputs", {})).items()
        }
        settings = dict(step.get("settings", {}))
        resolved_inputs = {name: values[reference] for name, reference in input_references.items()}
        try:
            value = _execute_step(model, operation, resolved_inputs, settings)
            value = _normalised_value(value, field=f"recipe step {step_id}")
            stored_elements += int(np.asarray(value).size)
            if stored_elements > MAX_RECIPE_STORED_ELEMENTS:
                raise AnalysisRecipeError(
                    "Recipe outputs exceed the bounded cumulative stored-value limit."
                )
        except (AnalysisRecipeError, KeyError, TypeError, ValueError) as exc:
            raise AnalysisRecipeError(f"Recipe '{item.identifier}' step '{step_id}' failed: {exc}") from exc
        digest = _step_digest(
            operation=operation,
            inputs=input_references,
            settings=settings,
            input_hashes={name: hashes[reference] for name, reference in input_references.items()},
            value=value,
        )
        values[step_id] = value
        hashes[step_id] = digest
        step_results.append(
            ComposedStepResult(
                step_id,
                operation,
                input_references,
                settings,
                value,
                digest,
            )
        )
    outputs = {
        str(label): ComposedOutput(str(reference), hashes[str(reference)], values[str(reference)])
        for label, reference in dict(item.properties["outputs"]).items()
    }
    return ComposedAnalysisResult(
        item.identifier,
        tuple(step_results),
        outputs,
        tuple(dict(view) for view in item.properties.get("views", [])),
    )


def _applicable(model: ModelIR) -> tuple[bool, str]:
    found = bool(objects_of_kind(model, RECIPE_KIND))
    return found, "" if found else "an executable declarative analysis recipe is required"


def _recipe_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(model, RECIPE_KIND, settings.get("object_id"), label="analysis recipe")
    step_shapes: dict[str, tuple[int, ...]] = {}
    total = 0
    for step in item.properties["steps"]:
        step_id = str(step["id"])
        operation = str(step["operation"])
        step_settings = dict(step.get("settings", {}))
        input_references = dict(step.get("inputs", {}))
        if operation == "matrix.evaluate":
            try:
                shape = tuple(
                    int(value)
                    for value in model.matrix_function(str(step_settings["matrix"])).shape
                )
            except KeyError:
                shape = (max(1, len(model.variables)),) * 2
            step_shapes[step_id] = shape
            total += max(1, max(shape, default=1) ** 3 * 10)
        elif operation == "matrix.generalized-eigenvalues":
            shape = step_shapes.get(str(input_references.get("matrix")), ())
            dimension = max(shape, default=max(1, len(model.variables)))
            step_shapes[step_id] = (dimension,)
            total += max(1, dimension ** 3 * 10)
        elif operation in {"array.linspace", "array.logspace"}:
            count = int(step_settings.get("count", 1))
            step_shapes[step_id] = (count,)
            total += count
        elif operation == "array.literal":
            value = np.asarray(step_settings.get("value"))
            step_shapes[step_id] = tuple(int(item) for item in value.shape)
            total += max(1, int(value.size))
        elif operation == "array.expression":
            total += MAX_VALUE_ELEMENTS
        else:
            total += max(1, len(model.variables) * 100)
    return total


COMPOSED_NUMERIC = "org.modellab.comparator.composed-analysis"


def _comparison_projection(value: object) -> object:
    """Remove only checksums derived from values compared elsewhere in the artifact."""
    if isinstance(value, Mapping):
        return {
            str(key): _comparison_projection(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if key != "value_sha256"
        }
    if isinstance(value, list):
        return [_comparison_projection(item) for item in value]
    return value


def _compare_composed_analysis(
    reference: ScientificArtifact,
    current: ScientificArtifact,
    rtol: float,
    atol: float,
) -> ComparisonOutcome:
    """Compare recipe semantics and values without treating derived hashes as inputs.

    Step and output hashes intentionally change when platform-level eigensolver rounding
    changes.  Strict reproduction still compares the complete artifact hash first; this
    tolerant path compares the underlying aligned values and all non-derived semantics.
    """
    return compare_numeric_data(
        _comparison_projection(reference.data),
        _comparison_projection(current.data),
        rtol=rtol,
        atol=atol,
        comparator_id=COMPOSED_NUMERIC,
        details={"ignored_derived_fields": ["value_sha256"]},
    )


comparator_registry.register(COMPOSED_NUMERIC, _compare_composed_analysis)

CAPABILITY_DESCRIPTORS = (
    CapabilityDescriptor(
        "org.modellab.composition.run-analysis-recipe",
        "1.0",
        PACK_ID,
        "Run declarative analysis recipe",
        "Compose bounded matrix, scalar, array, expression, threshold, and plotting primitives into a content-addressed scientific analysis graph.",
        {
            "type": "object",
            "properties": {"object_id": {"type": ["string", "null"]}},
        },
        "numpy+scipy",
        (
            ArtifactTypeDescriptor(
                "org.modellab.artifact.composed-analysis",
                "1.0",
                "Composed scientific analysis",
                COMPOSED_NUMERIC,
            ),
        ),
        _applicable,
        run_analysis_recipe,
        _recipe_units,
        ("org.modellab.renderer.plotly-composed-analysis",),
    ),
)


MANIFEST = PackManifest(
    PACK_ID,
    "1.0",
    "Declarative Analysis Composition",
    "Bounded reusable numerical primitives, safe derived expressions, content-addressed step graphs, and declarative multi-series figures.",
    ("org.modellab.pack.multidimensional-mathematics",),
    tuple(item.kind for item in KIND_DESCRIPTORS),
    tuple(item.identifier for item in CAPABILITY_DESCRIPTORS),
)
