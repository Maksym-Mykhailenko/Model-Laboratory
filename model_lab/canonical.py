"""Canonical serialisation helpers for Model Laboratory.

The canonical Model IR is owned by Model Laboratory.  Persisted mathematical expressions
use the versioned Model Laboratory expression AST and never SymPy's internal ``srepr``.
SymPy may change independently without changing the permanent model-file representation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .expression import (
    BinaryOperation,
    FunctionCall,
    IntegerLiteral,
    NamedConstant,
    RealLiteral,
    SymbolReference,
    UnaryOperation,
    PREVIOUS_EXPRESSION_AST_SCHEMA_VERSION,
    expression_ast_from_payload,
    expression_ast_payload,
)
from .model import ModelIR
from .provenance import Provenance


CANONICAL_MODEL_IR_SCHEMA = "model-laboratory-model-ir"
CANONICAL_MODEL_IR_SCHEMA_VERSION = "3.0"
PREVIOUS_CANONICAL_MODEL_IR_SCHEMA_VERSION = "2.1"
HYPERBOLIC_CANONICAL_MODEL_IR_SCHEMA_VERSION = "2.2"
DECIMAL_CONTEXT_CANONICAL_MODEL_IR_SCHEMA_VERSION = "2.0"
LEGACY_CANONICAL_MODEL_IR_SCHEMA_VERSION = "1.0"


def canonical_json_bytes(value: object) -> bytes:
    """Return the deterministic UTF-8 JSON encoding used for canonical fingerprints."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    """Return SHA-256 over :func:`canonical_json_bytes`."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def pretty_json_bytes(value: object) -> bytes:
    """Return stable, human-readable JSON bytes for portable bundle members."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def provenance_payload(provenance: Provenance) -> dict[str, object]:
    """Return a complete JSON-compatible provenance record."""
    return {
        "kind": provenance.kind.value,
        "source_location": provenance.source_location,
        "source_refs": list(provenance.source_refs),
        "operation": provenance.operation,
        "approval": provenance.approval.value,
        "note": provenance.note,
    }


def _element_metadata_payload(metadata: object) -> dict[str, object]:
    return {
        "label": getattr(metadata, "label", None),
        "description": getattr(metadata, "description", None),
        "unit": getattr(metadata, "unit", None),
        "tags": list(getattr(metadata, "tags", ())),
    }


def _domain_payload(domain: object) -> dict[str, object]:
    return {
        "lower": float(getattr(domain, "lower")),
        "upper": float(getattr(domain, "upper")),
        "lower_inclusive": bool(getattr(domain, "lower_inclusive")),
        "upper_inclusive": bool(getattr(domain, "upper_inclusive")),
    }


def canonical_model_ir_payload(model: ModelIR) -> dict[str, Any]:
    """Return the complete, library-independent canonical representation of ``model``."""
    return {
        "schema": CANONICAL_MODEL_IR_SCHEMA,
        "schema_version": CANONICAL_MODEL_IR_SCHEMA_VERSION,
        "name": model.name,
        "metadata": {
            "description": model.metadata.description,
            "notes": model.metadata.notes,
            "tags": list(model.metadata.tags),
        },
        "variables": [
            {
                "name": item.name,
                "domain": _domain_payload(item.domain),
                "initial_value": item.initial_value,
                "metadata": _element_metadata_payload(item.metadata),
                "provenance": provenance_payload(item.provenance),
            }
            for item in model.variables
        ],
        "parameters": [
            {
                "name": item.name,
                "default": item.default,
                "domain": _domain_payload(item.domain),
                "metadata": _element_metadata_payload(item.metadata),
                "provenance": provenance_payload(item.provenance),
            }
            for item in model.parameters
        ],
        "constants": [
            {
                "name": item.name,
                "value": item.value,
                "metadata": _element_metadata_payload(item.metadata),
                "provenance": provenance_payload(item.provenance),
            }
            for item in model.constants
        ],
        "derived_quantities": [
            {
                "name": item.name,
                "expression": expression_ast_payload(item.expression_ast),
                "source": item.source,
                "dependencies": list(item.dependencies),
                "metadata": _element_metadata_payload(item.metadata),
                "provenance": provenance_payload(item.provenance),
            }
            for item in model.derived_quantities
        ],
        "functions": [
            {
                "name": item.name,
                "expression": expression_ast_payload(item.expression_ast),
                "source": item.source,
                "dependencies": list(item.dependencies),
                "metadata": _element_metadata_payload(item.metadata),
                "provenance": provenance_payload(item.provenance),
            }
            for item in model.functions
        ],
        "vector_functions": [
            {
                "name": item.name,
                "components": [
                    {
                        "name": component.name,
                        "expression": expression_ast_payload(component.expression_ast),
                        "source": component.source,
                        "dependencies": list(component.dependencies),
                        "metadata": _element_metadata_payload(component.metadata),
                        "provenance": provenance_payload(component.provenance),
                    }
                    for component in item.components
                ],
                "metadata": _element_metadata_payload(item.metadata),
                "provenance": provenance_payload(item.provenance),
            }
            for item in model.vector_functions
        ],
        "matrix_functions": [
            {
                "name": item.name,
                "entries": [
                    [
                        {
                            "name": component.name,
                            "expression": expression_ast_payload(component.expression_ast),
                            "source": component.source,
                            "dependencies": list(component.dependencies),
                            "metadata": _element_metadata_payload(component.metadata),
                            "provenance": provenance_payload(component.provenance),
                        }
                        for component in row
                    ]
                    for row in item.entries
                ],
                "row_labels": list(item.row_labels),
                "column_labels": list(item.column_labels),
                "metadata": _element_metadata_payload(item.metadata),
                "provenance": provenance_payload(item.provenance),
            }
            for item in model.matrix_functions
        ],
        "constraints": [
            {
                "name": item.name,
                "left": expression_ast_payload(item.left_ast),
                "relation": item.relation.value,
                "right": expression_ast_payload(item.right_ast),
                "left_source": item.left_source,
                "right_source": item.right_source,
                "dependencies": list(item.dependencies),
                "metadata": _element_metadata_payload(item.metadata),
                "provenance": provenance_payload(item.provenance),
            }
            for item in model.constraints
        ],
        "assumptions": [
            {
                "name": item.name,
                "statement": item.statement,
                "affects": list(item.affects),
                "provenance": provenance_payload(item.provenance),
            }
            for item in model.assumptions
        ],
        "ambiguities": [
            {
                "name": item.name,
                "statement": item.statement,
                "options": list(item.options),
                "resolution": item.resolution,
                "blocking": item.blocking,
                "affects": list(item.affects),
                "provenance": provenance_payload(item.provenance),
            }
            for item in model.ambiguities
        ],
        "graph": model.graph.payload(),
    }


def canonical_model_ir_sha256(model: ModelIR) -> str:
    """Return the stable Model Laboratory semantic fingerprint of a validated model."""
    return canonical_json_sha256(canonical_model_ir_payload(model))


def _strip_legacy_expression_backend_fields(value: object) -> object:
    """Remove SymPy-owned expression printer data from a legacy Model IR document.

    Version-1 canonical Model IR documents stored ``sympy_srepr`` and ``normalised``
    strings.  Both were produced by SymPy and are therefore unsuitable as permanent
    cross-version identity.  Legacy migration validates every Model Laboratory-owned
    field while treating those expression printer fields as archival backend metadata.
    """
    if isinstance(value, list):
        return [_strip_legacy_expression_backend_fields(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"sympy_srepr", "normalised"}:
            if not all(isinstance(value[key], str) for key in ("sympy_srepr", "normalised")):
                raise ValueError("Legacy expression representation is malformed.")
            return {"legacy_expression_backend": True}
        return {
            str(key): _strip_legacy_expression_backend_fields(item)
            for key, item in value.items()
        }
    return value


def legacy_model_ir_projection_from_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the backend-independent projection of a legacy canonical IR document."""
    if document.get("schema") != CANONICAL_MODEL_IR_SCHEMA:
        raise ValueError("Unsupported legacy canonical Model IR schema.")
    if document.get("schema_version") != LEGACY_CANONICAL_MODEL_IR_SCHEMA_VERSION:
        raise ValueError("Unsupported legacy canonical Model IR schema version.")
    projected = _strip_legacy_expression_backend_fields(dict(document))
    assert isinstance(projected, dict)
    return projected


def legacy_model_ir_projection_from_model(model: ModelIR) -> dict[str, Any]:
    """Build the Model Laboratory-owned fields expected from a v1 canonical IR.

    Expression values are represented only by a placeholder because their former
    ``srepr``/pretty-printer contents were backend-owned.  Source, dependencies, metadata,
    provenance and all non-expression structure remain strictly checked.
    """
    placeholder = {"legacy_expression_backend": True}
    return {
        "schema": CANONICAL_MODEL_IR_SCHEMA,
        "schema_version": LEGACY_CANONICAL_MODEL_IR_SCHEMA_VERSION,
        "name": model.name,
        "metadata": {
            "description": model.metadata.description,
            "notes": model.metadata.notes,
            "tags": list(model.metadata.tags),
        },
        "variables": [
            {
                "name": item.name,
                "domain": _domain_payload(item.domain),
                "initial_value": item.initial_value,
                "metadata": _element_metadata_payload(item.metadata),
                "provenance": provenance_payload(item.provenance),
            }
            for item in model.variables
        ],
        "parameters": [
            {
                "name": item.name,
                "default": item.default,
                "domain": _domain_payload(item.domain),
                "metadata": _element_metadata_payload(item.metadata),
                "provenance": provenance_payload(item.provenance),
            }
            for item in model.parameters
        ],
        "constants": [
            {
                "name": item.name,
                "value": item.value,
                "metadata": _element_metadata_payload(item.metadata),
                "provenance": provenance_payload(item.provenance),
            }
            for item in model.constants
        ],
        "derived_quantities": [
            {
                "name": item.name,
                "expression": dict(placeholder),
                "source": item.source,
                "dependencies": list(item.dependencies),
                "metadata": _element_metadata_payload(item.metadata),
                "provenance": provenance_payload(item.provenance),
            }
            for item in model.derived_quantities
        ],
        "functions": [
            {
                "name": item.name,
                "expression": dict(placeholder),
                "source": item.source,
                "dependencies": list(item.dependencies),
                "metadata": _element_metadata_payload(item.metadata),
                "provenance": provenance_payload(item.provenance),
            }
            for item in model.functions
        ],
        "constraints": [
            {
                "name": item.name,
                "left": dict(placeholder),
                "relation": item.relation.value,
                "right": dict(placeholder),
                "left_source": item.left_source,
                "right_source": item.right_source,
                "dependencies": list(item.dependencies),
                "metadata": _element_metadata_payload(item.metadata),
                "provenance": provenance_payload(item.provenance),
            }
            for item in model.constraints
        ],
        "assumptions": [
            {
                "name": item.name,
                "statement": item.statement,
                "affects": list(item.affects),
                "provenance": provenance_payload(item.provenance),
            }
            for item in model.assumptions
        ],
        "ambiguities": [
            {
                "name": item.name,
                "statement": item.statement,
                "options": list(item.options),
                "resolution": item.resolution,
                "blocking": item.blocking,
                "affects": list(item.affects),
                "provenance": provenance_payload(item.provenance),
            }
            for item in model.ambiguities
        ],
    }


def legacy_model_ir_document_matches_model(
    document: Mapping[str, Any],
    model: ModelIR,
) -> bool:
    """Whether a v1 canonical Model IR matches ``model`` ignoring SymPy-owned printers."""
    try:
        return legacy_model_ir_projection_from_document(document) == legacy_model_ir_projection_from_model(model)
    except (TypeError, ValueError):
        return False


def _use_previous_expression_schema(document: dict[str, Any]) -> dict[str, Any]:
    """Project current compact expression payloads onto historical AST schema 1.1."""
    try:
        for item in document["derived_quantities"]:
            item["expression"]["schema_version"] = PREVIOUS_EXPRESSION_AST_SCHEMA_VERSION
        for item in document["functions"]:
            item["expression"]["schema_version"] = PREVIOUS_EXPRESSION_AST_SCHEMA_VERSION
        for item in document["constraints"]:
            item["left"]["schema_version"] = PREVIOUS_EXPRESSION_AST_SCHEMA_VERSION
            item["right"]["schema_version"] = PREVIOUS_EXPRESSION_AST_SCHEMA_VERSION
    except (KeyError, TypeError) as exc:
        raise ValueError("Canonical Model IR 2.1 document is malformed.") from exc
    return document


def compact_model_ir_projection_from_model(model: ModelIR) -> dict[str, Any]:
    """Build the exact Model IR 2.1 / expression-AST 1.1 representation."""
    projected = canonical_model_ir_payload(model)
    projected["schema_version"] = PREVIOUS_CANONICAL_MODEL_IR_SCHEMA_VERSION
    # Model IR 2.1 predates first-class vector and matrix outputs.
    projected.pop("vector_functions", None)
    projected.pop("matrix_functions", None)
    projected.pop("graph", None)
    return _use_previous_expression_schema(projected)


def compact_model_ir_document_matches_model(
    document: Mapping[str, Any],
    model: ModelIR,
) -> bool:
    """Whether a historical Model IR 2.1 document matches the reconstructed model."""
    try:
        if document.get("schema") != CANONICAL_MODEL_IR_SCHEMA:
            return False
        if document.get("schema_version") != PREVIOUS_CANONICAL_MODEL_IR_SCHEMA_VERSION:
            return False
        return dict(document) == compact_model_ir_projection_from_model(model)
    except (TypeError, ValueError):
        return False


def hyperbolic_model_ir_projection_from_model(model: ModelIR) -> dict[str, Any]:
    """Build historical Model IR 2.2 / expression-AST 1.2 representation."""
    projected = canonical_model_ir_payload(model)
    projected["schema_version"] = HYPERBOLIC_CANONICAL_MODEL_IR_SCHEMA_VERSION
    projected.pop("vector_functions", None)
    projected.pop("matrix_functions", None)
    projected.pop("graph", None)
    return projected


def hyperbolic_model_ir_document_matches_model(
    document: Mapping[str, Any], model: ModelIR
) -> bool:
    try:
        return (
            document.get("schema") == CANONICAL_MODEL_IR_SCHEMA
            and document.get("schema_version")
            == HYPERBOLIC_CANONICAL_MODEL_IR_SCHEMA_VERSION
            and dict(document) == hyperbolic_model_ir_projection_from_model(model)
        )
    except (TypeError, ValueError):
        return False


def _upgrade_expression_payload(value: object) -> dict[str, Any]:
    """Migrate a versioned expression payload to the current canonical AST format."""
    return expression_ast_payload(expression_ast_from_payload(value))


def _expression_shape_without_real_values(node: object) -> dict[str, Any]:
    """Return expression structure while masking legacy real-literal digits.

    Expression-AST 1.0 real literals were canonicalised through the mutable Decimal
    context.  Their stored digits can therefore differ from the exact source token even
    when the source is unchanged.  During format-1.1 migration we preserve every other
    part of the Model Laboratory expression structure and the exact source string while
    treating only the affected real-literal digits/exponent as historical metadata.
    Integer-vs-real type remains significant.
    """
    if isinstance(node, IntegerLiteral):
        return {"type": "integer", "value": node.value}
    if isinstance(node, RealLiteral):
        return {"type": "real", "legacy_decimal_value": True}
    if isinstance(node, SymbolReference):
        return {"type": "symbol", "name": node.name}
    if isinstance(node, NamedConstant):
        return {"type": "constant", "name": node.name}
    if isinstance(node, UnaryOperation):
        return {
            "type": "unary",
            "operator": node.operator,
            "operand": _expression_shape_without_real_values(node.operand),
        }
    if isinstance(node, BinaryOperation):
        return {
            "type": "binary",
            "operator": node.operator,
            "left": _expression_shape_without_real_values(node.left),
            "right": _expression_shape_without_real_values(node.right),
        }
    if isinstance(node, FunctionCall):
        return {
            "type": "call",
            "name": node.name,
            "arguments": [
                _expression_shape_without_real_values(item) for item in node.arguments
            ],
        }
    raise ValueError("Unsupported expression node in migration projection.")


def _previous_expression_shape(value: object) -> dict[str, Any]:
    return _expression_shape_without_real_values(expression_ast_from_payload(value))


def _project_model_document_expression_shapes(document: dict[str, Any]) -> dict[str, Any]:
    try:
        for item in document["derived_quantities"]:
            item["expression"] = _previous_expression_shape(item["expression"])
        for item in document["functions"]:
            item["expression"] = _previous_expression_shape(item["expression"])
        for item in document["constraints"]:
            item["left"] = _previous_expression_shape(item["left"])
            item["right"] = _previous_expression_shape(item["right"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Previous canonical Model IR document is malformed.") from exc
    return document


def previous_model_ir_projection_from_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable migration projection of a canonical Model IR 2.0 document.

    Model IR 2.0 used expression-AST 1.0.  Because AST-1.0 real-literal digits could be
    rounded by the mutable Decimal context, migration verifies the exact model source,
    dependency metadata and complete expression structure while masking only historical
    real-literal digits.  This allows affected 1.1 bundles to open rather than making the
    old canonicalisation bug permanent.
    """
    if document.get("schema") != CANONICAL_MODEL_IR_SCHEMA:
        raise ValueError("Unsupported previous canonical Model IR schema.")
    if document.get("schema_version") != DECIMAL_CONTEXT_CANONICAL_MODEL_IR_SCHEMA_VERSION:
        raise ValueError("Unsupported previous canonical Model IR schema version.")
    projected = json.loads(json.dumps(document, ensure_ascii=False, allow_nan=False))
    return _project_model_document_expression_shapes(projected)


def previous_model_ir_projection_from_model(model: ModelIR) -> dict[str, Any]:
    """Build the Model IR 2.0 migration projection expected from ``model``."""
    projected = canonical_model_ir_payload(model)
    projected["schema_version"] = DECIMAL_CONTEXT_CANONICAL_MODEL_IR_SCHEMA_VERSION
    projected.pop("vector_functions", None)
    projected.pop("matrix_functions", None)
    projected.pop("graph", None)
    return _project_model_document_expression_shapes(projected)


def previous_model_ir_document_matches_model(
    document: Mapping[str, Any],
    model: ModelIR,
) -> bool:
    """Whether a Model IR 2.0 document matches ``model`` under safe AST-1.0 migration."""
    try:
        return previous_model_ir_projection_from_document(document) == previous_model_ir_projection_from_model(model)
    except (TypeError, ValueError):
        return False
