"""Typed, extensible Model Graph primitives used by Model IR 3.0.

The graph is deliberately structural.  It records what model objects *are* and how they
relate without pretending that every discipline can be reduced to the scalar expression
AST.  Executable behaviour is supplied only by locally installed, registered capability
packs; an ``.mlab`` document never supplies executable plugin code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Iterable, Mapping


MODEL_GRAPH_SCHEMA = "model-laboratory-model-graph"
MODEL_GRAPH_SCHEMA_VERSION = "3.0"

_KIND_PATTERN = re.compile(
    r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)+(?:/[a-z][a-z0-9-]*)*$"
)
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,255}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ModelGraphError(ValueError):
    """Raised when graph structure or a kind registry is invalid."""


def validate_graph_identifier(value: str, *, field_name: str = "identifier") -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ModelGraphError(
            f"{field_name} must begin with a letter and contain at most 256 letters, "
            "digits, dots, colons, underscores, or hyphens."
        )
    return value


def validate_namespaced_kind(value: str) -> str:
    if not isinstance(value, str) or _KIND_PATTERN.fullmatch(value) is None:
        raise ModelGraphError(
            "Model-object kinds must be lowercase namespaced identifiers such as "
            "'org.modellab.core.variable'."
        )
    return value


def _json_value(value: Any, *, path: str = "properties", depth: int = 0) -> Any:
    """Validate and copy finite, bounded-depth JSON data."""
    if depth > 24:
        raise ModelGraphError(f"{path} exceeds the maximum nesting depth.")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ModelGraphError(f"{path} contains a non-finite number.")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or "\x00" in key:
                raise ModelGraphError(f"{path} contains an invalid object key.")
            result[key] = _json_value(item, path=f"{path}.{key}", depth=depth + 1)
        return result
    raise ModelGraphError(f"{path} contains a value that is not portable JSON data.")


def _schema_value(value: Any, schema: Mapping[str, Any], *, path: str) -> Any:
    """Validate the strict JSON-schema subset accepted for extension properties."""
    expected_type = schema.get("type")
    allowed_types = (expected_type,) if isinstance(expected_type, str) else tuple(expected_type or ())
    type_matches = {
        "null": value is None,
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "array": isinstance(value, (list, tuple)),
        "object": isinstance(value, Mapping),
    }
    if allowed_types and not any(type_matches.get(item, False) for item in allowed_types):
        raise ModelGraphError(f"{path} does not match extension property type {expected_type!r}.")
    if "enum" in schema and value not in schema["enum"]:
        raise ModelGraphError(f"{path} is outside the extension property's allowed values.")
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = set(schema.get("required", ()))
        if not isinstance(properties, Mapping) or not isinstance(schema.get("required", ()), (list, tuple)):
            raise ModelGraphError("Installed extension kind contains a malformed object schema.")
        missing = required - set(value)
        if missing:
            raise ModelGraphError(f"{path} is missing required properties: {', '.join(sorted(missing))}.")
        unknown = set(value) - set(properties)
        if schema.get("additionalProperties") is False and unknown:
            raise ModelGraphError(f"{path} contains unknown properties: {', '.join(sorted(unknown))}.")
        return {
            str(key): _schema_value(item, properties[key], path=f"{path}.{key}")
            if key in properties
            else _json_value(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        if len(value) < int(schema.get("minItems", 0)):
            raise ModelGraphError(f"{path} contains too few items.")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise ModelGraphError(f"{path} contains too many items.")
        item_schema = schema.get("items", {})
        if not isinstance(item_schema, Mapping):
            raise ModelGraphError("Installed extension kind contains a malformed array schema.")
        return [
            _schema_value(item, item_schema, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ModelGraphError(f"{path} is below its extension-schema minimum.")
        if "maximum" in schema and value > schema["maximum"]:
            raise ModelGraphError(f"{path} is above its extension-schema maximum.")
    return _json_value(value, path=path)


@dataclass(frozen=True, slots=True)
class DomainType:
    """A portable mathematical domain declaration used by Model Graph value types."""

    kind: str = "opaque"
    lower: int | float | None = None
    upper: int | float | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    values: tuple[Any, ...] = ()
    factors: tuple["DomainType", ...] = ()
    index_domain: "DomainType | None" = None
    value_domain: "DomainType | None" = None

    def __post_init__(self) -> None:
        allowed = {
            "real",
            "integer",
            "boolean",
            "complex",
            "interval",
            "discrete",
            "product",
            "indexed",
            "opaque",
        }
        if self.kind not in allowed:
            raise ModelGraphError(f"Unsupported domain kind '{self.kind}'.")
        for name, value in (("lower", self.lower), ("upper", self.upper)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))))
            ):
                raise ModelGraphError(f"Domain {name} must be a finite real number or null.")
        if self.kind == "interval":
            if self.lower is None or self.upper is None or self.lower >= self.upper:
                raise ModelGraphError("An interval domain requires finite lower < upper bounds.")
        elif self.lower is not None or self.upper is not None:
            raise ModelGraphError("Only interval domains may declare lower/upper bounds.")
        if self.kind == "discrete":
            if not self.values:
                raise ModelGraphError("A discrete domain requires at least one value.")
            object.__setattr__(self, "values", tuple(_json_value(item, path="domain.values") for item in self.values))
        elif self.values:
            raise ModelGraphError("Only discrete domains may declare values.")
        if self.kind == "product":
            if not self.factors:
                raise ModelGraphError("A product domain requires at least one factor.")
        elif self.factors:
            raise ModelGraphError("Only product domains may declare factors.")
        if self.kind == "indexed":
            if self.index_domain is None or self.value_domain is None:
                raise ModelGraphError("An indexed domain requires index_domain and value_domain.")
        elif self.index_domain is not None or self.value_domain is not None:
            raise ModelGraphError("Only indexed domains may declare index/value domains.")
        if _domain_depth(self) > 16:
            raise ModelGraphError("A domain declaration cannot exceed 16 nested levels.")

    def payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "lower": self.lower,
            "upper": self.upper,
            "lower_inclusive": self.lower_inclusive,
            "upper_inclusive": self.upper_inclusive,
            "values": [_json_value(item, path="domain.values") for item in self.values],
            "factors": [item.payload() for item in self.factors],
            "index_domain": None if self.index_domain is None else self.index_domain.payload(),
            "value_domain": None if self.value_domain is None else self.value_domain.payload(),
        }


def _domain_depth(value: DomainType) -> int:
    children = [*value.factors]
    if value.index_domain is not None:
        children.append(value.index_domain)
    if value.value_domain is not None:
        children.append(value.value_domain)
    return 1 + max((_domain_depth(item) for item in children), default=0)


@dataclass(frozen=True, slots=True)
class ValueType:
    """A portable shaped value declaration independent of a numerical backend."""

    kind: str = "real"
    shape: tuple[int | None, ...] = ()
    element_kind: str | None = None
    unit_dimension: str | None = None
    domain: DomainType | None = None
    input_types: tuple["ValueType", ...] = ()
    output_type: "ValueType | None" = None

    def __post_init__(self) -> None:
        allowed = {
            "real",
            "integer",
            "boolean",
            "complex",
            "categorical",
            "vector",
            "matrix",
            "tensor",
            "set",
            "graph",
            "function",
            "relation",
            "structured",
            "opaque",
        }
        if self.kind not in allowed:
            raise ModelGraphError(f"Unsupported value type '{self.kind}'.")
        if any(
            dimension is not None
            and (isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0)
            for dimension in self.shape
        ):
            raise ModelGraphError("Value-type dimensions must be non-negative integers or null.")
        expected_rank = {"vector": 1, "matrix": 2}
        if self.kind in expected_rank and len(self.shape) != expected_rank[self.kind]:
            raise ModelGraphError(
                f"Value type '{self.kind}' requires a rank-{expected_rank[self.kind]} shape."
            )
        if self.kind == "tensor" and not self.shape:
            raise ModelGraphError("A tensor value type requires a non-empty shape.")
        if self.kind in {"function", "relation"}:
            if self.output_type is None:
                raise ModelGraphError(f"Value type '{self.kind}' requires an output_type.")
            if self.shape or self.element_kind is not None:
                raise ModelGraphError(f"Value type '{self.kind}' cannot declare shape/element_kind.")
        elif self.input_types or self.output_type is not None:
            raise ModelGraphError("Only function/relation value types may declare input/output types.")
        if _value_type_depth(self) > 16:
            raise ModelGraphError("A value-type signature cannot exceed 16 nested levels.")

    def payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "shape": list(self.shape),
            "element_kind": self.element_kind,
            "unit_dimension": self.unit_dimension,
            "domain": None if self.domain is None else self.domain.payload(),
            "input_types": [item.payload() for item in self.input_types],
            "output_type": None if self.output_type is None else self.output_type.payload(),
        }


def _value_type_depth(value: ValueType) -> int:
    children = [*value.input_types]
    if value.output_type is not None:
        children.append(value.output_type)
    return 1 + max((_value_type_depth(item) for item in children), default=0)


@dataclass(frozen=True, slots=True)
class ModelObject:
    """One stable, typed object in a Model Graph."""

    identifier: str
    kind: str
    kind_version: str
    value_type: ValueType = field(default_factory=ValueType)
    properties: Mapping[str, Any] = field(default_factory=dict)
    references: tuple[str, ...] = ()
    executable: bool = False
    opaque: bool = False

    def __post_init__(self) -> None:
        validate_graph_identifier(self.identifier, field_name="Model-object identifier")
        validate_namespaced_kind(self.kind)
        if not isinstance(self.kind_version, str) or not self.kind_version.strip():
            raise ModelGraphError("Model-object kind_version must be a non-empty string.")
        for reference in self.references:
            validate_graph_identifier(reference, field_name="Model-object reference")
        object.__setattr__(self, "properties", _json_value(dict(self.properties)))
        if self.opaque and self.executable:
            raise ModelGraphError("Opaque model objects cannot be executable.")

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "kind": self.kind,
            "kind_version": self.kind_version,
            "value_type": self.value_type.payload(),
            "properties": _json_value(self.properties),
            "references": list(self.references),
            "executable": self.executable,
            "opaque": self.opaque,
        }


@dataclass(frozen=True, slots=True)
class ModelRelationship:
    """One typed directed relationship between graph objects."""

    identifier: str
    kind: str
    kind_version: str
    source: str
    target: str
    properties: Mapping[str, Any] = field(default_factory=dict)
    opaque: bool = False

    def __post_init__(self) -> None:
        validate_graph_identifier(self.identifier, field_name="Relationship identifier")
        validate_namespaced_kind(self.kind)
        validate_graph_identifier(self.source, field_name="Relationship source")
        validate_graph_identifier(self.target, field_name="Relationship target")
        if not isinstance(self.kind_version, str) or not self.kind_version.strip():
            raise ModelGraphError("Relationship kind_version must be a non-empty string.")
        object.__setattr__(self, "properties", _json_value(dict(self.properties)))

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "kind": self.kind,
            "kind_version": self.kind_version,
            "source": self.source,
            "target": self.target,
            "properties": _json_value(self.properties),
            "opaque": self.opaque,
        }


@dataclass(frozen=True, slots=True)
class StructuredAssetReference:
    """A non-expression model asset such as a mesh, table, image, or graph."""

    identifier: str
    kind: str
    kind_version: str
    media_type: str
    sha256: str | None = None
    size: int | None = None
    bundle_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    opaque: bool = False

    def __post_init__(self) -> None:
        validate_graph_identifier(self.identifier, field_name="Asset identifier")
        validate_namespaced_kind(self.kind)
        if not isinstance(self.kind_version, str) or not self.kind_version.strip():
            raise ModelGraphError("Asset kind_version must be a non-empty string.")
        if not isinstance(self.media_type, str) or "/" not in self.media_type:
            raise ModelGraphError("Asset media_type must be a MIME type.")
        if self.sha256 is not None and _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ModelGraphError("Asset sha256 must be a lowercase SHA-256 digest.")
        if self.size is not None and (
            isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0
        ):
            raise ModelGraphError("Asset size must be a non-negative integer.")
        if self.bundle_path is not None and (
            self.bundle_path.startswith("/")
            or ".." in self.bundle_path.split("/")
            or "\\" in self.bundle_path
        ):
            raise ModelGraphError("Asset bundle_path must be a safe relative path.")
        object.__setattr__(self, "metadata", _json_value(dict(self.metadata)))

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "kind": self.kind,
            "kind_version": self.kind_version,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "size": self.size,
            "bundle_path": self.bundle_path,
            "metadata": _json_value(self.metadata),
            "opaque": self.opaque,
        }


@dataclass(frozen=True, slots=True)
class ObjectKindDescriptor:
    """Locally installed schema/behaviour declaration for one object kind."""

    kind: str
    version: str
    title: str
    property_schema: Mapping[str, Any] = field(default_factory=dict)
    executable: bool = False

    def __post_init__(self) -> None:
        validate_namespaced_kind(self.kind)
        if not self.version.strip() or not self.title.strip():
            raise ModelGraphError("Kind descriptors require a version and title.")
        object.__setattr__(self, "property_schema", _json_value(dict(self.property_schema)))
        if self.property_schema:
            if self.property_schema.get("type") != "object":
                raise ModelGraphError("Extension property schemas must have object type.")
            if self.property_schema.get("additionalProperties") is not False:
                raise ModelGraphError(
                    "Installed extension property schemas must explicitly forbid unknown fields."
                )

    def validate_properties(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.property_schema:
            return _json_value(dict(value))
        checked = _schema_value(value, self.property_schema, path=f"{self.kind}.properties")
        if not isinstance(checked, Mapping):  # defensive: descriptor enforces object schema
            raise ModelGraphError("Extension properties did not validate as an object.")
        return checked


class ObjectKindRegistry:
    """Registry that controls which graph objects may be interpreted or executed."""

    def __init__(self, descriptors: Iterable[ObjectKindDescriptor] = ()) -> None:
        self._descriptors: dict[tuple[str, str], ObjectKindDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: ObjectKindDescriptor) -> None:
        key = (descriptor.kind, descriptor.version)
        if key in self._descriptors:
            raise ModelGraphError(
                f"Object kind '{descriptor.kind}' version '{descriptor.version}' is already registered."
            )
        self._descriptors[key] = descriptor

    def descriptor(self, kind: str, version: str) -> ObjectKindDescriptor | None:
        return self._descriptors.get((kind, version))

    def knows(self, kind: str, version: str) -> bool:
        return (kind, version) in self._descriptors

    @property
    def descriptors(self) -> tuple[ObjectKindDescriptor, ...]:
        return tuple(self._descriptors[key] for key in sorted(self._descriptors))

    def catalogue(self) -> list[dict[str, Any]]:
        return [
            {
                "kind": item.kind,
                "version": item.version,
                "title": item.title,
                "property_schema": _json_value(item.property_schema),
                "executable": item.executable,
            }
            for item in self.descriptors
        ]

    def validate_properties(
        self, kind: str, version: str, properties: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], bool, bool]:
        """Return validated properties, executable flag, and opacity.

        Unknown kinds are preserved after portable-JSON validation, marked opaque, and
        never made executable. Known kinds use their locally installed strict schema.
        """
        descriptor = self.descriptor(kind, version)
        if descriptor is None:
            return _json_value(dict(properties)), False, True
        return descriptor.validate_properties(properties), descriptor.executable, False


CORE_KIND_REGISTRY = ObjectKindRegistry(
    (
        ObjectKindDescriptor("org.modellab.core.variable", "1.0", "Scalar variable", executable=True),
        ObjectKindDescriptor("org.modellab.core.parameter", "1.0", "Scalar parameter", executable=True),
        ObjectKindDescriptor("org.modellab.core.constant", "1.0", "Scalar constant", executable=True),
        ObjectKindDescriptor("org.modellab.core.derived-quantity", "1.0", "Derived quantity", executable=True),
        ObjectKindDescriptor("org.modellab.core.scalar-function", "1.0", "Scalar function", executable=True),
        ObjectKindDescriptor("org.modellab.core.vector-function", "1.0", "Vector function", executable=True),
        ObjectKindDescriptor("org.modellab.core.matrix-function", "1.0", "Matrix function", executable=True),
        ObjectKindDescriptor("org.modellab.core.constraint", "1.0", "Constraint", executable=True),
        ObjectKindDescriptor("org.modellab.core.depends-on", "1.0", "Dependency relationship"),
    )
)


@dataclass(frozen=True, slots=True)
class ModelGraph:
    """Canonical structural graph embedded in Model IR 3.0."""

    objects: tuple[ModelObject, ...] = ()
    relationships: tuple[ModelRelationship, ...] = ()
    assets: tuple[StructuredAssetReference, ...] = ()

    def __post_init__(self) -> None:
        object_ids = [item.identifier for item in self.objects]
        relationship_ids = [item.identifier for item in self.relationships]
        asset_ids = [item.identifier for item in self.assets]
        if len(object_ids) != len(set(object_ids)):
            raise ModelGraphError("Model Graph contains duplicate object identifiers.")
        if len(relationship_ids) != len(set(relationship_ids)):
            raise ModelGraphError("Model Graph contains duplicate relationship identifiers.")
        if len(asset_ids) != len(set(asset_ids)):
            raise ModelGraphError("Model Graph contains duplicate asset identifiers.")
        known = set(object_ids) | set(asset_ids)
        for item in self.objects:
            unknown = set(item.references) - known
            if unknown:
                raise ModelGraphError(
                    f"Model object '{item.identifier}' references unknown identifier(s): "
                    + ", ".join(sorted(unknown))
                    + "."
                )
        for relationship in self.relationships:
            unknown = {relationship.source, relationship.target} - known
            if unknown:
                raise ModelGraphError(
                    f"Relationship '{relationship.identifier}' has unknown endpoint(s): "
                    + ", ".join(sorted(unknown))
                    + "."
                )

    def object(self, identifier: str) -> ModelObject:
        for item in self.objects:
            if item.identifier == identifier:
                return item
        raise KeyError(identifier)

    @property
    def opaque_objects(self) -> tuple[ModelObject, ...]:
        return tuple(item for item in self.objects if item.opaque)

    @property
    def unavailable_extension_kinds(self) -> tuple[tuple[str, str], ...]:
        kinds = {(item.kind, item.kind_version) for item in self.opaque_objects}
        kinds.update(
            (item.kind, item.kind_version) for item in self.relationships if item.opaque
        )
        kinds.update((item.kind, item.kind_version) for item in self.assets if item.opaque)
        return tuple(sorted(kinds))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": MODEL_GRAPH_SCHEMA,
            "schema_version": MODEL_GRAPH_SCHEMA_VERSION,
            "objects": [item.payload() for item in sorted(self.objects, key=lambda item: item.identifier)],
            "relationships": [
                item.payload() for item in sorted(self.relationships, key=lambda item: item.identifier)
            ],
            "assets": [item.payload() for item in sorted(self.assets, key=lambda item: item.identifier)],
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
