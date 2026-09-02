"""Pydantic schema for Model Laboratory model files."""

from __future__ import annotations

import keyword
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str) -> str:
    """Require a simple identifier that can safely represent a mathematical symbol."""
    if not _NAME_PATTERN.fullmatch(name) or keyword.iskeyword(name):
        raise ValueError(
            f"'{name}' is not a valid model identifier. Use letters, digits and underscores, "
            "and do not begin with a digit."
        )
    return name


def _validate_finite_interval(domain: tuple[float, float]) -> tuple[float, float]:
    lower, upper = domain
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("Domain bounds must be finite numbers in the current model schema.")
    if lower >= upper:
        raise ValueError("The lower domain bound must be smaller than the upper bound.")
    return domain


def _normalise_tags(tags: tuple[str, ...]) -> tuple[str, ...]:
    cleaned: list[str] = []
    for tag in tags:
        stripped = tag.strip()
        if not stripped:
            raise ValueError("Tags cannot be empty.")
        if stripped not in cleaned:
            cleaned.append(stripped)
    return tuple(cleaned)


class ModelMetadataSpec(BaseModel):
    """Optional human-facing metadata for a model document."""

    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    notes: str | None = None
    tags: tuple[str, ...] = ()

    @field_validator("description", "notes")
    @classmethod
    def normalise_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: tuple[str, ...]) -> tuple[str, ...]:
        return _normalise_tags(tags)


class ElementAnnotationsSpec(BaseModel):
    """Optional labels and descriptive information with no mathematical effect."""

    model_config = ConfigDict(extra="forbid")

    label: str | None = None
    description: str | None = None
    unit: str | None = None
    tags: tuple[str, ...] = ()

    @field_validator("label", "description", "unit")
    @classmethod
    def normalise_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: tuple[str, ...]) -> tuple[str, ...]:
        return _normalise_tags(tags)


class VariableSpec(ElementAnnotationsSpec):
    """External specification for one continuous scalar variable."""

    domain: tuple[float, float]
    initial: float | None = None

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, domain: tuple[float, float]) -> tuple[float, float]:
        return _validate_finite_interval(domain)

    @field_validator("initial")
    @classmethod
    def validate_initial_is_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("An initial value must be finite.")
        return value

    @model_validator(mode="after")
    def validate_initial_inside_domain(self) -> "VariableSpec":
        if self.initial is not None:
            lower, upper = self.domain
            if not lower <= self.initial <= upper:
                raise ValueError(
                    f"Initial value {self.initial} lies outside the variable domain "
                    f"[{lower}, {upper}]."
                )
        return self


class ParameterSpec(ElementAnnotationsSpec):
    """External specification for one adjustable scalar parameter."""

    default: float
    domain: tuple[float, float]

    @field_validator("default")
    @classmethod
    def validate_default_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("A parameter default must be finite.")
        return value

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, domain: tuple[float, float]) -> tuple[float, float]:
        return _validate_finite_interval(domain)

    @model_validator(mode="after")
    def validate_default(self) -> "ParameterSpec":
        lower, upper = self.domain
        if not lower <= self.default <= upper:
            raise ValueError(
                f"Default value {self.default} lies outside the parameter domain "
                f"[{lower}, {upper}]."
            )
        return self


class ConstantSpec(ElementAnnotationsSpec):
    """External specification for one fixed scalar constant."""

    value: float

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("A constant value must be finite.")
        return value


class FunctionSpec(ElementAnnotationsSpec):
    """Rich external specification for one scalar function."""

    expression: str

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, expression: str) -> str:
        if not expression.strip():
            raise ValueError("A function expression cannot be empty.")
        return expression.strip()


class VectorFunctionSpec(ElementAnnotationsSpec):
    """External specification for one named vector-valued function."""

    components: dict[str, str | FunctionSpec]

    @field_validator("components")
    @classmethod
    def validate_components(
        cls, components: dict[str, str | FunctionSpec]
    ) -> dict[str, str | FunctionSpec]:
        if not components:
            raise ValueError("A vector function requires at least one component.")
        for name, specification in components.items():
            _validate_identifier(name)
            if isinstance(specification, str) and not specification.strip():
                raise ValueError(f"Vector component '{name}' has an empty expression.")
        return components


class MatrixFunctionSpec(ElementAnnotationsSpec):
    """External specification for one rectangular matrix-valued function."""

    entries: tuple[tuple[str, ...], ...]
    row_labels: tuple[str, ...] = ()
    column_labels: tuple[str, ...] = ()

    @field_validator("entries")
    @classmethod
    def validate_entries(cls, entries: tuple[tuple[str, ...], ...]) -> tuple[tuple[str, ...], ...]:
        if not entries or not entries[0]:
            raise ValueError("A matrix function requires a non-empty rectangular entry array.")
        columns = len(entries[0])
        if any(len(row) != columns for row in entries):
            raise ValueError("Matrix-function entries must form a rectangular array.")
        cleaned: list[tuple[str, ...]] = []
        for row in entries:
            cleaned_row: list[str] = []
            for expression in row:
                if not isinstance(expression, str) or not expression.strip():
                    raise ValueError("Matrix-function expressions cannot be empty.")
                cleaned_row.append(expression.strip())
            cleaned.append(tuple(cleaned_row))
        return tuple(cleaned)

    @field_validator("row_labels", "column_labels")
    @classmethod
    def validate_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("Matrix row and column labels cannot be empty.")
        if len(values) != len(set(values)):
            raise ValueError("Matrix row and column labels must be unique within an axis.")
        return tuple(value.strip() for value in values)

    @model_validator(mode="after")
    def validate_label_lengths(self) -> "MatrixFunctionSpec":
        if self.row_labels and len(self.row_labels) != len(self.entries):
            raise ValueError("row_labels must contain one label per matrix row.")
        if self.column_labels and len(self.column_labels) != len(self.entries[0]):
            raise ValueError("column_labels must contain one label per matrix column.")
        return self


class DomainTypeSpec(BaseModel):
    """Strict portable domain declaration for a Model Graph value type."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "real",
        "integer",
        "boolean",
        "complex",
        "interval",
        "discrete",
        "product",
        "indexed",
        "opaque",
    ] = "opaque"
    lower: int | float | None = None
    upper: int | float | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    values: tuple[str | int | float | bool | None, ...] = ()
    factors: tuple["DomainTypeSpec", ...] = ()
    index_domain: "DomainTypeSpec | None" = None
    value_domain: "DomainTypeSpec | None" = None

    @model_validator(mode="after")
    def validate_domain_shape(self) -> "DomainTypeSpec":
        if self.kind == "interval":
            if self.lower is None or self.upper is None or self.lower >= self.upper:
                raise ValueError("An interval domain requires finite lower < upper bounds.")
        elif self.lower is not None or self.upper is not None:
            raise ValueError("Only interval domains may declare lower/upper bounds.")
        if self.kind == "discrete":
            if not self.values:
                raise ValueError("A discrete domain requires at least one value.")
        elif self.values:
            raise ValueError("Only discrete domains may declare values.")
        if self.kind == "product":
            if not self.factors:
                raise ValueError("A product domain requires at least one factor.")
        elif self.factors:
            raise ValueError("Only product domains may declare factors.")
        if self.kind == "indexed":
            if self.index_domain is None or self.value_domain is None:
                raise ValueError("An indexed domain requires index_domain and value_domain.")
        elif self.index_domain is not None or self.value_domain is not None:
            raise ValueError("Only indexed domains may declare index/value domains.")
        return self


class ValueTypeSpec(BaseModel):
    """Portable shaped type attached to a structured model object."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
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
    ] = "opaque"
    shape: tuple[int | None, ...] = ()
    element_kind: str | None = None
    unit_dimension: str | None = None
    domain: DomainTypeSpec | None = None
    input_types: tuple["ValueTypeSpec", ...] = ()
    output_type: "ValueTypeSpec | None" = None

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
        if len(shape) > 16:
            raise ValueError("A shaped type cannot have more than 16 dimensions.")
        if any(value is not None and value < 0 for value in shape):
            raise ValueError("Shape dimensions must be non-negative integers or null.")
        return shape

    @model_validator(mode="after")
    def validate_signature(self) -> "ValueTypeSpec":
        expected_rank = {"vector": 1, "matrix": 2}
        if self.kind in expected_rank and len(self.shape) != expected_rank[self.kind]:
            raise ValueError(f"Value type '{self.kind}' requires rank {expected_rank[self.kind]}.")
        if self.kind == "tensor" and not self.shape:
            raise ValueError("A tensor value type requires a non-empty shape.")
        if self.kind in {"function", "relation"}:
            if self.output_type is None:
                raise ValueError(f"Value type '{self.kind}' requires output_type.")
            if self.shape or self.element_kind is not None:
                raise ValueError(f"Value type '{self.kind}' cannot declare shape/element_kind.")
        elif self.input_types or self.output_type is not None:
            raise ValueError("Only function/relation value types may declare input/output types.")
        return self


class ModelObjectSpec(BaseModel):
    """One extension object whose semantics are owned by its namespaced kind."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    kind_version: str = "1.0"
    value_type: ValueTypeSpec = Field(default_factory=ValueTypeSpec)
    properties: dict[str, Any] = Field(default_factory=dict)
    references: tuple[str, ...] = ()

    @field_validator("kind", "kind_version")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Object kind and kind_version cannot be empty.")
        return value.strip()


class ModelRelationshipSpec(BaseModel):
    """One extension relationship between model objects or assets."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    kind_version: str = "1.0"
    source: str
    target: str
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind", "kind_version", "source", "target")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Relationship fields cannot be empty.")
        return value.strip()


class StructuredAssetSpec(BaseModel):
    """Reference to a structured, non-expression model asset."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    kind_version: str = "1.0"
    media_type: str
    sha256: str | None = None
    size: int | None = Field(default=None, ge=0)
    bundle_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind", "kind_version", "media_type")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Asset kind, kind_version and media_type cannot be empty.")
        return value.strip()


class DerivedQuantitySpec(ElementAnnotationsSpec):
    """External specification for one named deterministic derived quantity."""

    expression: str

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, expression: str) -> str:
        if not expression.strip():
            raise ValueError("A derived-quantity expression cannot be empty.")
        return expression.strip()


class ConstraintSpec(BaseModel):
    """External specification for one named mathematical constraint."""

    model_config = ConfigDict(extra="forbid")

    left: str
    relation: Literal["=", "==", "<", "<=", ">", ">="]
    right: str
    description: str | None = None
    tags: tuple[str, ...] = ()

    @field_validator("left", "right", mode="before")
    @classmethod
    def validate_side(cls, expression: object) -> str:
        if isinstance(expression, bool):
            raise ValueError("Constraint expressions must be mathematical expressions.")
        if isinstance(expression, (int, float)):
            expression = str(expression)
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("Constraint expressions cannot be empty.")
        return expression.strip()

    @field_validator("description")
    @classmethod
    def normalise_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: tuple[str, ...]) -> tuple[str, ...]:
        return _normalise_tags(tags)


class AssumptionSpec(BaseModel):
    """External specification for one explicit model assumption."""

    model_config = ConfigDict(extra="forbid")

    statement: str
    affects: tuple[str, ...] = ()

    @field_validator("statement")
    @classmethod
    def validate_statement(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("An assumption statement cannot be empty.")
        return stripped

    @field_validator("affects")
    @classmethod
    def validate_affects(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned: list[str] = []
        for value in values:
            name = _validate_identifier(value.strip())
            if name not in cleaned:
                cleaned.append(name)
        return tuple(cleaned)


class AmbiguitySpec(BaseModel):
    """External specification for one explicit model ambiguity."""

    model_config = ConfigDict(extra="forbid")

    statement: str
    options: tuple[str, ...] = ()
    resolution: str | None = None
    blocking: bool = True
    affects: tuple[str, ...] = ()

    @field_validator("statement")
    @classmethod
    def validate_statement(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("An ambiguity statement cannot be empty.")
        return stripped

    @field_validator("options")
    @classmethod
    def validate_options(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned: list[str] = []
        for value in values:
            stripped = value.strip()
            if not stripped:
                raise ValueError("Ambiguity options cannot be empty.")
            if stripped not in cleaned:
                cleaned.append(stripped)
        return tuple(cleaned)

    @field_validator("resolution")
    @classmethod
    def normalise_resolution(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("affects")
    @classmethod
    def validate_affects(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned: list[str] = []
        for value in values:
            name = _validate_identifier(value.strip())
            if name not in cleaned:
                cleaned.append(name)
        return tuple(cleaned)

    @model_validator(mode="after")
    def validate_resolution_against_options(self) -> "AmbiguitySpec":
        if self.resolution is not None and self.options and self.resolution not in self.options:
            raise ValueError(
                "Ambiguity resolution must match one of the declared options when options are supplied."
            )
        return self


class ModelSpec(BaseModel):
    """Validated external schema for a Model Laboratory YAML document."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    metadata: ModelMetadataSpec = Field(default_factory=ModelMetadataSpec)
    variables: dict[str, VariableSpec] = Field(default_factory=dict)
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    constants: dict[str, ConstantSpec] = Field(default_factory=dict)
    derived_quantities: dict[str, DerivedQuantitySpec] = Field(default_factory=dict)
    functions: dict[str, str | FunctionSpec] = Field(default_factory=dict)
    vector_functions: dict[str, VectorFunctionSpec] = Field(default_factory=dict)
    matrix_functions: dict[str, MatrixFunctionSpec] = Field(default_factory=dict)
    constraints: dict[str, ConstraintSpec] = Field(default_factory=dict)
    assumptions: dict[str, AssumptionSpec] = Field(default_factory=dict)
    ambiguities: dict[str, AmbiguitySpec] = Field(default_factory=dict)
    objects: dict[str, ModelObjectSpec] = Field(default_factory=dict)
    relationships: dict[str, ModelRelationshipSpec] = Field(default_factory=dict)
    assets: dict[str, StructuredAssetSpec] = Field(default_factory=dict)

    @field_validator("variables")
    @classmethod
    def validate_variables(
        cls, variables: dict[str, VariableSpec]
    ) -> dict[str, VariableSpec]:
        for name in variables:
            _validate_identifier(name)
        return variables

    @field_validator("parameters")
    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, ParameterSpec]
    ) -> dict[str, ParameterSpec]:
        for name in parameters:
            _validate_identifier(name)
        return parameters

    @field_validator("constants")
    @classmethod
    def validate_constants(cls, constants: dict[str, ConstantSpec]) -> dict[str, ConstantSpec]:
        for name in constants:
            _validate_identifier(name)
        return constants

    @field_validator("derived_quantities")
    @classmethod
    def validate_derived_quantities(
        cls, quantities: dict[str, DerivedQuantitySpec]
    ) -> dict[str, DerivedQuantitySpec]:
        for name in quantities:
            _validate_identifier(name)
        return quantities

    @field_validator("functions")
    @classmethod
    def validate_functions(
        cls, functions: dict[str, str | FunctionSpec]
    ) -> dict[str, str | FunctionSpec]:
        for name, specification in functions.items():
            _validate_identifier(name)
            if isinstance(specification, str) and not specification.strip():
                raise ValueError(f"Function '{name}' has an empty expression.")
        return functions

    @field_validator("vector_functions")
    @classmethod
    def validate_vector_functions(
        cls, functions: dict[str, VectorFunctionSpec]
    ) -> dict[str, VectorFunctionSpec]:
        for name in functions:
            _validate_identifier(name)
        return functions

    @field_validator("matrix_functions")
    @classmethod
    def validate_matrix_functions(
        cls, functions: dict[str, MatrixFunctionSpec]
    ) -> dict[str, MatrixFunctionSpec]:
        for name in functions:
            _validate_identifier(name)
        return functions

    @field_validator("objects", "relationships", "assets")
    @classmethod
    def validate_graph_identifiers(cls, values: dict[str, Any]) -> dict[str, Any]:
        for name in values:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{0,255}", name):
                raise ValueError(f"'{name}' is not a valid stable Model Graph identifier.")
        return values

    @field_validator("constraints")
    @classmethod
    def validate_constraints(
        cls, constraints: dict[str, ConstraintSpec]
    ) -> dict[str, ConstraintSpec]:
        for name in constraints:
            _validate_identifier(name)
        return constraints

    @field_validator("assumptions")
    @classmethod
    def validate_assumptions(
        cls, assumptions: dict[str, AssumptionSpec]
    ) -> dict[str, AssumptionSpec]:
        for name in assumptions:
            _validate_identifier(name)
        return assumptions

    @field_validator("ambiguities")
    @classmethod
    def validate_ambiguities(
        cls, ambiguities: dict[str, AmbiguitySpec]
    ) -> dict[str, AmbiguitySpec]:
        for name in ambiguities:
            _validate_identifier(name)
        return ambiguities

    @model_validator(mode="after")
    def validate_unique_names(self) -> "ModelSpec":
        groups = {
            "variables": set(self.variables),
            "parameters": set(self.parameters),
            "constants": set(self.constants),
            "derived quantities": set(self.derived_quantities),
            "functions": set(self.functions),
            "vector functions": set(self.vector_functions),
            "matrix functions": set(self.matrix_functions),
            "constraints": set(self.constraints),
        }

        seen: set[str] = set()
        collisions: set[str] = set()
        for names in groups.values():
            collisions |= seen & names
            seen |= names

        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(f"Model identifiers must be unique. Reused: {names}.")
        if not (
            self.functions
            or self.vector_functions
            or self.matrix_functions
            or self.objects
            or self.assets
        ):
            raise ValueError(
                "A model must declare at least one scalar/vector/matrix function, "
                "structured object, or asset."
            )
        if (self.functions or self.vector_functions or self.matrix_functions) and not self.variables:
            raise ValueError("Mathematical function models require at least one variable.")
        return self
