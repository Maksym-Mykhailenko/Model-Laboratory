"""Typed Model IR for Model Laboratory.

Scalar mathematical objects are native core kinds.  Discipline-specific connectivity,
materials, reactions, meshes, state machines, and other structured entities live in the
typed Model Graph instead of being flattened into anonymous numbers.  A local capability
pack may understand a registered kind; unavailable extension kinds remain preserved and
inspectable but cannot become executable merely because a document mentions them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import sympy as sp

from .expression import ExpressionNode
from .issues import Ambiguity, Assumption
from .model_graph import ModelGraph
from .provenance import Provenance, ProvenanceApproval, ProvenanceKind, model_ref


@dataclass(frozen=True, slots=True)
class ElementMetadata:
    """Optional human-facing metadata with no mathematical effect."""

    label: str | None = None
    description: str | None = None
    unit: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Optional metadata describing the model as a whole."""

    description: str | None = None
    notes: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Domain:
    """Numerical interval used by a scalar variable or adjustable parameter.

    The current external schema creates finite closed intervals, while the IR records
    boundary inclusion explicitly so later consumers do not have to infer it.
    """

    lower: float
    upper: float
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    def contains(self, value: float) -> bool:
        lower_ok = value >= self.lower if self.lower_inclusive else value > self.lower
        upper_ok = value <= self.upper if self.upper_inclusive else value < self.upper
        return lower_ok and upper_ok

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True, slots=True)
class Variable:
    """One continuous scalar variable declared by the supplied model."""

    name: str
    domain: Domain
    initial_value: float | None = None
    metadata: ElementMetadata = field(default_factory=ElementMetadata)
    provenance: Provenance = field(default_factory=lambda: Provenance.supplied("unspecified"))


@dataclass(frozen=True, slots=True)
class Parameter:
    """One adjustable scalar parameter."""

    name: str
    default: float
    domain: Domain
    metadata: ElementMetadata = field(default_factory=ElementMetadata)
    provenance: Provenance = field(default_factory=lambda: Provenance.supplied("unspecified"))


@dataclass(frozen=True, slots=True)
class Constant:
    """One fixed scalar constant supplied by the model."""

    name: str
    value: float
    metadata: ElementMetadata = field(default_factory=ElementMetadata)
    provenance: Provenance = field(default_factory=lambda: Provenance.supplied("unspecified"))


@dataclass(frozen=True, slots=True)
class ScalarFunction:
    """One named scalar-valued model function."""

    name: str
    expression: sp.Expr
    expression_ast: ExpressionNode
    source: str
    dependencies: tuple[str, ...] = ()
    metadata: ElementMetadata = field(default_factory=ElementMetadata)
    provenance: Provenance = field(default_factory=lambda: Provenance.supplied("unspecified"))


@dataclass(frozen=True, slots=True)
class VectorComponent:
    """One named scalar component of a vector-valued function."""

    name: str
    expression: sp.Expr
    expression_ast: ExpressionNode
    source: str
    dependencies: tuple[str, ...] = ()
    metadata: ElementMetadata = field(default_factory=ElementMetadata)
    provenance: Provenance = field(default_factory=lambda: Provenance.supplied("unspecified"))


@dataclass(frozen=True, slots=True)
class VectorFunction:
    """A first-class ordered function from R^n to R^m."""

    name: str
    components: tuple[VectorComponent, ...]
    metadata: ElementMetadata = field(default_factory=ElementMetadata)
    provenance: Provenance = field(default_factory=lambda: Provenance.supplied("unspecified"))

    @property
    def output_dimension(self) -> int:
        return len(self.components)


@dataclass(frozen=True, slots=True)
class MatrixFunction:
    """A first-class rectangular matrix-valued function."""

    name: str
    entries: tuple[tuple[VectorComponent, ...], ...]
    row_labels: tuple[str, ...] = ()
    column_labels: tuple[str, ...] = ()
    metadata: ElementMetadata = field(default_factory=ElementMetadata)
    provenance: Provenance = field(default_factory=lambda: Provenance.supplied("unspecified"))

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.entries), len(self.entries[0]) if self.entries else 0)


@dataclass(frozen=True, slots=True)
class DerivedQuantity:
    """A named quantity deterministically defined from supplied model elements."""

    name: str
    expression: sp.Expr
    expression_ast: ExpressionNode
    source: str
    dependencies: tuple[str, ...] = ()
    metadata: ElementMetadata = field(default_factory=ElementMetadata)
    provenance: Provenance = field(
        default_factory=lambda: Provenance.supplied("unspecified")
    )


class ConstraintRelation(str, Enum):
    """Supported mathematical relations for declared constraints."""

    EQUAL = "=="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="


@dataclass(frozen=True, slots=True)
class Constraint:
    """One named mathematical relation declared by the model."""

    name: str
    left: sp.Expr
    left_ast: ExpressionNode
    relation: ConstraintRelation
    right: sp.Expr
    right_ast: ExpressionNode
    left_source: str
    right_source: str
    dependencies: tuple[str, ...] = ()
    metadata: ElementMetadata = field(default_factory=ElementMetadata)
    provenance: Provenance = field(default_factory=lambda: Provenance.supplied("unspecified"))


@dataclass(frozen=True, slots=True)
class ModelIR:
    """Internal representation consumed by the deterministic laboratory core."""

    name: str
    variables: tuple[Variable, ...]
    parameters: tuple[Parameter, ...]
    functions: tuple[ScalarFunction, ...] = ()
    vector_functions: tuple[VectorFunction, ...] = ()
    matrix_functions: tuple[MatrixFunction, ...] = ()
    constants: tuple[Constant, ...] = ()
    derived_quantities: tuple[DerivedQuantity, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    assumptions: tuple[Assumption, ...] = ()
    ambiguities: tuple[Ambiguity, ...] = ()
    metadata: ModelMetadata = field(default_factory=ModelMetadata)
    graph: ModelGraph = field(default_factory=ModelGraph)

    def parameter_defaults(self) -> dict[str, float]:
        return {parameter.name: parameter.default for parameter in self.parameters}

    def constant_values(self) -> dict[str, float]:
        return {constant.name: constant.value for constant in self.constants}

    def parameter(self, name: str) -> Parameter:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(name)

    def variable(self, name: str) -> Variable:
        for variable in self.variables:
            if variable.name == name:
                return variable
        raise KeyError(name)

    def constant(self, name: str) -> Constant:
        for constant in self.constants:
            if constant.name == name:
                return constant
        raise KeyError(name)

    def function(self, name: str) -> ScalarFunction:
        for function in self.functions:
            if function.name == name:
                return function
        raise KeyError(name)

    def vector_function(self, name: str) -> VectorFunction:
        for function in self.vector_functions:
            if function.name == name:
                return function
        raise KeyError(name)

    def matrix_function(self, name: str) -> MatrixFunction:
        for function in self.matrix_functions:
            if function.name == name:
                return function
        raise KeyError(name)

    def derived_quantity(self, name: str) -> DerivedQuantity:
        for quantity in self.derived_quantities:
            if quantity.name == name:
                return quantity
        raise KeyError(name)

    def constraint(self, name: str) -> Constraint:
        for constraint in self.constraints:
            if constraint.name == name:
                return constraint
        raise KeyError(name)

    def transitive_derived_dependencies(self, dependencies: tuple[str, ...]) -> tuple[str, ...]:
        """Return declared derived quantities reachable from direct dependency names."""
        derived_by_name = {item.name: item for item in self.derived_quantities}
        ordered: list[str] = []
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting or name not in derived_by_name:
                return
            visiting.add(name)
            quantity = derived_by_name[name]
            for dependency in quantity.dependencies:
                visit(dependency)
            if name not in ordered:
                ordered.append(name)

        for dependency in dependencies:
            visit(dependency)
        return tuple(ordered)

    def assumption(self, name: str) -> Assumption:
        for assumption in self.assumptions:
            if assumption.name == name:
                return assumption
        raise KeyError(name)

    def ambiguity(self, name: str) -> Ambiguity:
        for ambiguity in self.ambiguities:
            if ambiguity.name == name:
                return ambiguity
        raise KeyError(name)


    @property
    def unresolved_ambiguities(self) -> tuple[Ambiguity, ...]:
        """Return ambiguities that do not yet have an explicit resolution."""
        return tuple(item for item in self.ambiguities if not item.is_resolved)

    @property
    def blocking_ambiguities(self) -> tuple[Ambiguity, ...]:
        """Return unresolved ambiguities that explicitly block computation."""
        return tuple(item for item in self.ambiguities if item.blocks_computation)

    @property
    def is_computation_ready(self) -> bool:
        """Whether deterministic computation may proceed without resolving ambiguity."""
        return not self.blocking_ambiguities

    def provenance_entries(self) -> tuple[tuple[str, Provenance], ...]:
        """Return every named Model IR element with its explicit provenance."""
        return (
            *((model_ref("variable", item.name), item.provenance) for item in self.variables),
            *((model_ref("parameter", item.name), item.provenance) for item in self.parameters),
            *((model_ref("constant", item.name), item.provenance) for item in self.constants),
            *((model_ref("derived_quantity", item.name), item.provenance) for item in self.derived_quantities),
            *((model_ref("function", item.name), item.provenance) for item in self.functions),
            *((model_ref("vector_function", item.name), item.provenance) for item in self.vector_functions),
            *((model_ref("matrix_function", item.name), item.provenance) for item in self.matrix_functions),
            *((model_ref("constraint", item.name), item.provenance) for item in self.constraints),
            *((model_ref("assumption", item.name), item.provenance) for item in self.assumptions),
            *((model_ref("ambiguity", item.name), item.provenance) for item in self.ambiguities),
        )

    def declared_names(self) -> tuple[str, ...]:
        """Return every named formal element in deterministic declaration order."""
        return (
            *(variable.name for variable in self.variables),
            *(parameter.name for parameter in self.parameters),
            *(constant.name for constant in self.constants),
            *(quantity.name for quantity in self.derived_quantities),
            *(function.name for function in self.functions),
            *(function.name for function in self.vector_functions),
            *(function.name for function in self.matrix_functions),
            *(constraint.name for constraint in self.constraints),
        )
