"""Deterministic mathematical validation and compilation into the Model IR."""

from __future__ import annotations

from typing import Mapping

import sympy as sp

from .expression import (
    FUNCTION_ARITIES,
    LEGACY_FUNCTION_ARITIES,
    NAMED_CONSTANTS,
    ExpressionError,
    ExpressionNode,
    expression_dependencies,
    expression_to_sympy,
    parse_expression,
)
from .issues import Ambiguity, Assumption
from .model import (
    Constant,
    Constraint,
    ConstraintRelation,
    DerivedQuantity,
    Domain,
    ElementMetadata,
    ModelIR,
    ModelMetadata,
    MatrixFunction,
    Parameter,
    Provenance,
    ScalarFunction,
    VectorComponent,
    VectorFunction,
    Variable,
)
from .model_graph import (
    CORE_KIND_REGISTRY,
    DomainType,
    ModelGraph,
    ModelGraphError,
    ModelObject,
    ObjectKindRegistry,
    ModelRelationship,
    StructuredAssetReference,
    ValueType,
)
from .official_packs import (
    OFFICIAL_KIND_DESCRIPTORS,
    OFFICIAL_KIND_REGISTRY,
    validate_official_graph,
)
from .schema import (
    AmbiguitySpec,
    AssumptionSpec,
    ConstraintSpec,
    DerivedQuantitySpec,
    ElementAnnotationsSpec,
    FunctionSpec,
    MatrixFunctionSpec,
    ModelSpec,
    VectorFunctionSpec,
)


class ModelValidationError(ValueError):
    """Raised when the mathematics of a schema-valid model is not valid."""


def _element_metadata(specification: ElementAnnotationsSpec) -> ElementMetadata:
    return ElementMetadata(
        label=specification.label,
        description=specification.description,
        unit=specification.unit,
        tags=tuple(specification.tags),
    )


def _function_parts(specification: str | FunctionSpec) -> tuple[str, ElementMetadata]:
    if isinstance(specification, str):
        return specification.strip(), ElementMetadata()
    return specification.expression, _element_metadata(specification)


def _compile_component(
    *,
    name: str,
    source: str,
    metadata: ElementMetadata,
    provenance_location: str,
    declared_symbol_names: set[str],
    symbols: dict[str, sp.Symbol],
    substitutions: dict[sp.Symbol, sp.Expr],
    function_arities: Mapping[str, tuple[int, int]],
) -> VectorComponent:
    expression_ast = _parse_ast(
        source,
        declared_symbol_names,
        context=f"Function component '{name}'",
        function_arities=function_arities,
    )
    try:
        expression = sp.simplify(expression_to_sympy(expression_ast, symbols).xreplace(substitutions))
    except ExpressionError as exc:
        raise ModelValidationError(f"Function component '{name}': {exc}") from exc
    return VectorComponent(
        name=name,
        expression=expression,
        expression_ast=expression_ast,
        source=source,
        dependencies=expression_dependencies(expression_ast),
        metadata=metadata,
        provenance=Provenance.supplied(provenance_location),
    )


def _parse_ast(
    source: str,
    declared_symbols: set[str],
    *,
    context: str,
    function_arities: Mapping[str, tuple[int, int]],
) -> ExpressionNode:
    try:
        return parse_expression(
            source,
            declared_symbols,
            function_arities=function_arities,
        )
    except ExpressionError as exc:
        raise ModelValidationError(f"{context}: {exc}") from exc


def _compile_derived_quantities(
    specifications: dict[str, DerivedQuantitySpec],
    declared_symbol_names: set[str],
    symbols: dict[str, sp.Symbol],
    function_arities: Mapping[str, tuple[int, int]],
) -> tuple[tuple[DerivedQuantity, ...], dict[sp.Symbol, sp.Expr]]:
    """Compile derived quantities, resolving dependencies without permitting cycles."""
    raw_asts: dict[str, ExpressionNode] = {}
    raw_expressions: dict[str, sp.Expr] = {}
    direct_dependencies: dict[str, tuple[str, ...]] = {}

    for name, specification in specifications.items():
        node = _parse_ast(
            specification.expression,
            declared_symbol_names,
            context=f"Derived quantity '{name}'",
            function_arities=function_arities,
        )
        raw_asts[name] = node
        direct_dependencies[name] = expression_dependencies(node)
        try:
            raw_expressions[name] = expression_to_sympy(node, symbols)
        except ExpressionError as exc:  # defensive: parser and adapter share the same grammar
            raise ModelValidationError(f"Derived quantity '{name}': {exc}") from exc

    derived_names = set(specifications)
    resolved: dict[str, sp.Expr] = {}
    visiting: list[str] = []

    def resolve(name: str) -> sp.Expr:
        if name in resolved:
            return resolved[name]
        if name in visiting:
            cycle_start = visiting.index(name)
            cycle = visiting[cycle_start:] + [name]
            raise ModelValidationError(
                "Cyclic derived-quantity dependency: " + " -> ".join(cycle) + "."
            )

        visiting.append(name)
        expression = raw_expressions[name]
        substitutions: dict[sp.Symbol, sp.Expr] = {}
        for dependency in direct_dependencies[name]:
            if dependency in derived_names:
                substitutions[symbols[dependency]] = resolve(dependency)
        if substitutions:
            expression = expression.xreplace(substitutions)
        expression = sp.simplify(expression)
        visiting.pop()
        resolved[name] = expression
        return expression

    quantities: list[DerivedQuantity] = []
    for name, specification in specifications.items():
        quantities.append(
            DerivedQuantity(
                name=name,
                expression=resolve(name),
                expression_ast=raw_asts[name],
                source=specification.expression,
                dependencies=direct_dependencies[name],
                metadata=_element_metadata(specification),
                provenance=Provenance.supplied(f"derived_quantities.{name}"),
            )
        )

    substitutions = {symbols[name]: expression for name, expression in resolved.items()}
    return tuple(quantities), substitutions


def _compile_constraints(
    specifications: dict[str, ConstraintSpec],
    declared_symbols: set[str],
    symbols: dict[str, sp.Symbol],
    substitutions: dict[sp.Symbol, sp.Expr],
    function_arities: Mapping[str, tuple[int, int]],
) -> tuple[Constraint, ...]:
    relation_map = {
        "=": ConstraintRelation.EQUAL,
        "==": ConstraintRelation.EQUAL,
        "<": ConstraintRelation.LESS_THAN,
        "<=": ConstraintRelation.LESS_EQUAL,
        ">": ConstraintRelation.GREATER_THAN,
        ">=": ConstraintRelation.GREATER_EQUAL,
    }

    constraints: list[Constraint] = []
    for name, specification in specifications.items():
        left_ast = _parse_ast(
            specification.left,
            declared_symbols,
            context=f"Constraint '{name}'",
            function_arities=function_arities,
        )
        right_ast = _parse_ast(
            specification.right,
            declared_symbols,
            context=f"Constraint '{name}'",
            function_arities=function_arities,
        )
        dependencies = tuple(
            sorted(set(expression_dependencies(left_ast)) | set(expression_dependencies(right_ast)))
        )
        try:
            raw_left = expression_to_sympy(left_ast, symbols)
            raw_right = expression_to_sympy(right_ast, symbols)
        except ExpressionError as exc:
            raise ModelValidationError(f"Constraint '{name}': {exc}") from exc

        constraints.append(
            Constraint(
                name=name,
                left=sp.simplify(raw_left.xreplace(substitutions)),
                left_ast=left_ast,
                relation=relation_map[specification.relation],
                right=sp.simplify(raw_right.xreplace(substitutions)),
                right_ast=right_ast,
                left_source=specification.left,
                right_source=specification.right,
                dependencies=dependencies,
                metadata=ElementMetadata(
                    description=specification.description,
                    tags=tuple(specification.tags),
                ),
                provenance=Provenance.supplied(f"constraints.{name}"),
            )
        )
    return tuple(constraints)


def _validate_issue_affects(
    category: str,
    name: str,
    affects: tuple[str, ...],
    declared_names: set[str],
) -> tuple[str, ...]:
    unknown = [item for item in affects if item not in declared_names]
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ModelValidationError(
            f"{category} '{name}' refers to unknown model element(s): {names}."
        )
    return affects


def _compile_assumptions(
    specifications: dict[str, AssumptionSpec],
    declared_names: set[str],
) -> tuple[Assumption, ...]:
    return tuple(
        Assumption(
            name=name,
            statement=specification.statement,
            affects=_validate_issue_affects(
                "Assumption", name, tuple(specification.affects), declared_names
            ),
            provenance=Provenance.supplied(f"assumptions.{name}"),
        )
        for name, specification in specifications.items()
    )


def _compile_ambiguities(
    specifications: dict[str, AmbiguitySpec],
    declared_names: set[str],
) -> tuple[Ambiguity, ...]:
    return tuple(
        Ambiguity(
            name=name,
            statement=specification.statement,
            options=tuple(specification.options),
            resolution=specification.resolution,
            blocking=specification.blocking,
            affects=_validate_issue_affects(
                "Ambiguity", name, tuple(specification.affects), declared_names
            ),
            provenance=Provenance.supplied(f"ambiguities.{name}"),
        )
        for name, specification in specifications.items()
    )


def validate_model(
    spec: ModelSpec,
    *,
    function_arities: Mapping[str, tuple[int, int]] | None = None,
    kind_registry: ObjectKindRegistry | None = None,
) -> ModelIR:
    """Validate mathematical content and compile a ModelSpec into ModelIR.

    Model source is parsed into the Model Laboratory expression AST first.  SymPy receives
    only ASTs that have already passed this deterministic validation layer.
    """
    active_function_arities = FUNCTION_ARITIES if function_arities is None else function_arities
    active_kind_registry = OFFICIAL_KIND_REGISTRY if kind_registry is None else kind_registry
    declared_symbol_names = (
        set(spec.variables)
        | set(spec.parameters)
        | set(spec.constants)
        | set(spec.derived_quantities)
    )
    all_declared_names = (
        declared_symbol_names
        | set(spec.functions)
        | set(spec.vector_functions)
        | set(spec.matrix_functions)
        | set(spec.constraints)
    )
    reserved_collisions = all_declared_names & (
        set(active_function_arities) | set(NAMED_CONSTANTS)
    )
    if reserved_collisions:
        names = ", ".join(sorted(reserved_collisions))
        raise ModelValidationError(
            f"These identifiers are reserved mathematical names and cannot be declared: {names}."
        )

    symbols = {name: sp.Symbol(name, real=True) for name in declared_symbol_names}

    variables = tuple(
        Variable(
            name=name,
            domain=Domain(*variable.domain),
            initial_value=variable.initial,
            metadata=_element_metadata(variable),
            provenance=Provenance.supplied(f"variables.{name}"),
        )
        for name, variable in spec.variables.items()
    )
    parameters = tuple(
        Parameter(
            name=name,
            default=parameter.default,
            domain=Domain(*parameter.domain),
            metadata=_element_metadata(parameter),
            provenance=Provenance.supplied(f"parameters.{name}"),
        )
        for name, parameter in spec.parameters.items()
    )
    constants = tuple(
        Constant(
            name=name,
            value=constant.value,
            metadata=_element_metadata(constant),
            provenance=Provenance.supplied(f"constants.{name}"),
        )
        for name, constant in spec.constants.items()
    )

    derived_quantities, derived_substitutions = _compile_derived_quantities(
        spec.derived_quantities,
        declared_symbol_names,
        symbols,
        active_function_arities,
    )

    functions: list[ScalarFunction] = []
    function_substitutions: dict[sp.Symbol, sp.Expr] = {}
    for name, specification in spec.functions.items():
        source, metadata = _function_parts(specification)
        expression_ast = _parse_ast(
            source,
            declared_symbol_names,
            context=f"Function '{name}'",
            function_arities=active_function_arities,
        )
        dependencies = expression_dependencies(expression_ast)
        try:
            raw_expression = expression_to_sympy(expression_ast, symbols)
        except ExpressionError as exc:
            raise ModelValidationError(f"Function '{name}': {exc}") from exc
        expression = sp.simplify(raw_expression.xreplace(derived_substitutions))
        functions.append(
            ScalarFunction(
                name=name,
                expression=expression,
                expression_ast=expression_ast,
                source=source,
                dependencies=dependencies,
                metadata=metadata,
                provenance=Provenance.supplied(f"functions.{name}"),
            )
        )
        function_substitutions[sp.Symbol(name, real=True)] = expression

    vector_functions: list[VectorFunction] = []
    for name, specification in spec.vector_functions.items():
        components: list[VectorComponent] = []
        for component_name, component_specification in specification.components.items():
            source, component_metadata = _function_parts(component_specification)
            components.append(
                _compile_component(
                    name=component_name,
                    source=source,
                    metadata=component_metadata,
                    provenance_location=f"vector_functions.{name}.components.{component_name}",
                    declared_symbol_names=declared_symbol_names,
                    symbols=symbols,
                    substitutions=derived_substitutions,
                    function_arities=active_function_arities,
                )
            )
        vector_functions.append(
            VectorFunction(
                name=name,
                components=tuple(components),
                metadata=_element_metadata(specification),
                provenance=Provenance.supplied(f"vector_functions.{name}"),
            )
        )

    matrix_functions: list[MatrixFunction] = []
    for name, specification in spec.matrix_functions.items():
        entries: list[tuple[VectorComponent, ...]] = []
        for row_index, row in enumerate(specification.entries):
            compiled_row: list[VectorComponent] = []
            for column_index, source in enumerate(row):
                component_name = f"r{row_index + 1}c{column_index + 1}"
                compiled_row.append(
                    _compile_component(
                        name=component_name,
                        source=source,
                        metadata=ElementMetadata(),
                        provenance_location=(
                            f"matrix_functions.{name}.entries.{row_index}.{column_index}"
                        ),
                        declared_symbol_names=declared_symbol_names,
                        symbols=symbols,
                        substitutions=derived_substitutions,
                        function_arities=active_function_arities,
                    )
                )
            entries.append(tuple(compiled_row))
        matrix_functions.append(
            MatrixFunction(
                name=name,
                entries=tuple(entries),
                row_labels=tuple(specification.row_labels),
                column_labels=tuple(specification.column_labels),
                metadata=_element_metadata(specification),
                provenance=Provenance.supplied(f"matrix_functions.{name}"),
            )
        )

    constraint_symbols = {
        **symbols,
        **{name: sp.Symbol(name, real=True) for name in spec.functions},
    }
    constraints = _compile_constraints(
        spec.constraints,
        set(constraint_symbols),
        constraint_symbols,
        {**derived_substitutions, **function_substitutions},
        active_function_arities,
    )

    issue_target_names = (
        set(spec.variables)
        | set(spec.parameters)
        | set(spec.constants)
        | set(spec.derived_quantities)
        | set(spec.functions)
        | set(spec.vector_functions)
        | set(spec.matrix_functions)
        | set(spec.constraints)
    )
    assumptions = _compile_assumptions(spec.assumptions, issue_target_names)
    ambiguities = _compile_ambiguities(spec.ambiguities, issue_target_names)

    graph_objects: list[ModelObject] = []
    graph_relationships: list[ModelRelationship] = []

    def domain_type(specification: object | None) -> DomainType | None:
        if specification is None:
            return None
        return DomainType(
            kind=specification.kind,  # type: ignore[attr-defined]
            lower=specification.lower,  # type: ignore[attr-defined]
            upper=specification.upper,  # type: ignore[attr-defined]
            lower_inclusive=specification.lower_inclusive,  # type: ignore[attr-defined]
            upper_inclusive=specification.upper_inclusive,  # type: ignore[attr-defined]
            values=tuple(specification.values),  # type: ignore[attr-defined]
            factors=tuple(domain_type(item) for item in specification.factors),  # type: ignore[attr-defined,arg-type]
            index_domain=domain_type(specification.index_domain),  # type: ignore[attr-defined]
            value_domain=domain_type(specification.value_domain),  # type: ignore[attr-defined]
        )

    def value_type(specification: object) -> ValueType:
        return ValueType(
            kind=specification.kind,  # type: ignore[attr-defined]
            shape=tuple(specification.shape),  # type: ignore[attr-defined]
            element_kind=specification.element_kind,  # type: ignore[attr-defined]
            unit_dimension=specification.unit_dimension,  # type: ignore[attr-defined]
            domain=domain_type(specification.domain),  # type: ignore[attr-defined]
            input_types=tuple(value_type(item) for item in specification.input_types),  # type: ignore[attr-defined]
            output_type=(
                None
                if specification.output_type is None  # type: ignore[attr-defined]
                else value_type(specification.output_type)  # type: ignore[attr-defined]
            ),
        )

    variable_input_types = tuple(
        ValueType(
            "real",
            domain=DomainType(
                "interval",
                lower=item.domain.lower,
                upper=item.domain.upper,
            ),
        )
        for item in variables
    )

    def core_object(
        identifier: str,
        kind: str,
        value_type: ValueType,
        properties: Mapping[str, object],
    ) -> None:
        graph_objects.append(
            ModelObject(
                identifier=identifier,
                kind=kind,
                kind_version="1.0",
                value_type=value_type,
                properties=properties,
                executable=True,
            )
        )

    for item in variables:
        core_object(
            f"variable:{item.name}",
            "org.modellab.core.variable",
            ValueType(
                "real",
                domain=DomainType("interval", lower=item.domain.lower, upper=item.domain.upper),
            ),
            {"name": item.name, "domain": [item.domain.lower, item.domain.upper]},
        )
    for item in parameters:
        core_object(
            f"parameter:{item.name}",
            "org.modellab.core.parameter",
            ValueType(
                "real",
                domain=DomainType("interval", lower=item.domain.lower, upper=item.domain.upper),
            ),
            {"name": item.name, "default": item.default, "domain": [item.domain.lower, item.domain.upper]},
        )
    for item in constants:
        core_object(
            f"constant:{item.name}",
            "org.modellab.core.constant",
            ValueType("real"),
            {"name": item.name, "value": item.value},
        )
    for item in derived_quantities:
        core_object(
            f"derived:{item.name}",
            "org.modellab.core.derived-quantity",
            ValueType("real"),
            {"name": item.name, "source": item.source},
        )
    for item in functions:
        core_object(
            f"function:{item.name}",
            "org.modellab.core.scalar-function",
            ValueType(
                "function",
                input_types=variable_input_types,
                output_type=ValueType("real"),
            ),
            {"name": item.name, "source": item.source},
        )
    for item in vector_functions:
        core_object(
            f"vector-function:{item.name}",
            "org.modellab.core.vector-function",
            ValueType(
                "function",
                input_types=variable_input_types,
                output_type=ValueType("vector", (len(item.components),), "real"),
            ),
            {"name": item.name, "components": [component.name for component in item.components]},
        )
    for item in matrix_functions:
        core_object(
            f"matrix-function:{item.name}",
            "org.modellab.core.matrix-function",
            ValueType(
                "function",
                input_types=variable_input_types,
                output_type=ValueType("matrix", item.shape, "real"),
            ),
            {"name": item.name, "row_labels": list(item.row_labels), "column_labels": list(item.column_labels)},
        )
    for item in constraints:
        core_object(
            f"constraint:{item.name}",
            "org.modellab.core.constraint",
            ValueType(
                "relation",
                input_types=variable_input_types,
                output_type=ValueType("boolean"),
            ),
            {"name": item.name, "relation": item.relation.value},
        )

    name_to_graph_id = {
        **{item.name: f"variable:{item.name}" for item in variables},
        **{item.name: f"parameter:{item.name}" for item in parameters},
        **{item.name: f"constant:{item.name}" for item in constants},
        **{item.name: f"derived:{item.name}" for item in derived_quantities},
        **{item.name: f"function:{item.name}" for item in functions},
        **{item.name: f"vector-function:{item.name}" for item in vector_functions},
        **{item.name: f"matrix-function:{item.name}" for item in matrix_functions},
        **{item.name: f"constraint:{item.name}" for item in constraints},
    }
    dependency_sources: list[tuple[str, tuple[str, ...]]] = [
        *((f"derived:{item.name}", item.dependencies) for item in derived_quantities),
        *((f"function:{item.name}", item.dependencies) for item in functions),
        *((
            f"vector-function:{item.name}",
            tuple(sorted({dependency for component in item.components for dependency in component.dependencies})),
        ) for item in vector_functions),
        *((
            f"matrix-function:{item.name}",
            tuple(sorted({dependency for row in item.entries for component in row for dependency in component.dependencies})),
        ) for item in matrix_functions),
        *((f"constraint:{item.name}", item.dependencies) for item in constraints),
    ]
    for source_id, dependencies in dependency_sources:
        for dependency in dependencies:
            target_id = name_to_graph_id.get(dependency)
            if target_id is None:
                continue
            graph_relationships.append(
                ModelRelationship(
                    identifier=f"dependency:{source_id}:{target_id}",
                    kind="org.modellab.core.depends-on",
                    kind_version="1.0",
                    source=source_id,
                    target=target_id,
                )
            )

    for identifier, specification in spec.objects.items():
        try:
            properties, executable, opaque = active_kind_registry.validate_properties(
                specification.kind,
                specification.kind_version,
                specification.properties,
            )
            graph_objects.append(
                ModelObject(
                    identifier=identifier,
                    kind=specification.kind,
                    kind_version=specification.kind_version,
                    value_type=value_type(specification.value_type),
                    properties=properties,
                    references=tuple(specification.references),
                    executable=executable,
                    opaque=opaque,
                )
            )
        except ModelGraphError as exc:
            raise ModelValidationError(f"Model Graph object '{identifier}': {exc}") from exc
    asset_values: list[StructuredAssetReference] = []
    for identifier, specification in spec.assets.items():
        try:
            metadata, _, opaque = active_kind_registry.validate_properties(
                specification.kind,
                specification.kind_version,
                specification.metadata,
            )
            asset_values.append(
                StructuredAssetReference(
                    identifier=identifier,
                    kind=specification.kind,
                    kind_version=specification.kind_version,
                    media_type=specification.media_type,
                    sha256=specification.sha256,
                    size=specification.size,
                    bundle_path=specification.bundle_path,
                    metadata=metadata,
                    opaque=opaque,
                )
            )
        except ModelGraphError as exc:
            raise ModelValidationError(f"Model Graph asset '{identifier}': {exc}") from exc
    assets = tuple(asset_values)
    for identifier, specification in spec.relationships.items():
        try:
            properties, _, opaque = active_kind_registry.validate_properties(
                specification.kind,
                specification.kind_version,
                specification.properties,
            )
            graph_relationships.append(
                ModelRelationship(
                    identifier=identifier,
                    kind=specification.kind,
                    kind_version=specification.kind_version,
                    source=specification.source,
                    target=specification.target,
                    properties=properties,
                    opaque=opaque,
                )
            )
        except ModelGraphError as exc:
            raise ModelValidationError(f"Model Graph relationship '{identifier}': {exc}") from exc

    try:
        graph = ModelGraph(
            objects=tuple(graph_objects),
            relationships=tuple(graph_relationships),
            assets=assets,
        )
        if any(
            active_kind_registry.knows(descriptor.kind, descriptor.version)
            for descriptor in OFFICIAL_KIND_DESCRIPTORS
        ):
            validate_official_graph(graph)
    except ModelGraphError as exc:
        raise ModelValidationError(f"Model Graph: {exc}") from exc

    return ModelIR(
        name=spec.name,
        variables=variables,
        parameters=parameters,
        functions=tuple(functions),
        vector_functions=tuple(vector_functions),
        matrix_functions=tuple(matrix_functions),
        constants=constants,
        derived_quantities=derived_quantities,
        constraints=constraints,
        assumptions=assumptions,
        ambiguities=ambiguities,
        metadata=ModelMetadata(
            description=spec.metadata.description,
            notes=spec.metadata.notes,
            tags=tuple(spec.metadata.tags),
        ),
        graph=graph,
    )


def validate_legacy_model(spec: ModelSpec) -> ModelIR:
    """Compile a pre-v1.8 model under its historical expression-name registry."""
    return validate_model(
        spec,
        function_arities=LEGACY_FUNCTION_ARITIES,
        kind_registry=CORE_KIND_REGISTRY,
    )
