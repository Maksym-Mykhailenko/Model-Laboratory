"""Built-in capability packs registered with the generic Run -> Artifact protocol."""

from __future__ import annotations

from typing import Any, Mapping

from .analysis import run_parameter_sweep, run_stationary_point_analysis
from .evaluator import (
    evaluate_scalar_at_points,
    evaluate_single_variable_function,
    evaluate_two_variable_function,
)
from .model import ModelIR
from .official_packs import OFFICIAL_CAPABILITY_DESCRIPTORS
from .protocol import (
    ArtifactTypeDescriptor,
    CapabilityDescriptor,
    CapabilityPackRegistry,
)
from .symbolic import analyse_scalar_function
from .vector_analysis import (
    analyse_matrix_function,
    analyse_vector_function,
    evaluate_vector_field_2d,
    optimize_scalar_with_constraints,
    solve_vector_system,
)


EXACT = "org.modellab.comparator.exact"
NUMERIC = "org.modellab.comparator.numeric"


def _artifact(identifier: str, title: str, *, numeric: bool = False) -> ArtifactTypeDescriptor:
    return ArtifactTypeDescriptor(identifier, "1.0", title, NUMERIC if numeric else EXACT)


def _applicable(
    predicate: bool, reason: str
) -> tuple[bool, str]:
    return (True, "") if predicate else (False, reason)


def _settings_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    del model
    return max(1, len(str(settings)))


def _point_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    points = settings.get("points", [])
    return max(1, len(points) * max(1, len(model.variables)))


def _grid_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    del model
    count = int(settings.get("points_per_axis", settings.get("points", 1000)))
    return count * count if "points_per_axis" in settings else count


def _seed_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    return int(settings.get("seeds", settings.get("root_seeds", 64))) * max(1, len(model.variables)) * 100


def _scalar_model(model: ModelIR) -> tuple[bool, str]:
    return _applicable(bool(model.variables and model.functions), "variables and a scalar function are required")


def _one_dimensional_scalar(model: ModelIR) -> tuple[bool, str]:
    return _applicable(
        len(model.variables) == 1 and len(model.functions) == 1,
        "exactly one variable and one scalar function are required",
    )


def _two_dimensional_scalar(model: ModelIR) -> tuple[bool, str]:
    return _applicable(
        len(model.variables) == 2 and len(model.functions) == 1,
        "exactly two variables and one scalar function are required",
    )


def _legacy_stationary(model: ModelIR) -> tuple[bool, str]:
    return _applicable(
        len(model.variables) in (1, 2) and len(model.functions) == 1 and not model.constraints,
        "the current stationary-point solver requires one/two variables, one scalar function, and no constraints",
    )


def _vector_model(model: ModelIR) -> tuple[bool, str]:
    return _applicable(bool(model.variables and model.vector_functions), "variables and a vector function are required")


def _vector_field_2d(model: ModelIR) -> tuple[bool, str]:
    return _applicable(
        len(model.variables) == 2
        and any(len(function.components) == 2 for function in model.vector_functions),
        "two variables and a two-component vector function are required",
    )


def _matrix_model(model: ModelIR) -> tuple[bool, str]:
    return _applicable(bool(model.variables and model.matrix_functions), "variables and a matrix function are required")


def _constrained_scalar(model: ModelIR) -> tuple[bool, str]:
    return _applicable(
        bool(model.variables and model.functions and model.constraints),
        "variables, a scalar objective, and declared constraints are required",
    )


run_registry = CapabilityPackRegistry(
    (
        CapabilityDescriptor(
            "org.modellab.scalar.evaluate-points",
            "1.0",
            "org.modellab.pack.scalar-calculus",
            "Scalar point evaluation",
            "Evaluate a selected scalar function at explicit points in arbitrary dimension.",
            {"type": "object", "required": ["points"], "properties": {"points": {"type": "array"}, "parameter_values": {"type": "object"}, "function_name": {"type": ["string", "null"]}}},
            "numpy",
            (_artifact("org.modellab.artifact.scalar-point-evaluation", "Scalar point evaluation", numeric=True),),
            _scalar_model,
            evaluate_scalar_at_points,
            _point_units,
        ),
        CapabilityDescriptor(
            "org.modellab.scalar.differential",
            "1.0",
            "org.modellab.pack.scalar-calculus",
            "Scalar gradient and Hessian",
            "Compute the symbolic gradient and Hessian in arbitrary finite dimension.",
            {"type": "object", "properties": {"function_name": {"type": ["string", "null"]}}},
            "sympy",
            (_artifact("org.modellab.artifact.scalar-differential", "Scalar differential analysis"),),
            _scalar_model,
            analyse_scalar_function,
            _settings_units,
        ),
        CapabilityDescriptor(
            "org.modellab.scalar.evaluate-curve",
            "1.0",
            "org.modellab.pack.scalar-calculus",
            "One-dimensional curve evaluation",
            "Evaluate a scalar function on a regular one-dimensional grid.",
            {"type": "object", "properties": {"points": {"type": "integer", "minimum": 2, "maximum": 200000}, "parameter_values": {"type": "object"}}},
            "numpy",
            (_artifact("org.modellab.artifact.scalar-curve", "Scalar curve", numeric=True),),
            _one_dimensional_scalar,
            evaluate_single_variable_function,
            _grid_units,
            ("org.modellab.renderer.plotly-curve",),
        ),
        CapabilityDescriptor(
            "org.modellab.scalar.evaluate-surface",
            "1.0",
            "org.modellab.pack.scalar-calculus",
            "Two-dimensional surface evaluation",
            "Evaluate a scalar field on a regular two-dimensional grid.",
            {"type": "object", "properties": {"points_per_axis": {"type": "integer", "minimum": 2, "maximum": 1000}, "parameter_values": {"type": "object"}}},
            "numpy",
            (_artifact("org.modellab.artifact.scalar-surface", "Scalar surface", numeric=True),),
            _two_dimensional_scalar,
            evaluate_two_variable_function,
            _grid_units,
            ("org.modellab.renderer.plotly-surface", "org.modellab.renderer.plotly-contour"),
        ),
        CapabilityDescriptor(
            "org.modellab.scalar.stationary-points",
            "1.0",
            "org.modellab.pack.scalar-calculus",
            "Stationary points",
            "Locate and classify stationary points for the current one/two-dimensional solver.",
            {"type": "object", "properties": {"parameter_values": {"type": "object"}, "samples_1d": {"type": "integer"}, "seeds_2d": {"type": "integer"}}},
            "scipy+sympy",
            (_artifact("org.modellab.artifact.stationary-points", "Stationary points", numeric=True),),
            _legacy_stationary,
            run_stationary_point_analysis,
            _seed_units,
        ),
        CapabilityDescriptor(
            "org.modellab.vector.differential",
            "1.0",
            "org.modellab.pack.vector-calculus",
            "Vector Jacobian and field operators",
            "Compute an arbitrary m-by-n Jacobian plus divergence/curl where defined.",
            {"type": "object", "properties": {"function_name": {"type": ["string", "null"]}}},
            "sympy",
            (_artifact("org.modellab.artifact.vector-differential", "Vector differential analysis"),),
            _vector_model,
            analyse_vector_function,
            _settings_units,
        ),
        CapabilityDescriptor(
            "org.modellab.vector.solve-roots",
            "1.0",
            "org.modellab.pack.vector-calculus",
            "Solve vector system",
            "Solve F(x)=0 with bounded deterministic seeds and classify square-field equilibria.",
            {"type": "object", "properties": {"function_name": {"type": ["string", "null"]}, "seeds": {"type": "integer", "minimum": 1, "maximum": 4096}, "parameter_values": {"type": "object"}, "residual_tolerance": {"type": "number"}}},
            "scipy+numpy+sympy",
            (_artifact("org.modellab.artifact.system-roots", "System roots and equilibria", numeric=True),),
            _vector_model,
            solve_vector_system,
            _seed_units,
        ),
        CapabilityDescriptor(
            "org.modellab.vector.field-2d",
            "1.0",
            "org.modellab.pack.vector-calculus",
            "Two-dimensional vector field",
            "Sample a 2D field for arrows, streamlines, nullclines, and equilibria.",
            {"type": "object", "properties": {"function_name": {"type": ["string", "null"]}, "points_per_axis": {"type": "integer", "minimum": 3, "maximum": 250}, "root_seeds": {"type": "integer"}, "parameter_values": {"type": "object"}}},
            "numpy+scipy",
            (_artifact("org.modellab.artifact.vector-field-2d", "Two-dimensional vector field", numeric=True),),
            _vector_field_2d,
            evaluate_vector_field_2d,
            _grid_units,
            ("org.modellab.renderer.plotly-vector-field",),
        ),
        CapabilityDescriptor(
            "org.modellab.matrix.analyse",
            "1.0",
            "org.modellab.pack.matrix-analysis",
            "Matrix analysis",
            "Evaluate a matrix and compute determinant, rank, singular values, condition, and eigenvalues where defined.",
            {"type": "object", "required": ["point"], "properties": {"point": {"type": "array"}, "function_name": {"type": ["string", "null"]}, "parameter_values": {"type": "object"}}},
            "numpy+sympy",
            (_artifact("org.modellab.artifact.matrix-analysis", "Matrix analysis", numeric=True),),
            _matrix_model,
            analyse_matrix_function,
            _settings_units,
        ),
        CapabilityDescriptor(
            "org.modellab.optimization.constrained",
            "1.0",
            "org.modellab.pack.optimization",
            "Constraint-aware optimisation",
            "Optimise a scalar objective over declared bounds and equality/inequality constraints.",
            {"type": "object", "properties": {"function_name": {"type": ["string", "null"]}, "objective": {"enum": ["minimize", "maximize"]}, "seeds": {"type": "integer"}, "parameter_values": {"type": "object"}}},
            "scipy+numpy+sympy",
            (_artifact("org.modellab.artifact.constrained-optimum", "Constrained optimum", numeric=True),),
            _constrained_scalar,
            optimize_scalar_with_constraints,
            _seed_units,
        ),
        *OFFICIAL_CAPABILITY_DESCRIPTORS,
    )
)
