from __future__ import annotations

from model_lab.analysis import run_parameter_sweep, run_stationary_point_analysis
from model_lab.evaluator import evaluate_single_variable_function
from model_lab.parser import parse_model_text
from model_lab.provenance import (
    Provenance,
    ProvenanceApproval,
    ProvenanceKind,
    analysis_ref,
    model_ref,
)
from model_lab.symbolic import analyse_one_variable_function, analyse_two_variable_function
from model_lab.validator import validate_model


def _compile(text: str):
    return validate_model(parse_model_text(text))


def test_supplied_model_elements_record_precise_source_locations() -> None:
    model = _compile(
        """
name: Provenance model
variables:
  x:
    domain: [-2, 2]
parameters:
  a:
    default: 1
    domain: [0, 2]
constants:
  c:
    value: 3
derived_quantities:
  q:
    expression: a*x
functions:
  y: q + c
constraints:
  upper:
    left: y
    relation: "<="
    right: 10
"""
    )

    expected = {
        model_ref("variable", "x"): "variables.x",
        model_ref("parameter", "a"): "parameters.a",
        model_ref("constant", "c"): "constants.c",
        model_ref("derived_quantity", "q"): "derived_quantities.q",
        model_ref("function", "y"): "functions.y",
        model_ref("constraint", "upper"): "constraints.upper",
    }

    for reference, provenance in model.provenance_entries():
        assert provenance.kind is ProvenanceKind.SUPPLIED
        assert provenance.source_location == expected[reference]
        assert provenance.source_refs == ()
        assert provenance.approval is ProvenanceApproval.NOT_REQUIRED


def test_symbolic_derivative_provenance_records_lineage() -> None:
    model = _compile(
        """
name: Derivative provenance
variables:
  x:
    domain: [-2, 2]
functions:
  y: x**3
"""
    )

    result = analyse_one_variable_function(model)

    assert result.first_derivative_provenance.kind is ProvenanceKind.DERIVED
    assert result.first_derivative_provenance.source_refs == (
        model_ref("function", "y"),
        model_ref("variable", "x"),
    )
    assert result.second_derivative_provenance.source_refs == (
        analysis_ref("first_derivative", "y", "x"),
        model_ref("variable", "x"),
    )


def test_gradient_and_hessian_provenance_are_distinct() -> None:
    model = _compile(
        """
name: Surface provenance
variables:
  x:
    domain: [-2, 2]
  y:
    domain: [-2, 2]
functions:
  z: x**2 + y**2
"""
    )

    result = analyse_two_variable_function(model)

    assert result.gradient_provenance.kind is ProvenanceKind.DERIVED
    assert result.gradient_provenance.source_refs == (
        model_ref("function", "z"),
        model_ref("variable", "x"),
        model_ref("variable", "y"),
    )
    assert result.hessian_provenance.source_refs == (analysis_ref("gradient", "z"),)


def test_numerical_evaluation_has_explicit_provenance() -> None:
    model = _compile(
        """
name: Evaluation provenance
variables:
  x:
    domain: [-2, 2]
parameters:
  a:
    default: 1
    domain: [0, 2]
functions:
  y: a*x
"""
    )

    result = evaluate_single_variable_function(model, points=5)

    assert result.provenance.kind is ProvenanceKind.DERIVED
    assert model_ref("function", "y") in result.provenance.source_refs
    assert model_ref("variable", "x") in result.provenance.source_refs
    assert model_ref("parameter", "a") in result.provenance.source_refs


def test_stationary_point_result_records_analysis_lineage() -> None:
    model = _compile(
        """
name: Stationary provenance
variables:
  x:
    domain: [-2, 2]
parameters:
  a:
    default: 1
    domain: [0.5, 2]
functions:
  y: a*x**2
"""
    )

    result = run_stationary_point_analysis(model)

    assert result.provenance.kind is ProvenanceKind.DERIVED
    assert model_ref("function", "y") in result.provenance.source_refs
    assert model_ref("parameter", "a") in result.provenance.source_refs
    assert "stationary-point" in result.provenance.operation


def test_parameter_sweep_provenance_references_analysis_and_swept_parameter() -> None:
    model = _compile(
        """
name: Sweep provenance
variables:
  x:
    domain: [-2, 2]
parameters:
  a:
    default: 1
    domain: [0.5, 2]
functions:
  y: a*x**2
"""
    )

    result = run_parameter_sweep(model, "a", start=0.5, end=2, step_count=5)

    assert result.provenance.kind is ProvenanceKind.DERIVED
    assert result.provenance.source_refs == (
        analysis_ref("stationary_points", "y"),
        model_ref("parameter", "a"),
    )


def test_suggested_provenance_is_not_active_until_approved() -> None:
    pending = Provenance.suggested(
        (model_ref("function", "y"),),
        "construct optional analytical quantity",
    )
    approved = Provenance.suggested(
        (model_ref("function", "y"),),
        "construct optional analytical quantity",
        approved=True,
    )

    assert pending.kind is ProvenanceKind.SUGGESTED
    assert pending.approval is ProvenanceApproval.PENDING
    assert pending.is_active is False
    assert approved.approval is ProvenanceApproval.APPROVED
    assert approved.is_active is True


def test_numerical_provenance_includes_declared_derived_quantity_lineage() -> None:
    model = _compile(
        """
name: Derived evaluation provenance
variables:
  x: {domain: [-2, 2]}
parameters:
  a: {default: 1, domain: [0, 2]}
derived_quantities:
  q: {expression: a*x}
functions:
  y: q**2
"""
    )

    evaluation = evaluate_single_variable_function(model, points=7)
    stationary = run_stationary_point_analysis(model)

    symbolic = analyse_one_variable_function(model)

    assert model_ref("derived_quantity", "q") in evaluation.provenance.source_refs
    assert model_ref("derived_quantity", "q") in stationary.provenance.source_refs
    assert model_ref("derived_quantity", "q") in symbolic.first_derivative_provenance.source_refs
