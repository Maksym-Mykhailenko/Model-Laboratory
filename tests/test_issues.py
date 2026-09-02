from __future__ import annotations

import pytest

from model_lab.analysis import AnalysisError, run_stationary_point_analysis
from model_lab.capabilities import AnalysisCapability, ControlCapability, ModelVisualisation, detect_capabilities
from model_lab.evaluator import EvaluationError, evaluate_single_variable_function
from model_lab.issues import AmbiguityStatus
from model_lab.parser import parse_model_text
from model_lab.provenance import ProvenanceKind, model_ref
from model_lab.validator import ModelValidationError, validate_model


def _compile(text: str):
    return validate_model(parse_model_text(text))


def test_assumptions_and_ambiguities_compile_into_explicit_ir() -> None:
    model = _compile(
        """
name: Explicit issues
variables:
  x:
    domain: [-1, 1]
parameters:
  a:
    default: 1
    domain: [0, 2]
functions:
  y: a*x
assumptions:
  local_linearity:
    statement: The relation is treated as linear over the declared domain.
    affects: [y]
ambiguities:
  parameter_role:
    statement: The role of a is not yet fixed.
    options: [fixed, adjustable]
    blocking: false
    affects: [a]
"""
    )

    assumption = model.assumption("local_linearity")
    ambiguity = model.ambiguity("parameter_role")

    assert assumption.statement.startswith("The relation")
    assert assumption.affects == ("y",)
    assert assumption.provenance.kind is ProvenanceKind.SUPPLIED
    assert assumption.provenance.source_location == "assumptions.local_linearity"

    assert ambiguity.status is AmbiguityStatus.UNRESOLVED
    assert ambiguity.options == ("fixed", "adjustable")
    assert ambiguity.blocking is False
    assert ambiguity.affects == ("a",)
    assert ambiguity.provenance.source_location == "ambiguities.parameter_role"


def test_unresolved_blocking_ambiguity_blocks_computation_capabilities() -> None:
    model = _compile(
        """
name: Blocking ambiguity
variables:
  x:
    domain: [-1, 1]
parameters:
  a:
    default: 1
    domain: [0, 2]
functions:
  y: a*x
ambiguities:
  interpretation:
    statement: The mathematical role of a is unresolved.
    blocking: true
    affects: [a]
"""
    )

    report = detect_capabilities(model)

    assert model.is_computation_ready is False
    assert len(model.blocking_ambiguities) == 1
    assert report.visualisations == ()
    assert report.analyses == ()
    assert report.controls == (ControlCapability.PARAMETER_CONTROLS,)


def test_resolved_blocking_ambiguity_no_longer_blocks_computation() -> None:
    model = _compile(
        """
name: Resolved ambiguity
variables:
  x:
    domain: [-1, 1]
functions:
  y: x**2
ambiguities:
  interpretation:
    statement: Two interpretations were initially possible.
    options: [first, second]
    resolution: first
    blocking: true
    affects: [y]
"""
    )

    report = detect_capabilities(model)

    assert model.ambiguity("interpretation").status is AmbiguityStatus.RESOLVED
    assert model.is_computation_ready is True
    assert ModelVisualisation.TWO_D_FUNCTION_PLOT in report.visualisations
    assert AnalysisCapability.STATIONARY_POINTS in report.analyses


def test_unknown_issue_target_is_rejected() -> None:
    specification = parse_model_text(
        """
name: Invalid issue target
variables:
  x:
    domain: [-1, 1]
functions:
  y: x
assumptions:
  bad_reference:
    statement: This assumption names something that is not in the model.
    affects: [missing]
"""
    )

    with pytest.raises(ModelValidationError, match="unknown model element.*missing"):
        validate_model(specification)


def test_assumption_and_ambiguity_provenance_entries_are_exposed() -> None:
    model = _compile(
        """
name: Issue provenance
variables:
  x:
    domain: [-1, 1]
functions:
  y: x
assumptions:
  a1:
    statement: One explicit assumption.
ambiguities:
  u1:
    statement: One unresolved non-blocking ambiguity.
    blocking: false
"""
    )

    entries = dict(model.provenance_entries())

    assert entries[model_ref("assumption", "a1")].source_location == "assumptions.a1"
    assert entries[model_ref("ambiguity", "u1")].source_location == "ambiguities.u1"


def test_blocking_ambiguity_is_enforced_by_direct_evaluation_and_analysis_calls() -> None:
    model = _compile(
        """
name: Blocking direct calls
variables:
  x:
    domain: [-1, 1]
functions:
  y: x**2
ambiguities:
  interpretation:
    statement: A blocking interpretation remains unresolved.
    blocking: true
    affects: [y]
"""
    )

    with pytest.raises(EvaluationError, match="blocking ambiguity"):
        evaluate_single_variable_function(model)
    with pytest.raises(AnalysisError, match="blocking ambiguity"):
        run_stationary_point_analysis(model)


def test_non_blocking_ambiguity_does_not_prevent_deterministic_evaluation() -> None:
    model = _compile(
        """
name: Non-blocking direct call
variables:
  x:
    domain: [-1, 1]
functions:
  y: x**2
ambiguities:
  interpretation:
    statement: A semantic question remains open but does not alter the supplied formalisation.
    blocking: false
    affects: [y]
"""
    )

    result = evaluate_single_variable_function(model, points=11)

    assert result.y.shape == (11,)
    assert model.is_computation_ready is True


def test_resolved_blocking_ambiguity_is_enforced_as_resolved_by_direct_analysis() -> None:
    model = _compile(
        """
name: Resolved direct call
variables:
  x:
    domain: [-1, 1]
functions:
  y: x**2
ambiguities:
  interpretation:
    statement: The interpretation has been fixed explicitly.
    options: [first, second]
    resolution: first
    blocking: true
    affects: [y]
"""
    )

    result = run_stationary_point_analysis(model)

    assert len(result.points) == 1
    assert result.numerical_status.value == "complete"


def test_resolution_must_match_declared_option_when_options_are_present() -> None:
    with pytest.raises(Exception, match="resolution must match one of the declared options"):
        parse_model_text(
            """
name: Invalid ambiguity resolution
variables:
  x: {domain: [-1, 1]}
functions:
  y: x
ambiguities:
  interpretation:
    statement: Choose one interpretation.
    options: [first, second]
    resolution: third
    affects: [y]
"""
        )


def test_free_form_resolution_is_allowed_when_no_options_are_declared() -> None:
    model = _compile(
        """
name: Free-form resolution
variables:
  x: {domain: [-1, 1]}
functions:
  y: x
ambiguities:
  interpretation:
    statement: Record an explicit resolution.
    resolution: resolved by supplied equation
    affects: [y]
"""
    )
    assert model.ambiguity("interpretation").resolution == "resolved by supplied equation"
