from __future__ import annotations

import pytest

from model_lab.analysis import ParameterSweepResult, StationaryPointAnalysisResult
from model_lab.capabilities import (
    AnalysisCapability,
    AnalysisVisualisation,
    ControlCapability,
    ModelVisualisation,
)
from model_lab.evaluator import FunctionEvaluation, SurfaceEvaluation
from model_lab.parser import parse_model_text
from model_lab.registry import (
    CapabilityDefinition,
    CapabilityKind,
    CapabilityRegistry,
    CapabilityRegistryError,
    AnalysisVisualisationMode,
    capability_registry,
)
from model_lab.symbolic import OneVariableSymbolicAnalysis
from model_lab.validator import validate_model


def _model(text: str):
    return validate_model(parse_model_text(text))


def test_every_public_capability_identifier_is_registered_once() -> None:
    registered = {definition.key for definition in capability_registry.definitions}
    expected = {
        *ModelVisualisation,
        *AnalysisCapability,
        *AnalysisVisualisation,
        *ControlCapability,
    }

    assert registered == expected
    assert len(capability_registry.definitions) == len(expected)


def test_analysis_definition_declares_runner_result_type_and_renderers() -> None:
    definition = capability_registry.definition(AnalysisCapability.PARAMETER_SWEEP)

    assert definition.kind == CapabilityKind.ANALYSIS
    assert callable(definition.runner)
    assert definition.result_type is ParameterSweepResult
    assert definition.compatible_visualisations == (
        AnalysisVisualisation.SWEEP_CLASSIFICATION_COUNTS,
        AnalysisVisualisation.SWEEP_POSITIONS,
        AnalysisVisualisation.SWEEP_FUNCTION_VALUES,
    )


def test_model_visualisations_declare_the_evaluation_types_they_accept() -> None:
    curve = capability_registry.definition(ModelVisualisation.TWO_D_FUNCTION_PLOT)
    surface = capability_registry.definition(ModelVisualisation.THREE_D_SURFACE)

    assert curve.accepted_result_types == (FunctionEvaluation,)
    assert surface.accepted_result_types == (SurfaceEvaluation,)


def test_registry_exposes_unmet_requirements_without_guessing() -> None:
    model = _model(
        """
name: Curve
variables:
  x:
    domain: [-2, 2]
parameters: {}
functions:
  y: x**2
"""
    )
    definition = capability_registry.definition(AnalysisCapability.PARAMETER_SWEEP)

    assert not definition.is_applicable_to(model)
    assert tuple(item.identifier for item in definition.unmet_requirements(model)) == (
        "has_parameters",
    )


def test_registry_can_execute_a_registered_analysis() -> None:
    model = _model(
        """
name: Curve
variables:
  x:
    domain: [-2, 2]
parameters: {}
functions:
  y: x**3
"""
    )

    result = capability_registry.run_analysis(AnalysisCapability.FIRST_DERIVATIVE, model)

    assert isinstance(result, OneVariableSymbolicAnalysis)
    assert str(result.first_derivative) == "3*x**2"


def test_analysis_visualisation_contract_matches_analysis_result_type() -> None:
    stationary = capability_registry.definition(AnalysisVisualisation.STATIONARY_POINT_OVERLAY)
    sweep = capability_registry.definition(AnalysisVisualisation.SWEEP_POSITIONS)

    assert stationary.accepted_result_types == (StationaryPointAnalysisResult,)
    assert stationary.visualisation_mode == AnalysisVisualisationMode.OVERLAY
    assert sweep.accepted_result_types == (ParameterSweepResult,)
    assert sweep.visualisation_mode == AnalysisVisualisationMode.STANDALONE


def test_duplicate_registration_is_rejected() -> None:
    registry = CapabilityRegistry()
    definition = CapabilityDefinition(
        key=ControlCapability.PARAMETER_CONTROLS,
        kind=CapabilityKind.CONTROL,
    )
    registry.register(definition)

    with pytest.raises(CapabilityRegistryError, match="already registered"):
        registry.register(definition)


def test_constraint_sensitive_analyses_are_not_offered_for_constrained_models() -> None:
    model = _model(
        """
name: Constrained curve
variables:
  x:
    domain: [-2, 2]
parameters:
  a:
    default: 1
    domain: [0, 2]
functions:
  y: a*x**2
constraints:
  feasible:
    left: x
    relation: ">="
    right: 0
"""
    )

    report = capability_registry.detect(model)
    assert AnalysisCapability.STATIONARY_POINTS not in report.analyses
    assert AnalysisCapability.PARAMETER_SWEEP not in report.analyses
    assert AnalysisCapability.FIRST_DERIVATIVE in report.analyses
    assert ModelVisualisation.TWO_D_FUNCTION_PLOT in report.visualisations

    definition = capability_registry.definition(AnalysisCapability.STATIONARY_POINTS)
    assert tuple(item.identifier for item in definition.unmet_requirements(model)) == (
        "no_constraints",
    )
