from __future__ import annotations

from model_lab.capabilities import (
    AnalysisCapability,
    AnalysisVisualisation,
    ControlCapability,
    ModelVisualisation,
    compatible_analysis_visualisations,
    detect_capabilities,
)
from model_lab.parser import parse_model_text
from model_lab.validator import validate_model


def _model(text: str):
    return validate_model(parse_model_text(text))


def test_two_variable_function_offers_three_model_visualisations() -> None:
    model = _model(
        """
name: Surface
variables:
  x:
    domain: [-2, 2]
  y:
    domain: [-3, 3]
parameters: {}
functions:
  z: x**2 + y**2
"""
    )
    report = detect_capabilities(model)

    assert ModelVisualisation.THREE_D_SURFACE in report.visualisations
    assert ModelVisualisation.CONTOUR_MAP in report.visualisations
    assert ModelVisualisation.HEAT_MAP in report.visualisations
    assert ModelVisualisation.TWO_D_FUNCTION_PLOT not in report.visualisations


def test_one_variable_function_offers_symbolic_derivatives() -> None:
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
    report = detect_capabilities(model)

    assert AnalysisCapability.FIRST_DERIVATIVE in report.analyses
    assert AnalysisCapability.SECOND_DERIVATIVE in report.analyses
    assert AnalysisCapability.GRADIENT not in report.analyses
    assert AnalysisCapability.HESSIAN not in report.analyses


def test_two_variable_function_offers_gradient_and_hessian() -> None:
    model = _model(
        """
name: Surface analysis
variables:
  x:
    domain: [-2, 2]
  y:
    domain: [-2, 2]
parameters: {}
functions:
  z: x**2 + y**2
"""
    )
    report = detect_capabilities(model)

    assert AnalysisCapability.GRADIENT in report.analyses
    assert AnalysisCapability.HESSIAN in report.analyses
    assert AnalysisCapability.FIRST_DERIVATIVE not in report.analyses
    assert AnalysisCapability.SECOND_DERIVATIVE not in report.analyses


def test_parameter_sweep_is_analysis_not_model_visualisation() -> None:
    model = _model(
        """
name: Parameterised curve
variables:
  x:
    domain: [-3, 3]
parameters:
  a:
    default: 1
    domain: [-2, 2]
functions:
  y: x**2 + a*x
"""
    )
    report = detect_capabilities(model)

    assert AnalysisCapability.PARAMETER_SWEEP in report.analyses
    assert ControlCapability.PARAMETER_CONTROLS in report.controls
    assert all("sweep" not in visualisation.value for visualisation in report.visualisations)


def test_parameter_sweep_reports_its_own_compatible_visualisations() -> None:
    model = _model(
        """
name: Parameterised curve
variables:
  x:
    domain: [-3, 3]
parameters:
  a:
    default: 1
    domain: [-2, 2]
functions:
  y: x**2 + a*x
"""
    )

    visualisations = compatible_analysis_visualisations(AnalysisCapability.PARAMETER_SWEEP, model)

    assert visualisations == (
        AnalysisVisualisation.SWEEP_CLASSIFICATION_COUNTS,
        AnalysisVisualisation.SWEEP_POSITIONS,
        AnalysisVisualisation.SWEEP_FUNCTION_VALUES,
    )
