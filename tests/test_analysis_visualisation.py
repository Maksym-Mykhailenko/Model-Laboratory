from __future__ import annotations

from model_lab.analysis import run_parameter_sweep
from model_lab.capabilities import AnalysisVisualisation
from model_lab.parser import parse_model_text
from model_lab.validator import validate_model
from model_lab.visualisation import render_analysis_visualisation


def _sweep_result():
    model = validate_model(
        parse_model_text(
            """
name: Quadratic family
variables:
  x:
    domain: [-3, 3]
parameters:
  a:
    default: 0
    domain: [-2, 2]
functions:
  y: x**2 + a*x
"""
        )
    )
    return run_parameter_sweep(model, "a", start=-2, end=2, step_count=3)


def test_sweep_count_visualisation_uses_precomputed_result() -> None:
    figure = render_analysis_visualisation(
        AnalysisVisualisation.SWEEP_CLASSIFICATION_COUNTS,
        _sweep_result(),
    )

    assert len(figure.data) == 1
    assert figure.data[0].name == "Local minimum"
    assert tuple(figure.data[0].y) == (1, 1, 1)


def test_sweep_position_visualisation_does_not_connect_inferred_branches() -> None:
    figure = render_analysis_visualisation(
        AnalysisVisualisation.SWEEP_POSITIONS,
        _sweep_result(),
    )

    assert len(figure.data) == 1
    assert figure.data[0].mode == "markers"
    assert tuple(figure.data[0].y) == (1.0, 0.0, -1.0)


def test_sweep_function_value_visualisation_is_available() -> None:
    figure = render_analysis_visualisation(
        AnalysisVisualisation.SWEEP_FUNCTION_VALUES,
        _sweep_result(),
    )

    assert len(figure.data) == 1
    assert figure.data[0].name == "Local minimum"
    assert tuple(round(value, 6) for value in figure.data[0].y) == (-1.0, 0.0, -1.0)


def test_sweep_visualisation_keeps_legend_outside_data_region() -> None:
    figure = render_analysis_visualisation(
        AnalysisVisualisation.SWEEP_CLASSIFICATION_COUNTS,
        _sweep_result(),
    )

    assert figure.layout.legend.orientation == "h"
    assert figure.layout.legend.y <= -0.20
    assert figure.layout.legend.title.text is None
    assert figure.layout.margin.b >= 120
