from __future__ import annotations

import numpy as np

from model_lab.critical_points import (
    OneDimensionalCriticalPoint,
    TwoDimensionalCriticalPoint,
)
from model_lab.evaluator import FunctionEvaluation, SurfaceEvaluation
from model_lab.provenance import Provenance
TEST_PROVENANCE = Provenance.derived(("model:function:test",), "test evaluation")


from model_lab.visualization import (
    create_2d_function_figure,
    create_3d_surface_figure,
    create_contour_figure,
)


def test_one_dimensional_stationary_point_is_overlaid() -> None:
    evaluation = FunctionEvaluation(
        x_name="x",
        y_name="y",
        x=np.array([-1.0, 0.0, 1.0]),
        y=np.array([1.0, 0.0, 1.0]),
        provenance=TEST_PROVENANCE,
    )
    point = OneDimensionalCriticalPoint(
        x=0.0,
        value=0.0,
        second_derivative=2.0,
        classification="local minimum",
    )

    figure = create_2d_function_figure(evaluation, (point,))

    assert len(figure.data) == 2
    assert figure.data[1].name == "Local minimum"
    assert tuple(figure.data[1].x) == (0.0,)
    assert tuple(figure.data[1].y) == (0.0,)


def _surface_evaluation() -> SurfaceEvaluation:
    x_values = np.array([-1.0, 0.0, 1.0])
    y_values = np.array([-1.0, 0.0, 1.0])
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
    return SurfaceEvaluation(
        x_name="x",
        y_name="y",
        z_name="z",
        x=x_grid,
        y=y_grid,
        z=x_grid**2 + y_grid**2,
        provenance=TEST_PROVENANCE,
    )


def _surface_point() -> TwoDimensionalCriticalPoint:
    return TwoDimensionalCriticalPoint(
        x=0.0,
        y=0.0,
        value=0.0,
        eigenvalues=(2.0, 2.0),
        classification="local minimum",
    )


def test_stationary_point_is_overlaid_on_3d_surface() -> None:
    figure = create_3d_surface_figure(_surface_evaluation(), (_surface_point(),))

    assert len(figure.data) == 2
    assert figure.data[1].type == "scatter3d"
    assert figure.data[1].name == "Local minimum"
    assert tuple(figure.data[1].z) == (0.0,)


def test_stationary_point_is_overlaid_on_contour_map() -> None:
    figure = create_contour_figure(_surface_evaluation(), (_surface_point(),))

    assert len(figure.data) == 2
    assert figure.data[1].type == "scatter"
    assert figure.data[1].name == "Local minimum"
    assert tuple(figure.data[1].customdata) == (0.0,)


def test_3d_stationary_point_legend_is_kept_clear_of_colour_bar() -> None:
    minimum = _surface_point()
    saddle = TwoDimensionalCriticalPoint(
        x=0.5,
        y=0.5,
        value=0.5,
        eigenvalues=(-1.0, 1.0),
        classification="saddle",
    )
    maximum = TwoDimensionalCriticalPoint(
        x=1.0,
        y=1.0,
        value=2.0,
        eigenvalues=(-2.0, -2.0),
        classification="local maximum",
    )

    figure = create_3d_surface_figure(_surface_evaluation(), (maximum, saddle, minimum))

    assert figure.layout.legend.orientation == "h"
    assert figure.layout.legend.y < 0
    assert figure.data[0].colorbar.x > 1
    assert [trace.name for trace in figure.data[1:]] == [
        "Local minimum",
        "Saddle",
        "Local maximum",
    ]


def test_one_dimensional_plot_keeps_stationary_legend_clear_of_axis_title() -> None:
    evaluation = FunctionEvaluation(
        x_name="x",
        y_name="y",
        x=np.array([-1.0, 0.0, 1.0]),
        y=np.array([1.0, 0.0, 1.0]),
        provenance=TEST_PROVENANCE,
    )
    point = OneDimensionalCriticalPoint(
        x=0.0,
        value=0.0,
        second_derivative=2.0,
        classification="local minimum",
    )

    figure = create_2d_function_figure(evaluation, (point,))

    assert figure.data[0].showlegend is False
    assert figure.layout.legend.orientation == "h"
    assert figure.layout.legend.y <= -0.20
    assert figure.layout.legend.title.text is None
    assert figure.layout.margin.b >= 120


def test_two_dimensional_plot_reserves_space_for_stationary_legend() -> None:
    figure = create_3d_surface_figure(_surface_evaluation(), (_surface_point(),))

    assert figure.layout.legend.y <= -0.20
    assert figure.layout.margin.b >= 120
    assert figure.layout.height >= 600
