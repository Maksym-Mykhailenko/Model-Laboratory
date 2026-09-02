"""Plotly visualisation functions for Model Laboratory.

This module renders already-computed evaluation and analysis result objects.  It performs
no symbolic or numerical analysis itself.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import math

import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator

from .analysis import ParameterSweepResult
from .capabilities import AnalysisVisualisation, ModelVisualisation
from .critical_points import OneDimensionalCriticalPoint, TwoDimensionalCriticalPoint
from .evaluator import FunctionEvaluation, SurfaceEvaluation
from .presentation import clean_display_number
from .registry import CapabilityRegistryError, capability_registry
from .vector_analysis import VectorFieldEvaluation


class VisualisationError(ValueError):
    """Raised when a result cannot be rendered by the requested visualisation."""


_MARKER_SYMBOLS_2D = {
    "local minimum": "circle",
    "local maximum": "diamond",
    "saddle": "x",
    "degenerate / inconclusive": "cross",
}

_MARKER_SYMBOLS_3D = {
    "local minimum": "circle",
    "local maximum": "diamond",
    "saddle": "x",
    "degenerate / inconclusive": "cross",
}

_CLASSIFICATION_LABELS = {
    "local minimum": "Local minimum",
    "local maximum": "Local maximum",
    "saddle": "Saddle",
    "degenerate / inconclusive": "Degenerate / inconclusive",
}

# Deliberately categorical colours: these marks represent classifications, not
# the scalar surface value encoded by the continuous surface colour scale.
_CLASSIFICATION_COLOURS = {
    "local minimum": "#00CC96",
    "local maximum": "#EF553B",
    "saddle": "#AB63FA",
    "degenerate / inconclusive": "#FFA15A",
}

_CLASSIFICATION_ORDER = (
    "local minimum",
    "saddle",
    "local maximum",
    "degenerate / inconclusive",
)


def _group_points(points):
    grouped = defaultdict(list)
    for point in points:
        grouped[point.classification].append(point)
    return grouped


def _ordered_groups(points):
    grouped = _group_points(points)
    for classification in _CLASSIFICATION_ORDER:
        group = grouped.pop(classification, None)
        if group:
            yield classification, group
    for classification in sorted(grouped):
        yield classification, grouped[classification]


def _stationary_point_legend_layout() -> dict:
    """Place stationary-point classifications below the plotting area."""
    return dict(
        orientation="h",
        yanchor="top",
        y=-0.20,
        xanchor="center",
        x=0.5,
        title=None,
        itemsizing="constant",
        tracegroupgap=8,
    )


def _analysis_legend_layout() -> dict:
    """Keep analysis legends outside the data region and away from axis titles."""
    return dict(
        orientation="h",
        yanchor="top",
        y=-0.22,
        xanchor="center",
        x=0.5,
        title=None,
        itemsizing="constant",
        tracegroupgap=8,
    )


def _cartesian_axis(title: str) -> dict:
    """Return an axis layout with enough room for labels in a wide desktop view."""
    return dict(title=dict(text=title, standoff=18), automargin=True)


def _add_one_dimensional_stationary_points(
    figure: go.Figure,
    evaluation: FunctionEvaluation,
    points: Sequence[OneDimensionalCriticalPoint],
) -> None:
    for classification, group in _ordered_groups(points):
        figure.add_scatter(
            x=[point.x for point in group],
            y=[point.value for point in group],
            mode="markers",
            name=_CLASSIFICATION_LABELS.get(classification, classification),
            marker=dict(
                size=11,
                symbol=_MARKER_SYMBOLS_2D.get(classification, "circle"),
                color=_CLASSIFICATION_COLOURS.get(classification),
                line=dict(width=1.5, color="white"),
            ),
            hovertemplate=(
                f"{evaluation.x_name}=%{{x:.8g}}<br>"
                f"{evaluation.y_name}=%{{y:.8g}}<br>"
                f"classification={classification}<extra></extra>"
            ),
        )


def _add_two_dimensional_stationary_points_3d(
    figure: go.Figure,
    evaluation: SurfaceEvaluation,
    points: Sequence[TwoDimensionalCriticalPoint],
) -> None:
    for classification, group in _ordered_groups(points):
        figure.add_trace(
            go.Scatter3d(
                x=[point.x for point in group],
                y=[point.y for point in group],
                z=[point.value for point in group],
                mode="markers",
                name=_CLASSIFICATION_LABELS.get(classification, classification),
                marker=dict(
                    size=6,
                    symbol=_MARKER_SYMBOLS_3D.get(classification, "circle"),
                    color=_CLASSIFICATION_COLOURS.get(classification),
                    line=dict(width=1.5, color="white"),
                ),
                hovertemplate=(
                    f"{evaluation.x_name}=%{{x:.8g}}<br>"
                    f"{evaluation.y_name}=%{{y:.8g}}<br>"
                    f"{evaluation.z_name}=%{{z:.8g}}<br>"
                    f"classification={classification}<extra></extra>"
                ),
            )
        )


def _add_two_dimensional_stationary_points_2d(
    figure: go.Figure,
    evaluation: SurfaceEvaluation,
    points: Sequence[TwoDimensionalCriticalPoint],
) -> None:
    for classification, group in _ordered_groups(points):
        figure.add_scatter(
            x=[point.x for point in group],
            y=[point.y for point in group],
            customdata=[point.value for point in group],
            mode="markers",
            name=_CLASSIFICATION_LABELS.get(classification, classification),
            marker=dict(
                size=11,
                symbol=_MARKER_SYMBOLS_2D.get(classification, "circle"),
                color=_CLASSIFICATION_COLOURS.get(classification),
                line=dict(width=1.5, color="white"),
            ),
            hovertemplate=(
                f"{evaluation.x_name}=%{{x:.8g}}<br>"
                f"{evaluation.y_name}=%{{y:.8g}}<br>"
                f"{evaluation.z_name}=%{{customdata:.8g}}<br>"
                f"classification={classification}<extra></extra>"
            ),
        )


def create_2d_function_figure(
    evaluation: FunctionEvaluation,
    stationary_points: Sequence[OneDimensionalCriticalPoint] = (),
) -> go.Figure:
    """Create an interactive 2D line plot from an evaluated scalar function."""
    figure = go.Figure()
    figure.add_scatter(
        x=evaluation.x,
        y=evaluation.y,
        mode="lines",
        name=evaluation.y_name,
        showlegend=False,
    )
    if stationary_points:
        _add_one_dimensional_stationary_points(figure, evaluation, stationary_points)
    figure.update_layout(
        xaxis=_cartesian_axis(evaluation.x_name),
        yaxis=_cartesian_axis(evaluation.y_name),
        legend=_stationary_point_legend_layout() if stationary_points else None,
        margin=dict(l=36, r=24, t=24, b=120 if stationary_points else 56),
        height=500,
    )
    return figure


def create_3d_surface_figure(
    evaluation: SurfaceEvaluation,
    stationary_points: Sequence[TwoDimensionalCriticalPoint] = (),
) -> go.Figure:
    """Create an interactive 3D surface from a two-variable scalar function."""
    figure = go.Figure(
        data=[
            go.Surface(
                x=evaluation.x,
                y=evaluation.y,
                z=evaluation.z,
                name=evaluation.z_name,
                colorbar=dict(title=evaluation.z_name, x=1.02, xanchor="left", len=0.78),
            )
        ]
    )
    if stationary_points:
        _add_two_dimensional_stationary_points_3d(figure, evaluation, stationary_points)
    figure.update_layout(
        scene=dict(
            xaxis_title=evaluation.x_name,
            yaxis_title=evaluation.y_name,
            zaxis_title=evaluation.z_name,
        ),
        legend=_stationary_point_legend_layout() if stationary_points else None,
        margin=dict(l=24, r=96, t=24, b=120 if stationary_points else 56),
        height=640,
    )
    return figure


def create_contour_figure(
    evaluation: SurfaceEvaluation,
    stationary_points: Sequence[TwoDimensionalCriticalPoint] = (),
) -> go.Figure:
    """Create an interactive contour map from a two-variable scalar function."""
    figure = go.Figure(
        data=[
            go.Contour(
                x=evaluation.x[0, :],
                y=evaluation.y[:, 0],
                z=evaluation.z,
                contours=dict(showlabels=True),
                colorbar=dict(title=evaluation.z_name, x=1.02, xanchor="left"),
            )
        ]
    )
    if stationary_points:
        _add_two_dimensional_stationary_points_2d(figure, evaluation, stationary_points)
    figure.update_layout(
        xaxis=_cartesian_axis(evaluation.x_name),
        yaxis=_cartesian_axis(evaluation.y_name),
        legend=_stationary_point_legend_layout() if stationary_points else None,
        margin=dict(l=36, r=96, t=24, b=120 if stationary_points else 56),
        height=560,
    )
    return figure


def create_heat_map_figure(
    evaluation: SurfaceEvaluation,
    stationary_points: Sequence[TwoDimensionalCriticalPoint] = (),
) -> go.Figure:
    """Create an interactive heat map from a two-variable scalar function."""
    figure = go.Figure(
        data=[
            go.Heatmap(
                x=evaluation.x[0, :],
                y=evaluation.y[:, 0],
                z=evaluation.z,
                colorbar=dict(title=evaluation.z_name, x=1.02, xanchor="left"),
            )
        ]
    )
    if stationary_points:
        _add_two_dimensional_stationary_points_2d(figure, evaluation, stationary_points)
    figure.update_layout(
        xaxis=_cartesian_axis(evaluation.x_name),
        yaxis=_cartesian_axis(evaluation.y_name),
        legend=_stationary_point_legend_layout() if stationary_points else None,
        margin=dict(l=36, r=96, t=24, b=120 if stationary_points else 56),
        height=560,
    )
    return figure


def _streamline_paths(
    evaluation: VectorFieldEvaluation,
    *,
    seed_count: int = 20,
    steps: int = 180,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Integrate deterministic normalised streamlines with fixed-step RK4."""
    x_axis = np.asarray(evaluation.x[0, :], dtype=float)
    y_axis = np.asarray(evaluation.y[:, 0], dtype=float)
    u_interpolator = RegularGridInterpolator(
        (y_axis, x_axis), evaluation.u, bounds_error=False, fill_value=np.nan
    )
    v_interpolator = RegularGridInterpolator(
        (y_axis, x_axis), evaluation.v, bounds_error=False, fill_value=np.nan
    )
    x_span = x_axis[-1] - x_axis[0]
    y_span = y_axis[-1] - y_axis[0]
    step_size = min(x_span, y_span) / 90.0
    edge_count = max(2, seed_count // 4)
    seeds = [
        *(np.asarray([x_axis[0], value]) for value in np.linspace(y_axis[0], y_axis[-1], edge_count)),
        *(np.asarray([x_axis[-1], value]) for value in np.linspace(y_axis[0], y_axis[-1], edge_count)),
        *(np.asarray([value, y_axis[0]]) for value in np.linspace(x_axis[0], x_axis[-1], edge_count)),
        *(np.asarray([value, y_axis[-1]]) for value in np.linspace(x_axis[0], x_axis[-1], edge_count)),
    ]

    def direction(point: np.ndarray) -> np.ndarray:
        query = np.asarray([point[1], point[0]])
        vector = np.asarray([u_interpolator(query), v_interpolator(query)], dtype=float).reshape(2)
        magnitude = float(np.linalg.norm(vector))
        return vector / magnitude if math.isfinite(magnitude) and magnitude > 1e-14 else np.full(2, np.nan)

    def integrate(seed: np.ndarray, sign: float) -> list[np.ndarray]:
        point = seed.astype(float)
        result = [point.copy()]
        for _ in range(steps):
            k1 = direction(point)
            k2 = direction(point + sign * step_size * k1 / 2.0)
            k3 = direction(point + sign * step_size * k2 / 2.0)
            k4 = direction(point + sign * step_size * k3)
            if not np.all(np.isfinite([*k1, *k2, *k3, *k4])):
                break
            point = point + sign * step_size * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            if not (x_axis[0] <= point[0] <= x_axis[-1] and y_axis[0] <= point[1] <= y_axis[-1]):
                break
            result.append(point.copy())
        return result

    paths: list[tuple[np.ndarray, np.ndarray]] = []
    for seed in seeds:
        backward = integrate(seed, -1.0)
        forward = integrate(seed, 1.0)
        combined = [*reversed(backward[1:]), *forward]
        if len(combined) >= 3:
            path = np.asarray(combined)
            paths.append((path[:, 0], path[:, 1]))
    return tuple(paths)


def create_vector_field_figure(evaluation: VectorFieldEvaluation) -> go.Figure:
    """Render arrows, deterministic streamlines, nullclines, and equilibria."""
    figure = go.Figure()
    x_span = float(evaluation.x[0, -1] - evaluation.x[0, 0])
    y_span = float(evaluation.y[-1, 0] - evaluation.y[0, 0])
    arrow_scale = min(x_span / max(1, evaluation.x.shape[1] - 1), y_span / max(1, evaluation.y.shape[0] - 1)) * 0.42
    speed = np.asarray(evaluation.speed, dtype=float)
    normaliser = np.where(speed > 1e-14, speed, np.nan)
    arrow_x: list[float | None] = []
    arrow_y: list[float | None] = []
    for x, y, u, v, magnitude in zip(
        evaluation.x.ravel(),
        evaluation.y.ravel(),
        evaluation.u.ravel(),
        evaluation.v.ravel(),
        normaliser.ravel(),
    ):
        if not np.isfinite(magnitude):
            continue
        arrow_x.extend([float(x), float(x + arrow_scale * u / magnitude), None])
        arrow_y.extend([float(y), float(y + arrow_scale * v / magnitude), None])
    figure.add_scatter(
        x=arrow_x,
        y=arrow_y,
        mode="lines",
        line=dict(color="#667085", width=1),
        opacity=0.72,
        name="Field direction",
        hoverinfo="skip",
    )
    for index, (path_x, path_y) in enumerate(_streamline_paths(evaluation)):
        figure.add_scatter(
            x=path_x,
            y=path_y,
            mode="lines",
            line=dict(color="#2E90FA", width=1.4),
            opacity=0.72,
            name="Streamlines",
            legendgroup="streamlines",
            showlegend=index == 0,
            hoverinfo="skip",
        )
    contour_common = dict(
        x=evaluation.x[0, :],
        y=evaluation.y[:, 0],
        contours=dict(start=0, end=0, size=1, coloring="none", showlabels=False),
        showscale=False,
        hoverinfo="skip",
    )
    figure.add_trace(
        go.Contour(
            z=evaluation.u,
            line=dict(color="#F04438", width=2),
            name=f"{evaluation.component_names[0]} = 0",
            **contour_common,
        )
    )
    figure.add_trace(
        go.Contour(
            z=evaluation.v,
            line=dict(color="#12B76A", width=2, dash="dash"),
            name=f"{evaluation.component_names[1]} = 0",
            **contour_common,
        )
    )
    if evaluation.equilibria:
        figure.add_scatter(
            x=[item.coordinates[0] for item in evaluation.equilibria],
            y=[item.coordinates[1] for item in evaluation.equilibria],
            customdata=[item.stability or "not classified" for item in evaluation.equilibria],
            mode="markers",
            marker=dict(size=11, symbol="x", color="#7F56D9", line=dict(width=2)),
            name="Equilibria",
            hovertemplate=(
                f"{evaluation.x_name}=%{{x:.8g}}<br>{evaluation.y_name}=%{{y:.8g}}"
                "<br>%{customdata}<extra></extra>"
            ),
        )
    figure.update_layout(
        xaxis={**_cartesian_axis(evaluation.x_name), "scaleanchor": "y", "scaleratio": 1},
        yaxis=_cartesian_axis(evaluation.y_name),
        legend=_analysis_legend_layout(),
        margin=dict(l=36, r=24, t=24, b=120),
        height=620,
    )
    return figure


_MODEL_RENDERERS = {
    ModelVisualisation.TWO_D_FUNCTION_PLOT: create_2d_function_figure,
    ModelVisualisation.THREE_D_SURFACE: create_3d_surface_figure,
    ModelVisualisation.CONTOUR_MAP: create_contour_figure,
    ModelVisualisation.HEAT_MAP: create_heat_map_figure,
}


def render_model_visualisation(
    visualisation: ModelVisualisation,
    evaluation: FunctionEvaluation | SurfaceEvaluation,
    stationary_points: Sequence[OneDimensionalCriticalPoint | TwoDimensionalCriticalPoint] = (),
) -> go.Figure:
    """Dispatch a model evaluation using the registry's declared type contract."""
    try:
        definition = capability_registry.definition(visualisation)
    except CapabilityRegistryError as exc:
        raise VisualisationError(str(exc)) from exc

    if not any(isinstance(evaluation, accepted) for accepted in definition.accepted_result_types):
        accepted_names = ", ".join(item.__name__ for item in definition.accepted_result_types)
        raise VisualisationError(
            f"'{visualisation.value}' cannot render {type(evaluation).__name__}; "
            f"expected {accepted_names}."
        )

    renderer = _MODEL_RENDERERS.get(visualisation)
    if renderer is None:
        raise VisualisationError(f"No renderer is implemented for '{visualisation.value}'.")
    return renderer(evaluation, stationary_points)  # type: ignore[arg-type]


def create_parameter_sweep_count_figure(result: ParameterSweepResult) -> go.Figure:
    """Plot the number of stationary points in each classification across a sweep."""
    figure = go.Figure()
    parameter_values = [step.parameter_value for step in result.steps]

    for classification in _CLASSIFICATION_ORDER:
        counts: list[int | None] = []
        for step in result.steps:
            if step.error is not None:
                counts.append(None)
            else:
                counts.append(sum(point.classification == classification for point in step.points))
        if any(value not in (0, None) for value in counts):
            figure.add_scatter(
                x=parameter_values,
                y=counts,
                mode="lines+markers",
                name=_CLASSIFICATION_LABELS[classification],
                line=dict(color=_CLASSIFICATION_COLOURS[classification]),
                marker=dict(symbol=_MARKER_SYMBOLS_2D[classification], size=8),
                connectgaps=False,
            )

    if not figure.data:
        figure.add_scatter(
            x=parameter_values,
            y=[None if step.error else 0 for step in result.steps],
            mode="lines+markers",
            name="Stationary points",
        )

    figure.update_layout(
        xaxis=_cartesian_axis(result.parameter_name),
        yaxis=_cartesian_axis("Stationary-point count"),
        legend=_analysis_legend_layout(),
        margin=dict(l=36, r=24, t=24, b=120),
        height=500,
        hovermode="x unified",
    )
    return figure


def create_parameter_sweep_position_figure(result: ParameterSweepResult) -> go.Figure:
    """Plot stationary-point coordinates against the swept parameter.

    Points are deliberately not connected into branches.  The current sweep does not infer
    correspondence between stationary points at adjacent parameter values.
    """
    figure = go.Figure()

    for variable_index, variable_name in enumerate(result.variable_names):
        for classification in _CLASSIFICATION_ORDER:
            x_values: list[float] = []
            y_values: list[float] = []
            custom_values: list[float] = []
            for step in result.steps:
                if step.error is not None:
                    continue
                for point in step.points:
                    if point.classification != classification:
                        continue
                    coordinate = point.x if variable_index == 0 else point.y  # type: ignore[attr-defined]
                    x_values.append(clean_display_number(step.parameter_value))
                    y_values.append(clean_display_number(float(coordinate)))
                    custom_values.append(clean_display_number(float(point.value)))

            if x_values:
                label = _CLASSIFICATION_LABELS[classification]
                trace_name = label if len(result.variable_names) == 1 else f"{variable_name} — {label}"
                figure.add_scatter(
                    x=x_values,
                    y=y_values,
                    customdata=custom_values,
                    mode="markers",
                    name=trace_name,
                    marker=dict(
                        color=_CLASSIFICATION_COLOURS[classification],
                        symbol=_MARKER_SYMBOLS_2D[classification],
                        size=8,
                        opacity=0.8 if variable_index == 0 else 0.55,
                    ),
                    hovertemplate=(
                        f"{result.parameter_name}=%{{x:.8g}}<br>"
                        f"{variable_name}=%{{y:.8g}}<br>"
                        f"{result.function_name}=%{{customdata:.8g}}<br>"
                        f"classification={classification}<extra></extra>"
                    ),
                )

    figure.update_layout(
        xaxis=_cartesian_axis(result.parameter_name),
        yaxis=_cartesian_axis("Stationary-point coordinate"),
        legend=_analysis_legend_layout(),
        margin=dict(l=36, r=24, t=24, b=120),
        height=500,
    )
    return figure


def create_parameter_sweep_value_figure(result: ParameterSweepResult) -> go.Figure:
    """Plot scalar-function values at stationary points across the swept parameter."""
    figure = go.Figure()

    for classification in _CLASSIFICATION_ORDER:
        x_values: list[float] = []
        y_values: list[float] = []
        for step in result.steps:
            if step.error is not None:
                continue
            for point in step.points:
                if point.classification == classification:
                    x_values.append(clean_display_number(step.parameter_value))
                    y_values.append(clean_display_number(float(point.value)))

        if x_values:
            figure.add_scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name=_CLASSIFICATION_LABELS[classification],
                marker=dict(
                    color=_CLASSIFICATION_COLOURS[classification],
                    symbol=_MARKER_SYMBOLS_2D[classification],
                    size=8,
                ),
                hovertemplate=(
                    f"{result.parameter_name}=%{{x:.8g}}<br>"
                    f"{result.function_name}=%{{y:.8g}}<br>"
                    f"classification={classification}<extra></extra>"
                ),
            )

    figure.update_layout(
        xaxis=_cartesian_axis(result.parameter_name),
        yaxis=_cartesian_axis(result.function_name),
        legend=_analysis_legend_layout(),
        margin=dict(l=36, r=24, t=24, b=120),
        height=500,
    )
    return figure


_ANALYSIS_RENDERERS = {
    AnalysisVisualisation.SWEEP_CLASSIFICATION_COUNTS: create_parameter_sweep_count_figure,
    AnalysisVisualisation.SWEEP_POSITIONS: create_parameter_sweep_position_figure,
    AnalysisVisualisation.SWEEP_FUNCTION_VALUES: create_parameter_sweep_value_figure,
}


def render_analysis_visualisation(
    visualisation: AnalysisVisualisation,
    result: object,
) -> go.Figure:
    """Dispatch a precomputed analysis result using the registry's type contract."""
    try:
        definition = capability_registry.definition(visualisation)
    except CapabilityRegistryError as exc:
        raise VisualisationError(str(exc)) from exc

    if not any(isinstance(result, accepted) for accepted in definition.accepted_result_types):
        accepted_names = ", ".join(item.__name__ for item in definition.accepted_result_types)
        raise VisualisationError(
            f"'{visualisation.value}' cannot render {type(result).__name__}; "
            f"expected {accepted_names}."
        )

    renderer = _ANALYSIS_RENDERERS.get(visualisation)
    if renderer is None:
        raise VisualisationError(
            f"'{visualisation.value}' is registered as an overlay or has no standalone renderer."
        )
    return renderer(result)  # type: ignore[arg-type]
