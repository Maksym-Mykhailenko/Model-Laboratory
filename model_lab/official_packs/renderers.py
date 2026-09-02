"""Deterministic Plotly renderers for official-pack scientific results."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import plotly.graph_objects as go

from .generative import (
    ActiveInferenceEvaluation, HiddenMarkovInference, MarkovBlanketAnalysis, POMDPDecision,
)
from .dynamics import (
    ODEEquilibriumAnalysis,
    ODETrajectory,
    StateSpaceAnalysis,
    StateSpaceTrajectory,
)
from .fields import (
    DiffusionSolution,
    PoissonSolution,
    ScalarFieldAnalysis,
    VectorFieldAnalysis,
)
from .geometry import MeshShortestPath, PointCloudAnalysis, TriangleMeshAnalysis
from .graphs import NetworkAnalysis
from .mechanics import (
    IsotropicMaterialAnalysis,
    TrussModalAnalysis,
    TrussStaticResult,
)
from .multidimensional import ArrayAnalysis, QuantityNormalisation, TensorContraction
from .statistics import DatasetAnalysis, GroupComparison, LinearModelFit
from .optimisation import LinearInverseSolution, NonlinearLeastSquaresFit, OptimisationResult
from .electrical import CircuitACAnalysis, CircuitDCAnalysis, DiodeCharacteristic, ElectrostaticAnalysis
from .reactions import (
    CompartmentAnalysis,
    CompartmentTrajectory,
    PopulationAnalysis,
    PopulationTrajectory,
    ReactionNetworkAnalysis,
    ReactionTrajectory,
)
from .learning import (
    ClusteringResult, FuzzyInference, NeuralNetworkEvaluation, SupervisedLearningFit,
)
from .probability import DistributionAnalysis, MarkovEvolution, MarkovSimulation


def _layout(figure: go.Figure, title: str, *, x_title: str = "", y_title: str = "") -> go.Figure:
    figure.update_layout(
        title=title, template="plotly_white", margin=dict(l=55, r=30, t=65, b=50),
        xaxis_title=x_title, yaxis_title=y_title, legend_title_text="",
    )
    return figure


def _array(values: np.ndarray, title: str) -> go.Figure:
    data = np.asarray(values)
    if data.ndim == 1:
        return _layout(go.Figure(go.Scatter(x=np.arange(data.size), y=data, mode="lines+markers")), title, x_title="Index", y_title="Value")
    if data.ndim == 2:
        return _layout(go.Figure(go.Heatmap(z=data, colorscale="Viridis")), title, x_title="Column", y_title="Row")
    flattened = data.reshape(-1)
    return _layout(go.Figure(go.Scattergl(x=np.arange(flattened.size), y=flattened, mode="markers")), f"{title} (flattened)", x_title="Flat index", y_title="Value")


def _state_probabilities(states: tuple[str, ...], values: np.ndarray, title: str) -> go.Figure:
    figure = go.Figure()
    for index, state in enumerate(states):
        figure.add_trace(go.Scatter(x=np.arange(values.shape[0]), y=values[:, index], mode="lines+markers", name=state))
    return _layout(figure, title, x_title="Step", y_title="Probability")


def _network(
    nodes: tuple[str, ...], edges: tuple[Mapping[str, Any], ...], title: str,
    highlighted: set[str] | None = None, target: str | None = None,
) -> go.Figure:
    count = len(nodes)
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    positions = {node: (float(np.cos(angle)), float(np.sin(angle))) for node, angle in zip(nodes, angles)}
    figure = go.Figure()
    for edge in edges:
        x0, y0 = positions[str(edge["source"])]
        x1, y1 = positions[str(edge["target"])]
        figure.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color="#8495a6", width=1.5), hoverinfo="skip", showlegend=False))
    highlighted = highlighted or set()
    colours = ["#c0392b" if node == target else "#0f9d8c" if node in highlighted else "#164b63" for node in nodes]
    figure.add_trace(go.Scatter(
        x=[positions[node][0] for node in nodes], y=[positions[node][1] for node in nodes],
        text=list(nodes), mode="markers+text", textposition="top center",
        marker=dict(size=16, color=colours, line=dict(color="white", width=1)),
        hovertemplate="%{text}<extra></extra>", showlegend=False,
    ))
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return _layout(figure, title)


def _trajectory(times: np.ndarray, values: np.ndarray, names: tuple[str, ...], title: str) -> go.Figure:
    figure = go.Figure()
    for index, name in enumerate(names):
        figure.add_trace(go.Scatter(x=times, y=values[:, index], mode="lines", name=name))
    return _layout(figure, title, x_title="Time", y_title="State")


def _complex_spectrum(values: tuple[complex, ...], title: str) -> go.Figure:
    figure = go.Figure(go.Scatter(
        x=[value.real for value in values], y=[value.imag for value in values],
        mode="markers", marker=dict(size=10, color="#0f9d8c"),
        hovertemplate="%{x:.6g} %{y:+.6g}i<extra></extra>",
    ))
    figure.add_vline(x=0.0, line_dash="dash", line_color="#c0392b")
    return _layout(figure, title, x_title="Real part", y_title="Imaginary part")


def _scalar_field_figure(result: ScalarFieldAnalysis) -> go.Figure:
    if len(result.axis_names) == 1:
        return _layout(
            go.Figure(go.Scatter(x=result.coordinates[0], y=result.values, mode="lines")),
            f"Scalar field · {result.object_id}", x_title=result.axis_names[0], y_title=result.value_name,
        )
    if len(result.axis_names) == 2:
        return _layout(
            go.Figure(go.Heatmap(
                x=result.coordinates[1], y=result.coordinates[0], z=result.values,
                colorscale="Viridis", colorbar_title=result.value_name,
            )), f"Scalar field · {result.object_id}",
            x_title=result.axis_names[1], y_title=result.axis_names[0],
        )
    middle = len(result.coordinates[0]) // 2
    return _layout(
        go.Figure(go.Heatmap(
            x=result.coordinates[2], y=result.coordinates[1], z=result.values[middle],
            colorscale="Viridis", colorbar_title=result.value_name,
        )), f"Scalar field · {result.object_id} · {result.axis_names[0]} slice {middle}",
        x_title=result.axis_names[2], y_title=result.axis_names[1],
    )


def _vector_field_figure(result: VectorFieldAnalysis) -> go.Figure:
    if len(result.axis_names) == 2:
        first, second = np.meshgrid(result.coordinates[0], result.coordinates[1], indexing="ij")
        stride = max(1, int(max(first.shape) / 24))
        x, y = first[::stride, ::stride], second[::stride, ::stride]
        u, v = result.values[0, ::stride, ::stride], result.values[1, ::stride, ::stride]
        scale = float(np.max(np.sqrt(u * u + v * v), initial=0.0))
        span = max(float(np.ptp(result.coordinates[0])), float(np.ptp(result.coordinates[1])))
        factor = 0.08 * span / scale if scale > 0.0 else 0.0
        figure = go.Figure()
        for x0, y0, du, dv in zip(x.ravel(), y.ravel(), u.ravel(), v.ravel()):
            figure.add_trace(go.Scatter(
                x=[x0, x0 + factor * du], y=[y0, y0 + factor * dv], mode="lines",
                line=dict(color="#164b63", width=1.5), hoverinfo="skip", showlegend=False,
            ))
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
        return _layout(figure, f"Vector field · {result.object_id}", x_title=result.axis_names[0], y_title=result.axis_names[1])
    x, y, z = np.meshgrid(*result.coordinates, indexing="ij")
    stride = max(1, int(max(x.shape) / 10))
    selection = (slice(None, None, stride),) * 3
    figure = go.Figure(go.Cone(
        x=x[selection].ravel(), y=y[selection].ravel(), z=z[selection].ravel(),
        u=result.values[0][selection].ravel(), v=result.values[1][selection].ravel(),
        w=result.values[2][selection].ravel(), colorscale="Viridis", sizemode="absolute",
    ))
    figure.update_layout(title=f"Vector field · {result.object_id}", template="plotly_white")
    return figure


def _point_cloud_figure(result: PointCloudAnalysis) -> go.Figure:
    if result.embedding_dimension == 2:
        return _layout(go.Figure(go.Scatter(
            x=result.points[:, 0], y=result.points[:, 1], text=list(result.labels), mode="markers",
            hovertemplate="%{text}<extra></extra>",
        )), f"Point cloud · {result.object_id}", x_title="x", y_title="y")
    figure = go.Figure(go.Scatter3d(
        x=result.points[:, 0], y=result.points[:, 1], z=result.points[:, 2],
        text=list(result.labels), mode="markers", hovertemplate="%{text}<extra></extra>",
    ))
    figure.update_layout(title=f"Point cloud · {result.object_id}", template="plotly_white")
    return figure


def _mesh_trace(vertices: np.ndarray, faces: np.ndarray, **kwargs: Any) -> go.Mesh3d:
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], **kwargs,
    )


def _mesh_figure(result: TriangleMeshAnalysis) -> go.Figure:
    figure = go.Figure(_mesh_trace(
        result.vertices, result.faces, intensity=result.face_areas, intensitymode="cell",
        colorscale="Viridis", colorbar_title="Face area", flatshading=True, opacity=0.9,
    ))
    figure.update_layout(title=f"Triangle mesh · {result.object_id}", template="plotly_white", scene_aspectmode="data")
    return figure


def _mesh_path_figure(result: MeshShortestPath, mesh: TriangleMeshAnalysis | None = None) -> go.Figure:
    figure = go.Figure()
    if mesh is not None:
        figure.add_trace(_mesh_trace(mesh.vertices, mesh.faces, color="#9fb3c8", opacity=0.35, showscale=False))
    figure.add_trace(go.Scatter3d(
        x=result.coordinates[:, 0], y=result.coordinates[:, 1], z=result.coordinates[:, 2],
        mode="lines+markers", text=list(result.vertex_labels), line=dict(color="#c0392b", width=6),
        marker=dict(size=4), hovertemplate="%{text}<extra></extra>",
    ))
    figure.update_layout(title=f"Mesh path · length {result.length:.6g}", template="plotly_white", scene_aspectmode="data")
    return figure


def _truss_lines(
    figure: go.Figure, coordinates: np.ndarray, elements: tuple[Any, ...], lookup: Mapping[str, int],
    *, colour: str, width: float, name: str, showlegend: bool,
) -> None:
    is_3d = coordinates.shape[1] == 3
    for index, element in enumerate(elements):
        points = coordinates[[lookup[element.start_node], lookup[element.end_node]]]
        common = dict(mode="lines", line=dict(color=colour, width=width), name=name, showlegend=showlegend and index == 0, hoverinfo="skip")
        if is_3d:
            figure.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], **common))
        else:
            figure.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], **common))


def _truss_static_figure(result: TrussStaticResult) -> go.Figure:
    lookup = {name: index for index, name in enumerate(result.node_ids)}
    span = max(1.0, float(np.max(np.ptp(result.coordinates, axis=0))))
    maximum = max(result.maximum_displacement, np.finfo(float).tiny)
    scale = min(1000.0, 0.12 * span / maximum)
    deformed = result.coordinates + scale * result.displacements
    figure = go.Figure()
    _truss_lines(figure, result.coordinates, result.elements, lookup, colour="#9fb3c8", width=2, name="Original", showlegend=True)
    _truss_lines(figure, deformed, result.elements, lookup, colour="#c0392b", width=4, name=f"Deformed ×{scale:.3g}", showlegend=True)
    if result.dimension == 2:
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
        return _layout(figure, f"Truss response · {result.load_case}", x_title="x", y_title="y")
    figure.update_layout(title=f"Truss response · {result.load_case}", template="plotly_white", scene_aspectmode="data")
    return figure


def _truss_mode_figure(result: TrussModalAnalysis) -> go.Figure:
    mode = result.mode_shapes[0]
    span = max(1.0, float(np.max(np.ptp(result.coordinates, axis=0))))
    amplitude = max(float(np.max(np.linalg.norm(mode, axis=1))), np.finfo(float).tiny)
    deformed = result.coordinates + 0.15 * span / amplitude * mode
    figure = go.Figure()
    if result.dimension == 2:
        for index, (left, right) in enumerate(result.element_connectivity):
            original = result.coordinates[[left, right]]
            displaced = deformed[[left, right]]
            figure.add_trace(go.Scatter(x=original[:, 0], y=original[:, 1], mode="lines", line=dict(color="#9fb3c8", width=2), name="Original", showlegend=index == 0, hoverinfo="skip"))
            figure.add_trace(go.Scatter(x=displaced[:, 0], y=displaced[:, 1], mode="lines", line=dict(color="#0f9d8c", width=4), name="Mode 1", showlegend=index == 0, hoverinfo="skip"))
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
        return _layout(figure, f"First mode · {result.frequencies_hz[0]:.6g} Hz", x_title="x", y_title="y")
    for index, (left, right) in enumerate(result.element_connectivity):
        original = result.coordinates[[left, right]]
        displaced = deformed[[left, right]]
        figure.add_trace(go.Scatter3d(x=original[:, 0], y=original[:, 1], z=original[:, 2], mode="lines", line=dict(color="#9fb3c8", width=2), name="Original", showlegend=index == 0, hoverinfo="skip"))
        figure.add_trace(go.Scatter3d(x=displaced[:, 0], y=displaced[:, 1], z=displaced[:, 2], mode="lines", line=dict(color="#0f9d8c", width=5), name="Mode 1", showlegend=index == 0, hoverinfo="skip"))
    figure.update_layout(title=f"First mode · {result.frequencies_hz[0]:.6g} Hz", template="plotly_white", scene_aspectmode="data")
    return figure


def _electrostatic_figure(result: ElectrostaticAnalysis) -> go.Figure:
    if result.dimension == 2:
        scale = max(float(np.max(result.field_magnitude, initial=0.0)), np.finfo(float).tiny)
        span = max(float(np.ptp(result.evaluation_points[:, 0])), float(np.ptp(result.evaluation_points[:, 1])), 1.0)
        factor = 0.08 * span / scale
        figure = go.Figure()
        for point, vector in zip(result.evaluation_points, result.electric_field):
            figure.add_trace(go.Scatter(
                x=[point[0], point[0] + factor * vector[0]],
                y=[point[1], point[1] + factor * vector[1]], mode="lines",
                line=dict(color="#164b63", width=1.5), hoverinfo="skip", showlegend=False,
            ))
        figure.add_trace(go.Scatter(
            x=result.charge_positions[:, 0], y=result.charge_positions[:, 1],
            text=list(result.charge_ids), mode="markers+text", textposition="top center",
            marker=dict(size=12, color=np.where(result.charges >= 0.0, "#c0392b", "#1769aa")),
            hovertemplate="%{text}<extra></extra>", showlegend=False,
        ))
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
        return _layout(figure, f"Electrostatic field · {result.object_id}", x_title="x", y_title="y")
    figure = go.Figure(go.Cone(
        x=result.evaluation_points[:, 0], y=result.evaluation_points[:, 1], z=result.evaluation_points[:, 2],
        u=result.electric_field[:, 0], v=result.electric_field[:, 1], w=result.electric_field[:, 2],
        colorscale="Viridis", sizemode="absolute",
    ))
    figure.update_layout(title=f"Electrostatic field · {result.object_id}", template="plotly_white", scene_aspectmode="data")
    return figure


def create_official_pack_figure(result: object) -> go.Figure | None:
    """Return a renderer-owned view; never alter artifact scientific identity."""
    if isinstance(result, ArrayAnalysis):
        return _array(result.values, f"Array analysis · {result.object_id}")
    if isinstance(result, TensorContraction):
        return _array(result.values, "Tensor contraction")
    if isinstance(result, QuantityNormalisation):
        rows = list(result.quantities)
        return _layout(go.Figure(go.Bar(x=[row["object_id"] for row in rows], y=[row["base_value"] for row in rows])), "Quantities in canonical base units", y_title="Base value")
    if isinstance(result, DistributionAnalysis):
        return _layout(go.Figure(go.Bar(x=list(result.outcomes), y=result.probabilities)), f"Distribution · {result.object_id}", x_title="Outcome", y_title="Probability")
    if isinstance(result, MarkovEvolution):
        return _state_probabilities(result.states, result.distributions, f"Markov evolution · {result.object_id}")
    if isinstance(result, MarkovSimulation):
        return _state_probabilities(result.states, result.empirical_distributions, f"Markov simulation · {result.object_id}")
    if isinstance(result, NetworkAnalysis):
        return _network(result.nodes, result.edges, f"Network · {result.object_id}")
    if isinstance(result, MarkovBlanketAnalysis):
        return _network(result.nodes, result.edges, f"Markov blanket · {result.target}", set(result.blanket), result.target)
    if isinstance(result, HiddenMarkovInference):
        return _state_probabilities(result.states, result.smoothed_probabilities, f"Hidden-state posterior · {result.object_id}")
    if isinstance(result, POMDPDecision):
        return _layout(go.Figure(go.Bar(x=list(result.actions), y=result.initial_action_values)), f"Decision values · recommended: {result.recommended_action}", x_title="Action", y_title="Initial expected value")
    if isinstance(result, ActiveInferenceEvaluation):
        figure = go.Figure(go.Bar(x=list(result.policy_names), y=result.policy_posterior, customdata=result.expected_free_energy, hovertemplate="Policy %{x}<br>Posterior %{y:.5g}<br>G %{customdata:.5g}<extra></extra>"))
        return _layout(figure, f"Active-inference policies · {result.object_id}", x_title="Policy", y_title="Posterior probability")
    if isinstance(result, ODETrajectory):
        return _trajectory(result.times, result.states, result.state_names, f"ODE trajectory · {result.object_id}")
    if isinstance(result, ODEEquilibriumAnalysis):
        eigenvalues = tuple(value for point in result.points for value in point.eigenvalues)
        return _complex_spectrum(eigenvalues, f"Equilibrium spectra · {result.object_id}") if eigenvalues else go.Figure().update_layout(title=f"No equilibria found · {result.object_id}", template="plotly_white")
    if isinstance(result, StateSpaceAnalysis):
        return _complex_spectrum(result.poles, f"State-space poles · {result.object_id}")
    if isinstance(result, StateSpaceTrajectory):
        return _trajectory(result.times, result.outputs, result.output_names, f"State-space outputs · {result.object_id}")
    if isinstance(result, ScalarFieldAnalysis):
        return _scalar_field_figure(result)
    if isinstance(result, VectorFieldAnalysis):
        return _vector_field_figure(result)
    if isinstance(result, DiffusionSolution):
        return _layout(go.Figure(go.Heatmap(
            x=result.coordinate, y=result.times, z=result.values, colorscale="Viridis",
            colorbar_title=result.value_name,
        )), f"Diffusion · {result.object_id}", x_title=result.coordinate_name, y_title="Time")
    if isinstance(result, PoissonSolution):
        figure = go.Figure(go.Surface(x=result.x_coordinates, y=result.y_coordinates, z=result.solution, colorscale="Viridis"))
        figure.update_layout(title=f"Poisson solution · {result.object_id}", template="plotly_white", scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title=result.solution_name))
        return figure
    if isinstance(result, PointCloudAnalysis):
        return _point_cloud_figure(result)
    if isinstance(result, TriangleMeshAnalysis):
        return _mesh_figure(result)
    if isinstance(result, MeshShortestPath):
        return _mesh_path_figure(result)
    if isinstance(result, IsotropicMaterialAnalysis):
        return _layout(go.Figure(go.Heatmap(z=result.stiffness_3d_voigt, colorscale="Viridis")), f"3D constitutive stiffness · {result.object_id}", x_title="Voigt column", y_title="Voigt row")
    if isinstance(result, TrussStaticResult):
        return _truss_static_figure(result)
    if isinstance(result, TrussModalAnalysis):
        return _truss_mode_figure(result)
    if isinstance(result, DatasetAnalysis):
        return _layout(go.Figure(go.Heatmap(
            z=result.correlation, x=list(result.columns), y=list(result.columns),
            zmin=-1.0, zmax=1.0, colorscale="RdBu", reversescale=True,
        )), f"Correlation matrix · {result.object_id}")
    if isinstance(result, LinearModelFit):
        observed = result.fitted_values + result.residuals
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=observed, y=result.fitted_values, mode="markers", name="Observations"))
        lower = float(min(np.min(observed), np.min(result.fitted_values)))
        upper = float(max(np.max(observed), np.max(result.fitted_values)))
        figure.add_trace(go.Scatter(x=[lower, upper], y=[lower, upper], mode="lines", name="Identity"))
        return _layout(figure, f"Linear-model fit · {result.object_id}", x_title="Observed", y_title="Fitted")
    if isinstance(result, GroupComparison):
        return _layout(go.Figure(go.Bar(
            x=list(result.group_names), y=result.means,
            error_y=dict(type="data", array=result.standard_deviations, visible=True),
        )), f"Group comparison · {result.measure_name}", x_title="Group", y_title=result.measure_name)
    if isinstance(result, OptimisationResult):
        return _layout(go.Figure(go.Bar(x=list(result.variable_names), y=result.optimum)), f"Optimisation result · {result.object_id}", x_title="Variable", y_title="Optimum")
    if isinstance(result, NonlinearLeastSquaresFit):
        return _layout(go.Figure(go.Scatter(
            x=list(result.residual_names), y=result.residuals, mode="markers",
        )), f"Nonlinear residuals · {result.object_id}", x_title="Residual", y_title="Value")
    if isinstance(result, LinearInverseSolution):
        observed = result.predicted_observations + result.residuals
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=list(result.observation_names), y=observed, mode="markers", name="Observed"))
        figure.add_trace(go.Scatter(x=list(result.observation_names), y=result.predicted_observations, mode="lines+markers", name="Predicted"))
        return _layout(figure, f"Inverse-problem fit · {result.object_id}", x_title="Observation", y_title="Value")
    if isinstance(result, CircuitDCAnalysis):
        return _layout(go.Figure(go.Bar(x=list(result.node_names), y=result.node_voltages)), f"DC node voltages · {result.object_id}", x_title="Node", y_title=result.voltage_unit)
    if isinstance(result, CircuitACAnalysis):
        figure = go.Figure()
        for index, node in enumerate(result.node_names):
            if node != result.ground_node:
                figure.add_trace(go.Scatter(x=result.frequencies_hz, y=result.node_voltage_magnitudes[:, index], mode="lines", name=node))
        figure.update_xaxes(type="log")
        return _layout(figure, f"AC magnitude response · {result.object_id}", x_title="Frequency (Hz)", y_title=result.voltage_unit)
    if isinstance(result, DiodeCharacteristic):
        return _layout(go.Figure(go.Scatter(x=result.voltages, y=result.currents, mode="lines")), f"Diode characteristic · {result.object_id}", x_title="Voltage (V)", y_title="Current (A)")
    if isinstance(result, ElectrostaticAnalysis):
        return _electrostatic_figure(result)
    if isinstance(result, ReactionNetworkAnalysis):
        return _layout(go.Figure(go.Heatmap(
            z=result.stoichiometric_matrix, x=list(result.reaction_ids), y=list(result.species), colorscale="RdBu", zmid=0,
        )), f"Reaction stoichiometry · {result.object_id}", x_title="Reaction", y_title="Species")
    if isinstance(result, ReactionTrajectory):
        return _trajectory(result.times, result.concentrations, result.species, f"Reaction kinetics · {result.object_id}")
    if isinstance(result, CompartmentAnalysis):
        return _complex_spectrum(result.eigenvalues, f"Compartment spectrum · {result.object_id}")
    if isinstance(result, CompartmentTrajectory):
        return _trajectory(result.times, result.amounts, result.compartments, f"Compartment dynamics · {result.object_id}")
    if isinstance(result, PopulationAnalysis):
        return _complex_spectrum(result.coexistence_eigenvalues, f"Population coexistence spectrum · {result.object_id}") if result.coexistence_eigenvalues else go.Figure().update_layout(title=result.coexistence_stability, template="plotly_white")
    if isinstance(result, PopulationTrajectory):
        return _trajectory(result.times, result.values, result.populations, f"Population dynamics · {result.object_id}")
    if isinstance(result, SupervisedLearningFit):
        if result.task == "classification":
            size = len(result.class_names)
            confusion = np.zeros((size, size), dtype=np.int64)
            for observed, predicted in zip(result.target_values.astype(int), result.predicted_class_indices):
                confusion[observed, predicted] += 1
            return _layout(go.Figure(go.Heatmap(
                z=confusion, x=list(result.class_names), y=list(result.class_names), colorscale="Blues",
            )), f"Classification · {result.object_id}", x_title="Predicted", y_title="Observed")
        return _layout(go.Figure(go.Scatter(
            x=result.target_values, y=result.predicted_values, mode="markers",
        )), f"Regression · {result.object_id}", x_title="Observed", y_title="Predicted")
    if isinstance(result, ClusteringResult):
        x = result.features[:, 0]
        y = result.assignments if result.features.shape[1] == 1 else result.features[:, 1]
        return _layout(go.Figure(go.Scatter(
            x=x, y=y, mode="markers", marker=dict(color=result.assignments, colorscale="Viridis"),
            text=list(result.sample_ids), hovertemplate="%{text}<extra></extra>",
        )), f"K-means clustering · {result.object_id}", x_title=result.feature_names[0], y_title=("Cluster" if result.features.shape[1] == 1 else result.feature_names[1]))
    if isinstance(result, NeuralNetworkEvaluation):
        return _array(result.outputs, f"Neural-network outputs · {result.network_object_id}")
    if isinstance(result, FuzzyInference):
        return _layout(go.Figure(go.Scatter(
            x=np.arange(result.outputs.size), y=result.outputs, mode="lines+markers",
        )), f"Fuzzy inference · {result.object_id}", x_title="Point", y_title=result.output_name)
    return None
