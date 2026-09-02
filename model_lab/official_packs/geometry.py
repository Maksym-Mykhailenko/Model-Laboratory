"""Geometry, Meshes and Spatial Computation official capability pack."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Mapping

import numpy as np
from scipy.spatial.distance import pdist

from ..model import ModelIR
from ..model_graph import ModelGraphError, ModelObject, ObjectKindDescriptor
from ..protocol import ArtifactTypeDescriptor, CapabilityDescriptor, SpectralDecomposition
from .common import MAX_INLINE_VALUES, PackManifest, objects_of_kind, select_object, symmetric_spectral_decomposition, unique_labels
from .units import LENGTH, to_si


PACK_ID = "org.modellab.pack.geometry-meshes-spatial-computation"
POINT_CLOUD_KIND = "org.modellab.geometry.point-cloud"
TRIANGLE_MESH_KIND = "org.modellab.geometry.triangle-mesh"

POINT_CLOUD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["points"],
    "properties": {
        "points": {
            "type": "array", "minItems": 1, "maxItems": 10000,
            "items": {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "number"}},
        },
        "labels": {"type": "array", "maxItems": 10000, "items": {"type": "string"}},
        "coordinate_unit": {"type": "string", "maxLength": 128},
    },
    "additionalProperties": False,
}

TRIANGLE_MESH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["vertices", "faces"],
    "properties": {
        "vertices": {
            "type": "array", "minItems": 3, "maxItems": 100000,
            "items": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
        },
        "faces": {
            "type": "array", "minItems": 1, "maxItems": 200000,
            "items": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "integer", "minimum": 0}},
        },
        "vertex_labels": {"type": "array", "maxItems": 100000, "items": {"type": "string"}},
        "coordinate_unit": {"type": "string", "maxLength": 128},
    },
    "additionalProperties": False,
}


KIND_DESCRIPTORS = (
    ObjectKindDescriptor(POINT_CLOUD_KIND, "1.1", "Finite Euclidean point cloud", POINT_CLOUD_SCHEMA, True),
    ObjectKindDescriptor(TRIANGLE_MESH_KIND, "1.1", "Indexed three-dimensional triangle mesh", TRIANGLE_MESH_SCHEMA, True),
)


def _points(item: ModelObject) -> tuple[np.ndarray, tuple[str, ...], str, str]:
    try:
        points = np.asarray(item.properties["points"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelGraphError(f"{item.identifier}.points must be a rectangular real matrix.") from exc
    if points.ndim != 2 or points.shape[1] not in {2, 3} or points.size > MAX_INLINE_VALUES or not np.all(np.isfinite(points)):
        raise ModelGraphError(f"{item.identifier}.points must contain finite two- or three-dimensional points.")
    raw_labels = item.properties.get("labels")
    labels = tuple(str(index) for index in range(len(points))) if raw_labels is None else unique_labels(raw_labels, field=f"{item.identifier}.labels")
    if len(labels) != len(points):
        raise ModelGraphError(f"{item.identifier}.labels must align with points.")
    declared = str(item.properties.get("coordinate_unit", "m"))
    return (
        to_si(points, declared, field=f"{item.identifier}.coordinate_unit", expected=LENGTH),
        labels,
        "m",
        declared,
    )


def _mesh(item: ModelObject) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], str, str]:
    try:
        vertices = np.asarray(item.properties["vertices"], dtype=np.float64)
        faces = np.asarray(item.properties["faces"], dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ModelGraphError(f"{item.identifier} vertices/faces must be rectangular arrays.") from exc
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.size > MAX_INLINE_VALUES or not np.all(np.isfinite(vertices)):
        raise ModelGraphError(f"{item.identifier}.vertices must contain finite 3D points.")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.size > MAX_INLINE_VALUES:
        raise ModelGraphError(f"{item.identifier}.faces must contain triangle index triples.")
    if np.any(faces < 0) or np.any(faces >= len(vertices)) or np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
        raise ModelGraphError(f"{item.identifier}.faces contain invalid or repeated vertex indices.")
    canonical_faces = np.sort(faces, axis=1)
    if len(np.unique(canonical_faces, axis=0)) != len(faces):
        raise ModelGraphError(f"{item.identifier}.faces contain a duplicate triangle.")
    if len(np.unique(faces)) != len(vertices):
        raise ModelGraphError(f"{item.identifier}.vertices contains an unreferenced vertex.")
    triangles = vertices[faces]
    double_areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    scale = max(1.0, float(np.max(np.ptp(vertices, axis=0))))
    if np.any(double_areas <= np.finfo(float).eps * scale * scale * 100):
        raise ModelGraphError(f"{item.identifier}.faces contain a degenerate triangle.")
    raw_labels = item.properties.get("vertex_labels")
    labels = tuple(str(index) for index in range(len(vertices))) if raw_labels is None else unique_labels(raw_labels, field=f"{item.identifier}.vertex_labels")
    if len(labels) != len(vertices):
        raise ModelGraphError(f"{item.identifier}.vertex_labels must align with vertices.")
    declared = str(item.properties.get("coordinate_unit", "m"))
    return (
        to_si(vertices, declared, field=f"{item.identifier}.coordinate_unit", expected=LENGTH),
        faces.copy(),
        labels,
        "m",
        declared,
    )


def validate_point_cloud(item: ModelObject) -> None:
    _points(item)


def validate_triangle_mesh(item: ModelObject) -> None:
    _mesh(item)


SEMANTIC_VALIDATORS = {POINT_CLOUD_KIND: validate_point_cloud, TRIANGLE_MESH_KIND: validate_triangle_mesh}


@dataclass(frozen=True, slots=True)
class PointCloudAnalysis:
    object_id: str
    labels: tuple[str, ...]
    points: np.ndarray
    coordinate_unit: str
    source_units: Mapping[str, str]
    point_count: int
    embedding_dimension: int
    affine_rank: int
    centroid: np.ndarray
    bounding_box_minimum: np.ndarray
    bounding_box_maximum: np.ndarray
    covariance: np.ndarray
    principal_values: np.ndarray
    principal_directions: SpectralDecomposition
    diameter: float
    diameter_pair: tuple[str, str]


def analyse_point_cloud(model: ModelIR, object_id: str | None = None) -> PointCloudAnalysis:
    item = select_object(model, POINT_CLOUD_KIND, object_id, label="point cloud")
    points, labels, unit, source_unit = _points(item)
    if len(points) > 5000:
        raise ValueError("Exact point-cloud diameter is limited to 5,000 points.")
    centroid = np.mean(points, axis=0)
    centred = points - centroid
    covariance = centred.T @ centred / max(1, len(points) - 1)
    values, directions = symmetric_spectral_decomposition(
        covariance, descending=True, basis_orientation="columns"
    )
    if len(points) == 1:
        diameter, pair = 0.0, (labels[0], labels[0])
    else:
        distances = pdist(points)
        flat_index = int(np.argmax(distances))
        diameter = float(distances[flat_index])
        # scipy.spatial.distance square-form condensed indexing, inverted deterministically.
        cursor = flat_index
        left = 0
        width = len(points) - 1
        while cursor >= width:
            cursor -= width
            left += 1
            width -= 1
        right = left + 1 + cursor
        pair = (labels[left], labels[right])
    return PointCloudAnalysis(
        item.identifier, labels, points, unit, {"coordinate": source_unit}, len(points), points.shape[1],
        int(np.linalg.matrix_rank(centred)), centroid, np.min(points, axis=0), np.max(points, axis=0),
        covariance, values, directions, diameter, pair,
    )


def _edge_table(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = {}
    for face_index, (a, b, c) in enumerate(faces):
        for left, right in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            edge = (min(left, right), max(left, right))
            result.setdefault(edge, []).append(face_index)
    return result


def _face_components(faces: np.ndarray, edges: Mapping[tuple[int, int], list[int]]) -> tuple[tuple[int, ...], ...]:
    adjacency = [set() for _ in range(len(faces))]
    for attached in edges.values():
        for left in attached:
            adjacency[left].update(index for index in attached if index != left)
    remaining = set(range(len(faces)))
    components: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        seen, stack = {start}, [start]
        while stack:
            current = stack.pop()
            for neighbour in sorted(adjacency[current]):
                if neighbour not in seen:
                    seen.add(neighbour); stack.append(neighbour)
        components.append(tuple(sorted(seen)))
        remaining.difference_update(seen)
    return tuple(components)


def _orientation_consistent(faces: np.ndarray) -> bool:
    directions: dict[tuple[int, int], list[int]] = {}
    for a, b, c in faces:
        for left, right in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            edge = (min(left, right), max(left, right))
            directions.setdefault(edge, []).append(1 if (left, right) == edge else -1)
    return all(len(signs) != 2 or signs[0] == -signs[1] for signs in directions.values())


def _vertex_manifold(vertices: int, faces: np.ndarray) -> bool:
    """Check that every vertex link is one connected path or cycle."""
    for vertex in range(vertices):
        link: dict[int, set[int]] = {}
        for face in faces:
            if vertex not in face:
                continue
            others = [int(value) for value in face if int(value) != vertex]
            left, right = others
            link.setdefault(left, set()).add(right)
            link.setdefault(right, set()).add(left)
        if not link:
            return False
        start = min(link)
        seen, stack = {start}, [start]
        while stack:
            current = stack.pop()
            for neighbour in sorted(link[current]):
                if neighbour not in seen:
                    seen.add(neighbour); stack.append(neighbour)
        degrees = [len(link[node]) for node in link]
        endpoints = sum(degree == 1 for degree in degrees)
        if len(seen) != len(link) or any(degree not in {1, 2} for degree in degrees) or endpoints not in {0, 2}:
            return False
    return True


@dataclass(frozen=True, slots=True)
class TriangleMeshAnalysis:
    object_id: str
    vertices: np.ndarray
    faces: np.ndarray
    vertex_labels: tuple[str, ...]
    coordinate_unit: str
    source_units: Mapping[str, str]
    vertex_count: int
    edge_count: int
    face_count: int
    connected_components: tuple[tuple[int, ...], ...]
    boundary_edges: tuple[tuple[int, int], ...]
    nonmanifold_edges: tuple[tuple[int, int], ...]
    orientation_consistent: bool
    vertex_manifold: bool
    manifold_with_boundary: bool
    watertight_two_manifold: bool
    euler_characteristic: int
    surface_area: float
    enclosed_volume: float | None
    area_centroid: np.ndarray
    face_normals: np.ndarray
    face_areas: np.ndarray
    minimum_edge_ratio: float
    bounding_box_minimum: np.ndarray
    bounding_box_maximum: np.ndarray


def analyse_triangle_mesh(model: ModelIR, object_id: str | None = None) -> TriangleMeshAnalysis:
    item = select_object(model, TRIANGLE_MESH_KIND, object_id, label="triangle mesh")
    vertices, faces, labels, unit, source_unit = _mesh(item)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    areas = 0.5 * double_area
    normals = cross / double_area[:, None]
    edge_table = _edge_table(faces)
    boundary = tuple(sorted(edge for edge, attached in edge_table.items() if len(attached) == 1))
    nonmanifold = tuple(sorted(edge for edge, attached in edge_table.items() if len(attached) > 2))
    orientation_consistent = _orientation_consistent(faces)
    vertex_manifold = _vertex_manifold(len(vertices), faces)
    manifold_with_boundary = not nonmanifold and vertex_manifold and orientation_consistent
    closed = not boundary and manifold_with_boundary
    signed_volume = float(np.sum(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))) / 6.0)
    face_centroids = np.mean(triangles, axis=1)
    centroid = np.sum(face_centroids * areas[:, None], axis=0) / float(np.sum(areas))
    lengths = np.stack((
        np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
        np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
        np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
    ), axis=1)
    ratios = np.min(lengths, axis=1) / np.max(lengths, axis=1)
    return TriangleMeshAnalysis(
        item.identifier, vertices, faces, labels, unit, {"coordinate": source_unit}, len(vertices), len(edge_table), len(faces),
        _face_components(faces, edge_table), boundary, nonmanifold,
        orientation_consistent, vertex_manifold, manifold_with_boundary, closed,
        len(vertices) - len(edge_table) + len(faces), float(np.sum(areas)),
        abs(signed_volume) if closed else None, centroid, normals, areas,
        float(np.min(ratios)), np.min(vertices, axis=0), np.max(vertices, axis=0),
    )


@dataclass(frozen=True, slots=True)
class MeshShortestPath:
    object_id: str
    source_vertex: int
    target_vertex: int
    source_label: str
    target_label: str
    vertex_indices: tuple[int, ...]
    vertex_labels: tuple[str, ...]
    coordinates: np.ndarray
    length: float
    coordinate_unit: str
    source_units: Mapping[str, str]


def shortest_mesh_path(
    model: ModelIR,
    object_id: str | None = None,
    source_vertex: int = 0,
    target_vertex: int = 1,
) -> MeshShortestPath:
    item = select_object(model, TRIANGLE_MESH_KIND, object_id, label="triangle mesh")
    vertices, faces, labels, unit, source_unit = _mesh(item)
    for name, value in (("source_vertex", source_vertex), ("target_vertex", target_vertex)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(vertices):
            raise ValueError(f"{name} must be a valid zero-based vertex index.")
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(vertices))]
    for left, right in _edge_table(faces):
        length = float(np.linalg.norm(vertices[left] - vertices[right]))
        adjacency[left].append((right, length)); adjacency[right].append((left, length))
    distance = [math.inf] * len(vertices)
    previous: list[int | None] = [None] * len(vertices)
    distance[source_vertex] = 0.0
    queue: list[tuple[float, int]] = [(0.0, source_vertex)]
    while queue:
        current_distance, vertex = heapq.heappop(queue)
        if current_distance != distance[vertex]:
            continue
        if vertex == target_vertex:
            break
        for neighbour, length in sorted(adjacency[vertex]):
            candidate = current_distance + length
            if candidate < distance[neighbour]:
                distance[neighbour], previous[neighbour] = candidate, vertex
                heapq.heappush(queue, (candidate, neighbour))
    if not math.isfinite(distance[target_vertex]):
        raise ValueError("The requested mesh vertices lie in disconnected components.")
    path: list[int] = []
    cursor: int | None = target_vertex
    while cursor is not None:
        path.append(cursor)
        cursor = previous[cursor]
    path.reverse()
    if path[0] != source_vertex:
        raise ValueError("The requested mesh vertices lie in disconnected components.")
    return MeshShortestPath(
        item.identifier, source_vertex, target_vertex, labels[source_vertex], labels[target_vertex],
        tuple(path), tuple(labels[index] for index in path), vertices[path].copy(), float(distance[target_vertex]),
        unit, {"coordinate": source_unit},
    )


def _kind_applicability(kind: str, label: str):
    def applicable(model: ModelIR) -> tuple[bool, str]:
        found = bool(objects_of_kind(model, kind))
        return found, "" if found else f"an executable {label} object is required"
    return applicable


def _point_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(model, POINT_CLOUD_KIND, settings.get("object_id"), label="point cloud")
    count = len(item.properties["points"])
    return max(1, count * count)


def _mesh_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(
        model, TRIANGLE_MESH_KIND, settings.get("object_id"), label="triangle mesh"
    )
    return max(1, len(item.properties["vertices"]) + 10 * len(item.properties["faces"]))


NUMERIC = "org.modellab.comparator.numeric"

CAPABILITY_DESCRIPTORS = (
    CapabilityDescriptor(
        "org.modellab.geometry.analyse-point-cloud", "1.2", PACK_ID,
        "Analyse point cloud", "Compute centroid, covariance, principal directions, affine rank, bounds and exact Euclidean diameter.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}}},
        "scipy+numpy", (ArtifactTypeDescriptor("org.modellab.artifact.point-cloud-analysis", "1.2", "Point-cloud analysis", NUMERIC),),
        _kind_applicability(POINT_CLOUD_KIND, "point-cloud"), analyse_point_cloud, _point_units,
        ("org.modellab.renderer.plotly-point-cloud",),
    ),
    CapabilityDescriptor(
        "org.modellab.geometry.analyse-triangle-mesh", "1.2", PACK_ID,
        "Analyse triangle mesh", "Compute mesh topology, boundary/non-manifold edges, area, enclosed volume, normals and triangle quality.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}}},
        "numpy", (ArtifactTypeDescriptor("org.modellab.artifact.triangle-mesh-analysis", "1.2", "Triangle-mesh analysis", NUMERIC),),
        _kind_applicability(TRIANGLE_MESH_KIND, "triangle-mesh"), analyse_triangle_mesh, _mesh_units,
        ("org.modellab.renderer.plotly-triangle-mesh",),
    ),
    CapabilityDescriptor(
        "org.modellab.geometry.shortest-mesh-path", "1.2", PACK_ID,
        "Compute mesh-edge shortest path", "Compute a deterministic Euclidean shortest path along triangle-mesh edges.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "source_vertex": {"type": "integer", "minimum": 0, "default": 0}, "target_vertex": {"type": "integer", "minimum": 0, "default": 1}}},
        "python+numpy", (ArtifactTypeDescriptor("org.modellab.artifact.mesh-shortest-path", "1.2", "Mesh shortest path", NUMERIC),),
        _kind_applicability(TRIANGLE_MESH_KIND, "triangle-mesh"), shortest_mesh_path, _mesh_units,
        ("org.modellab.renderer.plotly-mesh-path",),
    ),
)


MANIFEST = PackManifest(
    PACK_ID, "1.2", "Geometry, Meshes and Spatial Computation",
    "Finite Euclidean point clouds and explicit triangle meshes with geometric, topological and path analyses.",
    (
        "org.modellab.pack.multidimensional-mathematics",
        "org.modellab.pack.graphs-networks-discrete",
    ),
    tuple(item.kind for item in KIND_DESCRIPTORS),
    tuple(item.identifier for item in CAPABILITY_DESCRIPTORS),
)
