"""Graphs, Networks and Discrete Structures official capability pack."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Mapping

import numpy as np

from ..model import ModelIR
from ..model_graph import ModelGraphError, ModelObject, ObjectKindDescriptor
from ..protocol import ArtifactTypeDescriptor, CapabilityDescriptor
from .common import PackManifest, select_object, unique_labels


PACK_ID = "org.modellab.pack.graphs-networks-discrete"
NETWORK_KIND = "org.modellab.graph.network"

NETWORK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["nodes", "edges", "directed"],
    "properties": {
        "nodes": {
            "type": "array", "minItems": 1, "maxItems": 10000,
            "items": {"type": "string"},
        },
        "edges": {
            "type": "array", "maxItems": 200000,
            "items": {
                "type": "object", "required": ["source", "target"],
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "weight": {"type": "number", "minimum": 0},
                    "label": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "directed": {"type": "boolean"},
        "multigraph": {"type": "boolean"},
        "node_values": {
            "type": "array", "maxItems": 10000, "items": {"type": "number"},
        },
    },
    "additionalProperties": False,
}

KIND_DESCRIPTORS = (
    ObjectKindDescriptor(NETWORK_KIND, "1.0", "Finite weighted graph or network", NETWORK_SCHEMA, True),
)


def _network(item: ModelObject) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...], bool, bool]:
    nodes = unique_labels(item.properties["nodes"], field=f"{item.identifier}.nodes")
    known = set(nodes)
    directed = bool(item.properties["directed"])
    multigraph = bool(item.properties.get("multigraph", False))
    edges: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(item.properties["edges"]):
        source, target = str(raw["source"]), str(raw["target"])
        if source not in known or target not in known:
            raise ModelGraphError(f"{item.identifier}.edges[{index}] references an unknown node.")
        weight = float(raw.get("weight", 1.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ModelGraphError(f"{item.identifier}.edges[{index}].weight must be finite and non-negative.")
        key = (source, target) if directed else tuple(sorted((source, target)))
        if not multigraph and key in seen:
            raise ModelGraphError(f"{item.identifier} contains a duplicate edge {key}.")
        seen.add(key)
        edges.append({
            "source": source, "target": target, "weight": weight,
            "label": str(raw.get("label", "")),
        })
    values = item.properties.get("node_values")
    if values is not None:
        checked = np.asarray(values, dtype=np.float64)
        if checked.ndim != 1 or checked.size != len(nodes) or not np.all(np.isfinite(checked)):
            raise ModelGraphError(f"{item.identifier}.node_values must align with nodes and be finite.")
    return nodes, tuple(edges), directed, multigraph


def validate_network(item: ModelObject) -> None:
    _network(item)


SEMANTIC_VALIDATORS = {NETWORK_KIND: validate_network}


def _components(adjacency: np.ndarray, *, strong: bool) -> tuple[tuple[int, ...], ...]:
    size = adjacency.shape[0]

    def reach(start: int, matrix: np.ndarray) -> set[int]:
        seen, stack = {start}, [start]
        while stack:
            node = stack.pop()
            for child in np.flatnonzero(matrix[node]):
                value = int(child)
                if value not in seen:
                    seen.add(value)
                    stack.append(value)
        return seen

    if not strong:
        matrix = np.logical_or(adjacency, adjacency.T)
        remaining = set(range(size))
        groups: list[tuple[int, ...]] = []
        while remaining:
            group = tuple(sorted(reach(min(remaining), matrix)))
            groups.append(group)
            remaining.difference_update(group)
        return tuple(groups)
    forward = [reach(index, adjacency) for index in range(size)]
    remaining = set(range(size))
    groups = []
    while remaining:
        first = min(remaining)
        group = tuple(index for index in sorted(remaining) if index in forward[first] and first in forward[index])
        groups.append(group)
        remaining.difference_update(group)
    return tuple(groups)


def _has_directed_cycle(adjacency: np.ndarray) -> bool:
    colour = [0] * adjacency.shape[0]

    def visit(node: int) -> bool:
        colour[node] = 1
        for raw in np.flatnonzero(adjacency[node]):
            child = int(raw)
            if colour[child] == 1 or (colour[child] == 0 and visit(child)):
                return True
        colour[node] = 2
        return False

    return any(colour[node] == 0 and visit(node) for node in range(adjacency.shape[0]))


def _shortest_path(
    nodes: tuple[str, ...], adjacency: np.ndarray, source: str, target: str
) -> tuple[tuple[str, ...], float] | None:
    lookup = {name: index for index, name in enumerate(nodes)}
    if source not in lookup or target not in lookup:
        raise ValueError("source and target must name graph nodes.")
    start, goal = lookup[source], lookup[target]
    distance = [math.inf] * len(nodes)
    previous: list[int | None] = [None] * len(nodes)
    distance[start] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start)]
    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance != distance[node]:
            continue
        if node == goal:
            break
        for raw in np.flatnonzero(np.isfinite(adjacency[node])):
            child = int(raw)
            candidate = current_distance + float(adjacency[node, child])
            if candidate < distance[child]:
                distance[child], previous[child] = candidate, node
                heapq.heappush(queue, (candidate, child))
    if not math.isfinite(distance[goal]):
        return None
    route, cursor = [], goal
    while True:
        route.append(nodes[cursor])
        if cursor == start:
            break
        cursor = previous[cursor]  # type: ignore[assignment]
    return tuple(reversed(route)), float(distance[goal])


@dataclass(frozen=True, slots=True)
class NetworkAnalysis:
    object_id: str
    directed: bool
    multigraph: bool
    nodes: tuple[str, ...]
    edges: tuple[Mapping[str, Any], ...]
    adjacency: np.ndarray
    out_degree: np.ndarray
    in_degree: np.ndarray
    unique_out_neighbour_count: np.ndarray
    unique_in_neighbour_count: np.ndarray
    weighted_out_degree: np.ndarray
    weighted_in_degree: np.ndarray
    density: float
    weak_components: tuple[tuple[str, ...], ...]
    strong_components: tuple[tuple[str, ...], ...]
    has_cycle: bool
    is_dag: bool
    adjacency_eigenvalues: tuple[complex, ...]
    shortest_path: tuple[str, ...] | None
    shortest_path_length: float | None


def analyse_network(
    model: ModelIR,
    object_id: str | None = None,
    source: str | None = None,
    target: str | None = None,
) -> NetworkAnalysis:
    item = select_object(model, NETWORK_KIND, object_id, label="network")
    nodes, edges, directed, multigraph = _network(item)
    size = len(nodes)
    lookup = {name: index for index, name in enumerate(nodes)}
    adjacency = np.zeros((size, size), dtype=np.float64)
    lengths = np.full((size, size), np.inf, dtype=np.float64)
    for edge in edges:
        left, right, weight = lookup[edge["source"]], lookup[edge["target"]], float(edge["weight"])
        adjacency[left, right] += weight
        lengths[left, right] = min(lengths[left, right], weight)
        if not directed and left != right:
            adjacency[right, left] += weight
            lengths[right, left] = min(lengths[right, left], weight)
    binary = adjacency != 0.0
    # Zero-weight edges still represent topology.
    for edge in edges:
        left, right = lookup[edge["source"]], lookup[edge["target"]]
        binary[left, right] = True
        if not directed:
            binary[right, left] = True
    weak = tuple(tuple(nodes[index] for index in group) for group in _components(binary, strong=False))
    strong = tuple(tuple(nodes[index] for index in group) for group in _components(binary, strong=True))
    cycle = _has_directed_cycle(binary) if directed else (
        any(edge["source"] == edge["target"] for edge in edges)
        or len(edges) > size - len(weak)
    )
    route = None
    if source is not None or target is not None:
        if source is None or target is None:
            raise ValueError("source and target must either both be supplied or both omitted.")
        route = _shortest_path(nodes, lengths, str(source), str(target))
    possible = size * (size - 1) if directed else size * (size - 1) / 2
    nonloop_edges = int(np.count_nonzero(binary & ~np.eye(size, dtype=bool)))
    if not directed:
        nonloop_edges //= 2
    out_degree = np.zeros(size, dtype=np.int64)
    in_degree = np.zeros(size, dtype=np.int64)
    for edge in edges:
        left, right = lookup[edge["source"]], lookup[edge["target"]]
        if directed:
            out_degree[left] += 1; in_degree[right] += 1
        elif left == right:
            # The two ends of an undirected loop are both incident on the vertex.
            out_degree[left] += 2; in_degree[left] += 2
        else:
            out_degree[left] += 1; out_degree[right] += 1
            in_degree[left] += 1; in_degree[right] += 1
    weighted_out = np.sum(adjacency, axis=1)
    weighted_in = np.sum(adjacency, axis=0)
    if not directed:
        # adjacency stores an undirected loop once; weighted degree counts it twice.
        weighted_out = weighted_out + np.diag(adjacency)
        weighted_in = weighted_out.copy()
    return NetworkAnalysis(
        item.identifier, directed, multigraph, nodes, edges, adjacency,
        out_degree, in_degree,
        np.sum(binary, axis=1).astype(np.int64),
        np.sum(binary, axis=0).astype(np.int64),
        weighted_out, weighted_in,
        0.0 if possible == 0 else float(nonloop_edges / possible),
        weak, strong, cycle, bool(directed and not cycle),
        tuple(complex(value) for value in np.linalg.eigvals(adjacency)),
        None if route is None else route[0],
        None if route is None else route[1],
    )


def _applicable(model: ModelIR) -> tuple[bool, str]:
    available = any(item.kind == NETWORK_KIND and not item.opaque for item in model.graph.objects)
    return available, "an executable network object is required"


def _units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(model, NETWORK_KIND, settings.get("object_id"), label="network")
    nodes = len(item.properties["nodes"])
    edges = len(item.properties["edges"])
    return max(1, nodes * nodes + edges)


CAPABILITY_DESCRIPTORS = (
    CapabilityDescriptor(
        "org.modellab.graph.analyse-network", "1.1", PACK_ID,
        "Analyse graph or network",
        "Compute deterministic topology, components, degrees, spectrum and optional weighted shortest path.",
        {
            "type": "object",
            "properties": {
                "object_id": {"type": ["string", "null"]},
                "source": {"type": ["string", "null"]},
                "target": {"type": ["string", "null"]},
            },
        },
        "numpy",
        (ArtifactTypeDescriptor("org.modellab.artifact.network-analysis", "1.1", "Network analysis", "org.modellab.comparator.numeric"),),
        _applicable, analyse_network, _units,
        ("org.modellab.renderer.plotly-network",),
    ),
)


MANIFEST = PackManifest(
    PACK_ID, "1.1", "Graphs, Networks and Discrete Structures",
    "Finite graphs and networks with topological, spectral and path analysis plus deterministic visualisation.",
    ("org.modellab.pack.multidimensional-mathematics",),
    tuple(item.kind for item in KIND_DESCRIPTORS),
    tuple(item.identifier for item in CAPABILITY_DESCRIPTORS),
)
