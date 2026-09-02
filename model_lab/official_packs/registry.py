"""Installed registry and public catalogue for Model Laboratory official packs."""

from __future__ import annotations

from typing import Any

from ..model import ModelIR
from ..model_graph import CORE_KIND_REGISTRY, ModelGraph, ObjectKindRegistry
from . import (
    dynamics,
    electrical,
    fields,
    generative,
    geometry,
    graphs,
    learning,
    mechanics,
    multidimensional,
    optimisation,
    probability,
    reactions,
    statistics,
)
from .common import validate_all


_MODULES = (
    multidimensional,
    probability,
    graphs,
    generative,
    dynamics,
    fields,
    geometry,
    mechanics,
    statistics,
    optimisation,
    electrical,
    reactions,
    learning,
)

_MODULE_BY_PACK_ID = {module.MANIFEST.identifier: module for module in _MODULES}

OFFICIAL_KIND_DESCRIPTORS = tuple(
    descriptor for module in _MODULES for descriptor in module.KIND_DESCRIPTORS
)
OFFICIAL_CAPABILITY_DESCRIPTORS = tuple(
    descriptor for module in _MODULES for descriptor in module.CAPABILITY_DESCRIPTORS
)
OFFICIAL_PACK_MANIFESTS = tuple(module.MANIFEST for module in _MODULES)
OFFICIAL_KIND_REGISTRY = ObjectKindRegistry(
    (*CORE_KIND_REGISTRY.descriptors, *OFFICIAL_KIND_DESCRIPTORS)
)

_SEMANTIC_VALIDATORS = {
    kind: validator
    for module in _MODULES
    for kind, validator in module.SEMANTIC_VALIDATORS.items()
}


def validate_official_graph(graph: ModelGraph) -> None:
    """Apply scientific constraints that are more expressive than JSON Schema."""
    validate_all(graph, _SEMANTIC_VALIDATORS)


def official_pack_catalogue(model: ModelIR | None = None) -> list[dict[str, Any]]:
    """Return stable official-pack metadata and model-specific applicability counts."""
    rows: list[dict[str, Any]] = []
    for manifest in OFFICIAL_PACK_MANIFESTS:
        payload = manifest.payload()
        capabilities = [
            item for item in OFFICIAL_CAPABILITY_DESCRIPTORS
            if item.pack_id == manifest.identifier
        ]
        payload["installed"] = True
        payload["capability_count"] = len(capabilities)
        payload["applicable_capability_count"] = (
            0 if model is None else sum(item.applicability(model)[0] for item in capabilities)
        )
        rows.append(payload)
    return rows


def official_kind_catalogue(pack_ids: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """Return the installed kind catalogue for selected official packs.

    Dependency closure is included in manifest order.  The interpreter uses this bounded
    projection so adding official disciplines cannot force every schema into every prompt.
    """
    selected = set(_MODULE_BY_PACK_ID) if pack_ids is None else _pack_dependency_closure(pack_ids)
    return [
        row
        for module in _MODULES
        if module.MANIFEST.identifier in selected
        for row in ObjectKindRegistry(module.KIND_DESCRIPTORS).catalogue()
    ]


def official_capability_descriptors(
    pack_ids: tuple[str, ...] | None = None,
) -> tuple[Any, ...]:
    selected = set(_MODULE_BY_PACK_ID) if pack_ids is None else _pack_dependency_closure(pack_ids)
    return tuple(
        descriptor
        for module in _MODULES
        if module.MANIFEST.identifier in selected
        for descriptor in module.CAPABILITY_DESCRIPTORS
    )


def _pack_dependency_closure(pack_ids: tuple[str, ...]) -> set[str]:
    unknown = set(pack_ids) - set(_MODULE_BY_PACK_ID)
    if unknown:
        raise ValueError("Unknown official pack identifier(s): " + ", ".join(sorted(unknown)) + ".")
    selected = set(pack_ids)
    pending = list(pack_ids)
    while pending:
        manifest = _MODULE_BY_PACK_ID[pending.pop()].MANIFEST
        for dependency in manifest.dependencies:
            if dependency in _MODULE_BY_PACK_ID and dependency not in selected:
                selected.add(dependency)
                pending.append(dependency)
    return selected
