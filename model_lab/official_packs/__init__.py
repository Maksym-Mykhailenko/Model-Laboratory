"""Official structured scientific capability packs shipped with Model Laboratory.

The modules in this package own domain object schemas and deterministic scientific
implementations.  Documents may declare the corresponding namespaced kinds, but never
carry executable code.  The locally installed registry remains authoritative.
"""

from .registry import (
    OFFICIAL_CAPABILITY_DESCRIPTORS,
    OFFICIAL_KIND_DESCRIPTORS,
    OFFICIAL_KIND_REGISTRY,
    OFFICIAL_PACK_MANIFESTS,
    official_capability_descriptors,
    official_kind_catalogue,
    official_pack_catalogue,
    validate_official_graph,
)

__all__ = [
    "OFFICIAL_CAPABILITY_DESCRIPTORS",
    "OFFICIAL_KIND_DESCRIPTORS",
    "OFFICIAL_KIND_REGISTRY",
    "OFFICIAL_PACK_MANIFESTS",
    "official_capability_descriptors",
    "official_kind_catalogue",
    "official_pack_catalogue",
    "validate_official_graph",
]
