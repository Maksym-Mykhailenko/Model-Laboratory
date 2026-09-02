"""Namespaced Run -> Artifact protocol for extensible scientific capabilities.

Capability implementations are installed locally and registered explicitly.  Portable
documents may refer to capability identifiers and preserve artifacts, but cannot provide
or execute implementation code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import importlib.metadata
import math
import re
from copy import deepcopy
from typing import Any, Callable, Iterable, Mapping
import uuid

import numpy as np
import sympy as sp

from .canonical import canonical_json_sha256
from .issues import NumericalDiagnostic
from .model import ModelIR


RUN_SCHEMA = "model-laboratory-run"
RUN_SCHEMA_VERSION = "1.1"
PREVIOUS_RUN_SCHEMA_VERSION = "1.0"
ARTIFACT_SCHEMA = "model-laboratory-scientific-artifact"
ARTIFACT_SCHEMA_VERSION = "1.0"
VIEW_SCHEMA = "model-laboratory-artifact-view"
VIEW_SCHEMA_VERSION = "1.0"

_NAMESPACED_ID = re.compile(r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)+(?:/[a-z][a-z0-9-]*)*$")


class ProtocolError(ValueError):
    """Raised for malformed runs, artifacts, descriptors, or registry use."""


def _strict_settings_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the executable settings contract and make its root object fail closed."""
    copied = deepcopy(dict(schema))
    if copied.get("type") != "object":
        raise ProtocolError("Capability settings_schema must describe an object.")
    copied.setdefault("properties", {})
    copied.setdefault("additionalProperties", False)
    return copied


def _validate_setting(value: Any, schema: Mapping[str, Any], *, path: str) -> Any:
    """Validate the JSON-Schema subset used by capability settings and apply defaults."""
    expected = schema.get("type")
    allowed = (expected,) if isinstance(expected, str) else tuple(expected or ())
    matches = {
        "null": value is None,
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)),
        "string": isinstance(value, str),
        "array": isinstance(value, (list, tuple)),
        "object": isinstance(value, Mapping),
    }
    if allowed and not any(matches.get(item, False) for item in allowed):
        raise ProtocolError(f"{path} must match settings type {expected!r}.")
    if "enum" in schema and value not in schema["enum"]:
        raise ProtocolError(f"{path} is outside the allowed settings values.")
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise ProtocolError("Capability settings_schema contains malformed properties.")
        result = {str(key): item for key, item in value.items()}
        for key, child_schema in properties.items():
            if key not in result and isinstance(child_schema, Mapping) and "default" in child_schema:
                result[str(key)] = deepcopy(child_schema["default"])
        missing = set(schema.get("required", ())) - set(result)
        if missing:
            raise ProtocolError(f"{path} is missing required settings: {', '.join(sorted(missing))}.")
        unknown = set(result) - set(properties)
        additional = schema.get("additionalProperties", True)
        if additional is False and unknown:
            raise ProtocolError(f"{path} contains unknown settings: {', '.join(sorted(unknown))}.")
        checked: dict[str, Any] = {}
        for key, item in result.items():
            child = properties.get(key)
            if isinstance(child, Mapping):
                checked[key] = _validate_setting(item, child, path=f"{path}.{key}")
            elif isinstance(additional, Mapping):
                checked[key] = _validate_setting(item, additional, path=f"{path}.{key}")
            else:
                checked[key] = portable_value(item)
        return checked
    if isinstance(value, (list, tuple)):
        if len(value) < int(schema.get("minItems", 0)):
            raise ProtocolError(f"{path} contains too few items.")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise ProtocolError(f"{path} contains too many items.")
        child = schema.get("items", {})
        if not isinstance(child, Mapping):
            raise ProtocolError("Capability settings_schema contains malformed array items.")
        return [_validate_setting(item, child, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            raise ProtocolError(f"{path} is shorter than permitted.")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise ProtocolError(f"{path} is longer than permitted.")
    if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        number = float(value)
        if not math.isfinite(number):
            raise ProtocolError(f"{path} must be finite.")
        if "minimum" in schema and number < float(schema["minimum"]):
            raise ProtocolError(f"{path} is below the allowed minimum.")
        if "maximum" in schema and number > float(schema["maximum"]):
            raise ProtocolError(f"{path} is above the allowed maximum.")
        if "exclusiveMinimum" in schema and number <= float(schema["exclusiveMinimum"]):
            raise ProtocolError(f"{path} is not above the exclusive minimum.")
        if "exclusiveMaximum" in schema and number >= float(schema["exclusiveMaximum"]):
            raise ProtocolError(f"{path} is not below the exclusive maximum.")
    return portable_value(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _namespaced(value: str, field: str) -> str:
    if not isinstance(value, str) or _NAMESPACED_ID.fullmatch(value) is None:
        raise ProtocolError(f"{field} must be a lowercase namespaced identifier.")
    return value


@dataclass(frozen=True, slots=True)
class SubspaceRepresentation:
    """Basis-compatible view whose scientific identity is an orthogonal projector."""

    dimension: int
    ambient_dimension: int
    orthogonal_projector: np.ndarray
    display_basis: np.ndarray
    basis_orientation: str

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.display_basis.shape)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.display_basis, dtype=dtype)


@dataclass(frozen=True, slots=True)
class SpectralDecomposition:
    """Degeneracy-aware symmetric spectrum with basis-invariant eigenspaces."""

    eigenvalues: np.ndarray
    eigenspaces: tuple[Mapping[str, Any], ...]
    display_basis: np.ndarray
    basis_orientation: str

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.display_basis.shape)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.display_basis, dtype=dtype)


def portable_value(value: Any) -> Any:
    """Convert scientific values to deterministic, finite JSON data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"special_float": "nan"}
        if value == math.inf:
            return {"special_float": "+inf"}
        if value == -math.inf:
            return {"special_float": "-inf"}
        return value
    if isinstance(value, complex):
        return {"real": portable_value(float(value.real)), "imag": portable_value(float(value.imag))}
    if isinstance(value, np.generic):
        return portable_value(value.item())
    if isinstance(value, np.ndarray):
        return {
            "encoding": "inline-array",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": portable_value(value.tolist()),
        }
    if isinstance(value, sp.MatrixBase):
        return {
            "rows": value.rows,
            "columns": value.cols,
            "entries": [
                [sp.sstr(value[row, column], order="lex") for column in range(value.cols)]
                for row in range(value.rows)
            ],
        }
    if isinstance(value, sp.Basic):
        return {"symbolic_expression": sp.sstr(value, order="lex")}
    if isinstance(value, Enum):
        return portable_value(value.value)
    if isinstance(value, NumericalDiagnostic):
        # Presentation wording is persisted, but is deliberately outside the
        # scientific identity used by fingerprints and comparators.
        return {
            "scientific": {
                "code": value.code,
                "severity": value.severity.value,
                "details": [[key, item] for key, item in value.details],
            },
            "presentation": {"message": value.message},
        }
    if isinstance(value, SubspaceRepresentation):
        return {
            "scientific": {
                "dimension": value.dimension,
                "ambient_dimension": value.ambient_dimension,
                "orthogonal_projector": portable_value(value.orthogonal_projector),
            },
            "presentation": {
                "basis": portable_value(value.display_basis),
                "basis_orientation": value.basis_orientation,
            },
        }
    if isinstance(value, SpectralDecomposition):
        return {
            "scientific": {
                "eigenvalues": portable_value(value.eigenvalues),
                "eigenspaces": portable_value(value.eigenspaces),
            },
            "presentation": {
                "basis": portable_value(value.display_basis),
                "basis_orientation": value.basis_orientation,
            },
        }
    if is_dataclass(value):
        return {
            item.name: portable_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): portable_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [portable_value(item) for item in value]
    raise ProtocolError(f"Scientific value of type {type(value).__name__} is not portable.")


def _scientific_projection(value: Any) -> Any:
    """Remove presentation-only fields from an artifact identity.

    Diagnostic documents intentionally use the exact two-part shape emitted by
    :func:`portable_value`.  Other mappings containing a key named ``presentation``
    retain it, so domain artifacts cannot accidentally lose scientific data merely
    because an extension used the same word.
    """
    if isinstance(value, Mapping):
        if set(value) == {"scientific", "presentation"}:
            return _scientific_projection(value["scientific"])
        return {
            str(key): _scientific_projection(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [_scientific_projection(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ArtifactTypeDescriptor:
    identifier: str
    version: str
    title: str
    comparator_id: str
    media_type: str = "application/vnd.model-laboratory.artifact+json"

    def __post_init__(self) -> None:
        _namespaced(self.identifier, "Artifact type")
        _namespaced(self.comparator_id, "Comparator")
        if not self.version or not self.title or "/" not in self.media_type:
            raise ProtocolError("Artifact type requires version, title, and MIME media type.")

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    identifier: str
    version: str
    pack_id: str
    title: str
    description: str
    settings_schema: Mapping[str, Any]
    backend: str
    output_types: tuple[ArtifactTypeDescriptor, ...]
    applicability: Callable[[ModelIR], tuple[bool, str]]
    runner: Callable[..., object]
    workload_estimator: Callable[[Mapping[str, Any], ModelIR], int]
    renderer_ids: tuple[str, ...] = ()
    input_schemas: tuple[str, ...] = ("model-laboratory-model-ir@3.0",)

    def __post_init__(self) -> None:
        _namespaced(self.identifier, "Capability")
        _namespaced(self.pack_id, "Capability pack")
        for renderer in self.renderer_ids:
            _namespaced(renderer, "Renderer")
        if not self.input_schemas or any(not item for item in self.input_schemas):
            raise ProtocolError("Capability descriptors require at least one input schema.")
        if not self.version or not self.title or not self.backend or not self.output_types:
            raise ProtocolError("Capability descriptors require version, title, backend, and outputs.")
        object.__setattr__(self, "settings_schema", _strict_settings_schema(self.settings_schema))

    def public_payload(self, model: ModelIR | None = None) -> dict[str, Any]:
        applicable, reason = self.applicability(model) if model is not None else (False, "model not supplied")
        return {
            "id": self.identifier,
            "version": self.version,
            "pack_id": self.pack_id,
            "title": self.title,
            "description": self.description,
            "settings_schema": portable_value(self.settings_schema),
            "input_schemas": list(self.input_schemas),
            "backend": self.backend,
            "outputs": [item.payload() for item in self.output_types],
            "renderer_ids": list(self.renderer_ids),
            "applicable": applicable,
            "inapplicable_reason": None if applicable else reason,
        }


@dataclass(frozen=True, slots=True)
class ScientificArtifact:
    """One immutable typed scientific result."""

    artifact_type: str
    artifact_type_version: str
    capability_id: str
    capability_version: str
    model_ir_sha256: str
    data: Any
    metadata: Mapping[str, Any]
    artifact_id: str
    artifact_sha256: str

    @classmethod
    def create(
        cls,
        *,
        artifact_type: ArtifactTypeDescriptor,
        capability_id: str,
        capability_version: str,
        model_ir_sha256: str,
        data: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ScientificArtifact":
        portable_data = portable_value(data)
        portable_metadata = portable_value(dict(metadata or {}))
        body = {
            "schema": ARTIFACT_SCHEMA,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": artifact_type.identifier,
            "artifact_type_version": artifact_type.version,
            "capability_id": capability_id,
            "capability_version": capability_version,
            "model_ir_sha256": model_ir_sha256,
            "data": portable_data,
            "metadata": portable_metadata,
        }
        identity_body = dict(body)
        identity_body["data"] = _scientific_projection(portable_data)
        identity_body["metadata"] = _scientific_projection(portable_metadata)
        digest = canonical_json_sha256(identity_body)
        return cls(
            artifact_type=artifact_type.identifier,
            artifact_type_version=artifact_type.version,
            capability_id=capability_id,
            capability_version=capability_version,
            model_ir_sha256=model_ir_sha256,
            data=portable_data,
            metadata=portable_metadata,
            artifact_id=f"sha256:{digest}",
            artifact_sha256=digest,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": ARTIFACT_SCHEMA,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "artifact_type": self.artifact_type,
            "artifact_type_version": self.artifact_type_version,
            "capability_id": self.capability_id,
            "capability_version": self.capability_version,
            "model_ir_sha256": self.model_ir_sha256,
            "data": portable_value(self.data),
            "metadata": portable_value(self.metadata),
        }


def validate_artifact_document(value: object) -> dict[str, Any]:
    """Validate a persisted artifact without invoking a capability or renderer."""
    if not isinstance(value, Mapping):
        raise ProtocolError("Scientific artifact must be an object.")
    expected = {
        "schema",
        "schema_version",
        "artifact_id",
        "artifact_sha256",
        "artifact_type",
        "artifact_type_version",
        "capability_id",
        "capability_version",
        "model_ir_sha256",
        "data",
        "metadata",
    }
    if set(value) != expected:
        raise ProtocolError("Scientific artifact has missing or unknown fields.")
    if value.get("schema") != ARTIFACT_SCHEMA or value.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ProtocolError("Unsupported scientific artifact schema.")
    for field_name in ("artifact_type", "capability_id"):
        _namespaced(str(value[field_name]), field_name)
    body = {key: portable_value(item) for key, item in value.items() if key not in {"artifact_id", "artifact_sha256"}}
    body["data"] = _scientific_projection(body["data"])
    body["metadata"] = _scientific_projection(body["metadata"])
    digest = canonical_json_sha256(body)
    if value.get("artifact_sha256") != digest or value.get("artifact_id") != f"sha256:{digest}":
        raise ProtocolError("Scientific artifact checksum does not match its contents.")
    return {str(key): portable_value(item) for key, item in value.items()}


def artifact_from_document(value: object) -> ScientificArtifact:
    document = validate_artifact_document(value)
    return ScientificArtifact(
        artifact_type=str(document["artifact_type"]),
        artifact_type_version=str(document["artifact_type_version"]),
        capability_id=str(document["capability_id"]),
        capability_version=str(document["capability_version"]),
        model_ir_sha256=str(document["model_ir_sha256"]),
        data=document["data"],
        metadata=document["metadata"],
        artifact_id=str(document["artifact_id"]),
        artifact_sha256=str(document["artifact_sha256"]),
    )


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One bounded execution record producing one or more artifacts."""

    run_id: str
    capability_id: str
    capability_version: str
    model_ir_sha256: str
    settings: Mapping[str, Any]
    workload_units: int
    started_at_utc: str
    completed_at_utc: str
    status: str
    artifact_ids: tuple[str, ...]
    backend_identity: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[Mapping[str, Any], ...] = ()

    def payload(self) -> dict[str, Any]:
        return {
            "schema": RUN_SCHEMA,
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": self.run_id,
            "capability_id": self.capability_id,
            "capability_version": self.capability_version,
            "model_ir_sha256": self.model_ir_sha256,
            "settings": portable_value(self.settings),
            "workload_units": self.workload_units,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "status": self.status,
            "artifact_ids": list(self.artifact_ids),
            "backend_identity": portable_value(self.backend_identity),
            "diagnostics": portable_value(self.diagnostics),
        }


def validate_run_document(value: object) -> dict[str, Any]:
    """Validate a persisted run record without executing it."""
    if not isinstance(value, Mapping):
        raise ProtocolError("Run record must be an object.")
    base_expected = {
        "schema", "schema_version", "run_id", "capability_id", "capability_version",
        "model_ir_sha256", "settings", "workload_units", "started_at_utc",
        "completed_at_utc", "status", "artifact_ids", "diagnostics",
    }
    version = value.get("schema_version")
    expected = base_expected | ({"backend_identity"} if version == RUN_SCHEMA_VERSION else set())
    if (
        set(value) != expected
        or value.get("schema") != RUN_SCHEMA
        or version not in {PREVIOUS_RUN_SCHEMA_VERSION, RUN_SCHEMA_VERSION}
    ):
        raise ProtocolError("Unsupported or malformed run record schema.")
    try:
        uuid.UUID(str(value["run_id"]))
    except (ValueError, AttributeError) as exc:
        raise ProtocolError("Run record run_id must be a UUID.") from exc
    _namespaced(str(value["capability_id"]), "Run capability")
    if not isinstance(value["workload_units"], int) or value["workload_units"] < 0:
        raise ProtocolError("Run workload_units must be a non-negative integer.")
    if value["status"] not in {"completed", "failed", "refused", "cancelled"}:
        raise ProtocolError("Run status is unsupported.")
    if version == RUN_SCHEMA_VERSION and not isinstance(value["backend_identity"], Mapping):
        raise ProtocolError("Run backend_identity must be an object.")
    if not isinstance(value["artifact_ids"], list) or not all(
        isinstance(item, str) and item.startswith("sha256:") for item in value["artifact_ids"]
    ):
        raise ProtocolError("Run artifact_ids must contain content-addressed identifiers.")
    return {str(key): portable_value(item) for key, item in value.items()}


@dataclass(frozen=True, slots=True)
class RunOutcome:
    run: RunRecord
    artifacts: tuple[ScientificArtifact, ...]
    results: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class ArtifactView:
    """Renderer-owned view configuration separate from scientific result identity."""

    view_id: str
    renderer_id: str
    renderer_version: str
    artifact_ids: tuple[str, ...]
    configuration: Mapping[str, Any]

    def payload(self) -> dict[str, Any]:
        return {
            "schema": VIEW_SCHEMA,
            "schema_version": VIEW_SCHEMA_VERSION,
            "view_id": self.view_id,
            "renderer_id": self.renderer_id,
            "renderer_version": self.renderer_version,
            "artifact_ids": list(self.artifact_ids),
            "configuration": portable_value(self.configuration),
        }


def validate_view_document(value: object) -> dict[str, Any]:
    """Validate renderer configuration without instantiating the renderer."""
    if not isinstance(value, Mapping):
        raise ProtocolError("Artifact view must be an object.")
    expected = {
        "schema", "schema_version", "view_id", "renderer_id", "renderer_version",
        "artifact_ids", "configuration",
    }
    if set(value) != expected or value.get("schema") != VIEW_SCHEMA or value.get("schema_version") != VIEW_SCHEMA_VERSION:
        raise ProtocolError("Unsupported or malformed artifact-view schema.")
    _namespaced(str(value["renderer_id"]), "Renderer")
    if not isinstance(value["artifact_ids"], list) or not all(
        isinstance(item, str) and item.startswith("sha256:") for item in value["artifact_ids"]
    ):
        raise ProtocolError("Artifact view references invalid artifact identifiers.")
    return {str(key): portable_value(item) for key, item in value.items()}


class CapabilityPackRegistry:
    """Local executable capability registry; documents contain references only."""

    def __init__(self, descriptors: Iterable[CapabilityDescriptor] = ()) -> None:
        self._descriptors: dict[tuple[str, str], CapabilityDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: CapabilityDescriptor) -> None:
        key = (descriptor.identifier, descriptor.version)
        if key in self._descriptors:
            raise ProtocolError(
                f"Capability '{descriptor.identifier}' version '{descriptor.version}' is already registered."
            )
        self._descriptors[key] = descriptor

    def descriptor(self, identifier: str, version: str | None = None) -> CapabilityDescriptor:
        matches = [item for (name, _), item in self._descriptors.items() if name == identifier]
        if version is not None:
            matches = [item for item in matches if item.version == version]
        if len(matches) != 1:
            raise ProtocolError(
                f"Installed capability '{identifier}'"
                + (f" version '{version}'" if version else "")
                + " is unavailable or ambiguous."
            )
        return matches[0]

    def catalogue(self, model: ModelIR | None = None) -> list[dict[str, Any]]:
        return [
            descriptor.public_payload(model)
            for _, descriptor in sorted(self._descriptors.items())
        ]

    def run(
        self,
        identifier: str,
        model: ModelIR,
        settings: Mapping[str, Any] | None = None,
        *,
        version: str | None = None,
        maximum_workload_units: int = 10_000_000,
    ) -> RunOutcome:
        """Execute only an installed descriptor after applicability/workload checks."""
        from .canonical import canonical_model_ir_sha256

        descriptor = self.descriptor(identifier, version)
        applicable, reason = descriptor.applicability(model)
        if not applicable:
            raise ProtocolError(f"Capability '{identifier}' is not applicable: {reason}.")
        normalised_settings = _validate_setting(dict(settings or {}), descriptor.settings_schema, path="settings")
        workload = descriptor.workload_estimator(normalised_settings, model)
        if isinstance(workload, bool) or not isinstance(workload, int) or workload < 0:
            raise ProtocolError("Capability workload estimator returned an invalid value.")
        if workload > maximum_workload_units:
            raise ProtocolError(
                f"Capability workload {workload} exceeds the configured limit {maximum_workload_units}."
            )
        started = _utc_now()
        try:
            result = descriptor.runner(model, **dict(normalised_settings))
        except TypeError as exc:
            raise ProtocolError(f"Capability settings are invalid: {exc}") from exc
        model_hash = canonical_model_ir_sha256(model)
        results = result if isinstance(result, tuple) and len(descriptor.output_types) > 1 else (result,)
        if len(results) != len(descriptor.output_types):
            raise ProtocolError("Capability returned a different number of outputs than declared.")
        artifacts = tuple(
            ScientificArtifact.create(
                artifact_type=artifact_type,
                capability_id=descriptor.identifier,
                capability_version=descriptor.version,
                model_ir_sha256=model_hash,
                data=item,
            )
            for artifact_type, item in zip(descriptor.output_types, results)
        )
        completed = _utc_now()
        package_versions: dict[str, str] = {}
        for candidate in re.split(r"[+/,]", descriptor.backend):
            package = candidate.strip().lower()
            if package not in {"numpy", "scipy", "sympy"}:
                continue
            try:
                package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                package_versions[package] = "not installed"
        run = RunRecord(
            run_id=str(uuid.uuid4()),
            capability_id=descriptor.identifier,
            capability_version=descriptor.version,
            model_ir_sha256=model_hash,
            settings=normalised_settings,
            workload_units=workload,
            started_at_utc=started,
            completed_at_utc=completed,
            status="completed",
            artifact_ids=tuple(item.artifact_id for item in artifacts),
            backend_identity={
                "pack_id": descriptor.pack_id,
                "capability_contract": f"{descriptor.identifier}@{descriptor.version}",
                "capability_implementation": (
                    f"{descriptor.runner.__module__}:{descriptor.runner.__qualname__}"
                ),
                "backend": descriptor.backend,
                "package_versions": package_versions,
            },
        )
        return RunOutcome(run=run, artifacts=artifacts, results=tuple(results))


@dataclass(frozen=True, slots=True)
class ComparisonOutcome:
    reproduced: bool
    comparator_id: str
    maximum_absolute_deviation: float | None
    maximum_relative_deviation: float | None
    details: Mapping[str, Any]


Comparator = Callable[[ScientificArtifact, ScientificArtifact, float, float], ComparisonOutcome]


class ComparatorRegistry:
    def __init__(self) -> None:
        self._comparators: dict[str, Comparator] = {}

    def register(self, identifier: str, comparator: Comparator) -> None:
        _namespaced(identifier, "Comparator")
        if identifier in self._comparators:
            raise ProtocolError(f"Comparator '{identifier}' is already registered.")
        self._comparators[identifier] = comparator

    def compare(
        self,
        identifier: str,
        reference: ScientificArtifact,
        current: ScientificArtifact,
        *,
        rtol: float = 1e-9,
        atol: float = 1e-12,
    ) -> ComparisonOutcome:
        try:
            comparator = self._comparators[identifier]
        except KeyError as exc:
            raise ProtocolError(f"Comparator '{identifier}' is not installed.") from exc
        return comparator(reference, current, rtol, atol)


def _exact_comparator(
    reference: ScientificArtifact,
    current: ScientificArtifact,
    rtol: float,
    atol: float,
) -> ComparisonOutcome:
    del rtol, atol
    matched = reference.artifact_sha256 == current.artifact_sha256
    return ComparisonOutcome(
        reproduced=matched,
        comparator_id="org.modellab.comparator.exact",
        maximum_absolute_deviation=0.0 if matched else None,
        maximum_relative_deviation=0.0 if matched else None,
        details={"reference_sha256": reference.artifact_sha256, "current_sha256": current.artifact_sha256},
    )


def _numeric_leaves(value: Any, path: tuple[str, ...] = ()) -> dict[tuple[str, ...], float]:
    result: dict[tuple[str, ...], float] = {}
    if isinstance(value, bool):
        return result
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        result[path] = float(value)
    elif isinstance(value, Mapping):
        if value.get("encoding") == "inline-array":
            result.update(_numeric_leaves(value.get("values"), path + ("values",)))
        elif set(value) == {"real", "imag"}:
            result.update(_numeric_leaves(value["real"], path + ("real",)))
            result.update(_numeric_leaves(value["imag"], path + ("imag",)))
        else:
            for key, item in sorted(value.items()):
                result.update(_numeric_leaves(item, path + (str(key),)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(_numeric_leaves(item, path + (str(index),)))
    return result


def _numeric_skeleton(value: Any) -> Any:
    """Mask finite numbers while preserving every non-numerical semantic field."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return {"numeric": True}
    if isinstance(value, Mapping):
        return {str(key): _numeric_skeleton(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_numeric_skeleton(item) for item in value]
    return value


def _numeric_comparator(
    reference: ScientificArtifact,
    current: ScientificArtifact,
    rtol: float,
    atol: float,
) -> ComparisonOutcome:
    if rtol < 0 or atol < 0:
        raise ProtocolError("Numerical tolerances must be non-negative.")
    reference_science = _scientific_projection(reference.data)
    current_science = _scientific_projection(current.data)
    left = _numeric_leaves(reference_science)
    right = _numeric_leaves(current_science)
    structure_matches = _numeric_skeleton(reference_science) == _numeric_skeleton(current_science)
    if set(left) != set(right) or not left or not structure_matches:
        return ComparisonOutcome(
            False,
            "org.modellab.comparator.numeric",
            None,
            None,
            {"aligned": False, "semantic_structure_matches": structure_matches},
        )
    absolute = [abs(left[path] - right[path]) for path in left]
    relative = [
        difference / max(abs(left[path]), abs(right[path]), np.finfo(float).tiny)
        for path, difference in zip(left, absolute)
    ]
    reproduced = all(
        difference <= atol + rtol * abs(left[path])
        for path, difference in zip(left, absolute)
    )
    return ComparisonOutcome(
        reproduced,
        "org.modellab.comparator.numeric",
        max(absolute, default=0.0),
        max(relative, default=0.0),
        {
            "aligned": True,
            "semantic_structure_matches": True,
            "value_count": len(left),
            "rtol": rtol,
            "atol": atol,
        },
    )


def _stochastic_comparator(
    reference: ScientificArtifact,
    current: ScientificArtifact,
    rtol: float,
    atol: float,
) -> ComparisonOutcome:
    """Compare stochastic artifacts using their pre-frozen statistical contract.

    ``rtol`` and ``atol`` remain report inputs, but the scientific decision is made
    using the equivalence settings stored in the reference before reproduction.
    """
    del rtol, atol
    from scipy.stats import ks_2samp
    from .experiment import _decode_array_reference

    left = reference.data.get("stochastic_reference") if isinstance(reference.data, Mapping) else None
    right = current.data.get("stochastic_reference") if isinstance(current.data, Mapping) else None
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return ComparisonOutcome(False, "org.modellab.comparator.stochastic", None, None, {"status": "UNABLE", "reason": "stochastic_reference is missing"})
    exact = left.get("exact_sha256") == right.get("exact_sha256")
    if exact:
        return ComparisonOutcome(True, "org.modellab.comparator.stochastic", 0.0, 0.0, {"status": "EXACT STOCHASTIC REPRODUCTION", "exact_fingerprint_matches": True})
    try:
        if left.get("schema") != "model-laboratory-stochastic-reference" or right.get("schema") != left.get("schema"):
            raise ProtocolError("Unsupported stochastic-reference schema.")
        settings = left["statistical_equivalence_settings"]
        left_summary, right_summary = left["summary"], right["summary"]
        if not all(isinstance(value, Mapping) for value in (settings, left_summary, right_summary)):
            raise ProtocolError("Malformed stochastic equivalence settings or summaries.")
        minimum = int(settings["minimum_sample_count"])
        counts = (int(left_summary["sample_count"]), int(right_summary["sample_count"]))
        if min(counts) < minimum:
            raise ProtocolError("Sample count is below the frozen minimum.")
        def sample_values(document: object) -> np.ndarray:
            if not isinstance(document, Mapping):
                raise ProtocolError("Malformed stochastic sample reference.")
            selected = document.get("sample_values") if document.get("encoding") == "canonical-samples-v1" else document
            if not isinstance(selected, Mapping):
                raise ProtocolError("Malformed canonical stochastic samples.")
            return _decode_array_reference(selected).reshape(-1)

        left_values = sample_values(left["samples"])
        right_values = sample_values(right["samples"])
        if min(left_values.size, right_values.size) < minimum:
            raise ProtocolError("Portable sample selection is below the frozen minimum.")
        mean_deviation = abs(float(left_summary["mean"]) - float(right_summary["mean"]))
        variance_deviation = abs(float(left_summary["variance"]) - float(right_summary["variance"]))
        mean_matches = bool(np.isclose(
            float(right_summary["mean"]), float(left_summary["mean"]),
            rtol=float(settings["mean_relative_tolerance"]),
            atol=float(settings["mean_absolute_tolerance"]),
        ))
        variance_matches = bool(np.isclose(
            float(right_summary["variance"]), float(left_summary["variance"]),
            rtol=float(settings["variance_relative_tolerance"]),
            atol=float(settings["variance_absolute_tolerance"]),
        ))
        ks = ks_2samp(left_values, right_values, method="auto")
        distribution_matches = float(ks.pvalue) >= float(settings["significance_level"])
        reproduced = mean_matches and variance_matches and distribution_matches
        return ComparisonOutcome(
            reproduced,
            "org.modellab.comparator.stochastic",
            max(mean_deviation, variance_deviation),
            None,
            {
                "status": "STATISTICALLY EQUIVALENT" if reproduced else "NOT STATISTICALLY EQUIVALENT",
                "exact_fingerprint_matches": False,
                "rng_matches": left.get("rng") == right.get("rng"),
                "sampling_configuration_matches": left.get("sampling") == right.get("sampling"),
                "saved_sample_count": counts[0],
                "current_sample_count": counts[1],
                "mean_absolute_deviation": mean_deviation,
                "variance_absolute_deviation": variance_deviation,
                "kolmogorov_smirnov_statistic": float(ks.statistic),
                "kolmogorov_smirnov_p_value": float(ks.pvalue),
                "frozen_settings": portable_value(settings),
            },
        )
    except (KeyError, TypeError, ValueError, ProtocolError) as exc:
        return ComparisonOutcome(False, "org.modellab.comparator.stochastic", None, None, {"status": "UNABLE", "reason": str(exc)})


comparator_registry = ComparatorRegistry()
comparator_registry.register("org.modellab.comparator.exact", _exact_comparator)
comparator_registry.register("org.modellab.comparator.numeric", _numeric_comparator)
comparator_registry.register("org.modellab.comparator.stochastic", _stochastic_comparator)


SWEEP_ARTIFACT_TYPE = ArtifactTypeDescriptor(
    "org.modellab.artifact.capability-sweep",
    "1.0",
    "Generic capability sweep",
    "org.modellab.comparator.numeric",
)


def run_capability_sweep(
    registry: CapabilityPackRegistry,
    model: ModelIR,
    *,
    target_capability_id: str,
    coordinate_name: str,
    coordinate_values: Iterable[float],
    setting_path: tuple[str, ...],
    base_settings: Mapping[str, Any] | None = None,
    target_capability_version: str | None = None,
    maximum_total_workload_units: int = 10_000_000,
) -> RunOutcome:
    """Sweep one setting of any installed capability and preserve aligned child artifacts."""
    from copy import deepcopy
    from .canonical import canonical_model_ir_sha256

    if not setting_path or len(setting_path) > 8:
        raise ProtocolError("A sweep setting path must contain one to eight components.")
    values = tuple(float(item) for item in coordinate_values)
    if not values or len(values) > 10000 or not all(math.isfinite(item) for item in values):
        raise ProtocolError("A capability sweep requires 1 to 10000 finite coordinates.")
    settings_template = portable_value(dict(base_settings or {}))
    outcomes: list[RunOutcome] = []
    total_workload = 0
    started = _utc_now()
    for value in values:
        settings = deepcopy(settings_template)
        current = settings
        for component in setting_path[:-1]:
            child = current.setdefault(component, {})
            if not isinstance(child, dict):
                raise ProtocolError("The sweep setting path traverses a non-object value.")
            current = child
        current[setting_path[-1]] = value
        outcome = registry.run(
            target_capability_id,
            model,
            settings,
            version=target_capability_version,
            maximum_workload_units=maximum_total_workload_units,
        )
        total_workload += outcome.run.workload_units
        if total_workload > maximum_total_workload_units:
            raise ProtocolError(
                f"Capability sweep workload exceeds the configured limit {maximum_total_workload_units}."
            )
        outcomes.append(outcome)
    model_hash = canonical_model_ir_sha256(model)
    artifact = ScientificArtifact.create(
        artifact_type=SWEEP_ARTIFACT_TYPE,
        capability_id="org.modellab.protocol.capability-sweep",
        capability_version="1.0",
        model_ir_sha256=model_hash,
        data={
            "target_capability_id": target_capability_id,
            "target_capability_version": outcomes[0].run.capability_version,
            "coordinate_name": coordinate_name,
            "coordinate_values": list(values),
            "setting_path": list(setting_path),
            "steps": [
                {
                    "coordinate": value,
                    "run": outcome.run.payload(),
                    "artifacts": [item.payload() for item in outcome.artifacts],
                }
                for value, outcome in zip(values, outcomes)
            ],
        },
    )
    completed = _utc_now()
    run = RunRecord(
        run_id=str(uuid.uuid4()),
        capability_id="org.modellab.protocol.capability-sweep",
        capability_version="1.0",
        model_ir_sha256=model_hash,
        settings={
            "target_capability_id": target_capability_id,
            "target_capability_version": target_capability_version,
            "coordinate_name": coordinate_name,
            "coordinate_values": list(values),
            "setting_path": list(setting_path),
            "base_settings": settings_template,
        },
        workload_units=total_workload,
        started_at_utc=started,
        completed_at_utc=completed,
        status="completed",
        artifact_ids=(artifact.artifact_id,),
        backend_identity={
            "capability_implementation": "org.modellab.protocol.capability-sweep@1.0",
            "backend": "orchestrator",
            "child_backends": [outcome.run.backend_identity for outcome in outcomes],
        },
    )
    return RunOutcome(run, (artifact,), (tuple(outcomes),))
