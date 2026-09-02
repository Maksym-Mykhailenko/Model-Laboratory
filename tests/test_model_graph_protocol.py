from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
import sympy as sp

import desktop_engine
from model_lab.bundle import MlabBundleError, create_run_mlab_bundle, load_mlab_bundle
from model_lab.builtin_packs import run_registry
from model_lab.canonical import canonical_model_ir_payload, canonical_model_ir_sha256
from model_lab.experiment import create_run_experiment_state
from model_lab.issues import DiagnosticSeverity, NumericalDiagnostic
from model_lab.model_graph import (
    CORE_KIND_REGISTRY,
    ObjectKindDescriptor,
    ObjectKindRegistry,
)
from model_lab.parser import parse_model_text
from model_lab.protocol import (
    ArtifactTypeDescriptor,
    CapabilityPackRegistry,
    ScientificArtifact,
    comparator_registry,
    run_capability_sweep,
)
from model_lab.reproduction import ReproductionStatus, reproduce_run_experiment
from model_lab.symbolic import analyse_scalar_function
from model_lab.validator import ModelValidationError, validate_model
from model_lab.vector_analysis import (
    analyse_matrix_function,
    analyse_vector_function,
    evaluate_vector_field_2d,
    optimize_scalar_with_constraints,
    solve_vector_system,
)
from model_lab.visualisation import create_vector_field_figure


def _compile(source: str, *, registry: ObjectKindRegistry | None = None):
    return validate_model(parse_model_text(source), kind_registry=registry)


VECTOR_SOURCE = """\
name: Vector, matrix, and constrained model
variables:
  x: {domain: [-3, 3], initial: 0.5}
  y: {domain: [-3, 3], initial: 0.5}
  z: {domain: [-3, 3], initial: 0.5}
functions:
  objective: (x - 1)**2 + (y - 2)**2 + z**2
vector_functions:
  F:
    components:
      p: x + y
      q: y*z
  stable_field:
    components:
      dx: -x
      dy: -y
matrix_functions:
  A:
    entries:
      - [x, "1"]
      - ["0", y]
constraints:
  plane:
    left: x + y
    relation: "="
    right: "2"
  z_zero:
    left: z
    relation: "="
    right: "0"
"""


def test_model_graph_is_canonical_and_unknown_kinds_are_opaque() -> None:
    first = _compile(
        """\
name: Roof fragment
objects:
  support:
    kind: org.example.structures.support
    properties: {fixed: true}
  node:
    kind: org.example.structures.node
    value_type: {kind: vector, shape: [3], element_kind: real}
    properties: {coordinates: [0, 1, 2]}
    references: [support]
relationships:
  restraint:
    kind: org.example.structures.restrained-by
    source: node
    target: support
"""
    )
    second = _compile(
        """\
name: Roof fragment
relationships:
  restraint:
    target: support
    source: node
    kind: org.example.structures.restrained-by
objects:
  node:
    references: [support]
    properties: {coordinates: [0, 1, 2]}
    value_type: {element_kind: real, shape: [3], kind: vector}
    kind: org.example.structures.node
  support:
    properties: {fixed: true}
    kind: org.example.structures.support
"""
    )

    assert canonical_model_ir_sha256(first) == canonical_model_ir_sha256(second)
    assert canonical_model_ir_payload(first)["schema_version"] == "3.0"
    assert first.graph.object("node").opaque is True
    assert first.graph.object("node").executable is False
    assert first.graph.unavailable_extension_kinds == (
        ("org.example.structures.node", "1.0"),
        ("org.example.structures.restrained-by", "1.0"),
        ("org.example.structures.support", "1.0"),
    )


def test_installed_extension_kind_uses_strict_versioned_schema() -> None:
    registry = ObjectKindRegistry(
        (
            *CORE_KIND_REGISTRY.descriptors,
            ObjectKindDescriptor(
                "org.example.structures.node",
                "2.0",
                "Structural node",
                {
                    "type": "object",
                    "required": ["coordinates"],
                    "properties": {
                        "coordinates": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 3,
                            "items": {"type": "number"},
                        }
                    },
                    "additionalProperties": False,
                },
                executable=True,
            ),
        )
    )
    valid = _compile(
        """\
name: Typed node
objects:
  n1:
    kind: org.example.structures.node
    kind_version: "2.0"
    properties: {coordinates: [0, 1, 2]}
""",
        registry=registry,
    )
    assert valid.graph.object("n1").opaque is False
    assert valid.graph.object("n1").executable is True

    with pytest.raises(ModelValidationError, match="unknown properties"):
        _compile(
            """\
name: Invalid typed node
objects:
  n1:
    kind: org.example.structures.node
    kind_version: "2.0"
    properties: {coordinates: [0, 1, 2], surprise: 4}
""",
            registry=registry,
        )


def test_model_graph_rejects_dangling_references() -> None:
    with pytest.raises(ModelValidationError, match="references unknown"):
        _compile(
            """\
name: Dangling graph
objects:
  n1:
    kind: org.example.graph.node
    references: [missing]
"""
        )


def test_model_graph_types_carry_domains_and_function_signatures() -> None:
    model = _compile(
        """\
name: Typed signal
objects:
  signal:
    kind: org.example.signals.transfer
    value_type:
      kind: function
      input_types:
        - kind: real
          domain:
            kind: interval
            lower: -1
            upper: 1
      output_type:
        kind: vector
        shape: [2]
        element_kind: real
"""
    )
    value_type = model.graph.object("signal").value_type
    assert value_type.kind == "function"
    assert value_type.input_types[0].domain is not None
    assert value_type.input_types[0].domain.kind == "interval"
    assert value_type.output_type is not None
    assert value_type.output_type.shape == (2,)

    scalar_model = _compile(
        """\
name: Explicit scalar signature
variables:
  x: {domain: [-2, 4]}
functions:
  f: x**2
"""
    )
    function_type = scalar_model.graph.object("function:f").value_type
    assert function_type.kind == "function"
    assert function_type.input_types[0].domain is not None
    assert function_type.input_types[0].domain.payload()["lower"] == -2.0
    assert function_type.output_type is not None
    assert function_type.output_type.kind == "real"


def test_general_vector_matrix_and_higher_dimensional_scalar_analysis() -> None:
    model = _compile(VECTOR_SOURCE)
    vector = analyse_vector_function(model, "F")
    assert vector.jacobian == sp.Matrix([[1, 1, 0], [0, sp.Symbol("z", real=True), sp.Symbol("y", real=True)]])
    assert vector.divergence is None
    assert vector.curl is None

    scalar = analyse_scalar_function(model, "objective")
    assert scalar.variable_names == ("x", "y", "z")
    assert scalar.gradient == (
        2 * sp.Symbol("x", real=True) - 2,
        2 * sp.Symbol("y", real=True) - 4,
        2 * sp.Symbol("z", real=True),
    )
    assert scalar.hessian == 2 * sp.eye(3)

    matrix = analyse_matrix_function(model, (2.0, 3.0, 0.0), function_name="A")
    assert matrix.rank == 2
    assert matrix.determinant == pytest.approx(6.0)
    assert sorted(value.real for value in matrix.eigenvalues) == pytest.approx([2.0, 3.0])


def test_vector_roots_field_visualisation_and_local_stability() -> None:
    model = _compile(VECTOR_SOURCE)
    roots = solve_vector_system(model, function_name="stable_field", seeds=8)
    assert roots.roots
    assert all(root.coordinates[:2] == pytest.approx((0.0, 0.0), abs=1e-8) for root in roots.roots)
    # stable_field has two components but the model has three inputs, so it is a
    # rectangular root problem and intentionally has no equilibrium eigen-classification.
    assert all(root.stability is None for root in roots.roots)

    field_model = _compile(
        """\
name: Stable plane
variables:
  x: {domain: [-2, 2], initial: 1}
  y: {domain: [-2, 2], initial: 1}
vector_functions:
  flow:
    components: {dx: -x, dy: -y}
"""
    )
    field = evaluate_vector_field_2d(field_model, function_name="flow", points_per_axis=9, root_seeds=8)
    assert field.u.shape == (9, 9)
    assert field.equilibria[0].stability == "asymptotically stable"
    figure = create_vector_field_figure(field).to_plotly_json()
    trace_names = {trace.get("name") for trace in figure["data"]}
    assert {"Field direction", "Streamlines", "dx = 0", "dy = 0", "Equilibria"}.issubset(trace_names)


def test_constraint_aware_optimisation_is_independent_of_stationary_points() -> None:
    model = _compile(VECTOR_SOURCE)
    result = optimize_scalar_with_constraints(model, function_name="objective", seeds=12)
    assert result.optimum is not None
    optimum = result.optimum
    assert optimum.coordinates == pytest.approx((0.5, 1.5, 0.0), abs=1e-5)
    assert optimum.value == pytest.approx(0.5, abs=1e-7)
    assert optimum.maximum_constraint_violation <= 1e-8


def test_capability_registry_is_namespaced_bounded_and_backend_identified() -> None:
    model = _compile(VECTOR_SOURCE)
    catalogue = {item["id"]: item for item in run_registry.catalogue(model)}
    assert catalogue["org.modellab.vector.differential"]["applicable"] is True
    assert catalogue["org.modellab.vector.field-2d"]["applicable"] is False

    with pytest.raises(Exception, match="exceeds the configured limit"):
        run_registry.run(
            "org.modellab.vector.solve-roots",
            model,
            {"function_name": "F", "seeds": 100},
            maximum_workload_units=1,
        )

    outcome = run_registry.run(
        "org.modellab.vector.differential", model, {"function_name": "F"}
    )
    backend = outcome.run.backend_identity
    assert backend["pack_id"] == "org.modellab.pack.vector-calculus"
    assert backend["capability_implementation"] == "model_lab.vector_analysis:analyse_vector_function"
    assert backend["backend"] == "sympy"
    assert "sympy" in backend["package_versions"]


def test_diagnostic_presentation_does_not_affect_scientific_identity() -> None:
    artifact_type = ArtifactTypeDescriptor(
        "org.example.artifact.diagnostic",
        "1.0",
        "Diagnostic result",
        "org.modellab.comparator.numeric",
    )
    old = NumericalDiagnostic(
        "nonfinite",
        DiagnosticSeverity.WARNING,
        "Old wording",
        (("b", "2"), ("a", "1")),
    )
    new = NumericalDiagnostic(
        "nonfinite",
        DiagnosticSeverity.WARNING,
        "New wording",
        (("a", "1"), ("b", "2")),
    )
    left = ScientificArtifact.create(
        artifact_type=artifact_type,
        capability_id="org.example.capability.test",
        capability_version="1.0",
        model_ir_sha256="a" * 64,
        data={"value": 1.0, "diagnostic": old},
    )
    right = ScientificArtifact.create(
        artifact_type=artifact_type,
        capability_id="org.example.capability.test",
        capability_version="1.0",
        model_ir_sha256="a" * 64,
        data={"value": 1.0, "diagnostic": new},
    )

    assert left.artifact_sha256 == right.artifact_sha256
    assert left.data["diagnostic"]["presentation"]["message"] == "Old wording"
    assert right.data["diagnostic"]["presentation"]["message"] == "New wording"
    comparison = comparator_registry.compare(
        "org.modellab.comparator.numeric", left, right
    )
    assert comparison.reproduced is True


def _asset_experiment() -> tuple[str, object, object, bytes, str]:
    blob = b"node,x,y,z\n1,0,1,2\n"
    digest = hashlib.sha256(blob).hexdigest()
    source = f"""\
name: Model with structured asset
variables:
  x: {{domain: [-2, 2]}}
functions:
  f: x**2
assets:
  mesh:
    kind: org.example.structures.node-table
    kind_version: "1.0"
    media_type: text/csv
    sha256: {digest}
    size: {len(blob)}
"""
    model = _compile(source)
    outcome = run_registry.run("org.modellab.scalar.differential", model, {"function_name": "f"})
    state = create_run_experiment_state(
        model_source=source,
        model=model,
        parameter_values={},
        run_outcomes=(outcome,),
    )
    return source, model, state, blob, digest


def test_mlab_2_content_addresses_assets_runs_artifacts_and_views() -> None:
    _, model, state, blob, digest = _asset_experiment()
    artifact_content = b"portable-binary-result\x00\x01"
    content_digest = hashlib.sha256(artifact_content).hexdigest()
    artifact_id = state.artifacts[0]["artifact_id"]
    data = create_run_mlab_bundle(
        state=state,
        model=model,
        content_blobs={content_digest: artifact_content},
        content_descriptors={
            content_digest: {
                "media_type": "application/vnd.example.numeric-array",
                "schema": "org.example.numeric-array",
                "schema_version": "1.0",
                "artifact_ids": [artifact_id],
                "provenance": {"operation": "test fixture"},
                "chunk_size": 7,
            }
        },
        asset_blobs={digest: blob},
    )
    bundle = load_mlab_bundle(data)
    path = f"assets/sha256/{digest}.bin"
    content_path = f"artifacts/sha256/{content_digest}.bin"

    assert bundle.manifest["format_version"] == "2.0"
    assert bundle.asset_members[path] == blob
    assert bundle.content_members[content_path] == artifact_content
    assert bundle.content_descriptors[0]["artifact_ids"] == [artifact_id]
    assert len(bundle.content_descriptors[0]["chunks"]) == 4
    assert bundle.asset_documents[0]["chunks"]
    assert bundle.run_documents == state.run_records
    assert bundle.artifact_documents == state.artifacts
    assert bundle.view_documents == state.views
    assert "assets/index.json" in {item["path"] for item in bundle.manifest["members"]}
    manifest_members = {item["path"]: item for item in bundle.manifest["members"]}
    assert manifest_members[content_path]["media_type"] == "application/vnd.example.numeric-array"
    assert manifest_members[path]["media_type"] == "text/csv"

    inspection = desktop_engine.dispatch(
        {
            "action": "inspect_experiment",
            "payload": {
                "filename": "asset.mlab",
                "data_base64": base64.b64encode(data).decode("ascii"),
            },
        }
    )
    assert inspection["execution_performed"] is False
    handle = inspection["run_protocol"]["content_handle"]
    chunk = desktop_engine.dispatch(
        {
            "action": "read_content_chunk",
            "payload": {"handle": handle, "path": path, "offset": 5, "length": 7},
        }
    )
    assert base64.b64decode(chunk["data_base64"]) == blob[5:12]
    assert chunk["total_size"] == len(blob)


def test_mlab_asset_integrity_rejects_tampering() -> None:
    _, model, state, blob, digest = _asset_experiment()
    original = create_run_mlab_bundle(state=state, model=model, asset_blobs={digest: blob})
    source = BytesIO(original)
    target = BytesIO()
    path = f"assets/sha256/{digest}.bin"
    with ZipFile(source, "r") as reader, ZipFile(target, "w", ZIP_DEFLATED) as writer:
        for info in reader.infolist():
            payload = reader.read(info.filename)
            writer.writestr(info, b"tampered" if info.filename == path else payload)
    with pytest.raises(MlabBundleError, match="manifest.json"):
        load_mlab_bundle(target.getvalue())


def test_typed_experiment_reproduction_and_missing_pack_status() -> None:
    source = """\
name: Scalar differential run
variables:
  x: {domain: [-2, 2]}
functions:
  f: sinh(x) + cosh(x) + tanh(x)
"""
    model = _compile(source)
    outcome = run_registry.run("org.modellab.scalar.differential", model, {"function_name": "f"})
    state = create_run_experiment_state(
        model_source=source,
        model=model,
        parameter_values={},
        run_outcomes=(outcome,),
    )
    exact = reproduce_run_experiment(state, model=model)
    assert exact.report.status is ReproductionStatus.EXACT

    unavailable = reproduce_run_experiment(
        state,
        model=model,
        registry=CapabilityPackRegistry(),
    )
    assert unavailable.report.status is ReproductionStatus.UNABLE
    assert "unavailable" in " ".join(unavailable.report.execution_errors).lower()


def test_generic_sweep_can_orchestrate_any_registered_run() -> None:
    source = """\
name: Generic point sweep
variables:
  x: {domain: [-4, 4]}
parameters:
  a: {default: 1, domain: [-2, 2]}
functions:
  f: a*x**2
"""
    model = _compile(source)
    sweep = run_capability_sweep(
        run_registry,
        model,
        target_capability_id="org.modellab.scalar.evaluate-points",
        coordinate_name="a",
        coordinate_values=(-1.0, 0.0, 1.0),
        setting_path=("parameter_values", "a"),
        base_settings={"points": [[1.0]], "parameter_values": {"a": 1.0}},
    )
    assert sweep.run.capability_id == "org.modellab.protocol.capability-sweep"
    assert sweep.artifacts[0].artifact_type == "org.modellab.artifact.capability-sweep"
    assert len(sweep.artifacts[0].data["steps"]) == 3


def test_current_version_contracts_are_unambiguous() -> None:
    model = _compile("""\
name: Hyperbolic AST
variables:
  x: {domain: [-1, 1]}
functions:
  f: sinh(x)
""")
    payload = canonical_model_ir_payload(model)
    assert payload["schema_version"] == "3.0"
    assert payload["functions"][0]["expression"]["schema_version"] == "1.2"
    assert desktop_engine.dispatch({"action": "health"})["protocol_version"] == 7
