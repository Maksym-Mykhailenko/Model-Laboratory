"""Independent verification of Model Graph 3.0 and Run -> Artifact architecture."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import desktop_engine
from model_lab.bundle import create_run_mlab_bundle, load_mlab_bundle
from model_lab.builtin_packs import run_registry
from model_lab.canonical import canonical_model_ir_payload
from model_lab.experiment import create_run_experiment_state
from model_lab.interpreter import MAX_CONTEXT_PACKAGE_BYTES, create_interpreter_context
from model_lab.issues import DiagnosticSeverity, NumericalDiagnostic
from model_lab.parser import parse_model_text
from model_lab.protocol import (
    ArtifactTypeDescriptor,
    CapabilityPackRegistry,
    ScientificArtifact,
    comparator_registry,
)
from model_lab.reproduction import ReproductionStatus, reproduce_run_experiment
from model_lab.validator import validate_model
from model_lab.vector_analysis import (
    analyse_matrix_function,
    analyse_vector_function,
    evaluate_vector_field_2d,
    optimize_scalar_with_constraints,
)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


SOURCE = """\
name: Protocol verification model
variables:
  x: {domain: [-3, 3], initial: 1}
  y: {domain: [-3, 3], initial: 1}
functions:
  objective: (x - 1)**2 + (y - 2)**2
vector_functions:
  flow:
    components: {dx: -x, dy: -y}
matrix_functions:
  A:
    entries:
      - [x, "1"]
      - ["0", y]
constraints:
  plane: {left: x + y, relation: "=", right: "2"}
"""


def run() -> list[Check]:
    checks: list[Check] = []
    try:
        model = validate_model(parse_model_text(SOURCE))
        payload = canonical_model_ir_payload(model)
        checks.extend(
            (
                Check("canonical Model IR is 3.0", payload["schema_version"] == "3.0"),
                Check("Model Graph is embedded", payload["graph"]["schema_version"] == "3.0"),
                Check(
                    "scalar function has explicit signature",
                    model.graph.object("function:objective").value_type.kind == "function",
                ),
                Check(
                    "input domain is explicit",
                    model.graph.object("function:objective").value_type.input_types[0].domain is not None,
                ),
            )
        )

        differential = analyse_vector_function(model, "flow")
        checks.extend(
            (
                Check("general Jacobian", differential.jacobian == -sp.eye(2)),
                Check("divergence", differential.divergence == -2),
                Check("two-dimensional curl", differential.curl == 0),
            )
        )
        field = evaluate_vector_field_2d(model, function_name="flow", points_per_axis=7, root_seeds=8)
        checks.append(
            Check(
                "equilibrium stability",
                bool(field.equilibria)
                and field.equilibria[0].stability == "asymptotically stable",
            )
        )
        matrix = analyse_matrix_function(model, (2.0, 3.0), function_name="A")
        checks.extend(
            (
                Check("matrix determinant", abs(complex(matrix.determinant or 0) - 6) < 1e-10),
                Check("matrix rank", matrix.rank == 2),
                Check("matrix eigenvalues", sorted(round(item.real, 10) for item in matrix.eigenvalues) == [2.0, 3.0]),
            )
        )
        optimum = optimize_scalar_with_constraints(model, function_name="objective", seeds=12).optimum
        checks.append(
            Check(
                "constraint-aware optimum",
                optimum is not None
                and max(abs(optimum.coordinates[0] - 0.5), abs(optimum.coordinates[1] - 1.5)) < 1e-5,
            )
        )

        catalogue = {item["id"]: item for item in run_registry.catalogue(model)}
        checks.extend(
            (
                Check("namespaced vector capability", catalogue["org.modellab.vector.differential"]["applicable"]),
                Check("namespaced matrix capability", catalogue["org.modellab.matrix.analyse"]["applicable"]),
                Check("namespaced optimisation capability", catalogue["org.modellab.optimization.constrained"]["applicable"]),
            )
        )
        outcome = run_registry.run(
            "org.modellab.vector.differential", model, {"function_name": "flow"}
        )
        checks.extend(
            (
                Check("run schema is 1.1", outcome.run.payload()["schema_version"] == "1.1"),
                Check("run records implementation identity", bool(outcome.run.backend_identity.get("capability_implementation"))),
                Check("artifact is content-addressed", outcome.artifacts[0].artifact_id.startswith("sha256:")),
            )
        )

        diagnostic_type = ArtifactTypeDescriptor(
            "org.example.artifact.diagnostic",
            "1.0",
            "Diagnostic",
            "org.modellab.comparator.numeric",
        )
        left = ScientificArtifact.create(
            artifact_type=diagnostic_type,
            capability_id="org.example.capability.diagnostic",
            capability_version="1.0",
            model_ir_sha256="a" * 64,
            data={
                "value": 1.0,
                "diagnostic": NumericalDiagnostic(
                    "nonfinite",
                    DiagnosticSeverity.WARNING,
                    "Old wording",
                    (("b", "2"), ("a", "1")),
                ),
            },
        )
        right = ScientificArtifact.create(
            artifact_type=diagnostic_type,
            capability_id="org.example.capability.diagnostic",
            capability_version="1.0",
            model_ir_sha256="a" * 64,
            data={
                "value": 1.0,
                "diagnostic": NumericalDiagnostic(
                    "nonfinite",
                    DiagnosticSeverity.WARNING,
                    "New wording",
                    (("a", "1"), ("b", "2")),
                ),
            },
        )
        checks.extend(
            (
                Check("diagnostic message excluded from identity", left.artifact_sha256 == right.artifact_sha256),
                Check("diagnostic messages remain persisted", left.data != right.data),
                Check(
                    "diagnostic numeric comparison",
                    comparator_registry.compare(
                        "org.modellab.comparator.numeric", left, right
                    ).reproduced,
                ),
            )
        )

        state = create_run_experiment_state(
            model_source=SOURCE,
            model=model,
            parameter_values={},
            run_outcomes=(outcome,),
        )
        bundle_data = create_run_mlab_bundle(state=state, model=model)
        bundle = load_mlab_bundle(bundle_data)
        member_paths = {entry["path"] for entry in bundle.manifest["members"]}
        checks.extend(
            (
                Check("experiment state is 6", state.format_version == 6),
                Check(".mlab format is 2.0", bundle.manifest["format_version"] == "2.0"),
                Check("run index present", "runs/index.json" in member_paths),
                Check("artifact index present", "artifacts/index.json" in member_paths),
                Check("asset index present", "assets/index.json" in member_paths),
                Check("view index present", "views/index.json" in member_paths),
            )
        )
        exact = reproduce_run_experiment(state, model=model)
        unavailable = reproduce_run_experiment(
            state, model=model, registry=CapabilityPackRegistry()
        )
        checks.extend(
            (
                Check("typed exact reproduction", exact.report.status is ReproductionStatus.EXACT),
                Check("missing pack is unable", unavailable.report.status is ReproductionStatus.UNABLE),
            )
        )
        inspection = desktop_engine.dispatch(
            {
                "action": "inspect_experiment",
                "payload": {
                    "filename": "verification.mlab",
                    "data_base64": base64.b64encode(bundle_data).decode("ascii"),
                },
            }
        )
        checks.append(Check("inspection performs no execution", inspection["execution_performed"] is False))

        context = create_interpreter_context(
            instruction="Explain the installed model structure.",
            current_model_source=SOURCE,
        )
        context_bytes = json.dumps(context, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        checks.extend(
            (
                Check("interpreter context remains bounded", len(context_bytes) <= MAX_CONTEXT_PACKAGE_BYTES),
                Check(
                    "interpreter schema is generated",
                    len(
                        context["contract"]["model"]["generated_schema_catalogue"][
                            "authoritative_schema_sha256"
                        ]
                    )
                    == 64,
                ),
                Check(
                    "interpreter functions come from registry",
                    {"sinh", "cosh", "tanh"}.issubset(
                        context["contract"]["expression_language"]["functions"]
                    ),
                ),
            )
        )
    except Exception as exc:
        checks.append(Check("Model Graph / Run protocol verification completed", False, str(exc)))
    return checks


def main() -> int:
    checks = run()
    passed = sum(item.passed for item in checks)
    print(f"Model-Graph/Run-protocol verification: {passed}/{len(checks)} passed")
    for item in checks:
        if not item.passed:
            print(f"FAIL: {item.name}: {item.detail}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
