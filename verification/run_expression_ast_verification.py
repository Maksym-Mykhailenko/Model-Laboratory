"""Independent verification for the Model Laboratory-owned expression AST and migration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import getcontext
from io import BytesIO
import base64
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_lab.analysis import run_stationary_point_analysis
from model_lab.bundle import MLAB_FORMAT_VERSION, MlabBundleError, create_mlab_bundle, load_mlab_bundle
from model_lab.canonical import canonical_json_sha256, canonical_model_ir_payload, canonical_model_ir_sha256
from model_lab.capabilities import ModelVisualisation
from model_lab.evaluator import evaluate_single_variable_function, evaluate_two_variable_function
from model_lab.experiment import EvaluationSettings, StationaryPointSettings, create_experiment_state
from model_lab.expression import EXPRESSION_AST_SCHEMA, EXPRESSION_AST_SCHEMA_VERSION, expression_ast_payload, parse_expression
from model_lab.parser import parse_model_text
from model_lab.reproduction import ReproductionStatus, reproduce_mlab_bundle
from model_lab.validator import validate_model


MODEL_PATHS = (
    ROOT / "models" / "quadratic.yaml",
    ROOT / "models" / "surface.yaml",
    ROOT / "models" / "structured.yaml",
    ROOT / "models" / "issues.yaml",
)
LEGACY_FIXTURE = ROOT / "tests" / "fixtures" / "legacy-v1.0-expression-format.mlab"
PREVIOUS_AST_FIXTURE = ROOT / "tests" / "fixtures" / "legacy-v1.1-expression-ast.mlab"
LEGACY_HYPERBOLIC_IDENTIFIER_FIXTURE = (
    ROOT / "tests" / "fixtures" / "legacy-v1.4-sinh-identifier.mlab.b64"
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def _members(data: bytes) -> dict[str, bytes]:
    with ZipFile(BytesIO(data), "r") as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _rebuild(members: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(members):
            info = ZipInfo(name)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, members[name])
    return output.getvalue()


def _refresh_manifest(members: dict[str, bytes]) -> None:
    manifest = json.loads(members["manifest.json"])
    for entry in manifest["members"]:
        raw = members[entry["path"]]
        entry["size"] = len(raw)
        entry["sha256"] = hashlib.sha256(raw).hexdigest()
    members["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _all_expression_payloads(payload: dict) -> list[dict]:
    expressions: list[dict] = []
    expressions.extend(item["expression"] for item in payload["derived_quantities"])
    expressions.extend(item["expression"] for item in payload["functions"])
    for item in payload["constraints"]:
        expressions.extend((item["left"], item["right"]))
    return expressions


def _small_bundle(source: str):
    model = validate_model(parse_model_text(source))
    parameters = model.parameter_defaults()
    settings = EvaluationSettings(points_1d=81, points_per_axis_2d=25)
    stationary_settings = StationaryPointSettings(samples_1d=401, seeds_per_axis_2d=5, root_tolerance=1e-9)
    if len(model.variables) == 1:
        evaluation = evaluate_single_variable_function(model, parameters, points=settings.points_1d)
        visualisation = ModelVisualisation.TWO_D_FUNCTION_PLOT
    else:
        evaluation = evaluate_two_variable_function(model, parameters, points_per_axis=settings.points_per_axis_2d)
        visualisation = ModelVisualisation.THREE_D_SURFACE
    stationary = None
    if not model.constraints and model.is_computation_ready:
        stationary = run_stationary_point_analysis(
            model,
            parameters,
            samples_1d=stationary_settings.samples_1d,
            seeds_per_axis_2d=stationary_settings.seeds_per_axis_2d,
            root_tolerance=stationary_settings.root_tolerance,
        )
    state = create_experiment_state(
        model_source=source,
        model=model,
        parameter_values=parameters,
        selected_model_visualisation=visualisation,
        evaluation=evaluation,
        stationary_result=stationary,
        evaluation_settings=settings,
        stationary_settings=stationary_settings,
    )
    bundle = create_mlab_bundle(
        state=state,
        model=model,
        evaluation=evaluation,
        stationary_result=stationary,
        sweep_result=None,
    )
    return model, bundle


def run() -> list[Check]:
    checks: list[Check] = []

    for path in MODEL_PATHS:
        try:
            source = path.read_text(encoding="utf-8")
            model = validate_model(parse_model_text(source))
            payload = canonical_model_ir_payload(model)
            text = json.dumps(payload, sort_keys=True)
            checks.append(Check(f"{path.name}: canonical Model IR schema 3.0", payload["schema_version"] == "3.0"))
            checks.append(Check(f"{path.name}: no SymPy printer data persisted", "sympy_srepr" not in text and '"normalised"' not in text))
            expressions = _all_expression_payloads(payload)
            checks.append(
                Check(
                    f"{path.name}: all persisted expressions use versioned ML AST",
                    all(
                        item.get("schema") == EXPRESSION_AST_SCHEMA
                        and item.get("schema_version") == EXPRESSION_AST_SCHEMA_VERSION
                        for item in expressions
                    ),
                )
            )
        except Exception as exc:
            checks.append(Check(f"{path.name}: AST compilation", False, str(exc)))

    try:
        source = (ROOT / "models" / "quadratic.yaml").read_text(encoding="utf-8")
        model = validate_model(parse_model_text(source))
        function = model.functions[0]
        runtime_variant = replace(
            model,
            functions=(
                replace(
                    function,
                    expression=sp.Add(*function.expression.as_ordered_terms(), evaluate=False),
                ),
            ),
        )
        checks.append(
            Check(
                "canonical model fingerprint ignores runtime SymPy tree",
                canonical_model_ir_sha256(model) == canonical_model_ir_sha256(runtime_variant),
            )
        )
    except Exception as exc:
        checks.append(Check("canonical model fingerprint ignores runtime SymPy tree", False, str(exc)))

    try:
        source = (ROOT / "models" / "structured.yaml").read_text(encoding="utf-8")
        _, bundle = _small_bundle(source)
        members = _members(bundle)
        manifest = json.loads(members["manifest.json"])
        model_ir = json.loads(members["model_ir.json"])
        checks.append(Check("new .mlab format is 2.0", manifest["format_version"] == MLAB_FORMAT_VERSION == "2.0"))
        checks.append(Check("new .mlab model_ir contains no SymPy srepr", "sympy_srepr" not in members["model_ir.json"].decode("utf-8")))
        loaded = load_mlab_bundle(bundle)
        checks.append(Check("new .mlab AST bundle round-trip", canonical_model_ir_payload(loaded.model) == model_ir))
    except Exception as exc:
        checks.append(Check("new .mlab AST bundle", False, str(exc)))

    try:
        literal = "1.23456789012345678901234567890123456789"
        original_precision = getcontext().prec
        payloads = []
        try:
            for precision in (10, 28, 50):
                getcontext().prec = precision
                payloads.append(expression_ast_payload(parse_expression(literal, set())))
        finally:
            getcontext().prec = original_precision
        checks.append(Check("decimal context cannot change canonical AST", payloads[0] == payloads[1] == payloads[2]))
        checks.append(Check("long decimal retained exactly", payloads[0]["root"] == {
            "type": "real",
            "coefficient": "123456789012345678901234567890123456789",
            "exponent": -38,
        }))
        tiny = expression_ast_payload(parse_expression("1e-999999", set()))
        checks.append(Check("extreme exponent remains compact", tiny["root"] == {"type": "real", "coefficient": "1", "exponent": -999999} and len(json.dumps(tiny)) < 300))
        spellings = [expression_ast_payload(parse_expression(item, set())) for item in ("1.2300", "1.23", "1.23000e0")]
        checks.append(Check("equivalent decimal spellings canonicalise identically", spellings[0] == spellings[1] == spellings[2]))
        checks.append(Check("integer and real literal types remain distinct", expression_ast_payload(parse_expression("2", set())) != expression_ast_payload(parse_expression("2.0", set()))))
        checks.append(Check("negative real zero canonicalises to real zero", expression_ast_payload(parse_expression("-0.0", set())) == expression_ast_payload(parse_expression("0.0", set()))))
    except Exception as exc:
        checks.append(Check("compact decimal AST verification", False, str(exc)))

    try:
        previous = load_mlab_bundle(PREVIOUS_AST_FIXTURE.read_bytes())
        payload = canonical_model_ir_payload(previous.model)
        checks.append(Check("format-1.1 AST bundle migrates", previous.manifest["format_version"] == "1.1"))
        checks.append(Check("format-1.1 bundle reconstructs current Model IR 3.0", payload["schema_version"] == "3.0"))
        checks.append(Check("format-1.1 real literal becomes current compact AST 1.2", payload["functions"][0]["expression"]["schema_version"] == "1.2"))
        outcome = reproduce_mlab_bundle(previous)
        checks.append(Check("format-1.1 bundle remains reproducible", outcome.report.status in {ReproductionStatus.EXACT, ReproductionStatus.NUMERICAL}, outcome.report.status.value))
    except Exception as exc:
        checks.append(Check("format-1.1 AST bundle migration", False, str(exc)))

    try:
        archived = load_mlab_bundle(
            base64.b64decode(LEGACY_HYPERBOLIC_IDENTIFIER_FIXTURE.read_text(encoding="ascii"))
        )
        checks.append(
            Check(
                "format-1.4 historical sinh identifier remains a parameter",
                archived.manifest["format_version"] == "1.4"
                and "sinh" in archived.model.parameter_defaults()
                and archived.model.functions[0].dependencies == ("sinh", "x"),
            )
        )
    except Exception as exc:
        checks.append(Check("format-1.4 historical sinh identifier migration", False, str(exc)))

    try:
        members = _members(PREVIOUS_AST_FIXTURE.read_bytes())
        model_ir = json.loads(members["model_ir.json"])
        model_ir["functions"][0]["expression"]["root"]["right"]["value"] = "0.00012"
        members["model_ir.json"] = (
            json.dumps(model_ir, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        experiment = json.loads(members["experiment.json"])
        experiment["model"]["canonical_ir_sha256"] = canonical_json_sha256(model_ir)
        members["experiment.json"] = (
            json.dumps(experiment, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        _refresh_manifest(members)
        migrated = load_mlab_bundle(_rebuild(members))
        checks.append(Check("format-1.1 Decimal-context-rounded real literal does not block loading", migrated.model.name == "AST 1.0 migration fixture"))
    except Exception as exc:
        checks.append(Check("format-1.1 rounded-real migration", False, str(exc)))

    try:
        legacy_data = LEGACY_FIXTURE.read_bytes()
        legacy = load_mlab_bundle(legacy_data)
        checks.append(Check("real format-1.0 bundle migrates", legacy.manifest["format_version"] == "1.0"))
        checks.append(Check("legacy bundle reconstructs current Model IR", canonical_model_ir_payload(legacy.model)["schema_version"] == "3.0"))
        outcome = reproduce_mlab_bundle(legacy)
        checks.append(
            Check(
                "legacy bundle remains reproducible after migration",
                outcome.report.status in {ReproductionStatus.EXACT, ReproductionStatus.NUMERICAL},
                outcome.report.status.value,
            )
        )
    except Exception as exc:
        checks.append(Check("real format-1.0 bundle migration", False, str(exc)))

    try:
        members = _members(LEGACY_FIXTURE.read_bytes())
        model_ir = json.loads(members["model_ir.json"])
        model_ir["functions"][0]["expression"]["sympy_srepr"] = "DifferentFutureSymPyTree(...)"
        model_ir["functions"][0]["expression"]["normalised"] = "different future printer output"
        members["model_ir.json"] = (
            json.dumps(model_ir, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        experiment = json.loads(members["experiment.json"])
        experiment["model"]["canonical_ir_sha256"] = canonical_json_sha256(model_ir)
        members["experiment.json"] = (
            json.dumps(experiment, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        _refresh_manifest(members)
        load_mlab_bundle(_rebuild(members))
        checks.append(Check("legacy SymPy printer variation does not block loading", True))
    except Exception as exc:
        checks.append(Check("legacy SymPy printer variation does not block loading", False, str(exc)))

    try:
        members = _members(LEGACY_FIXTURE.read_bytes())
        model_ir = json.loads(members["model_ir.json"])
        model_ir["functions"][0]["dependencies"] = ["x"]
        members["model_ir.json"] = (
            json.dumps(model_ir, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        _refresh_manifest(members)
        try:
            load_mlab_bundle(_rebuild(members))
        except MlabBundleError:
            checks.append(Check("legacy Model Laboratory-owned IR tampering rejected", True))
        else:
            checks.append(Check("legacy Model Laboratory-owned IR tampering rejected", False))
    except Exception as exc:
        checks.append(Check("legacy Model Laboratory-owned IR tampering rejected", False, str(exc)))

    return checks


def main() -> int:
    checks = run()
    passed = sum(item.passed for item in checks)
    total = len(checks)
    print(f"Expression-AST verification: {passed}/{total} passed")
    for item in checks:
        if not item.passed:
            print(f"FAIL: {item.name}: {item.detail}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
