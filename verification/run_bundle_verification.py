"""Independent Phase-A verification for portable reproducible experiment bundles."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_lab.analysis import run_parameter_sweep, run_stationary_point_analysis
from model_lab.bundle import (
    MLAB_FORMAT_NAME,
    MLAB_FORMAT_VERSION,
    MlabBundleError,
    canonical_experiment_document,
    create_mlab_bundle,
    load_mlab_bundle,
)
from model_lab.canonical import canonical_model_ir_payload, canonical_model_ir_sha256
from model_lab.capabilities import AnalysisVisualisation, ModelVisualisation
from model_lab.evaluator import evaluate_single_variable_function, evaluate_two_variable_function
from model_lab.experiment import (
    EvaluationSettings,
    StationaryPointSettings,
    SweepConfiguration,
    compare_reproduced_results,
    create_experiment_state,
)
from model_lab.parser import parse_model_text
from model_lab.validator import validate_model


MODEL_PATHS = (
    ROOT / "models" / "quadratic.yaml",
    ROOT / "models" / "surface.yaml",
    ROOT / "models" / "structured.yaml",
    ROOT / "models" / "issues.yaml",
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def _experiment_for(path: Path):
    source = path.read_text(encoding="utf-8")
    model = validate_model(parse_model_text(source))
    parameters = model.parameter_defaults()
    evaluation_settings = EvaluationSettings(points_1d=101, points_per_axis_2d=31)
    stationary_settings = StationaryPointSettings(
        samples_1d=801,
        seeds_per_axis_2d=7,
        root_tolerance=1e-9,
    )
    if len(model.variables) == 1:
        evaluation = evaluate_single_variable_function(
            model, parameters, points=evaluation_settings.points_1d
        )
        visualisation = ModelVisualisation.TWO_D_FUNCTION_PLOT
    else:
        evaluation = evaluate_two_variable_function(
            model, parameters, points_per_axis=evaluation_settings.points_per_axis_2d
        )
        visualisation = ModelVisualisation.THREE_D_SURFACE

    stationary = run_stationary_point_analysis(
        model,
        parameters,
        samples_1d=stationary_settings.samples_1d,
        seeds_per_axis_2d=stationary_settings.seeds_per_axis_2d,
        root_tolerance=stationary_settings.root_tolerance,
    )

    sweep_configuration = None
    sweep = None
    if model.parameters:
        parameter = model.parameters[0]
        sweep = run_parameter_sweep(
            model,
            parameter.name,
            start=parameter.domain.lower,
            end=parameter.domain.upper,
            step_count=5,
            fixed_parameter_values={
                item.name: parameters[item.name]
                for item in model.parameters
                if item.name != parameter.name
            },
            samples_1d=stationary_settings.samples_1d,
            seeds_per_axis_2d=stationary_settings.seeds_per_axis_2d,
            root_tolerance=stationary_settings.root_tolerance,
        )
        sweep_configuration = SweepConfiguration(
            parameter_name=parameter.name,
            start=parameter.domain.lower,
            end=parameter.domain.upper,
            step_count=5,
            selected_visualisation=AnalysisVisualisation.SWEEP_CLASSIFICATION_COUNTS.value,
        )

    state = create_experiment_state(
        model_source=source,
        model=model,
        parameter_values=parameters,
        selected_model_visualisation=visualisation,
        evaluation=evaluation,
        stationary_result=stationary,
        evaluation_settings=evaluation_settings,
        stationary_settings=stationary_settings,
        sweep_configuration=sweep_configuration,
        sweep_result=sweep,
    )
    bundle = create_mlab_bundle(
        state=state,
        model=model,
        evaluation=evaluation,
        stationary_result=stationary,
        sweep_result=sweep,
    )
    return source, model, parameters, evaluation, stationary, sweep, state, bundle


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


def run() -> list[Check]:
    checks: list[Check] = []
    built: list[tuple] = []

    for path in MODEL_PATHS:
        try:
            values = _experiment_for(path)
            built.append(values)
            source, model, _, evaluation, stationary, sweep, state, bundle = values
            loaded = load_mlab_bundle(bundle)
            checks.append(Check(f"{path.name}: bundle loads", True))
            checks.append(
                Check(
                    f"{path.name}: exact state round-trip",
                    loaded.state.payload_dict() == state.payload_dict()
                    and loaded.state.state_sha256 == state.state_sha256,
                )
            )
            checks.append(
                Check(
                    f"{path.name}: canonical Model IR round-trip",
                    canonical_model_ir_payload(loaded.model) == canonical_model_ir_payload(model),
                )
            )
            checks.append(
                Check(
                    f"{path.name}: canonical experiment round-trip",
                    loaded.experiment_document == canonical_experiment_document(state, model),
                )
            )
            checks.append(
                Check(
                    f"{path.name}: model source exact",
                    loaded.state.model_source == source,
                )
            )
            reproduction = compare_reproduced_results(
                loaded.state,
                model=loaded.model,
                evaluation=evaluation,
                stationary_result=stationary,
                sweep_result=sweep,
            )
            checks.append(
                Check(
                    f"{path.name}: stored references reproduce",
                    reproduction.all_results_match and reproduction.all_numerical_results_match,
                )
            )
            strict_names = {name for name, _ in loaded.state.result_fingerprints}
            reference_names = {name for name, _ in loaded.state.numerical_reference_data}
            checks.append(
                Check(
                    f"{path.name}: reference class coverage",
                    {"evaluation", "stationary_points", "parameter_sweep"}.issubset(reference_names)
                    and "symbolic" in strict_names,
                )
            )
            checks.append(
                Check(
                    f"{path.name}: manifest/model identity",
                    loaded.manifest["format"] == MLAB_FORMAT_NAME
                    and loaded.manifest["format_version"] == MLAB_FORMAT_VERSION
                    and loaded.experiment_document["model"]["canonical_ir_sha256"]
                    == canonical_model_ir_sha256(model),
                )
            )
        except Exception as exc:  # pragma: no cover - verification runner reports detail
            checks.append(Check(f"{path.name}: construction", False, str(exc)))

    if built:
        source, model, _, evaluation, stationary, sweep, state, bundle = built[0]
        try:
            repeated = create_mlab_bundle(
                state=state,
                model=model,
                evaluation=evaluation,
                stationary_result=stationary,
                sweep_result=sweep,
            )
            checks.append(Check("deterministic bundle bytes", repeated == bundle))
        except Exception as exc:
            checks.append(Check("deterministic bundle bytes", False, str(exc)))

        members = _members(bundle)
        manifest = json.loads(members["manifest.json"])
        described = {item["path"]: item for item in manifest["members"]}
        manifest_ok = set(described) == set(members) - {"manifest.json"}
        if manifest_ok:
            for path, entry in described.items():
                manifest_ok &= entry["size"] == len(members[path])
                manifest_ok &= entry["sha256"] == hashlib.sha256(members[path]).hexdigest()
        checks.append(Check("manifest checksums and sizes", bool(manifest_ok)))

        tampered = dict(members)
        tampered["model.yaml"] += b"\n# tampered\n"
        try:
            load_mlab_bundle(_rebuild(tampered))
        except MlabBundleError:
            checks.append(Check("tampered member rejected", True))
        else:
            checks.append(Check("tampered member rejected", False))

        extras = dict(members)
        extras["extra.txt"] = b"unexpected"
        try:
            load_mlab_bundle(_rebuild(extras))
        except MlabBundleError:
            checks.append(Check("unexpected member rejected", True))
        else:
            checks.append(Check("unexpected member rejected", False))

        references = json.loads(members["results/references.json"])
        policy = references.get("storage_policy", {})
        checks.append(
            Check(
                "reference storage policy complete",
                set(policy) == {"evaluation", "stationary_points", "parameter_sweep", "symbolic"},
            )
        )

    return checks


def main() -> int:
    checks = run()
    passed = sum(item.passed for item in checks)
    total = len(checks)
    print(f"Phase-A bundle verification: {passed}/{total} passed")
    for item in checks:
        if not item.passed:
            print(f"FAIL: {item.name}: {item.detail}")
    report = {
        "title": "Model Laboratory Phase-A bundle verification",
        "passed": passed,
        "total": total,
        "checks": [item.__dict__ for item in checks],
    }
    (ROOT / "verification" / "phase_a_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = [
        "# Model Laboratory Phase-A bundle verification",
        "",
        f"**Result: {passed}/{total} checks passed.**",
        "",
        "The checks exercise portable `.mlab` creation, canonical round-tripping, reference-result storage, integrity validation and reconstruction across every included example model.",
        "",
    ]
    for item in checks:
        marker = "PASS" if item.passed else "FAIL"
        line = f"- **{marker}** — {item.name}"
        if item.detail:
            line += f": {item.detail}"
        markdown.append(line)
    markdown.append("")
    (ROOT / "verification" / "PHASE_A_REPORT.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
