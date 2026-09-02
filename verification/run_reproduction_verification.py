"""Independent Phase-B verification for mathematical experiment reproduction semantics."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_lab.analysis import ParameterSweepResult, run_parameter_sweep, run_stationary_point_analysis
from model_lab.bundle import create_mlab_bundle, load_mlab_bundle
from model_lab.capabilities import AnalysisVisualisation, ModelVisualisation
from model_lab.critical_points import TwoDimensionalCriticalPoint
from model_lab.evaluator import FunctionEvaluation, SurfaceEvaluation, evaluate_single_variable_function, evaluate_two_variable_function
from model_lab.experiment import EvaluationSettings, NumericalReproductionSettings, StationaryPointSettings, SweepConfiguration, _compare_evaluation_reference, create_experiment_state, fingerprint_evaluation, numerical_reference_evaluation
from model_lab.issues import DiagnosticSeverity, NumericalDiagnostic
from model_lab.parser import parse_model_text
from model_lab.reproduction import EnvironmentComparisonStatus, ReproductionStatus, build_reproduction_report, reproduce_experiment, reproduce_mlab_bundle
from model_lab.provenance import Provenance
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


def _build(source: str, *, sweep_steps: int = 5):
    model = validate_model(parse_model_text(source))
    parameters = model.parameter_defaults()
    evaluation_settings = EvaluationSettings(points_1d=101, points_per_axis_2d=31)
    stationary_settings = StationaryPointSettings(samples_1d=801, seeds_per_axis_2d=7, root_tolerance=1e-9)
    if len(model.variables) == 1:
        evaluation = evaluate_single_variable_function(model, parameters, points=evaluation_settings.points_1d)
        visualisation = ModelVisualisation.TWO_D_FUNCTION_PLOT
    else:
        evaluation = evaluate_two_variable_function(model, parameters, points_per_axis=evaluation_settings.points_per_axis_2d)
        visualisation = ModelVisualisation.THREE_D_SURFACE
    stationary = run_stationary_point_analysis(
        model,
        parameters,
        samples_1d=stationary_settings.samples_1d,
        seeds_per_axis_2d=stationary_settings.seeds_per_axis_2d,
        root_tolerance=stationary_settings.root_tolerance,
    )
    sweep = None
    sweep_configuration = None
    if model.parameters:
        parameter = model.parameters[0]
        sweep = run_parameter_sweep(
            model,
            parameter.name,
            start=parameter.domain.lower,
            end=parameter.domain.upper,
            step_count=sweep_steps,
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
            step_count=sweep_steps,
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
    return state, model, evaluation, stationary, sweep


def _within_tolerance_evaluation(evaluation: FunctionEvaluation | SurfaceEvaluation):
    if isinstance(evaluation, FunctionEvaluation):
        values = evaluation.y.copy()
        candidates = np.flatnonzero(np.isfinite(values) & (np.abs(values) > 0.5))
        index = int(candidates[0])
        delta = min(1e-9, abs(values[index]) * 1e-9)
        values[index] += delta
        return replace(evaluation, y=values)
    values = evaluation.z.copy()
    candidates = np.argwhere(np.isfinite(values) & (np.abs(values) > 0.5))
    index = tuple(int(item) for item in candidates[0])
    delta = min(1e-9, abs(values[index]) * 1e-9)
    values[index] += delta
    return replace(evaluation, z=values)


def _outside_tolerance_evaluation(evaluation: FunctionEvaluation | SurfaceEvaluation):
    if isinstance(evaluation, FunctionEvaluation):
        values = evaluation.y.copy()
        index = int(np.flatnonzero(np.isfinite(values))[0])
        values[index] += 0.1
        return replace(evaluation, y=values)
    values = evaluation.z.copy()
    index = tuple(int(item) for item in np.argwhere(np.isfinite(values))[0])
    values[index] += 0.1
    return replace(evaluation, z=values)


def run() -> list[Check]:
    checks: list[Check] = []
    built: list[tuple] = []

    # Exact reproduction and bundle-to-reproduction path across every included model.
    for path in MODEL_PATHS:
        try:
            source = path.read_text(encoding="utf-8")
            state, model, evaluation, stationary, sweep = _build(source)
            built.append((state, model, evaluation, stationary, sweep))
            report = build_reproduction_report(
                state,
                model=model,
                evaluation=evaluation,
                stationary_result=stationary,
                sweep_result=sweep,
            )
            checks.append(Check(f"{path.name}: exact status", report.status is ReproductionStatus.EXACT))
            checks.append(Check(f"{path.name}: every result strict", all(item.status is ReproductionStatus.EXACT for item in report.results)))
            checks.append(
                Check(
                    f"{path.name}: environment captured",
                    report.environment_status
                    in (
                        EnvironmentComparisonStatus.IDENTICAL,
                        EnvironmentComparisonStatus.COMPATIBLE,
                        EnvironmentComparisonStatus.DIFFERENT,
                    ),
                )
            )
            bundle = load_mlab_bundle(create_mlab_bundle(
                state=state,
                model=model,
                evaluation=evaluation,
                stationary_result=stationary,
                sweep_result=sweep,
            ))
            outcome = reproduce_mlab_bundle(bundle)
            checks.append(Check(f"{path.name}: .mlab explicit reproduction", outcome.report.status is ReproductionStatus.EXACT and outcome.report.experiment_id == bundle.experiment_id))
            document = json.loads(outcome.report.to_json())
            checks.append(Check(f"{path.name}: JSON report schema", document.get("schema") == "model-laboratory-reproduction-report" and document.get("status") == "EXACT REPRODUCTION"))
            checks.append(Check(f"{path.name}: text report", "EXACT REPRODUCTION" in outcome.report.to_text()))
        except Exception as exc:
            checks.append(Check(f"{path.name}: Phase-B construction", False, str(exc)))

    if built:
        state, model, evaluation, stationary, sweep = built[0]

        # Environment identity must not be conflated with exact mathematical result identity.
        different_environment = replace(state, laboratory_version="other", environment=(("python", "other"),))
        report = build_reproduction_report(
            different_environment,
            model=model,
            evaluation=evaluation,
            stationary_result=stationary,
            sweep_result=sweep,
        )
        checks.append(Check("exact result survives different environment", report.status is ReproductionStatus.EXACT))
        checks.append(Check("different environment is explicit", report.environment_status is EnvironmentComparisonStatus.DIFFERENT and bool(report.environment_differences)))
        checks.append(Check("laboratory version difference is explicit", not report.laboratory_version_matches))

        # Genuine tolerance comparison: strict bytes differ, numerical reference still agrees.
        near = _within_tolerance_evaluation(evaluation)
        report = build_reproduction_report(
            state,
            model=model,
            evaluation=near,
            stationary_result=stationary,
            sweep_result=sweep,
        )
        eval_report = next(item for item in report.results if item.name == "evaluation")
        checks.append(Check("within-tolerance result is numerical reproduction", report.status is ReproductionStatus.NUMERICAL))
        checks.append(Check("numerical result differs strictly", not eval_report.strict_fingerprint_matches))
        checks.append(Check("numerical comparator accepts saved tolerance", eval_report.numerical_comparison_matches is True))
        checks.append(Check("numerical deviation is reported", eval_report.max_absolute_deviation is not None and eval_report.max_absolute_deviation > 0))

        far = _outside_tolerance_evaluation(evaluation)
        report = build_reproduction_report(
            state,
            model=model,
            evaluation=far,
            stationary_result=stationary,
            sweep_result=sweep,
        )
        eval_report = next(item for item in report.results if item.name == "evaluation")
        checks.append(Check("outside-tolerance result is rejected", eval_report.status is ReproductionStatus.NOT_REPRODUCED))
        checks.append(Check("mixed reproduced/non-reproduced result is partial", report.status is ReproductionStatus.PARTIAL))
        checks.append(Check("outside-tolerance deviation is reported", eval_report.max_absolute_deviation is not None and eval_report.max_absolute_deviation >= 0.09))

        # Strict-only symbolic evidence must not be bypassed by numerical results.
        altered_fingerprints = tuple(
            (name, "0" * 64 if name == "symbolic" else digest)
            for name, digest in state.result_fingerprints
        )
        symbolic_bad = replace(state, result_fingerprints=altered_fingerprints).with_checksum()
        report = build_reproduction_report(
            symbolic_bad,
            model=model,
            evaluation=evaluation,
            stationary_result=stationary,
            sweep_result=sweep,
        )
        symbolic_report = next(item for item in report.results if item.name == "symbolic")
        checks.append(Check("strict-only symbolic mismatch is rejected", symbolic_report.status is ReproductionStatus.NOT_REPRODUCED))
        checks.append(Check("strict-only mismatch prevents numerical overall status", report.status is ReproductionStatus.PARTIAL))

        # Sweep comparison is aligned by parameter value, not arbitrary saved list order.
        if sweep is not None:
            reordered_sweep = replace(sweep, steps=tuple(reversed(sweep.steps)))
            report = build_reproduction_report(
                state,
                model=model,
                evaluation=evaluation,
                stationary_result=stationary,
                sweep_result=reordered_sweep,
            )
            sweep_report = next(item for item in report.results if item.name == "parameter_sweep")
            checks.append(Check("reordered sweep loses strict fingerprint", not sweep_report.strict_fingerprint_matches))
            checks.append(Check("reordered sweep reproduces numerically", sweep_report.status is ReproductionStatus.NUMERICAL))

        # Explicit execution failure is distinct from a conflicting computed result.
        report = build_reproduction_report(
            state,
            model=model,
            evaluation=evaluation,
            stationary_result=None,
            sweep_result=sweep,
            execution_errors={"stationary_points": "verification solver unavailable"},
        )
        stationary_report = next(item for item in report.results if item.name == "stationary_points")
        checks.append(Check("per-result inability has UNABLE status", stationary_report.status is ReproductionStatus.UNABLE))
        checks.append(Check("mixed inability produces partial overall status", report.status is ReproductionStatus.PARTIAL))
        checks.append(Check("execution failure detail preserved", "verification solver unavailable" in stationary_report.details))

        # Workload refusal must produce UNABLE TO REPRODUCE without beginning evaluation.
        oversized = replace(
            state,
            stationary_settings=replace(state.stationary_settings, samples_1d=200001),
            sweep=replace(state.sweep, step_count=201) if state.sweep is not None else None,
        ).with_checksum()
        outcome = reproduce_experiment(oversized, model=model)
        checks.append(Check("over-budget reproduction is refused", outcome.report.status is ReproductionStatus.UNABLE))
        checks.append(Check("over-budget reproduction performs no evaluation", outcome.evaluation is None))
        checks.append(Check("over-budget report explains refusal", any("budget" in item.lower() for item in outcome.report.execution_errors)))

        # A fully conflicting result set should be NOT REPRODUCED rather than PARTIAL.
        far = _outside_tolerance_evaluation(evaluation)
        all_bad_fingerprints = tuple((name, "f" * 64) for name, _ in state.result_fingerprints)
        all_bad_state = replace(state, result_fingerprints=all_bad_fingerprints).with_checksum()
        report = build_reproduction_report(
            all_bad_state,
            model=model,
            evaluation=far,
            stationary_result=stationary,
            sweep_result=sweep,
        )
        # Stationary/sweep still have numerical references and may agree, so force their execution
        # evidence unavailable to test the no-success status deterministically.
        report = build_reproduction_report(
            all_bad_state,
            model=model,
            evaluation=far,
            stationary_result=None,
            sweep_result=None,
            execution_errors={
                "stationary_points": "verification unavailable",
                "parameter_sweep": "verification unavailable",
            } if sweep is not None else {"stationary_points": "verification unavailable"},
        )
        # Symbolic and evaluation are genuine non-reproductions, while unavailable numerical
        # results make the overall attempt unable rather than falsely claiming a contradiction.
        checks.append(Check("no successful result with execution failures is unable", report.status is ReproductionStatus.UNABLE))

    # Numerical diagnostic identity excludes message wording and canonicalises details.
    try:
        saved_diag = NumericalDiagnostic(
            code="nonfinite",
            severity=DiagnosticSeverity.WARNING,
            message="Old wording",
            details=(("a", "1"), ("b", "2")),
        )
        current_diag = NumericalDiagnostic(
            code="nonfinite",
            severity=DiagnosticSeverity.WARNING,
            message="New wording",
            details=(("b", "2"), ("a", "1")),
        )
        provenance = Provenance.derived(("verification:diagnostic",), "diagnostic verification")
        saved_eval = FunctionEvaluation(
            x_name="x", y_name="y", x=np.array([0.0]), y=np.array([1.0]),
            provenance=provenance, diagnostics=(saved_diag,),
        )
        current_eval = replace(saved_eval, diagnostics=(current_diag,))
        reference = numerical_reference_evaluation(saved_eval)
        comparison = _compare_evaluation_reference(reference, current_eval, NumericalReproductionSettings())
        checks.append(Check("diagnostic message wording is presentation-only", comparison.matches))
        checks.append(Check("strict diagnostic fingerprint excludes message wording", fingerprint_evaluation(saved_eval) == fingerprint_evaluation(current_eval)))
        checks.append(Check("diagnostic detail order is scientifically irrelevant", saved_diag.details == current_diag.details))
        checks.append(Check("saved diagnostic message remains preserved", reference["diagnostic_records"][0]["message"] == "Old wording"))

        different_eval = replace(
            saved_eval,
            diagnostics=(NumericalDiagnostic.warning("nonfinite", "Any wording", a=1, b=3),),
        )
        different = _compare_evaluation_reference(reference, different_eval, NumericalReproductionSettings())
        checks.append(Check("different scientific diagnostic details are rejected", not different.matches))
    except Exception as exc:
        checks.append(Check("scientific diagnostic identity", False, str(exc)))

    # Semantic stationary-point comparison: Hessian eigenvalue order is not mathematical identity.
    try:
        source = """name: Eigenvalue order
variables:
  x: {domain: [-2, 2]}
  y: {domain: [-2, 2]}
functions:
  z: x**2 + 2*y**2
"""
        state, model, evaluation, stationary, _ = _build(source)
        point = stationary.points[0]
        if not isinstance(point, TwoDimensionalCriticalPoint):
            raise TypeError("expected a two-dimensional stationary point")
        reversed_point = replace(point, eigenvalues=tuple(reversed(point.eigenvalues)))
        reversed_stationary = replace(stationary, points=(reversed_point,))
        report = build_reproduction_report(
            state,
            model=model,
            evaluation=evaluation,
            stationary_result=reversed_stationary,
            sweep_result=None,
        )
        point_report = next(item for item in report.results if item.name == "stationary_points")
        checks.append(Check("Hessian eigenvalue order is semantically irrelevant", point_report.status is ReproductionStatus.NUMERICAL))
    except Exception as exc:
        checks.append(Check("Hessian eigenvalue order is semantically irrelevant", False, str(exc)))

    return checks


def main() -> int:
    checks = run()
    passed = sum(item.passed for item in checks)
    total = len(checks)
    print(f"Phase-B reproduction verification: {passed}/{total} passed")
    for item in checks:
        if not item.passed:
            print(f"FAIL: {item.name}: {item.detail}")
    report = {
        "title": "Model Laboratory Phase-B reproduction verification",
        "passed": passed,
        "total": total,
        "checks": [item.__dict__ for item in checks],
    }
    (ROOT / "verification" / "phase_b_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = [
        "# Model Laboratory Phase-B reproduction verification",
        "",
        f"**Result: {passed}/{total} checks passed.**",
        "",
        "The checks exercise exact fingerprint reproduction, explicit numerical tolerance comparison, result-specific semantic comparison, status classification, environment separation, explicit execution refusal, `.mlab` reproduction and JSON/text reproduction reports.",
        "",
    ]
    for item in checks:
        marker = "PASS" if item.passed else "FAIL"
        line = f"- **{marker}** — {item.name}"
        if item.detail:
            line += f": {item.detail}"
        markdown.append(line)
    markdown.append("")
    (ROOT / "verification" / "PHASE_B_REPORT.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
