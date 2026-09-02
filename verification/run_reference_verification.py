"""Extensive independent reference verification for Model Laboratory.

Run from the project root with:

    python verification/run_reference_verification.py

The script compares deterministic laboratory outputs with analytically known results for
many scalar models.  It writes a Markdown and JSON report and exits non-zero if any
reference check fails.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import pi, sqrt
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_lab import __version__
from model_lab.analysis import run_parameter_sweep, run_stationary_point_analysis
from model_lab.capabilities import AnalysisVisualisation, ModelVisualisation
from model_lab.critical_points import search_one_variable_stationary_points, search_two_variable_stationary_points
from model_lab.evaluator import evaluate_single_variable_function, evaluate_two_variable_function
from model_lab.experiment import (
    EvaluationSettings,
    ExperimentState,
    StationaryPointSettings,
    SweepConfiguration,
    compare_reproduced_results,
    create_experiment_state,
)
from model_lab.issues import NumericalStatus
from model_lab.parser import parse_model_text
from model_lab.symbolic import analyse_one_variable_function, analyse_two_variable_function
from model_lab.validator import validate_model


@dataclass
class CheckResult:
    name: str
    category: str
    passed: bool
    detail: str


results: list[CheckResult] = []


def compile_model(text: str):
    return validate_model(parse_model_text(text))


def record(name: str, category: str, check: Callable[[], None]) -> None:
    try:
        check()
    except Exception as exc:  # verification deliberately records every failure
        results.append(CheckResult(name, category, False, f"{type(exc).__name__}: {exc}"))
    else:
        results.append(CheckResult(name, category, True, "matched reference result"))


def assert_close(actual: float, expected: float, tol: float = 1e-6) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"expected {expected:.12g}, got {actual:.12g}")


def assert_1d_points(model_text: str, expected: list[tuple[float, str]], *, status=NumericalStatus.COMPLETE) -> None:
    model = compile_model(model_text)
    found = search_one_variable_stationary_points(model, samples=4001)
    if found.status is not status:
        raise AssertionError(f"expected status {status.value}, got {found.status.value}")
    if len(found.points) != len(expected):
        raise AssertionError(f"expected {len(expected)} points, got {len(found.points)}")
    actual = sorted(found.points, key=lambda point: point.x)
    reference = sorted(expected, key=lambda item: item[0])
    for point, (x, classification) in zip(actual, reference):
        assert_close(point.x, x, 2e-6)
        if point.classification != classification:
            raise AssertionError(f"at x={x}, expected {classification}, got {point.classification}")


def assert_2d_points(model_text: str, expected: list[tuple[float, float, str]], *, status=NumericalStatus.COMPLETE, seeds=15) -> None:
    model = compile_model(model_text)
    found = search_two_variable_stationary_points(model, seeds_per_axis=seeds)
    if found.status is not status:
        raise AssertionError(f"expected status {status.value}, got {found.status.value}")
    if len(found.points) != len(expected):
        raise AssertionError(f"expected {len(expected)} points, got {len(found.points)}")
    unmatched = list(found.points)
    for x, y, classification in expected:
        candidates = [
            point for point in unmatched
            if abs(point.x - x) <= 5e-5 and abs(point.y - y) <= 5e-5
            and point.classification == classification
        ]
        if not candidates:
            raise AssertionError(f"missing {classification} near ({x}, {y})")
        unmatched.remove(candidates[0])


# ---- One-dimensional stationary-point references ---------------------------------

one_d_cases = [
    ("parabola", "x**2", [(0.0, "local minimum")]),
    ("inverted parabola", "-x**2", [(0.0, "local maximum")]),
    ("shifted quadratic", "(x-1.2345)**2", [(1.2345, "local minimum")]),
    ("cubic turning points", "x**3-3*x", [(-1.0, "local maximum"), (1.0, "local minimum")]),
    ("double-well quartic", "(x**2-1)**2", [(-1.0, "local minimum"), (0.0, "local maximum"), (1.0, "local minimum")]),
    ("flat quartic", "(x-0.3333)**4", [(0.3333, "degenerate / inconclusive")]),
    ("sixth-order polynomial", "x**6-x**4", [(-sqrt(2/3), "local minimum"), (0.0, "degenerate / inconclusive"), (sqrt(2/3), "local minimum")]),
    ("monotone exponential", "exp(x)", []),
    ("bounded rational", "x**2/(1+x**2)", [(0.0, "local minimum")]),
]
for case_name, expression, expected in one_d_cases:
    record(
        case_name,
        "1D stationary points",
        lambda expression=expression, expected=expected, case_name=case_name: assert_1d_points(
            f"""name: {case_name}\nvariables:\n  x:\n    domain: [-3, 3]\nfunctions:\n  y: {expression}\n""",
            expected,
        ),
    )

record(
    "sine over two periods",
    "1D stationary points",
    lambda: assert_1d_points(
        f"""name: sine\nvariables:\n  x:\n    domain: [{-2*pi}, {2*pi}]\nfunctions:\n  y: sin(x)\n""",
        [
            (-3*pi/2, "local maximum"),
            (-pi/2, "local minimum"),
            (pi/2, "local maximum"),
            (3*pi/2, "local minimum"),
        ],
    ),
)
record(
    "cosine including boundary stationary points",
    "1D stationary points",
    lambda: assert_1d_points(
        f"""name: cosine\nvariables:\n  x:\n    domain: [{-2*pi}, {2*pi}]\nfunctions:\n  y: cos(x)\n""",
        [
            (-2*pi, "local maximum"),
            (-pi, "local minimum"),
            (0.0, "local maximum"),
            (pi, "local minimum"),
            (2*pi, "local maximum"),
        ],
    ),
)


def check_absolute_value_cusp() -> None:
    model = compile_model("""name: cusp\nvariables:\n  x: {domain: [-1, 1]}\nfunctions:\n  y: abs(x)\n""")
    result = search_one_variable_stationary_points(model, samples=1001)
    if result.status is not NumericalStatus.PARTIAL or result.points:
        raise AssertionError("absolute-value cusp should be excluded as non-differentiable")
    if not any(item.code == "non_differentiable_candidates" for item in result.diagnostics):
        raise AssertionError("missing non-differentiability diagnostic")

record("absolute-value cusp", "1D pathological", check_absolute_value_cusp)


def check_parameterised_quadratic() -> None:
    model = compile_model("""name: parameterised quadratic\nvariables:\n  x: {domain: [-2, 2]}\nparameters:\n  a: {default: 1, domain: [-2, 2]}\nfunctions:\n  y: a*x**2\n""")
    positive = run_stationary_point_analysis(model, {"a": 2.0})
    negative = run_stationary_point_analysis(model, {"a": -2.0})
    zero = run_stationary_point_analysis(model, {"a": 0.0})
    if positive.points[0].classification != "local minimum":
        raise AssertionError("a>0 should give a local minimum")
    if negative.points[0].classification != "local maximum":
        raise AssertionError("a<0 should give a local maximum")
    if zero.numerical_status is not NumericalStatus.INDETERMINATE:
        raise AssertionError("a=0 should give a stationary continuum")

record("parameterised quadratic sign change", "1D parameter behaviour", check_parameterised_quadratic)


# ---- Two-dimensional stationary-point references ---------------------------------

two_d_cases = [
    ("circular bowl", "x**2+y**2", [(0, 0, "local minimum")]),
    ("inverted circular bowl", "-(x**2+y**2)", [(0, 0, "local maximum")]),
    ("hyperbolic saddle", "x**2-y**2", [(0, 0, "saddle")]),
    ("shifted bowl", "(x-1.2)**2+(y+0.7)**2", [(1.2, -0.7, "local minimum")]),
    ("two-dimensional double well", "(x**2-1)**2+y**2", [(-1, 0, "local minimum"), (0, 0, "saddle"), (1, 0, "local minimum")]),
    ("Rosenbrock", "(1-x)**2+100*(y-x**2)**2", [(1, 1, "local minimum")]),
    ("Booth", "(x+2*y-7)**2+(2*x+y-5)**2", [(1, 3, "local minimum")]),
    ("Matyas", "0.26*(x**2+y**2)-0.48*x*y", [(0, 0, "local minimum")]),
]
for case_name, expression, expected in two_d_cases:
    record(
        case_name,
        "2D stationary points",
        lambda expression=expression, expected=expected, case_name=case_name: assert_2d_points(
            f"""name: {case_name}\nvariables:\n  x: {{domain: [-5, 5]}}\n  y: {{domain: [-5, 5]}}\nfunctions:\n  z: {expression}\n""",
            expected,
        ),
    )

four_well_expected = [
    (-1, -1, "local minimum"), (-1, 0, "saddle"), (-1, 1, "local minimum"),
    (0, -1, "saddle"), (0, 0, "local maximum"), (0, 1, "saddle"),
    (1, -1, "local minimum"), (1, 0, "saddle"), (1, 1, "local minimum"),
]
record(
    "four-well surface",
    "2D stationary points",
    lambda: assert_2d_points(
        """name: four well\nvariables:\n  x: {domain: [-2, 2]}\n  y: {domain: [-2, 2]}\nfunctions:\n  z: (x**2-1)**2+(y**2-1)**2\n""",
        four_well_expected,
    ),
)

himmelblau_expected = [
    (-3.779310, -3.283186, "local minimum"),
    (-3.073026, -0.081353, "saddle"),
    (-2.805118, 3.131313, "local minimum"),
    (-0.270845, -0.923039, "local maximum"),
    (-0.127961, -1.953715, "saddle"),
    (0.086678, 2.884255, "saddle"),
    (3.0, 2.0, "local minimum"),
    (3.385154, 0.073852, "saddle"),
    (3.584428, -1.848127, "local minimum"),
]
record(
    "Himmelblau nine critical points",
    "2D stationary points",
    lambda: assert_2d_points(
        """name: Himmelblau\nvariables:\n  x: {domain: [-5, 5]}\n  y: {domain: [-5, 5]}\nfunctions:\n  z: (x**2+y-11)**2+(x+y**2-7)**2\n""",
        himmelblau_expected,
        seeds=15,
    ),
)


def check_nonisolated_rank_one() -> None:
    model = compile_model("""name: rank one\nvariables:\n  x: {domain: [-2, 2]}\n  y: {domain: [-2, 2]}\nfunctions:\n  z: (x+y)**2\n""")
    result = search_two_variable_stationary_points(model, seeds_per_axis=11)
    if result.status is not NumericalStatus.INDETERMINATE or result.points:
        raise AssertionError("stationary line x+y=0 must be reported as non-isolated")
    if not any(item.code == "non_isolated_stationary_structure" for item in result.diagnostics):
        raise AssertionError("missing non-isolated stationary-set diagnostic")

record("rank-deficient stationary line", "2D pathological", check_nonisolated_rank_one)


def check_two_dimensional_cusp() -> None:
    model = compile_model("""name: cusp 2D\nvariables:\n  x: {domain: [-1, 1]}\n  y: {domain: [-1, 1]}\nfunctions:\n  z: abs(x)+y**2\n""")
    result = search_two_variable_stationary_points(model, seeds_per_axis=9)
    if result.status is not NumericalStatus.PARTIAL or result.points:
        raise AssertionError("non-differentiable 2D cusp must not be returned as stationary")

record("two-dimensional cusp", "2D pathological", check_two_dimensional_cusp)


def check_linear_no_stationary_points() -> None:
    model = compile_model("""name: linear 2D\nvariables:\n  x: {domain: [-2, 2]}\n  y: {domain: [-2, 2]}\nfunctions:\n  z: x+y\n""")
    result = search_two_variable_stationary_points(model)
    if result.status is not NumericalStatus.COMPLETE or result.points:
        raise AssertionError("linear plane should have no stationary points")

record("linear plane", "2D pathological", check_linear_no_stationary_points)


# ---- Direct numerical evaluation references ---------------------------------------

def check_linear_grid() -> None:
    model = compile_model("""name: linear grid\nvariables:\n  x: {domain: [-2, 2]}\nfunctions:\n  y: 2*x+1\n""")
    result = evaluate_single_variable_function(model, points=5)
    np.testing.assert_allclose(result.x, [-2, -1, 0, 1, 2], atol=0, rtol=0)
    np.testing.assert_allclose(result.y, [-3, -1, 1, 3, 5], atol=1e-14, rtol=0)

record("exact linear grid evaluation", "numerical evaluation", check_linear_grid)


def check_parameter_grid() -> None:
    model = compile_model("""name: parameter grid\nvariables:\n  x: {domain: [-1, 1]}\nparameters:\n  a: {default: 2, domain: [-5, 5]}\n  b: {default: 1, domain: [-5, 5]}\nfunctions:\n  y: a*x+b\n""")
    result = evaluate_single_variable_function(model, {"a": 3, "b": -2}, points=3)
    np.testing.assert_allclose(result.y, [-5, -2, 1], atol=1e-14, rtol=0)

record("parameter override evaluation", "numerical evaluation", check_parameter_grid)


def check_constant_broadcast() -> None:
    model = compile_model("""name: constant broadcast\nvariables:\n  x: {domain: [-1, 1]}\nconstants:\n  c: {value: 7}\nfunctions:\n  y: c\n""")
    result = evaluate_single_variable_function(model, points=17)
    np.testing.assert_allclose(result.y, np.full(17, 7.0), atol=0, rtol=0)

record("constant scalar broadcast", "numerical evaluation", check_constant_broadcast)


def check_surface_grid() -> None:
    model = compile_model("""name: plane grid\nvariables:\n  x: {domain: [-1, 1]}\n  y: {domain: [-1, 1]}\nfunctions:\n  z: x+2*y\n""")
    result = evaluate_two_variable_function(model, points_per_axis=3)
    expected = np.array([[-3, -2, -1], [-1, 0, 1], [1, 2, 3]], dtype=float)
    np.testing.assert_allclose(result.z, expected, atol=1e-14, rtol=0)

record("exact 2D plane grid evaluation", "numerical evaluation", check_surface_grid)


def check_derived_quantity_evaluation() -> None:
    model = compile_model("""name: derived evaluation\nvariables:\n  x: {domain: [-1, 1]}\nparameters:\n  a: {default: 2, domain: [0, 4]}\nconstants:\n  c: {value: 3}\nderived_quantities:\n  q: {expression: a*x+c}\nfunctions:\n  y: q**2\n""")
    result = evaluate_single_variable_function(model, {"a": 2}, points=3)
    np.testing.assert_allclose(result.y, [1, 9, 25], atol=1e-14, rtol=0)

record("derived quantity and constant evaluation", "numerical evaluation", check_derived_quantity_evaluation)


def check_partial_log_evaluation() -> None:
    model = compile_model("""name: partial log\nvariables:\n  x: {domain: [-1, 1]}\nfunctions:\n  y: log(x)\n""")
    result = evaluate_single_variable_function(model, points=101)
    if result.numerical_status is not NumericalStatus.PARTIAL:
        raise AssertionError("log over [-1,1] should be a partial evaluation")
    if np.count_nonzero(np.isnan(result.y)) != 51:
        raise AssertionError("unexpected count of non-finite log samples")

record("partially undefined logarithm", "numerical evaluation", check_partial_log_evaluation)


# ---- Symbolic references -----------------------------------------------------------

def check_symbolic_polynomial() -> None:
    model = compile_model("""name: symbolic polynomial\nvariables:\n  x: {domain: [-2, 2]}\nfunctions:\n  y: x**4-3*x**2+2*x\n""")
    result = analyse_one_variable_function(model)
    x = sp.Symbol("x", real=True)
    if sp.simplify(result.first_derivative - (4*x**3-6*x+2)) != 0:
        raise AssertionError("first derivative mismatch")
    if sp.simplify(result.second_derivative - (12*x**2-6)) != 0:
        raise AssertionError("second derivative mismatch")

record("polynomial symbolic derivatives", "symbolic analysis", check_symbolic_polynomial)


def check_symbolic_trigonometric() -> None:
    model = compile_model("""name: symbolic trig\nvariables:\n  x: {domain: [-2, 2]}\nfunctions:\n  y: sin(x)*exp(x)\n""")
    result = analyse_one_variable_function(model)
    x = sp.Symbol("x", real=True)
    expected = sp.exp(x)*(sp.sin(x)+sp.cos(x))
    if sp.simplify(result.first_derivative - expected) != 0:
        raise AssertionError("trigonometric derivative mismatch")

record("trigonometric symbolic derivative", "symbolic analysis", check_symbolic_trigonometric)


def check_gradient_hessian() -> None:
    model = compile_model("""name: gradient hessian\nvariables:\n  x: {domain: [-2, 2]}\n  y: {domain: [-2, 2]}\nfunctions:\n  z: x**2+3*x*y+2*y**2\n""")
    result = analyse_two_variable_function(model)
    x, y = sp.symbols("x y", real=True)
    expected_gradient = (2*x+3*y, 3*x+4*y)
    expected_hessian = sp.Matrix([[2, 3], [3, 4]])
    if any(sp.simplify(a-b) != 0 for a, b in zip(result.gradient, expected_gradient)):
        raise AssertionError("gradient mismatch")
    if result.hessian != expected_hessian:
        raise AssertionError("Hessian mismatch")

record("2D gradient and Hessian", "symbolic analysis", check_gradient_hessian)


# ---- Parameter-sweep references ----------------------------------------------------

def check_quadratic_sweep_positions() -> None:
    model = compile_model("""name: sweep positions\nvariables:\n  x: {domain: [-3, 3]}\nparameters:\n  a: {default: 0, domain: [-2, 2]}\nfunctions:\n  y: x**2+a*x\n""")
    result = run_parameter_sweep(model, "a", start=-2, end=2, step_count=5)
    expected_a = [-2, -1, 0, 1, 2]
    expected_x = [1, 0.5, 0, -0.5, -1]
    for step, a, x in zip(result.steps, expected_a, expected_x):
        assert_close(step.parameter_value, a, 1e-12)
        if len(step.points) != 1:
            raise AssertionError("quadratic sweep should have one stationary point per step")
        assert_close(step.points[0].x, x, 1e-6)

record("quadratic stationary-position sweep", "parameter sweep", check_quadratic_sweep_positions)


def check_bifurcation_sweep() -> None:
    model = compile_model("""name: quartic sweep\nvariables:\n  x: {domain: [-3, 3]}\nparameters:\n  a: {default: 0, domain: [-2, 2]}\nfunctions:\n  y: x**4+a*x**2\n""")
    result = run_parameter_sweep(model, "a", start=-2, end=2, step_count=3)
    left, centre, right = result.steps
    if sorted(point.classification for point in left.points) != ["local maximum", "local minimum", "local minimum"]:
        raise AssertionError("a=-2 stationary classifications are wrong")
    if len(centre.points) != 1 or centre.points[0].classification != "degenerate / inconclusive":
        raise AssertionError("a=0 should have one degenerate stationary point")
    if len(right.points) != 1 or right.points[0].classification != "local minimum":
        raise AssertionError("a=2 should have one minimum")

record("quartic structural-transition sweep", "parameter sweep", check_bifurcation_sweep)


# ---- Experiment-state reproducibility ---------------------------------------------

def check_experiment_roundtrip() -> None:
    source = """# source comment must survive\nname: experiment reference\nvariables:\n  x: {domain: [-2, 2]}\nparameters:\n  a: {default: 1, domain: [-2, 2]}\nfunctions:\n  y: (x-a)**2\n"""
    model = compile_model(source)
    params = {"a": 0.75}
    evaluation_settings = EvaluationSettings(points_1d=257)
    stationary_settings = StationaryPointSettings(samples_1d=1001)
    evaluation = evaluate_single_variable_function(model, params, points=257)
    stationary = run_stationary_point_analysis(model, params, samples_1d=1001)
    sweep = run_parameter_sweep(model, "a", start=-1, end=1, step_count=5, samples_1d=1001)
    state = create_experiment_state(
        model_source=source,
        model=model,
        parameter_values=params,
        selected_model_visualisation=ModelVisualisation.TWO_D_FUNCTION_PLOT,
        evaluation=evaluation,
        stationary_result=stationary,
        evaluation_settings=evaluation_settings,
        stationary_settings=stationary_settings,
        sweep_configuration=SweepConfiguration(
            "a", -1, 1, 5, AnalysisVisualisation.SWEEP_POSITIONS.value
        ),
        sweep_result=sweep,
    )
    restored = ExperimentState.from_json(state.to_json())
    if restored.model_source != source:
        raise AssertionError("exact model source was not preserved")
    check = compare_reproduced_results(
        restored,
        model=model,
        evaluation=evaluation,
        stationary_result=stationary,
        sweep_result=sweep,
    )
    if not check.all_results_match:
        raise AssertionError("replayed result fingerprints do not match")

record("experiment-state round trip", "reproducibility", check_experiment_roundtrip)



# ---- Additional robustness references ---------------------------------------------

record(
    "shifted cubic flat stationary point",
    "1D stationary points",
    lambda: assert_1d_points(
        """name: shifted cubic\nvariables:\n  x: {domain: [-2, 2]}\nfunctions:\n  y: (x-0.4)**3\n""",
        [(0.4, "degenerate / inconclusive")],
    ),
)


def check_positive_log_no_stationary() -> None:
    model = compile_model("""name: positive log\nvariables:\n  x: {domain: [0.1, 5]}\nfunctions:\n  y: log(x)\n""")
    result = search_one_variable_stationary_points(model, samples=2001)
    if result.status is not NumericalStatus.COMPLETE or result.points:
        raise AssertionError("log(x) on a positive domain should have no stationary points")

record("positive logarithm", "1D stationary points", check_positive_log_no_stationary)


def check_square_root_boundary_singularity() -> None:
    model = compile_model("""name: square root\nvariables:\n  x: {domain: [0, 4]}\nfunctions:\n  y: sqrt(x)\n""")
    result = search_one_variable_stationary_points(model, samples=2001)
    if result.status is not NumericalStatus.PARTIAL or result.points:
        raise AssertionError("sqrt(x) should report a boundary derivative singularity without stationary points")
    if not any(item.code == "non_finite_derivative_samples" for item in result.diagnostics):
        raise AssertionError("missing non-finite derivative diagnostic")

record("square-root boundary singularity", "1D pathological", check_square_root_boundary_singularity)

record(
    "cross-coupled positive-definite quadratic",
    "2D stationary points",
    lambda: assert_2d_points(
        """name: cross quadratic\nvariables:\n  x: {domain: [-2, 2]}\n  y: {domain: [-2, 2]}\nfunctions:\n  z: 3*x**2+2*x*y+4*y**2\n""",
        [(0, 0, "local minimum")],
    ),
)

record(
    "two-dimensional flat quartic",
    "2D stationary points",
    lambda: assert_2d_points(
        """name: quartic 2D\nvariables:\n  x: {domain: [-2, 2]}\n  y: {domain: [-2, 2]}\nfunctions:\n  z: x**4+y**4\n""",
        [(0, 0, "degenerate / inconclusive")],
    ),
)


def check_rotated_stationary_line() -> None:
    model = compile_model("""name: rotated stationary line\nvariables:\n  x: {domain: [-2, 2]}\n  y: {domain: [-2, 2]}\nfunctions:\n  z: (2*x-y)**2\n""")
    result = search_two_variable_stationary_points(model, seeds_per_axis=11)
    if result.status is not NumericalStatus.INDETERMINATE or result.points:
        raise AssertionError("2x-y=0 stationary line must be reported as non-isolated")

record("rotated rank-deficient stationary line", "2D pathological", check_rotated_stationary_line)


def check_constant_surface_stationary_continuum() -> None:
    model = compile_model("""name: constant surface\nvariables:\n  x: {domain: [-2, 2]}\n  y: {domain: [-2, 2]}\nfunctions:\n  z: "1"\n""")
    result = search_two_variable_stationary_points(model)
    if result.status is not NumericalStatus.INDETERMINATE or result.points:
        raise AssertionError("constant surface should be an indeterminate stationary continuum")
    if not any(item.code == "stationary_continuum" for item in result.diagnostics):
        raise AssertionError("missing stationary-continuum diagnostic")

record("constant surface", "2D pathological", check_constant_surface_stationary_continuum)


def check_parameterised_two_dimensional_curvature() -> None:
    model = compile_model("""name: parameterised 2D\nvariables:\n  x: {domain: [-2, 2]}\n  y: {domain: [-2, 2]}\nparameters:\n  a: {default: 1, domain: [-1, 1]}\nfunctions:\n  z: a*x**2+y**2\n""")
    positive = run_stationary_point_analysis(model, {"a": 1})
    negative = run_stationary_point_analysis(model, {"a": -1})
    zero = run_stationary_point_analysis(model, {"a": 0})
    if positive.points[0].classification != "local minimum":
        raise AssertionError("a=1 should produce a local minimum")
    if negative.points[0].classification != "saddle":
        raise AssertionError("a=-1 should produce a saddle")
    if zero.numerical_status is not NumericalStatus.INDETERMINATE:
        raise AssertionError("a=0 should produce a non-isolated stationary set")

record("parameterised 2D curvature change", "2D parameter behaviour", check_parameterised_two_dimensional_curvature)


def check_quadratic_grid_exact_values() -> None:
    model = compile_model("""name: exact quadratic grid\nvariables:\n  x: {domain: [-2, 2]}\nfunctions:\n  y: x**2-2*x+3\n""")
    result = evaluate_single_variable_function(model, points=5)
    np.testing.assert_allclose(result.y, [11, 6, 3, 2, 3], atol=1e-14, rtol=0)

record("exact quadratic grid evaluation", "numerical evaluation", check_quadratic_grid_exact_values)


def check_surface_parameter_override() -> None:
    model = compile_model("""name: surface parameter\nvariables:\n  x: {domain: [-1, 1]}\n  y: {domain: [-1, 1]}\nparameters:\n  a: {default: 1, domain: [-3, 3]}\nfunctions:\n  z: a*x-y\n""")
    result = evaluate_two_variable_function(model, {"a": 2}, points_per_axis=3)
    expected = np.array([[-1, 1, 3], [-2, 0, 2], [-3, -1, 1]], dtype=float)
    np.testing.assert_allclose(result.z, expected, atol=1e-14, rtol=0)

record("2D parameter override evaluation", "numerical evaluation", check_surface_parameter_override)


def check_symbolic_derived_chain() -> None:
    model = compile_model("""name: symbolic derived chain\nvariables:\n  x: {domain: [-2, 2]}\nparameters:\n  a: {default: 2, domain: [0, 4]}\nderived_quantities:\n  q: {expression: a*x}\n  r: {expression: q**2+1}\nfunctions:\n  y: r+q\n""")
    result = analyse_one_variable_function(model)
    x = sp.Symbol("x", real=True)
    a = sp.Symbol("a", real=True)
    expected = 2*a**2*x + a
    if sp.simplify(result.first_derivative - expected) != 0:
        raise AssertionError("derived-quantity chain was not resolved before differentiation")

record("symbolic derived-quantity chain", "symbolic analysis", check_symbolic_derived_chain)

# ---- Write report -----------------------------------------------------------------

passed = sum(item.passed for item in results)
failed = len(results) - passed
report = {
    "laboratory_version": __version__,
    "checks": len(results),
    "passed": passed,
    "failed": failed,
    "results": [asdict(item) for item in results],
}

report_path = ROOT / "verification" / "report.json"
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

markdown = [
    f"# Model Laboratory {__version__} — Reference Verification Report",
    "",
    f"**Checks:** {len(results)}  ",
    f"**Passed:** {passed}  ",
    f"**Failed:** {failed}",
    "",
    "The checks compare laboratory outputs with independently specified analytical or exact numerical reference results.",
    "",
    "| Category | Check | Result |",
    "|---|---|---|",
]
for item in results:
    markdown.append(
        f"| {item.category} | {item.name} | {'PASS' if item.passed else 'FAIL — ' + item.detail} |"
    )
markdown.extend(["", "## Failures", ""])
if failed:
    for item in results:
        if not item.passed:
            markdown.append(f"- **{item.name}:** {item.detail}")
else:
    markdown.append("None.")
markdown.append("")
(ROOT / "verification" / "REPORT.md").write_text("\n".join(markdown), encoding="utf-8")

print(f"Reference verification: {passed}/{len(results)} passed")
for item in results:
    if not item.passed:
        print(f"FAIL [{item.category}] {item.name}: {item.detail}")

if failed:
    raise SystemExit(1)
