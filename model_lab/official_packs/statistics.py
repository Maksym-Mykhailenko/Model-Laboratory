"""Statistical Inference and Data Modelling official capability pack.

The pack keeps observations, fitted models, inferential settings and diagnostics
explicit.  It deliberately does not treat a plot or a loose matrix as a statistical
study.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping
import warnings

import numpy as np
from scipy import stats

from ..model import ModelIR
from ..model_graph import ModelGraphError, ModelObject, ObjectKindDescriptor
from ..protocol import ArtifactTypeDescriptor, CapabilityDescriptor, SpectralDecomposition
from .common import PackManifest, finite_matrix, finite_vector, objects_of_kind, select_object, symmetric_spectral_decomposition, unique_labels


PACK_ID = "org.modellab.pack.statistical-inference-data-modelling"
DATASET_KIND = "org.modellab.statistics.dataset"
LINEAR_MODEL_KIND = "org.modellab.statistics.linear-model-study"
GROUPED_SAMPLES_KIND = "org.modellab.statistics.grouped-samples"

_NUMBER_ROW = {
    "type": "array", "minItems": 1, "maxItems": 512,
    "items": {"type": "number"},
}

DATASET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["columns", "data"],
    "properties": {
        "columns": {"type": "array", "minItems": 1, "maxItems": 512, "items": {"type": "string"}},
        "data": {"type": "array", "minItems": 2, "maxItems": 100000, "items": _NUMBER_ROW},
        "observation_ids": {"type": "array", "maxItems": 100000, "items": {"type": "string"}},
        "column_units": {"type": "array", "maxItems": 512, "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

LINEAR_MODEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["response_name", "predictor_names", "response", "predictors", "include_intercept"],
    "properties": {
        "response_name": {"type": "string", "minLength": 1, "maxLength": 128},
        "predictor_names": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "string"}},
        "response": {"type": "array", "minItems": 2, "maxItems": 100000, "items": {"type": "number"}},
        "predictors": {"type": "array", "minItems": 2, "maxItems": 100000, "items": _NUMBER_ROW},
        "include_intercept": {"type": "boolean"},
        "weights": {"type": "array", "maxItems": 100000, "items": {"type": "number"}},
        "observation_ids": {"type": "array", "maxItems": 100000, "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

_GROUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["name", "values"],
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 128},
        "values": {"type": "array", "minItems": 2, "maxItems": 100000, "items": {"type": "number"}},
    },
    "additionalProperties": False,
}

GROUPED_SAMPLES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["measure_name", "groups"],
    "properties": {
        "measure_name": {"type": "string", "minLength": 1, "maxLength": 128},
        "unit": {"type": "string", "maxLength": 128},
        "groups": {"type": "array", "minItems": 2, "maxItems": 128, "items": _GROUP_SCHEMA},
    },
    "additionalProperties": False,
}

KIND_DESCRIPTORS = (
    ObjectKindDescriptor(DATASET_KIND, "1.0", "Finite complete numerical dataset", DATASET_SCHEMA, True),
    ObjectKindDescriptor(LINEAR_MODEL_KIND, "1.0", "Linear statistical model study", LINEAR_MODEL_SCHEMA, True),
    ObjectKindDescriptor(GROUPED_SAMPLES_KIND, "1.0", "Independent grouped numerical samples", GROUPED_SAMPLES_SCHEMA, True),
)


@dataclass(frozen=True, slots=True)
class _Dataset:
    object_id: str
    columns: tuple[str, ...]
    data: np.ndarray
    observation_ids: tuple[str, ...]
    column_units: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _LinearStudy:
    object_id: str
    response_name: str
    predictor_names: tuple[str, ...]
    response: np.ndarray
    predictors: np.ndarray
    include_intercept: bool
    weights: np.ndarray
    observation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _GroupedSamples:
    object_id: str
    measure_name: str
    unit: str
    names: tuple[str, ...]
    values: tuple[np.ndarray, ...]


def _optional_labels(value: object, *, field: str, expected: int, prefix: str) -> tuple[str, ...]:
    if value is None:
        return tuple(f"{prefix}{index + 1}" for index in range(expected))
    labels = unique_labels(value, field=field)
    if len(labels) != expected:
        raise ModelGraphError(f"{field} must contain exactly {expected} labels.")
    return labels


def _dataset(item: ModelObject) -> _Dataset:
    columns = unique_labels(item.properties["columns"], field=f"{item.identifier}.columns")
    data = finite_matrix(item.properties["data"], field=f"{item.identifier}.data")
    if data.shape[1] != len(columns):
        raise ModelGraphError(f"{item.identifier}.data columns must align with columns.")
    observations = _optional_labels(
        item.properties.get("observation_ids"), field=f"{item.identifier}.observation_ids",
        expected=data.shape[0], prefix="row-",
    )
    raw_units = item.properties.get("column_units")
    if raw_units is None:
        units = tuple("" for _ in columns)
    else:
        if not isinstance(raw_units, list) or len(raw_units) != len(columns):
            raise ModelGraphError(f"{item.identifier}.column_units must align with columns.")
        units = tuple(str(value) for value in raw_units)
    return _Dataset(item.identifier, columns, data, observations, units)


def _linear_study(item: ModelObject) -> _LinearStudy:
    names = unique_labels(item.properties["predictor_names"], field=f"{item.identifier}.predictor_names")
    response = finite_vector(item.properties["response"], field=f"{item.identifier}.response")
    predictors = finite_matrix(item.properties["predictors"], field=f"{item.identifier}.predictors")
    if predictors.shape != (response.size, len(names)):
        raise ModelGraphError(f"{item.identifier}.predictors must align with response and predictor_names.")
    raw_weights = item.properties.get("weights")
    weights = np.ones(response.size, dtype=np.float64) if raw_weights is None else finite_vector(
        raw_weights, field=f"{item.identifier}.weights"
    )
    if weights.size != response.size or np.any(weights <= 0.0):
        raise ModelGraphError(f"{item.identifier}.weights must be positive and align with response.")
    observations = _optional_labels(
        item.properties.get("observation_ids"), field=f"{item.identifier}.observation_ids",
        expected=response.size, prefix="observation-",
    )
    response_name = str(item.properties["response_name"])
    if not response_name.strip():
        raise ModelGraphError(f"{item.identifier}.response_name cannot be empty.")
    return _LinearStudy(
        item.identifier, response_name, names, response, predictors,
        bool(item.properties["include_intercept"]), weights, observations,
    )


def _grouped(item: ModelObject) -> _GroupedSamples:
    raw_groups = item.properties["groups"]
    if not isinstance(raw_groups, list):
        raise ModelGraphError(f"{item.identifier}.groups must be a list.")
    names: list[str] = []
    values: list[np.ndarray] = []
    total = 0
    for index, raw in enumerate(raw_groups):
        if not isinstance(raw, Mapping):
            raise ModelGraphError(f"{item.identifier}.groups[{index}] must be an object.")
        name = str(raw["name"])
        if not name.strip() or name in names:
            raise ModelGraphError(f"{item.identifier}.groups must have unique non-empty names.")
        sample = finite_vector(raw["values"], field=f"{item.identifier}.groups[{index}].values")
        if sample.size < 2:
            raise ModelGraphError(f"{item.identifier}.groups[{index}] requires at least two observations.")
        names.append(name); values.append(sample); total += sample.size
    if total > 100000:
        raise ModelGraphError(f"{item.identifier}.groups exceeds the observation limit.")
    return _GroupedSamples(
        item.identifier, str(item.properties["measure_name"]), str(item.properties.get("unit", "")),
        tuple(names), tuple(values),
    )


SEMANTIC_VALIDATORS = {
    DATASET_KIND: lambda item: _dataset(item),
    LINEAR_MODEL_KIND: lambda item: _linear_study(item),
    GROUPED_SAMPLES_KIND: lambda item: _grouped(item),
}


@dataclass(frozen=True, slots=True)
class DatasetAnalysis:
    object_id: str
    columns: tuple[str, ...]
    column_units: tuple[str, ...]
    observation_count: int
    column_count: int
    degrees_of_freedom_correction: int
    means: np.ndarray
    standard_deviations: np.ndarray
    minima: np.ndarray
    first_quartiles: np.ndarray
    medians: np.ndarray
    third_quartiles: np.ndarray
    maxima: np.ndarray
    covariance: np.ndarray
    correlation: np.ndarray
    principal_coordinate_mode: str
    principal_coordinate_center: np.ndarray
    principal_coordinate_scale: np.ndarray
    constant_columns: tuple[str, ...]
    singular_values: np.ndarray
    principal_components: SpectralDecomposition
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray


def analyse_dataset(
    model: ModelIR,
    object_id: str | None = None,
    ddof: int = 1,
    feature_scaling: str = "standardized",
) -> DatasetAnalysis:
    if isinstance(ddof, bool) or ddof not in {0, 1}:
        raise ValueError("ddof must be 0 or 1.")
    if feature_scaling not in {"raw", "standardized"}:
        raise ValueError("feature_scaling must be raw or standardized.")
    data = _dataset(select_object(model, DATASET_KIND, object_id, label="numerical dataset"))
    if data.data.shape[0] <= ddof:
        raise ValueError("The dataset has insufficient observations for the requested ddof.")
    centered = data.data - np.mean(data.data, axis=0)
    covariance = centered.T @ centered / (data.data.shape[0] - ddof)
    standard = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(standard, standard)
    correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0.0)
    np.fill_diagonal(correlation, np.where(standard > 0.0, 1.0, 0.0))
    safe_scale = np.where(standard > 0.0, standard, 1.0)
    analysis_coordinates = centered / safe_scale if feature_scaling == "standardized" else centered
    analysis_scale = safe_scale if feature_scaling == "standardized" else np.ones_like(standard)
    analysis_covariance = (
        analysis_coordinates.T @ analysis_coordinates / (data.data.shape[0] - ddof)
    )
    singular = np.linalg.svd(analysis_coordinates, compute_uv=False, full_matrices=False)
    variance, components = symmetric_spectral_decomposition(
        analysis_covariance, descending=True, basis_orientation="rows"
    )
    variance = np.maximum(variance, 0.0)
    total = float(np.sum(variance))
    ratio = variance / total if total > 0.0 else np.zeros_like(variance)
    quantiles = np.quantile(data.data, [0.25, 0.5, 0.75], axis=0)
    return DatasetAnalysis(
        data.object_id, data.columns, data.column_units, data.data.shape[0], data.data.shape[1], ddof,
        np.mean(data.data, axis=0), standard, np.min(data.data, axis=0), quantiles[0], quantiles[1],
        quantiles[2], np.max(data.data, axis=0), covariance, correlation,
        feature_scaling, np.mean(data.data, axis=0), analysis_scale,
        tuple(data.columns[index] for index in np.flatnonzero(standard == 0.0)),
        singular, components,
        variance, ratio,
    )


@dataclass(frozen=True, slots=True)
class LinearModelFit:
    object_id: str
    response_name: str
    term_names: tuple[str, ...]
    observation_ids: tuple[str, ...]
    coefficients: np.ndarray
    standard_errors: np.ndarray
    t_statistics: np.ndarray
    p_values: np.ndarray
    confidence_intervals: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    weighted_residual_sum_squares: float
    residual_standard_error: float
    r_squared: float
    adjusted_r_squared: float
    r_squared_definition: str
    degrees_of_freedom: int
    design_rank: int
    condition_number: float
    confidence_level: float
    aic: float
    bic: float
    information_criterion_likelihood: str


def fit_linear_model(
    model: ModelIR, object_id: str | None = None, confidence_level: float = 0.95,
) -> LinearModelFit:
    if not math.isfinite(float(confidence_level)) or not 0.5 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must lie strictly between 0.5 and 1.")
    study = _linear_study(select_object(model, LINEAR_MODEL_KIND, object_id, label="linear-model study"))
    design = study.predictors
    terms = study.predictor_names
    if study.include_intercept:
        design = np.column_stack((np.ones(study.response.size), design))
        terms = ("intercept", *terms)
    root_weight = np.sqrt(study.weights)
    weighted_design = design * root_weight[:, None]
    weighted_response = study.response * root_weight
    coefficients, _, rank, singular = np.linalg.lstsq(weighted_design, weighted_response, rcond=None)
    if rank < weighted_design.shape[1]:
        raise ValueError(
            "Linear-model coefficient inference is not identifiable because the weighted design is rank-deficient."
        )
    fitted = design @ coefficients
    residuals = study.response - fitted
    rss = float(np.sum(study.weights * residuals * residuals))
    dof = int(study.response.size - rank)
    if dof <= 0:
        raise ValueError("Linear-model inference requires positive residual degrees of freedom.")
    variance = rss / dof
    covariance = variance * np.linalg.pinv(weighted_design.T @ weighted_design, hermitian=True)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_values = np.divide(coefficients, standard_errors, out=np.zeros_like(coefficients), where=standard_errors > 0.0)
    p_values = 2.0 * stats.t.sf(np.abs(t_values), dof)
    critical = float(stats.t.ppf((1.0 + float(confidence_level)) / 2.0, dof))
    intervals = np.column_stack((coefficients - critical * standard_errors, coefficients + critical * standard_errors))
    if study.include_intercept:
        weighted_mean = float(np.average(study.response, weights=study.weights))
        total = float(np.sum(study.weights * (study.response - weighted_mean) ** 2))
        r_squared_definition = "centred-weighted"
    else:
        total = float(np.sum(study.weights * study.response * study.response))
        r_squared_definition = "uncentred-weighted-through-origin"
    r_squared = 1.0 - rss / total if total > 0.0 else (1.0 if rss <= np.finfo(float).eps else 0.0)
    adjustment_numerator = study.response.size - 1 if study.include_intercept else study.response.size
    adjusted = 1.0 - (1.0 - r_squared) * adjustment_numerator / dof
    sigma2_mle = max(rss / study.response.size, np.finfo(float).tiny)
    # Weights are relative inverse variances: Var(e_i)=sigma^2/w_i.  Including
    # sum(log(w_i)) makes AIC/BIC invariant to a common rescaling of all weights.
    log_likelihood = -0.5 * (
        study.response.size * (math.log(2.0 * math.pi * sigma2_mle) + 1.0)
        - float(np.sum(np.log(study.weights)))
    )
    parameter_count = len(coefficients) + 1
    condition = float(np.inf if singular.size == 0 or singular[-1] == 0.0 else singular[0] / singular[-1])
    return LinearModelFit(
        study.object_id, study.response_name, tuple(terms), study.observation_ids, coefficients,
        standard_errors, t_values, np.asarray(p_values), intervals, fitted, residuals, rss,
        math.sqrt(variance), float(r_squared), float(adjusted), r_squared_definition,
        dof, int(rank), condition,
        float(confidence_level), float(2 * parameter_count - 2 * log_likelihood),
        float(math.log(study.response.size) * parameter_count - 2 * log_likelihood),
        "Gaussian independent errors with relative inverse-variance weights and fitted common scale",
    )


@dataclass(frozen=True, slots=True)
class GroupComparison:
    object_id: str
    measure_name: str
    unit: str
    group_names: tuple[str, ...]
    sample_sizes: np.ndarray
    means: np.ndarray
    standard_deviations: np.ndarray
    anova_f: float | None
    anova_p: float | None
    kruskal_h: float | None
    kruskal_p: float | None
    eta_squared: float | None
    test_statuses: Mapping[str, Mapping[str, str]]
    pairwise_welch: tuple[Mapping[str, Any], ...]
    correction: str
    significance_level: float


def compare_groups(
    model: ModelIR, object_id: str | None = None, significance_level: float = 0.05,
) -> GroupComparison:
    if not math.isfinite(float(significance_level)) or not 0.0 < float(significance_level) < 0.5:
        raise ValueError("significance_level must lie strictly between 0 and 0.5.")
    grouped = _grouped(select_object(model, GROUPED_SAMPLES_KIND, object_id, label="grouped-samples"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        anova = stats.f_oneway(*grouped.values)
        try:
            kruskal = stats.kruskal(*grouped.values)
        except ValueError:
            kruskal = None
    anova_defined = math.isfinite(float(anova.statistic)) and math.isfinite(float(anova.pvalue))
    kruskal_defined = (
        kruskal is not None
        and math.isfinite(float(kruskal.statistic))
        and math.isfinite(float(kruskal.pvalue))
    )
    all_values = np.concatenate(grouped.values)
    grand = float(np.mean(all_values))
    between = sum(sample.size * (float(np.mean(sample)) - grand) ** 2 for sample in grouped.values)
    total = float(np.sum((all_values - grand) ** 2))
    raw: list[tuple[int, int, float | None, float | None, float, str, str]] = []
    for left in range(len(grouped.values)):
        for right in range(left + 1, len(grouped.values)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                test = stats.ttest_ind(grouped.values[left], grouped.values[right], equal_var=False)
            statistic, probability = float(test.statistic), float(test.pvalue)
            defined = math.isfinite(statistic) and math.isfinite(probability)
            raw.append((
                left, right, statistic if defined else None, probability if defined else None,
                float(np.mean(grouped.values[left]) - np.mean(grouped.values[right])),
                "defined" if defined else "undefined",
                "" if defined else "Welch's test is undefined because the samples have degenerate variance.",
            ))
    defined_indices = [index for index, row in enumerate(raw) if row[3] is not None]
    order = sorted(defined_indices, key=lambda index: (float(raw[index][3]), raw[index][0], raw[index][1]))
    adjusted: list[float | None] = [None] * len(raw)
    running = 0.0
    for position, index in enumerate(order):
        value = min(1.0, (len(order) - position) * float(raw[index][3]))
        running = max(running, value)
        adjusted[index] = running
    pairwise = tuple({
        "left": grouped.names[left], "right": grouped.names[right], "mean_difference": difference,
        "status": status, "reason": reason or None,
        "t_statistic": statistic, "raw_p": probability, "holm_adjusted_p": adjusted[index],
        "reject": bool(adjusted[index] is not None and adjusted[index] <= significance_level),
    } for index, (left, right, statistic, probability, difference, status, reason) in enumerate(raw))
    test_statuses = {
        "anova": {
            "status": "defined" if anova_defined else "undefined",
            "reason": "" if anova_defined else "One-way ANOVA is undefined because the grouped samples have degenerate variance.",
        },
        "kruskal_wallis": {
            "status": "defined" if kruskal_defined else "undefined",
            "reason": "" if kruskal_defined else "Kruskal-Wallis is undefined because all ranked observations are tied.",
        },
        "eta_squared": {
            "status": "defined" if total > 0.0 else "undefined",
            "reason": "" if total > 0.0 else "Eta squared is undefined because the observations have no total variation.",
        },
    }
    return GroupComparison(
        grouped.object_id, grouped.measure_name, grouped.unit, grouped.names,
        np.asarray([sample.size for sample in grouped.values], dtype=np.int64),
        np.asarray([np.mean(sample) for sample in grouped.values], dtype=np.float64),
        np.asarray([np.std(sample, ddof=1) for sample in grouped.values], dtype=np.float64),
        float(anova.statistic) if anova_defined else None,
        float(anova.pvalue) if anova_defined else None,
        float(kruskal.statistic) if kruskal_defined else None,
        float(kruskal.pvalue) if kruskal_defined else None,
        float(between / total) if total > 0.0 else None,
        test_statuses, pairwise, "Holm over defined pairwise tests", float(significance_level),
    )


def _kind_applicability(kind: str, label: str):
    def applicable(model: ModelIR) -> tuple[bool, str]:
        found = bool(objects_of_kind(model, kind))
        return found, "" if found else f"an executable {label} object is required"
    return applicable


def _dataset_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(model, DATASET_KIND, settings.get("object_id"), label="numerical dataset")
    rows = len(item.properties["data"]); columns = len(item.properties["columns"])
    return max(1, rows * columns * columns)


def _linear_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(
        model, LINEAR_MODEL_KIND, settings.get("object_id"), label="linear-model study"
    )
    rows = len(item.properties["response"]); columns = len(item.properties["predictor_names"]) + 1
    return max(1, rows * columns * columns)


def _group_units(settings: Mapping[str, Any], model: ModelIR) -> int:
    item = select_object(
        model, GROUPED_SAMPLES_KIND, settings.get("object_id"), label="grouped samples"
    )
    groups = item.properties["groups"]
    return max(1, sum(len(group["values"]) for group in groups) * len(groups))


NUMERIC = "org.modellab.comparator.numeric"

CAPABILITY_DESCRIPTORS = (
    CapabilityDescriptor(
        "org.modellab.statistics.analyse-dataset", "1.2", PACK_ID,
        "Analyse numerical dataset", "Compute complete-data summaries, covariance, correlation and principal components in explicit raw or standardized analysis coordinates.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "ddof": {"type": "integer", "enum": [0, 1], "default": 1}, "feature_scaling": {"type": "string", "enum": ["raw", "standardized"], "default": "standardized"}}},
        "numpy+scipy", (ArtifactTypeDescriptor("org.modellab.artifact.dataset-analysis", "1.2", "Dataset analysis", NUMERIC),),
        _kind_applicability(DATASET_KIND, "numerical-dataset"), analyse_dataset, _dataset_units,
        ("org.modellab.renderer.plotly-correlation-matrix",),
    ),
    CapabilityDescriptor(
        "org.modellab.statistics.fit-linear-model", "1.1", PACK_ID,
        "Fit linear statistical model", "Fit ordinary or positive-weighted least squares with coefficient uncertainty, residual diagnostics and information criteria.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "confidence_level": {"type": "number", "exclusiveMinimum": 0.5, "exclusiveMaximum": 1.0, "default": 0.95}}},
        "numpy+scipy", (ArtifactTypeDescriptor("org.modellab.artifact.linear-model-fit", "1.1", "Linear-model fit", NUMERIC),),
        _kind_applicability(LINEAR_MODEL_KIND, "linear-model-study"), fit_linear_model, _linear_units,
        ("org.modellab.renderer.plotly-regression-diagnostics",),
    ),
    CapabilityDescriptor(
        "org.modellab.statistics.compare-groups", "1.1", PACK_ID,
        "Compare independent groups", "Compute one-way ANOVA, Kruskal-Wallis, eta squared and Holm-corrected pairwise Welch tests, with explicit undefined results for degenerate samples.",
        {"type": "object", "properties": {"object_id": {"type": ["string", "null"]}, "significance_level": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 0.5, "default": 0.05}}},
        "scipy+numpy", (ArtifactTypeDescriptor("org.modellab.artifact.group-comparison", "1.1", "Group comparison", NUMERIC),),
        _kind_applicability(GROUPED_SAMPLES_KIND, "grouped-samples"), compare_groups, _group_units,
        ("org.modellab.renderer.plotly-group-comparison",),
    ),
)

MANIFEST = PackManifest(
    PACK_ID, "1.2", "Statistical Inference and Data Modelling",
    "Complete numerical datasets, regression studies and independent-group inference with explicit assumptions and reproducible diagnostics.",
    ("org.modellab.pack.multidimensional-mathematics", "org.modellab.pack.probability-stochastic-systems"),
    tuple(item.kind for item in KIND_DESCRIPTORS), tuple(item.identifier for item in CAPABILITY_DESCRIPTORS),
)
