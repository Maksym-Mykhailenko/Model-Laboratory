from __future__ import annotations

from dataclasses import replace

import pytest

from model_lab.authoring import (
    ExperimentAuthoringError,
    create_publishable_mlab_bundle,
    freeze_experiment,
    prepare_experiment_review,
)
from model_lab.bundle import create_mlab_bundle, load_mlab_bundle
from model_lab.capabilities import ModelVisualisation
from model_lab.evaluator import evaluate_single_variable_function
from model_lab.experiment import (
    EvaluationSettings,
    NumericalReproductionSettings,
    StationaryPointSettings,
    create_experiment_state,
)
from model_lab.parser import parse_model_text
from model_lab.validator import validate_model


MODEL_TEXT = """name: Frozen experiment
variables:
  x:
    domain: [-2, 2]
parameters:
  a:
    default: 1
    domain: [0, 2]
functions:
  y: x**2 + a
"""


def _experiment():
    model = validate_model(parse_model_text(MODEL_TEXT))
    evaluation_settings = EvaluationSettings(points_1d=101)
    stationary_settings = StationaryPointSettings(samples_1d=501)
    evaluation = evaluate_single_variable_function(model, {"a": 1.0}, points=101)
    state = create_experiment_state(
        model_source=MODEL_TEXT,
        model=model,
        parameter_values={"a": 1.0},
        selected_model_visualisation=ModelVisualisation.TWO_D_FUNCTION_PLOT,
        evaluation=evaluation,
        stationary_result=None,
        evaluation_settings=evaluation_settings,
        stationary_settings=stationary_settings,
        sweep_configuration=None,
        sweep_result=None,
    )
    return state, model, evaluation


def test_publication_requires_exact_explicit_review_approval() -> None:
    state, model, _ = _experiment()
    review = prepare_experiment_review(state, model)

    with pytest.raises(ExperimentAuthoringError, match="does not match"):
        freeze_experiment(review, approved_review_sha256="0" * 64)


def test_publishable_bundle_preserves_and_validates_frozen_author_review() -> None:
    state, model, evaluation = _experiment()
    review = prepare_experiment_review(state, model)
    frozen = freeze_experiment(review, approved_review_sha256=review.review_sha256)
    data = create_publishable_mlab_bundle(
        frozen=frozen,
        state=state,
        model=model,
        evaluation=evaluation,
        stationary_result=None,
        sweep_result=None,
    )

    loaded = load_mlab_bundle(data)
    assert loaded.authoring_document is not None
    assert loaded.author_approved_for_publication
    assert loaded.authoring_document["status"] == "FROZEN"
    assert loaded.authoring_document["review_sha256"] == review.review_sha256


def test_state_or_tolerance_change_invalidates_author_approval() -> None:
    state, model, evaluation = _experiment()
    review = prepare_experiment_review(state, model)
    frozen = freeze_experiment(review, approved_review_sha256=review.review_sha256)
    changed = replace(
        state,
        numerical_reproduction_settings=NumericalReproductionSettings(
            relative_tolerance=5e-9,
            absolute_tolerance=1e-11,
        ),
        state_sha256="",
    ).with_checksum()

    with pytest.raises(ExperimentAuthoringError, match="changed after author approval"):
        create_publishable_mlab_bundle(
            frozen=frozen,
            state=changed,
            model=model,
            evaluation=evaluation,
            stationary_result=None,
            sweep_result=None,
        )


def test_low_level_bundle_is_explicitly_unpublished_without_authoring_record() -> None:
    state, model, evaluation = _experiment()
    data = create_mlab_bundle(
        state=state,
        model=model,
        evaluation=evaluation,
        stationary_result=None,
        sweep_result=None,
    )

    loaded = load_mlab_bundle(data)
    assert loaded.authoring_document is None
    assert not loaded.author_approved_for_publication
