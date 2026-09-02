from __future__ import annotations

import json

import pytest

from model_lab.capabilities import ModelVisualisation
from model_lab.commit import (
    COMMITTED_EXPERIMENT_SCHEMA,
    CommittedExperiment,
    CommittedExperimentError,
    commit_experiment_state,
)
from model_lab.evaluator import evaluate_single_variable_function
from model_lab.experiment import EvaluationSettings, StationaryPointSettings, create_experiment_state
from model_lab.parser import parse_model_text
from model_lab.validator import validate_model


MODEL_TEXT = """name: Committed state example
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
    return state, model


def test_commit_is_separate_immutable_envelope_over_exact_experiment_state() -> None:
    state, model = _experiment()
    committed = commit_experiment_state(
        state,
        model,
        committed_at_utc="2026-08-27T00:30:00.000000Z",
    )

    document = committed.to_document()
    assert document["schema"] == COMMITTED_EXPERIMENT_SCHEMA
    assert document["status"] == "COMMITTED"
    assert document["experiment_state_sha256"] == state.with_checksum().state_sha256
    assert document["execution_policy"] == {
        "state_source": "embedded_committed_experiment_state",
        "live_state_authority": False,
        "external_execution_requires_commit_sha256": True,
    }
    assert len(committed.commit_sha256) == 64

    reconstructed = CommittedExperiment.from_json(committed.to_json())
    verified_state = reconstructed.verify_for_model(model)
    assert verified_state.state_sha256 == state.with_checksum().state_sha256


def test_returned_experiment_document_cannot_mutate_committed_object() -> None:
    state, model = _experiment()
    committed = commit_experiment_state(
        state,
        model,
        committed_at_utc="2026-08-27T00:30:00.000000Z",
    )
    detached = committed.experiment_state_document()
    detached["parameter_values"][0]["value"] = 1.75

    assert committed.verify_for_model(model).state_sha256 == state.with_checksum().state_sha256
    assert committed.experiment_state_document()["parameter_values"][0]["value"] == 1.0


def test_commit_event_can_chain_without_mutating_previous_commit() -> None:
    state, model = _experiment()
    first = commit_experiment_state(
        state,
        model,
        committed_at_utc="2026-08-27T00:30:00.000000Z",
    )
    second = commit_experiment_state(
        state,
        model,
        parent_commit_sha256=first.commit_sha256,
        committed_at_utc="2026-08-27T00:31:00.000000Z",
    )

    assert second.parent_commit_sha256 == first.commit_sha256
    assert second.commit_sha256 != first.commit_sha256
    assert second.experiment_state_sha256 == first.experiment_state_sha256
    assert first.parent_commit_sha256 is None


def test_tampered_embedded_state_is_rejected_even_if_outer_identity_is_unchanged() -> None:
    state, model = _experiment()
    committed = commit_experiment_state(
        state,
        model,
        committed_at_utc="2026-08-27T00:30:00.000000Z",
    )
    document = json.loads(committed.to_json())
    document["experiment_state"]["parameter_values"][0]["value"] = 1.5

    with pytest.raises(CommittedExperimentError, match="checksum"):
        CommittedExperiment.from_document(document)


def test_wrong_model_ir_cannot_be_used_as_committed_execution_authority() -> None:
    state, model = _experiment()
    committed = commit_experiment_state(
        state,
        model,
        committed_at_utc="2026-08-27T00:30:00.000000Z",
    )
    other = validate_model(parse_model_text(MODEL_TEXT.replace("x**2 + a", "x**2 + 2*a")))

    with pytest.raises(CommittedExperimentError, match="Canonical Model IR"):
        committed.verify_for_model(other)
