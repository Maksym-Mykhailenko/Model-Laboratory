"""Deliberate review and freeze workflow for publishable experiments.

The workflow is front-end independent so a desktop application can render the review and
ask for approval without changing the scientific contract.  Low-level bundle creation is
still available for drafts and internal interchange; a bundle intended for publication is
created through :func:`create_publishable_mlab_bundle` and carries its frozen review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .analysis import ParameterSweepResult, StationaryPointAnalysisResult
from .bundle import (
    AUTHORING_DOCUMENT_SCHEMA,
    AUTHORING_DOCUMENT_SCHEMA_VERSION,
    MlabBundleError,
    _create_mlab_bundle,
    canonical_authoring_review,
)
from .canonical import canonical_json_sha256, canonical_model_ir_sha256
from .evaluator import FunctionEvaluation, SurfaceEvaluation
from .experiment import ExperimentState, ExperimentStateError, validate_experiment_state_for_model
from .model import ModelIR


class ExperimentAuthoringError(ValueError):
    """Raised when an experiment has not passed the explicit freeze workflow."""


@dataclass(frozen=True, slots=True)
class ExperimentReview:
    """Canonical material shown to an author before reference state is frozen."""

    review: Mapping[str, Any]
    review_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.review, dict):
            raise ExperimentAuthoringError("experiment review must be a JSON object.")
        if canonical_json_sha256(self.review) != self.review_sha256:
            raise ExperimentAuthoringError("experiment review checksum is invalid.")


@dataclass(frozen=True, slots=True)
class FrozenExperiment:
    """An author-approved experiment state that can be published as ``.mlab``."""

    review: ExperimentReview
    experiment_state_sha256: str
    model_ir_sha256: str

    @property
    def authoring_document(self) -> dict[str, object]:
        return {
            "schema": AUTHORING_DOCUMENT_SCHEMA,
            "schema_version": AUTHORING_DOCUMENT_SCHEMA_VERSION,
            "status": "FROZEN",
            "review_sha256": self.review.review_sha256,
            "frozen_experiment_state_sha256": self.experiment_state_sha256,
            "model_ir_sha256": self.model_ir_sha256,
            "review": dict(self.review.review),
        }

    def verify(self, state: ExperimentState, model: ModelIR) -> None:
        checked = state.with_checksum()
        if checked.state_sha256 != self.experiment_state_sha256:
            raise ExperimentAuthoringError(
                "experiment state changed after author approval; prepare and approve a new review."
            )
        if canonical_model_ir_sha256(model) != self.model_ir_sha256:
            raise ExperimentAuthoringError(
                "canonical model changed after author approval; prepare and approve a new review."
            )
        if canonical_json_sha256(self.review.review) != self.review.review_sha256:
            raise ExperimentAuthoringError("the approved experiment review has been modified.")


def prepare_experiment_review(state: ExperimentState, model: ModelIR) -> ExperimentReview:
    """Create the exact, deterministic review that must be approved before publication."""
    checked = state.with_checksum()
    try:
        validate_experiment_state_for_model(checked, model, enforce_workload_budget=False)
    except ExperimentStateError as exc:
        raise ExperimentAuthoringError(str(exc)) from exc
    review = canonical_authoring_review(checked, model)
    return ExperimentReview(review=review, review_sha256=canonical_json_sha256(review))


def freeze_experiment(
    review: ExperimentReview, *, approved_review_sha256: str
) -> FrozenExperiment:
    """Freeze only the exact review digest explicitly approved by the author."""
    if approved_review_sha256 != review.review_sha256:
        raise ExperimentAuthoringError(
            "approved review SHA-256 does not match the current experiment review."
        )
    experiment = review.review.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ExperimentAuthoringError("experiment review is missing its canonical experiment.")
    state_sha256 = experiment.get("experiment_state_sha256")
    model = experiment.get("model")
    model_sha256 = model.get("canonical_ir_sha256") if isinstance(model, Mapping) else None
    if not isinstance(state_sha256, str) or not isinstance(model_sha256, str):
        raise ExperimentAuthoringError("experiment review identities are malformed.")
    return FrozenExperiment(review, state_sha256, model_sha256)


def create_publishable_mlab_bundle(
    *,
    frozen: FrozenExperiment,
    state: ExperimentState,
    model: ModelIR,
    evaluation: FunctionEvaluation | SurfaceEvaluation,
    stationary_result: StationaryPointAnalysisResult | None,
    sweep_result: ParameterSweepResult | None,
    asset_blobs: Mapping[str, bytes] | None = None,
) -> bytes:
    """Create a bundle carrying proof of the explicit author review and freeze step."""
    frozen.verify(state, model)
    try:
        return _create_mlab_bundle(
            state=state,
            model=model,
            evaluation=evaluation,
            stationary_result=stationary_result,
            sweep_result=sweep_result,
            authoring_document=frozen.authoring_document,
            asset_blobs=asset_blobs,
        )
    except MlabBundleError as exc:
        raise ExperimentAuthoringError(str(exc)) from exc


def create_publishable_run_mlab_bundle(
    *,
    frozen: FrozenExperiment,
    state: ExperimentState,
    model: ModelIR,
    content_blobs: Mapping[str, bytes] | None = None,
    content_descriptors: Mapping[str, Mapping[str, Any]] | None = None,
    asset_blobs: Mapping[str, bytes] | None = None,
) -> bytes:
    """Publish an author-approved generic Run -> Artifact experiment."""
    if not state.run_records or not state.artifacts:
        raise ExperimentAuthoringError("The experiment has no frozen Run -> Artifact state.")
    frozen.verify(state, model)
    try:
        return _create_mlab_bundle(
            state=state,
            model=model,
            evaluation=None,
            stationary_result=None,
            sweep_result=None,
            authoring_document=frozen.authoring_document,
            content_blobs=content_blobs,
            content_descriptors=content_descriptors,
            asset_blobs=asset_blobs,
        )
    except MlabBundleError as exc:
        raise ExperimentAuthoringError(str(exc)) from exc
