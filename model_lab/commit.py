"""Immutable committed experiment-state boundary.

A committed experiment is deliberately distinct from the mutable/live authoring state.
The commit envelope contains the complete checksummed :class:`ExperimentState`, binds it
to the canonical Model IR, and has its own event checksum. Future external execution or
physicalisation layers should accept only a validated :class:`CommittedExperiment` (or its
``commit_sha256`` identity), never values read directly from interactive UI controls.

Committing does not execute an experiment and is not the same as publication approval.
Publication remains governed by :mod:`model_lab.authoring`; this module supplies the
operational state boundary that can remain stable while the live model continues to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Mapping

from .canonical import canonical_model_ir_sha256
from .experiment import ExperimentState, ExperimentStateError, validate_experiment_state_for_model
from .model import ModelIR


COMMITTED_EXPERIMENT_SCHEMA = "model-laboratory.committed-experiment"
COMMITTED_EXPERIMENT_SCHEMA_VERSION = "1.0"
_COMMITTED_STATUS = "COMMITTED"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXECUTION_POLICY = {
    "state_source": "embedded_committed_experiment_state",
    "live_state_authority": False,
    "external_execution_requires_commit_sha256": True,
}


class CommittedExperimentError(ValueError):
    """Raised when a committed-state envelope is malformed or fails validation."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_text(value: object) -> str:
    return _canonical_json_bytes(value).decode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _normalise_timestamp(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if not isinstance(value, str) or not value.endswith("Z"):
        raise CommittedExperimentError("committed_at_utc must be an RFC 3339 UTC timestamp ending in Z.")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise CommittedExperimentError("committed_at_utc is not a valid RFC 3339 timestamp.") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise CommittedExperimentError("committed_at_utc must use UTC.")
    return value


def _state_document(state: ExperimentState) -> dict[str, Any]:
    return json.loads(state.with_checksum().to_json(indent=None))


def _commit_payload(
    *,
    committed_at_utc: str,
    experiment_state_sha256: str,
    model_source_sha256: str,
    model_ir_sha256: str,
    parent_commit_sha256: str | None,
    experiment_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": COMMITTED_EXPERIMENT_SCHEMA,
        "schema_version": COMMITTED_EXPERIMENT_SCHEMA_VERSION,
        "status": _COMMITTED_STATUS,
        "committed_at_utc": committed_at_utc,
        "experiment_state_sha256": experiment_state_sha256,
        "model_source_sha256": model_source_sha256,
        "model_ir_sha256": model_ir_sha256,
        "parent_commit_sha256": parent_commit_sha256,
        "execution_policy": dict(_EXECUTION_POLICY),
        "experiment_state": dict(experiment_state),
    }


@dataclass(frozen=True, slots=True)
class CommittedExperiment:
    """Immutable operational commit of one exact experiment state.

    The embedded experiment is stored internally as canonical JSON text so the frozen
    dataclass cannot be mutated through a nested dictionary reference. ``experiment_state_sha256``
    identifies the scientific state; ``commit_sha256`` identifies the commit event itself,
    including timestamp and optional parent commit.
    """

    committed_at_utc: str
    experiment_state_sha256: str
    model_source_sha256: str
    model_ir_sha256: str
    parent_commit_sha256: str | None
    experiment_state_json: str
    commit_sha256: str

    def __post_init__(self) -> None:
        _normalise_timestamp(self.committed_at_utc)
        for name, value in (
            ("experiment_state_sha256", self.experiment_state_sha256),
            ("model_source_sha256", self.model_source_sha256),
            ("model_ir_sha256", self.model_ir_sha256),
            ("commit_sha256", self.commit_sha256),
        ):
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise CommittedExperimentError(f"{name} must be a lowercase SHA-256 digest.")
        if self.parent_commit_sha256 is not None and (
            not isinstance(self.parent_commit_sha256, str)
            or _SHA256_RE.fullmatch(self.parent_commit_sha256) is None
        ):
            raise CommittedExperimentError("parent_commit_sha256 must be null or a lowercase SHA-256 digest.")
        try:
            embedded = json.loads(self.experiment_state_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise CommittedExperimentError("experiment_state must be a JSON object.") from exc
        if not isinstance(embedded, dict):
            raise CommittedExperimentError("experiment_state must be a JSON object.")
        if _canonical_json_text(embedded) != self.experiment_state_json:
            raise CommittedExperimentError("Internal experiment state must use canonical JSON encoding.")
        if self.payload_sha256() != self.commit_sha256:
            raise CommittedExperimentError("Committed experiment checksum does not match its contents.")

    def experiment_state_document(self) -> dict[str, Any]:
        """Return a fresh copy of the embedded experiment document."""
        value = json.loads(self.experiment_state_json)
        assert isinstance(value, dict)
        return value

    def payload_dict(self) -> dict[str, Any]:
        return _commit_payload(
            committed_at_utc=self.committed_at_utc,
            experiment_state_sha256=self.experiment_state_sha256,
            model_source_sha256=self.model_source_sha256,
            model_ir_sha256=self.model_ir_sha256,
            parent_commit_sha256=self.parent_commit_sha256,
            experiment_state=self.experiment_state_document(),
        )

    def payload_sha256(self) -> str:
        return _sha256_json(self.payload_dict())

    def to_document(self) -> dict[str, Any]:
        document = self.payload_dict()
        document["commit_sha256"] = self.commit_sha256
        return document

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_document(),
            sort_keys=True,
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        ) + ("\n" if indent is not None else "")

    @classmethod
    def from_json(cls, text: str) -> "CommittedExperiment":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CommittedExperimentError(f"Committed experiment is not valid JSON: {exc.msg}.") from exc
        if not isinstance(raw, dict):
            raise CommittedExperimentError("Committed experiment must be a JSON object.")
        return cls.from_document(raw)

    @classmethod
    def from_document(cls, raw: Mapping[str, Any]) -> "CommittedExperiment":
        required = {
            "schema",
            "schema_version",
            "status",
            "committed_at_utc",
            "experiment_state_sha256",
            "model_source_sha256",
            "model_ir_sha256",
            "parent_commit_sha256",
            "execution_policy",
            "experiment_state",
            "commit_sha256",
        }
        extra = set(raw) - required
        missing = required - set(raw)
        if extra:
            raise CommittedExperimentError(
                "Committed experiment contains unknown field(s): " + ", ".join(sorted(extra)) + "."
            )
        if missing:
            raise CommittedExperimentError(
                "Committed experiment is missing field(s): " + ", ".join(sorted(missing)) + "."
            )
        if raw.get("schema") != COMMITTED_EXPERIMENT_SCHEMA:
            raise CommittedExperimentError("Committed experiment schema identifier is unsupported.")
        if raw.get("schema_version") != COMMITTED_EXPERIMENT_SCHEMA_VERSION:
            raise CommittedExperimentError("Committed experiment schema version is unsupported.")
        if raw.get("status") != _COMMITTED_STATUS:
            raise CommittedExperimentError("Committed experiment status must be COMMITTED.")
        if raw.get("execution_policy") != _EXECUTION_POLICY:
            raise CommittedExperimentError("Committed experiment execution policy is malformed.")
        experiment_state = raw.get("experiment_state")
        if not isinstance(experiment_state, Mapping):
            raise CommittedExperimentError("experiment_state must be a JSON object.")
        return cls(
            committed_at_utc=str(raw["committed_at_utc"]),
            experiment_state_sha256=str(raw["experiment_state_sha256"]),
            model_source_sha256=str(raw["model_source_sha256"]),
            model_ir_sha256=str(raw["model_ir_sha256"]),
            parent_commit_sha256=(
                None if raw["parent_commit_sha256"] is None else str(raw["parent_commit_sha256"])
            ),
            experiment_state_json=_canonical_json_text(dict(experiment_state)),
            commit_sha256=str(raw["commit_sha256"]),
        )

    def experiment(self) -> ExperimentState:
        """Reconstruct and checksum-validate the embedded experiment state."""
        try:
            state = ExperimentState.from_json(self.experiment_state_json)
        except ExperimentStateError as exc:
            raise CommittedExperimentError(f"Embedded experiment state is invalid: {exc}") from exc
        if state.state_sha256 != self.experiment_state_sha256:
            raise CommittedExperimentError(
                "Embedded experiment-state SHA-256 does not match the committed identity."
            )
        if state.model_sha256 != self.model_source_sha256:
            raise CommittedExperimentError(
                "Embedded model-source SHA-256 does not match the committed identity."
            )
        return state

    def verify_for_model(self, model: ModelIR) -> ExperimentState:
        """Validate the commit and return its state for a trusted executor."""
        state = self.experiment()
        if canonical_model_ir_sha256(model) != self.model_ir_sha256:
            raise CommittedExperimentError(
                "Canonical Model IR SHA-256 does not match the committed identity."
            )
        try:
            validate_experiment_state_for_model(state, model, enforce_workload_budget=False)
        except ExperimentStateError as exc:
            raise CommittedExperimentError(str(exc)) from exc
        return state


def commit_experiment_state(
    state: ExperimentState,
    model: ModelIR,
    *,
    parent_commit_sha256: str | None = None,
    committed_at_utc: str | None = None,
) -> CommittedExperiment:
    """Create an immutable commit envelope from one exact validated experiment state."""
    checked = state.with_checksum()
    try:
        validate_experiment_state_for_model(checked, model, enforce_workload_budget=False)
    except ExperimentStateError as exc:
        raise CommittedExperimentError(str(exc)) from exc
    if parent_commit_sha256 is not None and _SHA256_RE.fullmatch(parent_commit_sha256) is None:
        raise CommittedExperimentError("parent_commit_sha256 must be null or a lowercase SHA-256 digest.")
    timestamp = _normalise_timestamp(committed_at_utc)
    state_document = _state_document(checked)
    model_ir_sha256 = canonical_model_ir_sha256(model)
    payload = _commit_payload(
        committed_at_utc=timestamp,
        experiment_state_sha256=checked.state_sha256,
        model_source_sha256=checked.model_sha256,
        model_ir_sha256=model_ir_sha256,
        parent_commit_sha256=parent_commit_sha256,
        experiment_state=state_document,
    )
    return CommittedExperiment(
        committed_at_utc=timestamp,
        experiment_state_sha256=checked.state_sha256,
        model_source_sha256=checked.model_sha256,
        model_ir_sha256=model_ir_sha256,
        parent_commit_sha256=parent_commit_sha256,
        experiment_state_json=_canonical_json_text(state_document),
        commit_sha256=_sha256_json(payload),
    )
