"""Checksum-bound interpreter candidate lifecycle, promotion, and rollback registry.

The registry is intentionally broader than the active runtime selector: experimental,
trained, evaluated, and approved candidates are retained as immutable-identity lifecycle
records, while only approved entries can ever become active.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import tempfile
from typing import Any, Mapping

from .canonical import canonical_json_sha256
from .interpreter import LOCAL_BASE_MODEL, LOCAL_MODEL, LOCAL_QUANTIZATION
from .interpreter_candidate import (
    CandidateEvaluationError,
    OllamaCandidateClient,
    compare_candidate_to_baseline,
    validate_candidate_evaluation_report,
)
from .interpreter_export import InterpreterExportError, validate_export_receipt
from .interpreter_finetuning import (
    DEFAULT_BASELINE_REPORT,
    DEFAULT_CONFIG,
    FineTuningError,
    baseline_gate,
    validate_training_report,
)
from .interpreter_promotion import default_promotion_ledger_path


REGISTRY_SCHEMA = "model-laboratory-interpreter-model-registry"
REGISTRY_VERSION = "1.3"
REGISTRY_FILENAME = "interpreter-model-registry.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LIFECYCLE_STATES = {"experimental", "trained", "evaluated", "approved"}


class InterpreterRegistryError(RuntimeError):
    """A registry integrity, lifecycle, promotion, concurrency, or rollback gate failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_registry_path() -> Path:
    configured = os.environ.get("MODEL_LAB_INTERPRETER_REGISTRY", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    system = platform.system()
    if system == "Windows":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif system == "Darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "org.modellaboratory.desktop" / REGISTRY_FILENAME


def _without_sha(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if key != field}


def _base_active() -> dict[str, Any]:
    return {"role": "frozen_base", "entry_id": None}


def empty_registry() -> dict[str, Any]:
    registry: dict[str, Any] = {
        "schema": REGISTRY_SCHEMA,
        "schema_version": REGISTRY_VERSION,
        "revision": 0,
        "active": _base_active(),
        "entries": [],
        "history": [],
    }
    registry["registry_sha256"] = canonical_json_sha256(registry)
    return registry


def _digest_or_none(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise InterpreterRegistryError(f"Registry candidate {field} is invalid.")
    return value


def _validate_entry(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InterpreterRegistryError("Registry candidate entry must be an object.")
    required = {
        "entry_id", "lifecycle_state", "training_status", "candidate_id",
        "candidate_tag", "observed_model_digest", "model_size_bytes", "quantization",
        "base_model", "base_snapshot_receipt_sha256", "quantized_gguf_sha256",
        "training_report_sha256", "export_receipt_sha256",
        "candidate_evaluation_report_sha256", "eligible_for_promotion",
        "evaluation_purpose", "promotion_campaign_id", "evaluation_benchmark_sha256",
        "promotion_seal_sha256",
        "created_at_utc", "evaluated_at_utc", "approved_at_utc", "entry_sha256",
    }
    if set(value) != required:
        raise InterpreterRegistryError("Registry candidate entry has missing or unknown fields.")
    entry_id = value.get("entry_id")
    if not isinstance(entry_id, str) or not entry_id.startswith("candidate-"):
        raise InterpreterRegistryError("Registry candidate entry_id is invalid.")
    state = value.get("lifecycle_state")
    if state not in _LIFECYCLE_STATES:
        raise InterpreterRegistryError("Registry candidate lifecycle state is invalid.")
    status = value.get("training_status")
    if status not in {"candidate_adapter_experimental", "candidate_adapter_unpromoted"}:
        raise InterpreterRegistryError("Registry candidate training status is invalid.")
    if state == "experimental" and status != "candidate_adapter_experimental":
        raise InterpreterRegistryError("Experimental registry state requires an experimental training report.")
    if state == "trained" and status != "candidate_adapter_unpromoted":
        raise InterpreterRegistryError("Trained registry state requires a non-experimental training report.")
    if state == "approved" and status != "candidate_adapter_unpromoted":
        raise InterpreterRegistryError("Only non-experimental training reports may be approved.")

    training_sha = _digest_or_none(value.get("training_report_sha256"), "training_report_sha256")
    if training_sha is None or entry_id != "candidate-" + training_sha[:20]:
        raise InterpreterRegistryError("Registry entry_id is not derived from its training report identity.")
    _digest_or_none(value.get("base_snapshot_receipt_sha256"), "base_snapshot_receipt_sha256")
    for field in ("quantized_gguf_sha256", "export_receipt_sha256", "candidate_evaluation_report_sha256"):
        _digest_or_none(value.get(field), field)
    _digest_or_none(value.get("entry_sha256"), "entry_sha256")
    if not isinstance(value.get("created_at_utc"), str) or not value["created_at_utc"]:
        raise InterpreterRegistryError("Registry candidate creation timestamp is missing.")

    advanced = state in {"evaluated", "approved"}
    advanced_fields = {
        "candidate_id": value.get("candidate_id"),
        "candidate_tag": value.get("candidate_tag"),
        "observed_model_digest": value.get("observed_model_digest"),
        "model_size_bytes": value.get("model_size_bytes"),
        "quantization": value.get("quantization"),
        "base_model": value.get("base_model"),
        "quantized_gguf_sha256": value.get("quantized_gguf_sha256"),
        "export_receipt_sha256": value.get("export_receipt_sha256"),
        "candidate_evaluation_report_sha256": value.get("candidate_evaluation_report_sha256"),
        "eligible_for_promotion": value.get("eligible_for_promotion"),
        "evaluation_purpose": value.get("evaluation_purpose"),
        "promotion_campaign_id": value.get("promotion_campaign_id"),
        "evaluation_benchmark_sha256": value.get("evaluation_benchmark_sha256"),
        "promotion_seal_sha256": value.get("promotion_seal_sha256"),
        "evaluated_at_utc": value.get("evaluated_at_utc"),
    }
    if advanced:
        if not isinstance(advanced_fields["candidate_id"], str) or not str(advanced_fields["candidate_id"]).startswith("candidate-"):
            raise InterpreterRegistryError("Evaluated registry candidate_id is invalid.")
        tag = advanced_fields["candidate_tag"]
        if not isinstance(tag, str) or not tag or tag == LOCAL_MODEL:
            raise InterpreterRegistryError("Evaluated registry candidate tag is invalid.")
        if not isinstance(advanced_fields["observed_model_digest"], str) or not advanced_fields["observed_model_digest"]:
            raise InterpreterRegistryError("Evaluated registry candidate Ollama digest is missing.")
        size = advanced_fields["model_size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise InterpreterRegistryError("Evaluated registry candidate size is invalid.")
        if advanced_fields["quantization"] != LOCAL_QUANTIZATION or advanced_fields["base_model"] != LOCAL_BASE_MODEL:
            raise InterpreterRegistryError("Evaluated registry candidate has incompatible model identity.")
        for field in ("quantized_gguf_sha256", "export_receipt_sha256", "candidate_evaluation_report_sha256"):
            if advanced_fields[field] is None:
                raise InterpreterRegistryError(f"Evaluated registry candidate lacks {field}.")
        if not isinstance(advanced_fields["eligible_for_promotion"], bool):
            raise InterpreterRegistryError("Evaluated registry candidate eligibility is missing.")
        purpose = advanced_fields["evaluation_purpose"]
        if purpose not in {"development", "promotion"}:
            raise InterpreterRegistryError("Evaluated registry candidate purpose is invalid.")
        if _digest_or_none(
            advanced_fields["evaluation_benchmark_sha256"], "evaluation_benchmark_sha256"
        ) is None:
            raise InterpreterRegistryError("Evaluated registry candidate benchmark identity is missing.")
        campaign_id = advanced_fields["promotion_campaign_id"]
        if purpose == "promotion":
            try:
                import uuid
                uuid.UUID(str(campaign_id))
            except (ValueError, AttributeError) as exc:
                raise InterpreterRegistryError("Promotion campaign ID is invalid.") from exc
            if _digest_or_none(
                advanced_fields["promotion_seal_sha256"], "promotion_seal_sha256"
            ) is None:
                raise InterpreterRegistryError("Promotion evaluation lacks its benchmark seal.")
        elif campaign_id is not None or advanced_fields["promotion_seal_sha256"] is not None:
            raise InterpreterRegistryError("Development evaluation cannot consume a promotion campaign ID.")
        if not isinstance(advanced_fields["evaluated_at_utc"], str) or not advanced_fields["evaluated_at_utc"]:
            raise InterpreterRegistryError("Evaluated registry candidate timestamp is missing.")
    else:
        for field, item in advanced_fields.items():
            if item is not None:
                raise InterpreterRegistryError(f"Pre-evaluation registry candidate unexpectedly defines {field}.")

    if state == "evaluated" and status == "candidate_adapter_experimental" and value.get("eligible_for_promotion") is not False:
        raise InterpreterRegistryError("An evaluated experimental candidate must remain non-promotable.")
    if state == "approved":
        if value.get("eligible_for_promotion") is not True:
            raise InterpreterRegistryError("Approved registry candidate was not promotion-eligible.")
        if not isinstance(value.get("approved_at_utc"), str) or not value["approved_at_utc"]:
            raise InterpreterRegistryError("Approved registry candidate timestamp is missing.")
    elif value.get("approved_at_utc") is not None:
        raise InterpreterRegistryError("Only approved registry candidates may have an approval timestamp.")

    computed = canonical_json_sha256(_without_sha(value, "entry_sha256"))
    if computed != value["entry_sha256"]:
        raise InterpreterRegistryError("Registry candidate entry checksum is invalid.")
    return dict(value)


def validate_registry(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InterpreterRegistryError("Interpreter model registry must be an object.")
    required = {"schema", "schema_version", "revision", "active", "entries", "history", "registry_sha256"}
    if set(value) != required or value.get("schema") != REGISTRY_SCHEMA or value.get("schema_version") != REGISTRY_VERSION:
        raise InterpreterRegistryError("Unsupported interpreter model registry schema.")
    revision = value.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise InterpreterRegistryError("Interpreter registry revision is invalid.")
    entries_raw = value.get("entries")
    if not isinstance(entries_raw, list):
        raise InterpreterRegistryError("Interpreter registry entries must be a list.")
    entries = [_validate_entry(item) for item in entries_raw]
    identifiers = [item["entry_id"] for item in entries]
    if len(identifiers) != len(set(identifiers)):
        raise InterpreterRegistryError("Interpreter registry entry ids must be unique.")
    candidate_ids = [item["candidate_id"] for item in entries if item["candidate_id"] is not None]
    tags = [item["candidate_tag"] for item in entries if item["candidate_tag"] is not None]
    if len(candidate_ids) != len(set(candidate_ids)) or len(tags) != len(set(tags)):
        raise InterpreterRegistryError("Evaluated candidate ids/tags must be unique.")
    promotion_benchmarks = [
        item["evaluation_benchmark_sha256"]
        for item in entries
        if item.get("evaluation_purpose") == "promotion"
    ]
    if len(promotion_benchmarks) != len(set(promotion_benchmarks)):
        raise InterpreterRegistryError(
            "A sequestered promotion benchmark may be consumed by only one candidate."
        )
    active = value.get("active")
    if not isinstance(active, Mapping) or set(active) != {"role", "entry_id"}:
        raise InterpreterRegistryError("Interpreter registry active selection is malformed.")
    role, entry_id = active.get("role"), active.get("entry_id")
    if role == "frozen_base":
        if entry_id is not None:
            raise InterpreterRegistryError("Frozen-base selection cannot name a candidate entry.")
    elif role == "approved_candidate":
        entry = next((item for item in entries if item["entry_id"] == entry_id), None)
        if entry is None or entry["lifecycle_state"] != "approved":
            raise InterpreterRegistryError("Active candidate is absent or not approved.")
    else:
        raise InterpreterRegistryError("Interpreter registry active role is invalid.")
    history = value.get("history")
    if not isinstance(history, list) or not all(isinstance(item, Mapping) for item in history):
        raise InterpreterRegistryError("Interpreter registry history is malformed.")
    declared = value.get("registry_sha256")
    computed = canonical_json_sha256(_without_sha(value, "registry_sha256"))
    if not isinstance(declared, str) or _SHA256.fullmatch(declared) is None or declared != computed:
        raise InterpreterRegistryError("Interpreter registry checksum is invalid.")
    return {
        **_without_sha(value, "registry_sha256"),
        "entries": entries,
        "active": dict(active),
        "history": [dict(item) for item in history],
        "registry_sha256": computed,
    }


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path).resolve() if path is not None else default_registry_path()
    if not target.exists():
        return empty_registry()
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InterpreterRegistryError(f"Cannot read interpreter model registry: {exc}") from exc
    return validate_registry(value)


def _atomic_registry(path: Path, registry: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(registry), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()


def _commit(
    registry: Mapping[str, Any],
    *,
    entries: list[Mapping[str, Any]] | None = None,
    active: Mapping[str, Any] | None = None,
    history_item: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "schema": REGISTRY_SCHEMA,
        "schema_version": REGISTRY_VERSION,
        "revision": int(registry["revision"]) + 1,
        "active": dict(active if active is not None else registry["active"]),
        "entries": [dict(item) for item in (entries if entries is not None else registry["entries"])],
        "history": [*[dict(item) for item in registry["history"]], dict(history_item)],
    }
    result["registry_sha256"] = canonical_json_sha256(result)
    return validate_registry(result)


def _entry_id(training_report_sha256: str) -> str:
    return "candidate-" + training_report_sha256[:20]


def _replace_entry(registry: Mapping[str, Any], entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = [dict(item) for item in registry["entries"] if item["entry_id"] != entry["entry_id"]]
    entries.append(dict(entry))
    entries.sort(key=lambda item: item["entry_id"])
    return entries


def register_training_candidate(
    *,
    registry_path: str | Path | None,
    adapter_directory: str | Path,
    base_model_directory: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    baseline_report: str | Path = DEFAULT_BASELINE_REPORT,
) -> dict[str, Any]:
    """Register both experimental and promotable training outputs as lifecycle records."""
    target = Path(registry_path).resolve() if registry_path is not None else default_registry_path()
    registry = load_registry(target)
    try:
        training = validate_training_report(
            adapter_directory,
            config_path=config_path,
            baseline_report=baseline_report,
            base_model_directory=base_model_directory,
            require_promotable=False,
        )
    except FineTuningError as exc:
        raise InterpreterRegistryError(str(exc)) from exc
    identifier = _entry_id(training["report_sha256"])
    existing = next((item for item in registry["entries"] if item["entry_id"] == identifier), None)
    if existing is not None and existing["lifecycle_state"] in {"evaluated", "approved"}:
        return registry
    state = "experimental" if training["status"] == "candidate_adapter_experimental" else "trained"
    created = existing["created_at_utc"] if existing is not None else _utc_now()
    entry: dict[str, Any] = {
        "entry_id": identifier,
        "lifecycle_state": state,
        "training_status": training["status"],
        "candidate_id": None,
        "candidate_tag": None,
        "observed_model_digest": None,
        "model_size_bytes": None,
        "quantization": None,
        "base_model": None,
        "base_snapshot_receipt_sha256": training["base_model"]["receipt_sha256"],
        "quantized_gguf_sha256": None,
        "training_report_sha256": training["report_sha256"],
        "export_receipt_sha256": None,
        "candidate_evaluation_report_sha256": None,
        "eligible_for_promotion": None,
        "evaluation_purpose": None,
        "promotion_campaign_id": None,
        "evaluation_benchmark_sha256": None,
        "promotion_seal_sha256": None,
        "created_at_utc": created,
        "evaluated_at_utc": None,
        "approved_at_utc": None,
    }
    entry["entry_sha256"] = canonical_json_sha256(entry)
    updated = _commit(
        registry,
        entries=_replace_entry(registry, entry),
        history_item={
            "event": "register_training_candidate",
            "at_utc": _utc_now(),
            "entry_id": identifier,
            "lifecycle_state": state,
            "training_report_sha256": training["report_sha256"],
            "entry_sha256": entry["entry_sha256"],
        },
    )
    _atomic_registry(target, updated)
    return updated


def register_evaluated_candidate(
    *,
    registry_path: str | Path | None,
    adapter_directory: str | Path,
    base_model_directory: str | Path,
    candidate_evaluation_report: str | Path,
    export_receipt: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    training_baseline_report: str | Path = DEFAULT_BASELINE_REPORT,
    evaluation_baseline_report: str | Path,
    benchmark_path: str | Path,
    promotion_seal: str | Path | None = None,
    promotion_ledger: str | Path | None = None,
) -> dict[str, Any]:
    """Register a completed evaluation, including failed/non-promotable evaluations."""
    target = Path(registry_path).resolve() if registry_path is not None else default_registry_path()
    registry = load_registry(target)
    try:
        training = validate_training_report(
            adapter_directory,
            config_path=config_path,
            baseline_report=training_baseline_report,
            base_model_directory=base_model_directory,
            require_promotable=False,
        )
        report = validate_candidate_evaluation_report(
            candidate_evaluation_report,
            benchmark_path=benchmark_path,
            baseline_report_path=evaluation_baseline_report,
            export_receipt_path=export_receipt,
            promotion_seal_path=promotion_seal,
            promotion_ledger_path=(
                promotion_ledger
                if promotion_ledger is not None
                else default_promotion_ledger_path()
            ),
        )
        receipt = validate_export_receipt(export_receipt, verify_artifacts=True)
    except (FineTuningError, CandidateEvaluationError, InterpreterExportError) as exc:
        raise InterpreterRegistryError(str(exc)) from exc
    candidate = report.get("candidate")
    if not isinstance(candidate, Mapping) or (
        candidate.get("training_report_sha256") != training["report_sha256"]
        or candidate.get("training_status") != training["status"]
        or receipt.get("training_report_sha256") != training["report_sha256"]
        or receipt.get("training_status") != training["status"]
        or candidate.get("export_receipt_sha256") != receipt.get("receipt_sha256")
        or candidate.get("candidate_id") != receipt.get("candidate_id")
        or candidate.get("candidate_tag") != receipt.get("candidate_tag")
        or candidate.get("quantized_gguf_sha256") != receipt["artifacts"]["quantized_gguf"]["sha256"]
    ):
        raise InterpreterRegistryError("Training, export, and evaluation identities do not form one candidate chain.")
    baseline_valid, baseline_reason, baseline = baseline_gate(
        evaluation_baseline_report, benchmark_path=benchmark_path
    )
    if not baseline_valid:
        # Experimental candidates may be evaluated specifically because the baseline gate was
        # bypassed at training time, but their evaluation still cannot claim promotion eligibility.
        if training["status"] != "candidate_adapter_experimental":
            raise InterpreterRegistryError(baseline_reason + ".")
    else:
        expected_comparison = compare_candidate_to_baseline(
            baseline,
            report["metrics"],
            training_status=training["status"],
            evaluation_purpose=report["evaluation"]["purpose"],
        )
        if expected_comparison != report["comparison"]:
            raise InterpreterRegistryError("Candidate evaluation comparison does not reproduce from baseline evidence.")
    if training["status"] == "candidate_adapter_experimental" and report["comparison"]["eligible_for_promotion"] is not False:
        raise InterpreterRegistryError("Experimental candidate evaluation incorrectly claims promotion eligibility.")

    identifier = _entry_id(training["report_sha256"])
    existing = next((item for item in registry["entries"] if item["entry_id"] == identifier), None)
    if existing is not None and existing["lifecycle_state"] == "approved":
        raise InterpreterRegistryError("An approved candidate lifecycle record cannot be downgraded.")
    created = existing["created_at_utc"] if existing is not None else _utc_now()
    runtime = report["runtime_identity"]
    evaluation = report["evaluation"]
    benchmark_sha = report["benchmark"]["benchmark_sha256"]
    if evaluation["purpose"] == "promotion":
        consumed = next(
            (
                item for item in registry["entries"]
                if item.get("evaluation_purpose") == "promotion"
                and item.get("evaluation_benchmark_sha256") == benchmark_sha
            ),
            None,
        )
        if consumed is not None and (
            consumed["entry_id"] != identifier
            or consumed["candidate_evaluation_report_sha256"] != report["report_sha256"]
        ):
            raise InterpreterRegistryError(
                "The sequestered promotion benchmark has already been consumed; create and review a new benchmark version before evaluating another candidate or retrying with changed evidence."
            )
    entry: dict[str, Any] = {
        "entry_id": identifier,
        "lifecycle_state": "evaluated",
        "training_status": training["status"],
        "candidate_id": receipt["candidate_id"],
        "candidate_tag": receipt["candidate_tag"],
        "observed_model_digest": runtime["observed_model_digest"],
        "model_size_bytes": runtime["model_size_bytes"],
        "quantization": LOCAL_QUANTIZATION,
        "base_model": LOCAL_BASE_MODEL,
        "base_snapshot_receipt_sha256": training["base_model"]["receipt_sha256"],
        "quantized_gguf_sha256": receipt["artifacts"]["quantized_gguf"]["sha256"],
        "training_report_sha256": training["report_sha256"],
        "export_receipt_sha256": receipt["receipt_sha256"],
        "candidate_evaluation_report_sha256": report["report_sha256"],
        "eligible_for_promotion": bool(report["comparison"]["eligible_for_promotion"]),
        "evaluation_purpose": evaluation["purpose"],
        "promotion_campaign_id": evaluation["promotion_campaign_id"],
        "evaluation_benchmark_sha256": benchmark_sha,
        "promotion_seal_sha256": evaluation.get("promotion_seal_sha256"),
        "created_at_utc": created,
        "evaluated_at_utc": _utc_now(),
        "approved_at_utc": None,
    }
    entry["entry_sha256"] = canonical_json_sha256(entry)
    updated = _commit(
        registry,
        entries=_replace_entry(registry, entry),
        history_item={
            "event": "register_evaluated_candidate",
            "at_utc": _utc_now(),
            "entry_id": identifier,
            "eligible_for_promotion": entry["eligible_for_promotion"],
            "evaluation_purpose": entry["evaluation_purpose"],
            "promotion_campaign_id": entry["promotion_campaign_id"],
            "evaluation_benchmark_sha256": entry["evaluation_benchmark_sha256"],
            "promotion_seal_sha256": entry["promotion_seal_sha256"],
            "candidate_evaluation_report_sha256": report["report_sha256"],
            "entry_sha256": entry["entry_sha256"],
        },
    )
    _atomic_registry(target, updated)
    return updated


def promote_candidate(
    *,
    registry_path: str | Path | None,
    candidate_evaluation_report: str | Path,
    export_receipt: str | Path,
    adapter_directory: str | Path,
    base_model_directory: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    training_baseline_report: str | Path = DEFAULT_BASELINE_REPORT,
    evaluation_baseline_report: str | Path,
    benchmark_path: str | Path,
    promotion_seal: str | Path,
    promotion_ledger: str | Path | None = None,
    expected_candidate_report_sha256: str,
    expected_registry_sha256: str,
    verify_runtime: bool = True,
) -> dict[str, Any]:
    """Atomically activate only an already-evaluated, non-experimental candidate."""
    target = Path(registry_path).resolve() if registry_path is not None else default_registry_path()
    registry = load_registry(target)
    if registry["registry_sha256"] != expected_registry_sha256:
        raise InterpreterRegistryError("Registry changed after review; reload before promotion.")
    try:
        # Final boundary independently reopens Stage-7 evidence.  Evaluation-report claims are
        # never sufficient to upgrade an experimental adapter into production.
        training = validate_training_report(
            adapter_directory,
            config_path=config_path,
            baseline_report=training_baseline_report,
            base_model_directory=base_model_directory,
            require_promotable=True,
        )
        report = validate_candidate_evaluation_report(
            candidate_evaluation_report,
            benchmark_path=benchmark_path,
            baseline_report_path=evaluation_baseline_report,
            export_receipt_path=export_receipt,
            promotion_seal_path=promotion_seal,
            promotion_ledger_path=(
                promotion_ledger
                if promotion_ledger is not None
                else default_promotion_ledger_path()
            ),
        )
        receipt = validate_export_receipt(export_receipt, verify_artifacts=True)
    except (FineTuningError, CandidateEvaluationError, InterpreterExportError) as exc:
        raise InterpreterRegistryError(str(exc)) from exc
    if training.get("status") != "candidate_adapter_unpromoted":
        raise InterpreterRegistryError(
            "Promotion requires the exact candidate_adapter_unpromoted training state."
        )
    if report["report_sha256"] != expected_candidate_report_sha256:
        raise InterpreterRegistryError("The explicitly confirmed candidate report hash does not match.")
    if report["comparison"]["eligible_for_promotion"] is not True:
        raise InterpreterRegistryError("Candidate failed one or more frozen promotion gates.")
    if report.get("evaluation", {}).get("purpose") != "promotion":
        raise InterpreterRegistryError(
            "A repeatable development evaluation can never authorize production promotion."
        )
    baseline_valid, baseline_reason, baseline = baseline_gate(
        evaluation_baseline_report, benchmark_path=benchmark_path
    )
    if not baseline_valid:
        raise InterpreterRegistryError(baseline_reason + ".")
    if (
        report.get("baseline", {}).get("report_sha256") != baseline.get("report_sha256")
        or compare_candidate_to_baseline(
            baseline,
            report["metrics"],
            training_status=training["status"],
            evaluation_purpose="promotion",
        ) != report["comparison"]
    ):
        raise InterpreterRegistryError(
            "Candidate promotion comparison does not reproduce against the valid baseline."
        )
    candidate = report.get("candidate")
    if not isinstance(candidate, Mapping) or (
        candidate.get("training_report_sha256") != training["report_sha256"]
        or candidate.get("training_status") != training["status"]
        or receipt.get("training_report_sha256") != training["report_sha256"]
        or receipt.get("training_status") != training["status"]
        or candidate.get("candidate_id") != receipt.get("candidate_id")
        or candidate.get("candidate_tag") != receipt.get("candidate_tag")
        or candidate.get("export_receipt_sha256") != receipt.get("receipt_sha256")
        or candidate.get("quantized_gguf_sha256") != receipt["artifacts"]["quantized_gguf"]["sha256"]
    ):
        raise InterpreterRegistryError("Training, export, and evaluation identities do not form one candidate chain.")

    identifier = _entry_id(training["report_sha256"])
    existing = next((item for item in registry["entries"] if item["entry_id"] == identifier), None)
    if existing is None or existing["lifecycle_state"] != "evaluated":
        raise InterpreterRegistryError(
            "Candidate must be registered in evaluated lifecycle state before promotion."
        )
    if (
        existing["candidate_evaluation_report_sha256"] != report["report_sha256"]
        or existing["export_receipt_sha256"] != receipt["receipt_sha256"]
        or existing["eligible_for_promotion"] is not True
    ):
        raise InterpreterRegistryError("Evaluated registry state does not match the promotion evidence.")

    if verify_runtime:
        try:
            identity = OllamaCandidateClient(receipt).identity()
        except CandidateEvaluationError as exc:
            raise InterpreterRegistryError(str(exc)) from exc
        digest, size = identity.model_digest, identity.model_size_bytes
    else:
        digest = str(receipt["ollama"]["model_digest"])
        size = int(receipt["ollama"]["model_size_bytes"])
    if digest != existing["observed_model_digest"] or size != existing["model_size_bytes"]:
        raise InterpreterRegistryError("Current runtime identity differs from the evaluated registry identity.")

    entry = deepcopy(existing)
    entry["lifecycle_state"] = "approved"
    entry["approved_at_utc"] = _utc_now()
    entry.pop("entry_sha256", None)
    entry["entry_sha256"] = canonical_json_sha256(entry)
    previous = dict(registry["active"])
    active = {"role": "approved_candidate", "entry_id": identifier}
    updated = _commit(
        registry,
        entries=_replace_entry(registry, entry),
        active=active,
        history_item={
            "event": "promote",
            "at_utc": _utc_now(),
            "from": previous,
            "to": active,
            "training_report_sha256": training["report_sha256"],
            "candidate_evaluation_report_sha256": report["report_sha256"],
            "entry_sha256": entry["entry_sha256"],
        },
    )
    _atomic_registry(target, updated)
    return updated


def rollback_interpreter_model(
    *,
    registry_path: str | Path | None,
    target_entry_id: str | None,
    expected_registry_sha256: str,
) -> dict[str, Any]:
    """Atomically switch to a prior approved candidate, or to the frozen base model."""
    target = Path(registry_path).resolve() if registry_path is not None else default_registry_path()
    registry = load_registry(target)
    if registry["registry_sha256"] != expected_registry_sha256:
        raise InterpreterRegistryError("Registry changed after review; reload before rollback.")
    if target_entry_id is None or target_entry_id == "frozen_base":
        active = _base_active()
        entry_sha = None
    else:
        entry = next(
            (
                item for item in registry["entries"]
                if item["entry_id"] == target_entry_id and item["lifecycle_state"] == "approved"
            ),
            None,
        )
        if entry is None:
            raise InterpreterRegistryError("Rollback target is not a previously approved candidate.")
        active = {"role": "approved_candidate", "entry_id": target_entry_id}
        entry_sha = entry["entry_sha256"]
    if active == registry["active"]:
        raise InterpreterRegistryError("Requested rollback target is already active.")
    updated = _commit(
        registry,
        active=active,
        history_item={
            "event": "rollback",
            "at_utc": _utc_now(),
            "from": dict(registry["active"]),
            "to": active,
            "entry_sha256": entry_sha,
        },
    )
    _atomic_registry(target, updated)
    return updated


def _frozen_base_runtime_lock() -> dict[str, str]:
    path = Path(__file__).with_name("baseline") / "base_model_lock_v1.1.json"
    raw = path.read_bytes()
    lock = json.loads(raw)
    ollama = lock.get("ollama") if isinstance(lock, Mapping) else None
    if not isinstance(ollama, Mapping):
        raise InterpreterRegistryError("Bundled frozen-base model lock is malformed.")
    manifest_sha256 = ollama.get("registry_manifest_sha256")
    layers = ollama.get("layers")
    model_layer = next(
        (
            item for item in layers
            if isinstance(item, Mapping)
            and item.get("media_type") == "application/vnd.ollama.image.model"
        ),
        None,
    ) if isinstance(layers, list) else None
    model_blob_sha256 = model_layer.get("sha256") if isinstance(model_layer, Mapping) else None
    if (
        not isinstance(manifest_sha256, str)
        or _SHA256.fullmatch(manifest_sha256) is None
        or not isinstance(model_blob_sha256, str)
        or _SHA256.fullmatch(model_blob_sha256) is None
    ):
        raise InterpreterRegistryError("Bundled frozen-base artifact identity is malformed.")
    return {
        "base_model_lock_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_manifest_sha256": manifest_sha256,
        "expected_model_blob_sha256": model_blob_sha256,
    }


def runtime_selection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact native-runtime selection from one validated registry."""
    registry = validate_registry(value)
    if registry["active"]["role"] == "frozen_base":
        lock = _frozen_base_runtime_lock()
        return {
            "model_role": "frozen_base",
            "model_tag": LOCAL_MODEL,
            "expected_digest": None,
            "model_size_bytes": None,
            "base_model_lock_sha256": lock["base_model_lock_sha256"],
            "expected_manifest_sha256": lock["expected_manifest_sha256"],
            "expected_model_blob_sha256": lock["expected_model_blob_sha256"],
            "registry_entry_sha256": None,
            "training_report_sha256": None,
            "candidate_evaluation_report_sha256": None,
            "candidate_lifecycle_state": None,
            "identity_verification": "exact_local_manifest_and_all_blobs_sha256",
        }
    entry = next(
        item for item in registry["entries"]
        if item["entry_id"] == registry["active"]["entry_id"]
    )
    if entry["lifecycle_state"] != "approved":
        raise InterpreterRegistryError("Only approved registry entries may be selected at runtime.")
    return {
        "model_role": "approved_candidate",
        "model_tag": entry["candidate_tag"],
        "expected_digest": entry["observed_model_digest"],
        "model_size_bytes": entry["model_size_bytes"],
        "expected_model_blob_sha256": entry["quantized_gguf_sha256"],
        "registry_entry_sha256": entry["entry_sha256"],
        "training_report_sha256": entry["training_report_sha256"],
        "candidate_evaluation_report_sha256": entry["candidate_evaluation_report_sha256"],
        "candidate_lifecycle_state": entry["lifecycle_state"],
        "identity_verification": "approved_registry_exact_manifest_and_model_blob_sha256",
    }
