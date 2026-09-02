"""Candidate-bound promotion benchmark sealing and one-shot consumption evidence."""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import tempfile
from typing import Any, Mapping
import uuid

from .canonical import canonical_json_sha256
from .interpreter_baseline import _source_identity, baseline_contract, load_benchmark
from .interpreter_export import InterpreterExportError, validate_export_receipt
from .interpreter_finetuning import FineTuningError, baseline_gate


PROMOTION_SEAL_SCHEMA = "model-laboratory-interpreter-promotion-benchmark-seal"
PROMOTION_SEAL_VERSION = "1.0"
PROMOTION_LEDGER_SCHEMA = "model-laboratory-interpreter-promotion-consumption-ledger"
PROMOTION_LEDGER_VERSION = "1.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PromotionEvidenceError(RuntimeError):
    """Promotion benchmark authorization, sealing, or consumption is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as stream:
            json.dump(dict(value), stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()


@contextmanager
def _ledger_lock(path: Path):
    """Serialize ledger mutations across desktop/CLI processes."""
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def default_promotion_ledger_path() -> Path:
    configured = os.environ.get("MODEL_LAB_PROMOTION_LEDGER", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    system = platform.system()
    if system == "Windows":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif system == "Darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "org.modellaboratory.desktop" / "promotion-consumption-ledger.json"


def _candidate_freeze(receipt: Mapping[str, Any]) -> dict[str, Any]:
    candidate = {
        "candidate_id": receipt.get("candidate_id"),
        "candidate_tag": receipt.get("candidate_tag"),
        "training_status": receipt.get("training_status"),
        "training_report_sha256": receipt.get("training_report_sha256"),
        "export_receipt_sha256": receipt.get("receipt_sha256"),
        "quantized_gguf_sha256": receipt.get("artifacts", {}).get(
            "quantized_gguf", {}
        ).get("sha256") if isinstance(receipt.get("artifacts"), Mapping) else None,
    }
    if (
        not isinstance(candidate["candidate_id"], str)
        or not candidate["candidate_id"].startswith("candidate-")
        or not isinstance(candidate["candidate_tag"], str)
        or candidate["training_status"] != "candidate_adapter_unpromoted"
        or any(
            not isinstance(candidate[field], str)
            or _SHA256.fullmatch(candidate[field]) is None
            for field in (
                "training_report_sha256", "export_receipt_sha256",
                "quantized_gguf_sha256",
            )
        )
    ):
        raise PromotionEvidenceError(
            "Promotion seal requires one fully exported, non-experimental frozen candidate."
        )
    candidate["candidate_freeze_sha256"] = canonical_json_sha256(candidate)
    return candidate


def create_promotion_seal(
    *,
    benchmark_path: str | Path,
    baseline_report_path: str | Path,
    export_receipt_path: str | Path,
    campaign_id: str,
    reviewer: str,
    provenance: str,
) -> dict[str, Any]:
    """Seal a private benchmark only after a concrete candidate has been frozen/exported."""
    try:
        campaign = str(uuid.UUID(campaign_id))
    except (ValueError, AttributeError) as exc:
        raise PromotionEvidenceError("Promotion campaign ID must be a UUID.") from exc
    if not reviewer.strip() or not provenance.strip():
        raise PromotionEvidenceError("Promotion benchmark reviewer and provenance are required.")
    benchmark_file = Path(benchmark_path).resolve()
    baseline_file = Path(baseline_report_path).resolve()
    try:
        benchmark = load_benchmark(benchmark_file)
        valid, reason, baseline = baseline_gate(
            baseline_file, benchmark_path=benchmark_file
        )
        receipt = validate_export_receipt(export_receipt_path, verify_artifacts=True)
    except (OSError, ValueError, FineTuningError, InterpreterExportError) as exc:
        raise PromotionEvidenceError(str(exc)) from exc
    if not valid:
        raise PromotionEvidenceError(reason + ".")
    if benchmark["review"].get("status") != "independent-expert-reviewed":
        raise PromotionEvidenceError(
            "Promotion benchmark must be independently expert-reviewed before sealing."
        )
    candidate = _candidate_freeze(receipt)
    seal: dict[str, Any] = {
        "schema": PROMOTION_SEAL_SCHEMA,
        "schema_version": PROMOTION_SEAL_VERSION,
        "campaign_id": campaign,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "authorization": {
            "status": "independent-reviewed-private-promotion-benchmark",
            "reviewer": reviewer.strip(),
            "provenance": provenance.strip(),
        },
        "benchmark": {
            "benchmark_sha256": benchmark["benchmark_sha256"],
            "schema": benchmark["schema"],
            "schema_version": benchmark["schema_version"],
            "case_count": len(benchmark["cases"]),
            "family_counts": benchmark["family_counts"],
            "review": benchmark["review"],
            "training_exclusion": True,
        },
        "baseline": {
            "report_file_sha256": _sha256_file(baseline_file),
            "report_sha256": baseline["report_sha256"],
        },
        "candidate": candidate,
        "evaluator": {
            "baseline_contract_sha256": canonical_json_sha256(baseline_contract()),
            "implementation_identity": _source_identity(),
        },
        "consumption_policy": "one benchmark, one frozen candidate, one campaign",
    }
    seal["seal_sha256"] = canonical_json_sha256(seal)
    return seal


def write_promotion_seal(seal: Mapping[str, Any], path: str | Path) -> None:
    declared = seal.get("seal_sha256")
    without_sha = {key: item for key, item in seal.items() if key != "seal_sha256"}
    if (
        not isinstance(declared, str)
        or _SHA256.fullmatch(declared) is None
        or declared != canonical_json_sha256(without_sha)
    ):
        raise PromotionEvidenceError("Promotion seal checksum is invalid.")
    _atomic_json(Path(path).resolve(), seal)


def validate_promotion_seal(
    value: str | Path | Mapping[str, Any],
    *,
    benchmark_path: str | Path,
    baseline_report_path: str | Path,
    export_receipt_path: str | Path,
) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        try:
            raw = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PromotionEvidenceError(f"Cannot read promotion seal: {exc}") from exc
    else:
        raw = dict(value)
    required = {
        "schema", "schema_version", "campaign_id", "created_at_utc", "authorization",
        "benchmark", "baseline", "candidate", "evaluator", "consumption_policy",
        "seal_sha256",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != required
        or raw.get("schema") != PROMOTION_SEAL_SCHEMA
        or raw.get("schema_version") != PROMOTION_SEAL_VERSION
    ):
        raise PromotionEvidenceError("Unsupported or malformed promotion seal.")
    declared = raw.get("seal_sha256")
    without_sha = {key: item for key, item in raw.items() if key != "seal_sha256"}
    if (
        not isinstance(declared, str)
        or _SHA256.fullmatch(declared) is None
        or declared != canonical_json_sha256(without_sha)
    ):
        raise PromotionEvidenceError("Promotion seal checksum is invalid.")
    try:
        uuid.UUID(str(raw.get("campaign_id")))
        created = datetime.fromisoformat(str(raw.get("created_at_utc")))
        if created.tzinfo is None:
            raise ValueError("timezone missing")
        benchmark = load_benchmark(benchmark_path)
        valid, reason, baseline = baseline_gate(
            baseline_report_path, benchmark_path=benchmark_path
        )
        receipt = validate_export_receipt(export_receipt_path, verify_artifacts=True)
    except (OSError, ValueError, FineTuningError, InterpreterExportError) as exc:
        raise PromotionEvidenceError(str(exc)) from exc
    if not valid:
        raise PromotionEvidenceError(reason + ".")
    if benchmark["review"].get("status") != "independent-expert-reviewed":
        raise PromotionEvidenceError(
            "Promotion benchmark is not independently expert-reviewed."
        )
    expected_candidate = _candidate_freeze(receipt)
    try:
        exported_at = datetime.fromisoformat(str(receipt.get("created_at_utc")))
    except (TypeError, ValueError) as exc:
        raise PromotionEvidenceError("Candidate export timestamp is malformed.") from exc
    if exported_at.tzinfo is None or created < exported_at:
        raise PromotionEvidenceError(
            "Promotion benchmark seal must be created after the candidate export is frozen."
        )
    expected_benchmark = {
        "benchmark_sha256": benchmark["benchmark_sha256"],
        "schema": benchmark["schema"],
        "schema_version": benchmark["schema_version"],
        "case_count": len(benchmark["cases"]),
        "family_counts": benchmark["family_counts"],
        "review": benchmark["review"],
        "training_exclusion": True,
    }
    baseline_path = Path(baseline_report_path).resolve()
    if (
        raw.get("benchmark") != expected_benchmark
        or raw.get("baseline") != {
            "report_file_sha256": _sha256_file(baseline_path),
            "report_sha256": baseline["report_sha256"],
        }
        or raw.get("candidate") != expected_candidate
        or raw.get("evaluator") != {
            "baseline_contract_sha256": canonical_json_sha256(baseline_contract()),
            "implementation_identity": _source_identity(),
        }
        or raw.get("consumption_policy")
        != "one benchmark, one frozen candidate, one campaign"
    ):
        raise PromotionEvidenceError(
            "Promotion seal differs from the supplied benchmark, baseline, candidate, or evaluator."
        )
    authorization = raw.get("authorization")
    if (
        not isinstance(authorization, Mapping)
        or set(authorization) != {"status", "reviewer", "provenance"}
        or authorization.get("status")
        != "independent-reviewed-private-promotion-benchmark"
        or not isinstance(authorization.get("reviewer"), str)
        or not authorization["reviewer"].strip()
        or not isinstance(authorization.get("provenance"), str)
        or not authorization["provenance"].strip()
    ):
        raise PromotionEvidenceError("Promotion seal authorization is incomplete.")
    return dict(raw)


def _empty_ledger() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": PROMOTION_LEDGER_SCHEMA,
        "schema_version": PROMOTION_LEDGER_VERSION,
        "revision": 0,
        "reservations": [],
    }
    value["ledger_sha256"] = canonical_json_sha256(value)
    return value


def _load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_ledger()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PromotionEvidenceError(f"Cannot read promotion consumption ledger: {exc}") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != {
            "schema", "schema_version", "revision", "reservations", "ledger_sha256"
        }
        or value.get("schema") != PROMOTION_LEDGER_SCHEMA
        or value.get("schema_version") != PROMOTION_LEDGER_VERSION
    ):
        raise PromotionEvidenceError("Promotion consumption ledger is malformed.")
    if (
        isinstance(value.get("revision"), bool)
        or not isinstance(value.get("revision"), int)
        or value["revision"] < 0
        or not isinstance(value.get("ledger_sha256"), str)
        or _SHA256.fullmatch(value["ledger_sha256"]) is None
    ):
        raise PromotionEvidenceError("Promotion consumption ledger identity is invalid.")
    declared = value.get("ledger_sha256")
    without_sha = {key: item for key, item in value.items() if key != "ledger_sha256"}
    if declared != canonical_json_sha256(without_sha):
        raise PromotionEvidenceError("Promotion consumption ledger checksum is invalid.")
    reservations = value.get("reservations")
    if not isinstance(reservations, list):
        raise PromotionEvidenceError("Promotion consumption reservations are malformed.")
    seen_campaigns: set[str] = set()
    seen_benchmarks: set[str] = set()
    for item in reservations:
        if not isinstance(item, Mapping) or set(item) != {
            "campaign_id", "benchmark_sha256", "seal_sha256", "candidate_freeze_sha256",
            "reserved_at_utc", "status", "completed_at_utc", "candidate_report_sha256",
            "reservation_sha256",
        }:
            raise PromotionEvidenceError("Promotion reservation is malformed.")
        campaign = str(item.get("campaign_id"))
        benchmark = str(item.get("benchmark_sha256"))
        try:
            uuid.UUID(campaign)
        except ValueError as exc:
            raise PromotionEvidenceError("Promotion reservation campaign ID is invalid.") from exc
        for field in (
            "benchmark_sha256", "seal_sha256", "candidate_freeze_sha256",
            "reservation_sha256",
        ):
            if (
                not isinstance(item.get(field), str)
                or _SHA256.fullmatch(item[field]) is None
            ):
                raise PromotionEvidenceError(f"Promotion reservation {field} is invalid.")
        if campaign in seen_campaigns or benchmark in seen_benchmarks:
            raise PromotionEvidenceError("Promotion ledger reuses a campaign or benchmark.")
        seen_campaigns.add(campaign)
        seen_benchmarks.add(benchmark)
        fixed = {
            key: item[key]
            for key in (
                "campaign_id", "benchmark_sha256", "seal_sha256",
                "candidate_freeze_sha256", "reserved_at_utc",
            )
        }
        if item.get("reservation_sha256") != canonical_json_sha256(fixed):
            raise PromotionEvidenceError("Promotion reservation checksum is invalid.")
        try:
            reserved_at = datetime.fromisoformat(str(item.get("reserved_at_utc")))
            completed_at = (
                datetime.fromisoformat(str(item.get("completed_at_utc")))
                if item.get("completed_at_utc") is not None
                else None
            )
        except (TypeError, ValueError) as exc:
            raise PromotionEvidenceError("Promotion reservation timestamp is invalid.") from exc
        if reserved_at.tzinfo is None or (
            completed_at is not None
            and (completed_at.tzinfo is None or completed_at < reserved_at)
        ):
            raise PromotionEvidenceError("Promotion reservation timestamp order is invalid.")
        if item.get("status") not in {"reserved", "completed"}:
            raise PromotionEvidenceError("Promotion reservation status is invalid.")
        completed = item.get("status") == "completed"
        if completed is not (
            isinstance(item.get("completed_at_utc"), str)
            and isinstance(item.get("candidate_report_sha256"), str)
            and _SHA256.fullmatch(item["candidate_report_sha256"]) is not None
        ):
            raise PromotionEvidenceError("Promotion completion evidence is inconsistent.")
    return dict(value)


def reserve_promotion_benchmark(
    ledger_path: str | Path,
    seal: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(ledger_path).resolve()
    with _ledger_lock(path):
        ledger = _load_ledger(path)
        benchmark_sha = str(seal["benchmark"]["benchmark_sha256"])
        campaign_id = str(seal["campaign_id"])
        candidate_freeze = str(seal["candidate"]["candidate_freeze_sha256"])
        existing = next(
            (
                item for item in ledger["reservations"]
                if item["benchmark_sha256"] == benchmark_sha
                or item["campaign_id"] == campaign_id
            ),
            None,
        )
        if existing is not None:
            if (
                existing["campaign_id"] != campaign_id
                or existing["benchmark_sha256"] != benchmark_sha
                or existing["seal_sha256"] != seal["seal_sha256"]
                or existing["candidate_freeze_sha256"] != candidate_freeze
            ):
                raise PromotionEvidenceError(
                    "Promotion benchmark or campaign has already been reserved for another candidate."
                )
            if existing["status"] == "completed":
                raise PromotionEvidenceError(
                    "Promotion benchmark has already been consumed by a completed candidate evaluation."
                )
            return dict(existing)
        fixed = {
            "campaign_id": campaign_id,
            "benchmark_sha256": benchmark_sha,
            "seal_sha256": seal["seal_sha256"],
            "candidate_freeze_sha256": candidate_freeze,
            "reserved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        reservation = {
            **fixed,
            "status": "reserved",
            "completed_at_utc": None,
            "candidate_report_sha256": None,
            "reservation_sha256": canonical_json_sha256(fixed),
        }
        updated = {
            "schema": PROMOTION_LEDGER_SCHEMA,
            "schema_version": PROMOTION_LEDGER_VERSION,
            "revision": int(ledger["revision"]) + 1,
            "reservations": [*ledger["reservations"], reservation],
        }
        updated["ledger_sha256"] = canonical_json_sha256(updated)
        _atomic_json(path, updated)
        return reservation


def complete_promotion_benchmark(
    ledger_path: str | Path,
    *,
    reservation_sha256: str,
    candidate_report_sha256: str,
) -> dict[str, Any]:
    if _SHA256.fullmatch(candidate_report_sha256) is None:
        raise PromotionEvidenceError("Candidate report SHA-256 is malformed.")
    path = Path(ledger_path).resolve()
    with _ledger_lock(path):
        ledger = _load_ledger(path)
        reservations = [dict(item) for item in ledger["reservations"]]
        reservation = next(
            (item for item in reservations if item["reservation_sha256"] == reservation_sha256),
            None,
        )
        if reservation is None:
            raise PromotionEvidenceError("Promotion reservation is absent.")
        if reservation["status"] == "completed":
            if reservation["candidate_report_sha256"] == candidate_report_sha256:
                return reservation
            raise PromotionEvidenceError(
                "Promotion reservation was completed by a different candidate report."
            )
        reservation["status"] = "completed"
        reservation["completed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        reservation["candidate_report_sha256"] = candidate_report_sha256
        updated = {
            "schema": PROMOTION_LEDGER_SCHEMA,
            "schema_version": PROMOTION_LEDGER_VERSION,
            "revision": int(ledger["revision"]) + 1,
            "reservations": reservations,
        }
        updated["ledger_sha256"] = canonical_json_sha256(updated)
        _atomic_json(path, updated)
        return reservation


def validate_promotion_consumption(
    ledger_path: str | Path,
    *,
    reservation_sha256: str,
    candidate_report_sha256: str,
) -> dict[str, Any]:
    ledger = _load_ledger(Path(ledger_path).resolve())
    item = next(
        (
            reservation for reservation in ledger["reservations"]
            if reservation["reservation_sha256"] == reservation_sha256
        ),
        None,
    )
    if (
        item is None
        or item["status"] != "completed"
        or item["candidate_report_sha256"] != candidate_report_sha256
    ):
        raise PromotionEvidenceError(
            "Promotion ledger does not contain the completed one-shot evaluation."
        )
    return dict(item)
