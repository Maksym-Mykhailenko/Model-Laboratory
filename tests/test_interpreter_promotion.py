from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_lab.canonical import canonical_json_sha256
from model_lab.interpreter_candidate import (
    CandidateEvaluationError,
    finalize_candidate_evaluation_report,
    run_candidate_evaluation,
)
from model_lab.interpreter_promotion import (
    PromotionEvidenceError,
    complete_promotion_benchmark,
    reserve_promotion_benchmark,
    validate_promotion_consumption,
)


def _seal(*, campaign: str, benchmark: str, candidate: str) -> dict:
    return {
        "campaign_id": campaign,
        "seal_sha256": "a" * 64,
        "benchmark": {"benchmark_sha256": benchmark},
        "candidate": {"candidate_freeze_sha256": candidate},
    }


def test_promotion_ledger_is_resumable_but_one_shot(tmp_path: Path) -> None:
    path = tmp_path / "promotion-ledger.json"
    seal = _seal(
        campaign="7b79d7c3-1482-4bf7-a61e-c999fa9eb7cf",
        benchmark="b" * 64,
        candidate="c" * 64,
    )
    reserved = reserve_promotion_benchmark(path, seal)
    assert reserve_promotion_benchmark(path, seal) == reserved

    with pytest.raises(PromotionEvidenceError, match="already been reserved"):
        reserve_promotion_benchmark(
            path,
            _seal(
                campaign="1e6c695d-b4b8-481f-b8a0-a5c0c913f652",
                benchmark="b" * 64,
                candidate="d" * 64,
            ),
        )

    completed = complete_promotion_benchmark(
        path,
        reservation_sha256=reserved["reservation_sha256"],
        candidate_report_sha256="e" * 64,
    )
    assert completed["status"] == "completed"
    assert validate_promotion_consumption(
        path,
        reservation_sha256=reserved["reservation_sha256"],
        candidate_report_sha256="e" * 64,
    )["campaign_id"] == seal["campaign_id"]
    with pytest.raises(PromotionEvidenceError, match="already been consumed"):
        reserve_promotion_benchmark(path, seal)


def test_promotion_ledger_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "promotion-ledger.json"
    reserve_promotion_benchmark(
        path,
        _seal(
            campaign="7b79d7c3-1482-4bf7-a61e-c999fa9eb7cf",
            benchmark="b" * 64,
            candidate="c" * 64,
        ),
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["reservations"][0]["candidate_freeze_sha256"] = "f" * 64
    # Even recomputing only the outer checksum cannot forge the reservation identity.
    value["ledger_sha256"] = canonical_json_sha256(
        {key: item for key, item in value.items() if key != "ledger_sha256"}
    )
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(PromotionEvidenceError, match="reservation checksum"):
        reserve_promotion_benchmark(
            path,
            _seal(
                campaign="7b79d7c3-1482-4bf7-a61e-c999fa9eb7cf",
                benchmark="b" * 64,
                candidate="c" * 64,
            ),
        )


def test_candidate_evaluation_has_no_shared_benchmark_default(tmp_path: Path) -> None:
    with pytest.raises(CandidateEvaluationError, match="explicit benchmark"):
        run_candidate_evaluation(
            adapter_directory=tmp_path / "adapter",
            base_model_directory=tmp_path / "base",
            export_receipt_path=tmp_path / "export.json",
        )


def test_report_is_durable_before_promotion_completion(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    reservation = reserve_promotion_benchmark(
        ledger,
        _seal(
            campaign="7b79d7c3-1482-4bf7-a61e-c999fa9eb7cf",
            benchmark="b" * 64,
            candidate="c" * 64,
        ),
    )
    report = {
        "evaluation": {
            "purpose": "promotion",
            "promotion_reservation_sha256": reservation["reservation_sha256"],
        }
    }
    report["report_sha256"] = canonical_json_sha256(report)
    output = tmp_path / "candidate-report.json"
    finalize_candidate_evaluation_report(
        report, output, promotion_ledger_path=ledger
    )
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert validate_promotion_consumption(
        ledger,
        reservation_sha256=reservation["reservation_sha256"],
        candidate_report_sha256=report["report_sha256"],
    )["status"] == "completed"
    # Retrying finalization after a process interruption is idempotent for the same report.
    finalize_candidate_evaluation_report(
        report, output, promotion_ledger_path=ledger
    )
