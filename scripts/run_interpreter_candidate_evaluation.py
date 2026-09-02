from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_lab.interpreter_candidate import (
    CandidateEvaluationError,
    finalize_candidate_evaluation_report,
    run_candidate_evaluation,
)
from model_lab.interpreter_registry import (
    InterpreterRegistryError,
    register_evaluated_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the complete frozen Stage-5 benchmark against one exact exported adapter candidate."
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "training" / "interpreter_qlora_v1.7.json")
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Explicit development or external private-promotion benchmark.",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        required=True,
        help="Untouched-base report produced against --benchmark.",
    )
    parser.add_argument(
        "--training-baseline-report",
        type=Path,
        default=ROOT / "verification" / "interpreter_baseline_report.json",
        help="Development baseline bound into the QLoRA training report.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--purpose", choices=("development", "promotion"), default="development",
        help="Development runs are repeatable and never promotable; promotion consumes the sequestered benchmark once.",
    )
    parser.add_argument(
        "--promotion-campaign-id",
        default=None,
        help="Required UUID for an explicitly authorized one-shot promotion evaluation.",
    )
    parser.add_argument(
        "--promotion-seal",
        type=Path,
        default=None,
        help="Required for promotion: candidate-bound private benchmark seal.",
    )
    parser.add_argument(
        "--promotion-ledger",
        type=Path,
        default=None,
        help="Optional ledger location; the platform application-data default is used otherwise.",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint or args.output.with_name(args.output.name + ".checkpoint.json")
    if args.restart:
        checkpoint.unlink(missing_ok=True)
    started = time.monotonic()

    def progress(index: int, total: int, result: object) -> None:
        elapsed = time.monotonic() - started
        average = elapsed / max(1, index)
        remaining = average * max(0, total - index)
        case_id = result.get("id", "unknown") if isinstance(result, dict) else "unknown"
        status = result.get("terminal_status", "unknown") if isinstance(result, dict) else "unknown"
        print(
            f"[{index}/{total}] {case_id}: {status}; elapsed {elapsed / 60:.1f} min; "
            f"estimated remaining {remaining / 60:.1f} min",
            flush=True,
        )

    try:
        report = run_candidate_evaluation(
            benchmark_path=args.benchmark,
            baseline_report_path=args.baseline_report,
            training_baseline_report_path=args.training_baseline_report,
            config_path=args.config,
            adapter_directory=args.adapter,
            base_model_directory=args.base_model,
            export_receipt_path=args.export_receipt,
            checkpoint_path=checkpoint,
            resume=not args.restart,
            progress=progress,
            evaluation_purpose=args.purpose,
            promotion_campaign_id=args.promotion_campaign_id,
            promotion_seal_path=args.promotion_seal,
            promotion_ledger_path=args.promotion_ledger,
        )
    except CandidateEvaluationError as exc:
        print(f"Candidate evaluation blocked: {exc}", file=sys.stderr)
        return 2
    try:
        finalize_candidate_evaluation_report(
            report,
            args.output,
            promotion_ledger_path=args.promotion_ledger,
        )
    except CandidateEvaluationError as exc:
        print(f"Candidate report finalization blocked: {exc}", file=sys.stderr)
        return 2
    try:
        register_evaluated_candidate(
            registry_path=args.registry,
            adapter_directory=args.adapter,
            base_model_directory=args.base_model,
            candidate_evaluation_report=args.output,
            export_receipt=args.export_receipt,
            config_path=args.config,
            training_baseline_report=args.training_baseline_report,
            evaluation_baseline_report=args.baseline_report,
            benchmark_path=args.benchmark,
            promotion_seal=args.promotion_seal,
            promotion_ledger=args.promotion_ledger,
        )
    except InterpreterRegistryError as exc:
        print(f"Evaluation completed but lifecycle registration was blocked: {exc}", file=sys.stderr)
        return 2
    checkpoint.unlink(missing_ok=True)
    print(json.dumps(report["comparison"], indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")
    if report["evaluation"]["purpose"] == "development":
        return 0
    return 0 if report["comparison"]["eligible_for_promotion"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
