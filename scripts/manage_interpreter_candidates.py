from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_lab.interpreter_registry import (
    InterpreterRegistryError,
    default_registry_path,
    load_registry,
    promote_candidate,
    register_evaluated_candidate,
    register_training_candidate,
    rollback_interpreter_model,
    runtime_selection,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and manage checksum-bound interpreter candidate lifecycle states."
    )
    parser.add_argument("--registry", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")

    register_training = subparsers.add_parser("register-training")
    register_training.add_argument("--adapter", type=Path, required=True)
    register_training.add_argument("--base-model", type=Path, required=True)
    register_training.add_argument(
        "--config", type=Path, default=ROOT / "training" / "interpreter_qlora_v1.7.json"
    )
    register_training.add_argument(
        "--baseline-report",
        type=Path,
        default=ROOT / "verification" / "interpreter_baseline_report.json",
    )

    register_evaluated = subparsers.add_parser("register-evaluated")
    register_evaluated.add_argument("--adapter", type=Path, required=True)
    register_evaluated.add_argument("--base-model", type=Path, required=True)
    register_evaluated.add_argument("--candidate-report", type=Path, required=True)
    register_evaluated.add_argument("--export-receipt", type=Path, required=True)
    register_evaluated.add_argument(
        "--config", type=Path, default=ROOT / "training" / "interpreter_qlora_v1.7.json"
    )
    register_evaluated.add_argument(
        "--baseline-report",
        type=Path,
        default=ROOT / "verification" / "interpreter_baseline_report.json",
    )
    register_evaluated.add_argument("--benchmark", type=Path, required=True)
    register_evaluated.add_argument(
        "--evaluation-baseline-report", type=Path, required=True
    )
    register_evaluated.add_argument("--promotion-seal", type=Path, default=None)
    register_evaluated.add_argument("--promotion-ledger", type=Path, default=None)

    promote = subparsers.add_parser("promote")
    promote.add_argument("--adapter", type=Path, required=True)
    promote.add_argument("--base-model", type=Path, required=True)
    promote.add_argument("--candidate-report", type=Path, required=True)
    promote.add_argument("--export-receipt", type=Path, required=True)
    promote.add_argument(
        "--config", type=Path, default=ROOT / "training" / "interpreter_qlora_v1.7.json"
    )
    promote.add_argument(
        "--baseline-report",
        type=Path,
        default=ROOT / "verification" / "interpreter_baseline_report.json",
    )
    promote.add_argument("--benchmark", type=Path, required=True)
    promote.add_argument("--evaluation-baseline-report", type=Path, required=True)
    promote.add_argument("--promotion-seal", type=Path, required=True)
    promote.add_argument("--promotion-ledger", type=Path, default=None)
    promote.add_argument("--confirm-report-sha256", required=True)
    promote.add_argument("--confirm-registry-sha256", required=True)

    rollback = subparsers.add_parser("rollback")
    rollback.add_argument(
        "--target",
        required=True,
        help="Previously approved candidate registry entry ID, or frozen_base.",
    )
    rollback.add_argument("--confirm-registry-sha256", required=True)
    args = parser.parse_args()
    path = args.registry.resolve() if args.registry is not None else default_registry_path()
    try:
        if args.command == "status":
            registry = load_registry(path)
        elif args.command == "register-training":
            registry = register_training_candidate(
                registry_path=path,
                adapter_directory=args.adapter,
                base_model_directory=args.base_model,
                config_path=args.config,
                baseline_report=args.baseline_report,
            )
        elif args.command == "register-evaluated":
            registry = register_evaluated_candidate(
                registry_path=path,
                adapter_directory=args.adapter,
                base_model_directory=args.base_model,
                candidate_evaluation_report=args.candidate_report,
                export_receipt=args.export_receipt,
                config_path=args.config,
                training_baseline_report=args.baseline_report,
                evaluation_baseline_report=args.evaluation_baseline_report,
                benchmark_path=args.benchmark,
                promotion_seal=args.promotion_seal,
                promotion_ledger=args.promotion_ledger,
            )
        elif args.command == "promote":
            registry = promote_candidate(
                registry_path=path,
                candidate_evaluation_report=args.candidate_report,
                export_receipt=args.export_receipt,
                adapter_directory=args.adapter,
                base_model_directory=args.base_model,
                config_path=args.config,
                training_baseline_report=args.baseline_report,
                evaluation_baseline_report=args.evaluation_baseline_report,
                benchmark_path=args.benchmark,
                promotion_seal=args.promotion_seal,
                promotion_ledger=args.promotion_ledger,
                expected_candidate_report_sha256=args.confirm_report_sha256,
                expected_registry_sha256=args.confirm_registry_sha256,
            )
        else:
            registry = rollback_interpreter_model(
                registry_path=path,
                target_entry_id=args.target,
                expected_registry_sha256=args.confirm_registry_sha256,
            )
    except InterpreterRegistryError as exc:
        print(f"Registry operation blocked: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "path": str(path),
                "registry": registry,
                "runtime_selection": runtime_selection(registry),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
