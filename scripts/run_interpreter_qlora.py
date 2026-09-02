#!/usr/bin/env python3
"""Preflight or explicitly start the gated Stage-7 QLoRA candidate training run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Hash randomisation is fixed before any project or ML module is imported. Setting this from
# inside an already-running interpreter would not alter that process's hash seed.
if os.environ.get("PYTHONHASHSEED") != "0":
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_lab.interpreter_finetuning import (
    DEFAULT_BASELINE_REPORT,
    DEFAULT_CONFIG,
    FineTuningError,
    preflight_report,
    run_qlora,
)
from model_lab.interpreter_registry import (
    InterpreterRegistryError,
    register_training_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Stage-7 gates or train one unpromoted QLoRA adapter candidate."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument(
        "--preflight-output",
        type=Path,
        default=ROOT / "verification" / "interpreter_qlora_preflight.json",
    )
    parser.add_argument(
        "--shallow-corpus",
        action="store_true",
        help="Fast diagnostic only. A shallow preflight is deliberately never READY for training.",
    )
    parser.add_argument(
        "--corpus-workers",
        type=int,
        default=None,
        help="Worker count for the deep 6,300-record compiler replay (use 1 to minimise RAM).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Start training. Without this flag the command performs only a non-mutating preflight.",
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-validation-examples", type=int, default=None)
    parser.add_argument(
        "--experimental-allow-unreviewed",
        action="store_true",
        help="Explicitly mark a test candidate experimental and bypass the human-review gate.",
    )
    parser.add_argument(
        "--experimental-allow-missing-baseline",
        action="store_true",
        help="Explicitly mark a test candidate experimental and bypass the completed-baseline gate.",
    )
    args = parser.parse_args()
    if args.corpus_workers is not None and args.corpus_workers < 1:
        parser.error("--corpus-workers must be a positive integer")

    try:
        preflight = preflight_report(
            args.config,
            baseline_report=args.baseline_report,
            deep_corpus=not args.shallow_corpus,
            corpus_workers=args.corpus_workers,
            allow_unreviewed=args.experimental_allow_unreviewed,
            allow_missing_baseline=args.experimental_allow_missing_baseline,
        )
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_output.write_text(
            json.dumps(preflight, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        print(f"Wrote {args.preflight_output}")
        if not args.train:
            return 0 if preflight["status"] == "READY" else 2
        if args.model_dir is None or args.output_dir is None:
            parser.error("--train requires both --model-dir and --output-dir")
        if (
            (args.max_train_examples is not None or args.max_validation_examples is not None)
            and not (
                args.experimental_allow_unreviewed
                or args.experimental_allow_missing_baseline
            )
        ):
            parser.error("example limits are smoke-test controls and require an experimental override")
        report = run_qlora(
            config_path=args.config,
            baseline_report=args.baseline_report,
            model_directory=args.model_dir,
            output_directory=args.output_dir,
            allow_unreviewed=args.experimental_allow_unreviewed,
            allow_missing_baseline=args.experimental_allow_missing_baseline,
            maximum_train_examples=args.max_train_examples,
            maximum_validation_examples=args.max_validation_examples,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        registry = register_training_candidate(
            registry_path=args.registry,
            adapter_directory=args.output_dir,
            base_model_directory=args.model_dir,
            config_path=args.config,
            baseline_report=args.baseline_report,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Candidate adapter written to {args.output_dir}")
        print(
            f"Registered lifecycle state {next(item['lifecycle_state'] for item in registry['entries'] if item['training_report_sha256'] == report['report_sha256'])}."
        )
        return 0
    except (FineTuningError, InterpreterRegistryError) as exc:
        print(f"Stage-7 QLoRA gate failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
