from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_lab.interpreter_baseline import (
    BaselineInfrastructureError, load_benchmark, run_baseline, validation_report
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the untouched Qwen baseline against an explicitly selected evaluation benchmark."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "verification" / "interpreter_baseline_v1.6.json",
        help="Evaluation benchmark JSON. The default shipped file is development-only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "verification" / "interpreter_baseline_report.json",
        help="Machine-readable report destination.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional deterministic prefix for smoke runs.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path. Defaults to <output>.checkpoint.json for live runs.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard any existing live-run checkpoint and start from case 1.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate benchmark/prompt assets without invoking Ollama.",
    )
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark)
    checkpoint = args.checkpoint
    if checkpoint is None and not args.validate_only:
        checkpoint = args.output.with_name(args.output.name + ".checkpoint.json")
    if args.restart and checkpoint is not None:
        checkpoint.unlink(missing_ok=True)

    if args.validate_only:
        report = validation_report(benchmark)
        exit_code = 0
    else:
        try:
            started = time.monotonic()
            resumed_count = 0
            if checkpoint is not None and checkpoint.exists() and not args.restart:
                try:
                    checkpoint_value = json.loads(checkpoint.read_text(encoding="utf-8"))
                    resumed_count = int(checkpoint_value.get("completed_count", 0))
                except Exception:
                    resumed_count = 0

            def show_progress(index: int, total: int, result: object) -> None:
                newly_completed = max(1, index - resumed_count)
                elapsed = time.monotonic() - started
                average = elapsed / newly_completed
                remaining = average * max(0, total - index)
                case_id = result.get("id", "unknown") if isinstance(result, dict) else "unknown"
                status = result.get("terminal_status", "unknown") if isinstance(result, dict) else "unknown"
                print(
                    f"[{index}/{total}] {case_id}: {status}; "
                    f"elapsed {elapsed / 60:.1f} min; estimated remaining {remaining / 60:.1f} min",
                    flush=True,
                )

            report = run_baseline(
                benchmark,
                limit=args.limit,
                checkpoint_path=checkpoint,
                resume=not args.restart,
                progress=show_progress,
            )
            exit_code = 0
        except BaselineInfrastructureError as exc:
            report = validation_report(benchmark)
            report["status"] = "campaign_invalid_infrastructure"
            report["reason"] = str(exc)
            if checkpoint is not None and checkpoint.exists():
                report["checkpoint"] = str(checkpoint)
            exit_code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if exit_code == 0 and checkpoint is not None:
        checkpoint.unlink(missing_ok=True)
    print(json.dumps(report.get("metrics", report), indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
