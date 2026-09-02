from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_lab.interpreter_export import (
    InterpreterExportError,
    create_export_plan,
    export_interpreter_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge, GGUF-convert, quantize, and import one unpromoted interpreter candidate."
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "training" / "interpreter_qlora_v1.7.json")
    parser.add_argument("--baseline-report", type=Path, default=ROOT / "verification" / "interpreter_baseline_report.json")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--convert-hf-to-gguf", type=Path, required=True)
    parser.add_argument("--llama-quantize", type=Path, required=True)
    parser.add_argument("--ollama", default="ollama")
    parser.add_argument("--merge-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Verify all existing inputs and print the plan without loading model weights.",
    )
    args = parser.parse_args()
    try:
        common = {
            "adapter_directory": args.adapter,
            "base_model_directory": args.base_model,
            "output_directory": args.output,
            "candidate_tag": args.tag,
            "convert_hf_to_gguf": args.convert_hf_to_gguf,
            "llama_quantize": args.llama_quantize,
            "config_path": args.config,
            "baseline_report": args.baseline_report,
        }
        result = (
            create_export_plan(**common)
            if args.plan_only
            else export_interpreter_candidate(
                **common,
                ollama_executable=args.ollama,
                merge_device=args.merge_device,
            )
        )
    except InterpreterExportError as exc:
        print(f"Export blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
