#!/usr/bin/env python3
"""Freeze the complete Stage-7 Python/CUDA environment before empirical training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_lab.interpreter_finetuning import (  # noqa: E402
    FineTuningError,
    freeze_training_environment_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hash and freeze the complete QLoRA environment before training."
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "training" / "interpreter_training_environment_v1.2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "training" / "interpreter_training_environment_receipt.json",
    )
    args = parser.parse_args()
    try:
        receipt = freeze_training_environment_receipt(args.policy, args.output)
    except FineTuningError as exc:
        print(f"Environment freeze blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
