#!/usr/bin/env python3
"""Seal one private promotion benchmark to one already-frozen candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_lab.interpreter_promotion import (  # noqa: E402
    PromotionEvidenceError,
    create_promotion_seal,
    write_promotion_seal,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bind a private promotion benchmark to one frozen candidate and campaign."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        seal = create_promotion_seal(
            benchmark_path=args.benchmark,
            baseline_report_path=args.baseline_report,
            export_receipt_path=args.export_receipt,
            campaign_id=args.campaign_id,
            reviewer=args.reviewer,
            provenance=args.provenance,
        )
    except PromotionEvidenceError as exc:
        print(f"Promotion seal blocked: {exc}", file=sys.stderr)
        return 2
    write_promotion_seal(seal, args.output)
    print(json.dumps(seal, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
