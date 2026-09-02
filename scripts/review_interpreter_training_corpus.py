#!/usr/bin/env python3
"""Inspect and record genuine human decisions for the frozen Stage-6 review queue."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_lab.interpreter_finetuning import (
    FineTuningError,
    update_review_manifest,
    validate_review_records,
)
from model_lab.interpreter_training import load_jsonl


def _write_jsonl_atomic(path: Path, records: list[dict[str, object]]) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        record,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode("utf-8")
                    + b"\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()


def _summary(records: list[dict[str, object]]) -> dict[str, int]:
    result = {"pending-human-review": 0, "accepted": 0, "corrected": 0, "rejected": 0}
    for record in records:
        status = record["review"]["status"]  # type: ignore[index]
        result[str(status)] += 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect or update one Stage-6 review decision.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "training" / "interpreter_corpus_v1.5",
    )
    parser.add_argument("--id", help="Exact review-record ID.")
    parser.add_argument("--next", action="store_true", help="Show the first pending review record.")
    parser.add_argument(
        "--decision", choices=("accepted", "corrected", "rejected"), default=None
    )
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--corrected-target", type=Path, default=None)
    parser.add_argument(
        "--replace", action="store_true", help="Permit replacement of an existing completed decision."
    )
    args = parser.parse_args()
    queue_path = args.corpus / "review_queue.jsonl"
    try:
        records = [dict(item) for item in load_jsonl(queue_path)]
        if args.id is None and not args.next:
            print(json.dumps(_summary(records), indent=2))
            return 0
        selected = None
        if args.next:
            selected = next(
                (item for item in records if item["review"]["status"] == "pending-human-review"),  # type: ignore[index]
                None,
            )
        else:
            selected = next((item for item in records if item.get("id") == args.id), None)
        if selected is None:
            raise FineTuningError("No matching review record was found.")
        if args.decision is None:
            print(json.dumps(selected, indent=2, ensure_ascii=False))
            return 0
        if not args.reviewer.strip():
            raise FineTuningError("A completed decision requires --reviewer.")
        if selected["review"]["status"] != "pending-human-review" and not args.replace:  # type: ignore[index]
            raise FineTuningError("This record was already reviewed; use --replace deliberately.")
        if args.decision in {"corrected", "rejected"} and not args.notes.strip():
            raise FineTuningError("Corrected and rejected decisions require explanatory --notes.")
        corrected_target = None
        if args.decision == "corrected":
            if args.corrected_target is None:
                raise FineTuningError("A corrected decision requires --corrected-target JSON.")
            corrected_target = json.loads(args.corrected_target.read_text(encoding="utf-8"))
            if not isinstance(corrected_target, dict):
                raise FineTuningError("The corrected target must be one JSON object.")
        elif args.corrected_target is not None:
            raise FineTuningError("Only a corrected decision may include --corrected-target.")
        selected["review"] = {
            "status": args.decision,
            "reviewer": args.reviewer.strip(),
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "notes": args.notes,
            "corrected_target": corrected_target,
        }
        validate_review_records(args.corpus, records)
        manifest_path = args.corpus / "manifest.json"
        original_queue = queue_path.read_bytes()
        original_manifest = manifest_path.read_bytes()
        try:
            _write_jsonl_atomic(queue_path, records)
            manifest = update_review_manifest(args.corpus)
        except Exception:
            _write_bytes_atomic(queue_path, original_queue)
            _write_bytes_atomic(manifest_path, original_manifest)
            raise
        print(json.dumps(_summary(records), indent=2))
        print(f"Recorded {args.decision} for {selected['id']}; review status: {manifest['review']['status']}")
        return 0
    except (FineTuningError, OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"Review update failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
