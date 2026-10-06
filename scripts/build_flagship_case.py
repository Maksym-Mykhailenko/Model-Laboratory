#!/usr/bin/env python3
"""Prepare a review, then freeze that explicitly selected review through the desktop API."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import desktop_engine
from model_lab.bundle import load_mlab_bundle

CASE_FILES = (
    "models/gaussian-hierarchy-dispersion.yaml",
    "examples/gaussian-hierarchy-dispersion.mlab",
    "examples/gaussian-hierarchy-dispersion-reference-results.json",
    "docs/assets/gaussian-hierarchy-dispersion.png",
    "docs/cases/gaussian-hierarchy-dispersion.md",
)


def write_case(bundle_data: bytes) -> None:
    bundle = load_mlab_bundle(bundle_data)
    if not bundle.author_approved_for_publication:
        raise ValueError("The desktop API did not produce a frozen author review.")
    source_path, bundle_path, summary_path, _, _ = (ROOT / name for name in CASE_FILES)
    if bundle.state.model_source != source_path.read_text(encoding="utf-8"):
        raise ValueError("The reviewed source no longer matches the case model.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifact = bundle.state.artifacts[0]
    outputs = artifact["data"]["outputs"]
    summary["reference"] = {
        "laboratory_version": bundle.state.laboratory_version,
        "experiment_id": bundle.experiment_id,
        "experiment_state_sha256": bundle.state.state_sha256,
        "artifact_id": artifact["artifact_id"],
        "model_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "bundle_sha256": hashlib.sha256(bundle_data).hexdigest(),
        "author_approved_for_publication": True,
        "review_sha256": bundle.authoring_document["review_sha256"],
        "relative_tolerance": bundle.state.numerical_reproduction_settings.relative_tolerance,
        "absolute_tolerance": bundle.state.numerical_reproduction_settings.absolute_tolerance,
    }
    output_names = {
        "relaxation_rates": "relaxation rates", "minimum_rate": "minimum rate",
        "geometric_mean_rate": "geometric mean", "maximum_rate": "maximum rate",
        "condition_number": "condition number", "log_spectral_dispersion": "dispersion",
        "t_0_90": "t0.90", "t_0_10": "t0.10",
        "recovery_breadth": "recovery breadth",
        "effective_dimension_at_t_0_4": "effective dimension at t=0.4",
    }
    for profile in ("A", "B"):
        for field, label in output_names.items():
            summary["profiles"][profile][field] = outputs[f"Profile {profile} {label}"]["value"]
        values = summary["profiles"][profile]
        values["recovery_ratio"] = values["t_0_10"] / values["t_0_90"]
    bundle_path.write_bytes(bundle_data)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f"{hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}  {name}" for name in CASE_FILES]
    (ROOT / "examples/gaussian-hierarchy-dispersion-SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n", encoding="ascii", newline="\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", type=Path, help="Save the current reference state and full canonical review.")
    actions.add_argument("--freeze", type=Path, help="Read the previously prepared state and review.")
    parser.add_argument("--approved-review-sha256", help="Exact digest of the reviewed state to freeze.")
    args = parser.parse_args()
    if args.prepare:
        prepared = desktop_engine.dispatch({"action": "prepare_run_experiment", "payload": {
            "source": (ROOT / CASE_FILES[0]).read_text(encoding="utf-8"),
            "runs": [{"capability_id": "org.modellab.composition.run-analysis-recipe",
                      "settings": {"object_id": "matched-relaxation-profiles"}}],
            "reproduction_settings": {"relative_tolerance": 1e-8, "absolute_tolerance": 1e-11},
        }})
        # The full review is retained, rather than approving a newly generated state later.
        prepared.pop("draft_bundle_base64", None)
        args.prepare.write_text(json.dumps(prepared, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Prepared canonical review {prepared['review_sha256']} in {args.prepare}")
    else:
        if not args.approved_review_sha256:
            parser.error("--freeze requires --approved-review-sha256")
        prepared = json.loads(args.freeze.read_text(encoding="utf-8"))
        if args.approved_review_sha256 != prepared["review_sha256"]:
            parser.error("Approved digest does not match the saved review.")
        final = desktop_engine.dispatch({"action": "finalize_experiment", "payload": {
            "state_json": prepared["state_json"],
            "approved_review_sha256": args.approved_review_sha256,
        }})
        write_case(base64.b64decode(final["bundle_base64"], validate=True))
        print(f"Frozen flagship case: {final['review_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
