#!/usr/bin/env python3
"""Reject a draft, stale, mismatched, incomplete or non-reproducing flagship case."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_lab import __version__
from model_lab.bundle import load_mlab_bundle
from model_lab.reproduction import ReproductionStatus, reproduce_run_experiment
from scripts.build_flagship_case import CASE_FILES


def verify_case(root: Path = ROOT) -> dict[str, str]:
    source_path, bundle_path, summary_path, _, _ = (root / name for name in CASE_FILES)
    data = bundle_path.read_bytes()
    bundle = load_mlab_bundle(data)
    if not bundle.author_approved_for_publication:
        raise ValueError("Flagship bundle is a draft; a validated frozen author review is required.")
    if bundle.state.laboratory_version != __version__:
        raise ValueError("Flagship reference was not created by the current release.")
    if bundle.state.model_source != source_path.read_text(encoding="utf-8"):
        raise ValueError("Flagship model source differs from its bundle.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "laboratory_version": __version__,
        "experiment_id": bundle.experiment_id,
        "experiment_state_sha256": bundle.state.state_sha256,
        "artifact_id": bundle.state.artifacts[0]["artifact_id"],
        "model_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "bundle_sha256": hashlib.sha256(data).hexdigest(),
        "author_approved_for_publication": True,
        "review_sha256": bundle.authoring_document["review_sha256"],
        "relative_tolerance": bundle.state.numerical_reproduction_settings.relative_tolerance,
        "absolute_tolerance": bundle.state.numerical_reproduction_settings.absolute_tolerance,
    }
    if summary["reference"] != expected:
        raise ValueError("Reference summary does not identify the frozen bundle exactly.")
    if summary["scientific_source"]["doi"] != "10.2139/ssrn.5853487":
        raise ValueError("Flagship scientific citation is missing or incorrect.")
    names = set()
    manifest = root / "examples/gaussian-hierarchy-dispersion-SHA256SUMS.txt"
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        if name not in CASE_FILES or name in names:
            raise ValueError("Case checksum manifest has an unexpected or duplicate file.")
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Case checksum mismatch: {name}")
        names.add(name)
    if names != set(CASE_FILES):
        raise ValueError("Case checksum manifest is incomplete.")
    reproduction = reproduce_run_experiment(bundle.state, model=bundle.model)
    if reproduction.report.status not in (ReproductionStatus.EXACT, ReproductionStatus.NUMERICAL):
        raise ValueError(f"Flagship failed reproduction: {reproduction.report.status.value}")
    return {"version": __version__, "authoring_status": "FROZEN",
            "review_sha256": expected["review_sha256"],
            "reproduction": reproduction.report.status.value}


if __name__ == "__main__":
    try:
        print(json.dumps(verify_case(), indent=2))
    except (KeyError, OSError, ValueError) as error:
        print(f"Flagship verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
