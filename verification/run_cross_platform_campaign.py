"""Produce or merge real cross-platform reproduction observations.

Each runner reproduces the archived ``.mlab`` fixtures and writes one checksummed
observation.  The merge mode never manufactures missing rows; it creates the publishable
compatibility matrix only from downloaded runner artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_lab.canonical import canonical_json_sha256
from model_lab.bundle import load_mlab_bundle
from model_lab.compatibility import (
    CompatibilityCaseOutcome,
    CompatibilityObservation,
    CompatibilityTarget,
    merge_compatibility_observations,
)
from model_lab.experiment import current_environment
from model_lab.reproduction import ReproductionStatus, reproduce_mlab_bundle


FIXTURES = tuple(sorted((ROOT / "tests" / "fixtures").glob("*.mlab")))


def produce_observation() -> CompatibilityObservation:
    environment = current_environment()
    environment_map = dict(environment)
    target = CompatibilityTarget(
        environment_map["operating_system"],
        environment_map["cpu_architecture"],
        environment_map["python"],
    )
    cases: list[CompatibilityCaseOutcome] = []
    for path in FIXTURES:
        bundle = load_mlab_bundle(path.read_bytes())
        report = reproduce_mlab_bundle(bundle).report
        cases.append(
            CompatibilityCaseOutcome(
                name=path.name,
                experiment_id=bundle.experiment_id,
                status=report.status,
                environment_status=report.environment_status.value,
                report_sha256=canonical_json_sha256(report.to_dict()),
            )
        )
    return CompatibilityObservation(target, environment, tuple(cases))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="write one runner observation")
    parser.add_argument("--merge", type=Path, nargs="+", help="merge runner observations")
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-markdown", type=Path)
    args = parser.parse_args()
    if bool(args.output) == bool(args.merge):
        parser.error("choose exactly one of --output or --merge")

    if args.output:
        observation = produce_observation()
        args.output.write_text(observation.to_json(), encoding="utf-8")
        print(f"recorded {observation.target.key}: {len(observation.cases)} cases")
        return 0 if observation.successful else 1

    if args.report_json is None or args.report_markdown is None:
        parser.error("--merge requires --report-json and --report-markdown")
    observations = [
        CompatibilityObservation.from_json(path.read_text(encoding="utf-8"))
        for path in args.merge
    ]
    report = merge_compatibility_observations(observations)
    args.report_json.write_text(report.to_json(), encoding="utf-8")
    args.report_markdown.write_text(report.to_markdown(), encoding="utf-8")
    print(
        f"merged {len(observations)} targets; "
        f"missing={len(report.missing_targets)}; backends={report.backend_count}"
    )
    return 0 if report.campaign_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
