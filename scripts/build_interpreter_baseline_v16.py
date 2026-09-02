#!/usr/bin/env python3
"""Add capability-boundary cases to the v1.5 held-out benchmark."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "verification" / "interpreter_baseline_v1.5.json"
TARGET = ROOT / "verification" / "interpreter_baseline_v1.6.json"

REQUESTS = (
    ("unsupported-sde", "Construct a stochastic differential equation with multiplicative noise."),
    ("unsupported-neural-training", "Train a neural network on these labelled observations."),
    ("unsupported-gnn", "Create a graph neural network for node classification."),
    ("unsupported-cnn", "Build a convolutional neural network for image classification."),
    ("unsupported-neural-controller", "Build a neural controller for an ODE system."),
    ("unsupported-truss-optimisation", "Optimise a truss for minimum mass while limiting displacement."),
    ("unsupported-ode-estimation", "Estimate the parameters of this ODE from observations."),
    ("unsupported-reaction-fit", "Fit reaction-network rate constants to concentration measurements."),
    ("unsupported-mesh-pde", "Solve Laplace's equation on an unstructured triangle mesh."),
    ("unsupported-circuit-fit", "Fit an RLC circuit model to voltage measurements."),
)


def main() -> None:
    benchmark = json.loads(SOURCE.read_text(encoding="utf-8"))
    additions = [
        {
            "id": case_id,
            "family": "scientific-capability-boundary",
            "instruction": instruction,
            "current_model_source": "",
            "expected": {"terminal_status": "unable", "first_action": "unable"},
            "notes": (
                "Fail-closed boundary case: nearby installed schemas do not provide the "
                "requested specialised or solver-coupled capability."
            ),
        }
        for case_id, instruction in REQUESTS
    ]
    existing_ids = {case["id"] for case in benchmark["cases"]}
    collisions = existing_ids.intersection(case["id"] for case in additions)
    if collisions:
        raise ValueError(f"v1.6 benchmark IDs collide: {sorted(collisions)}")
    benchmark["schema_version"] = "1.6"
    benchmark["cases"].extend(additions)
    if len(benchmark["cases"]) != 150:
        raise ValueError("The v1.6 development benchmark must contain exactly 150 cases")
    benchmark["title"] = "Model Laboratory v1.16.1 capability-boundary AI-interpreter benchmark"
    benchmark["review"] = {
        "status": "internal-curated",
        "provenance": (
            "The 140 v1.5 cases are retained and ten deterministic capability-boundary cases are added. "
            "The benchmark is compiler-checked but is not claimed to be independently expert-reviewed."
        ),
    }
    TARGET.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
