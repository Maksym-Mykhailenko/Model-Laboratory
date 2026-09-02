#!/usr/bin/env python3
"""Build the v1.5 development benchmark directly from the v1.3 full-release benchmark.

The builder deliberately does not depend on the superseded v1.4 update artifact, so the
result can be reproduced from an untouched Model Laboratory v1.16.1 tree.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "verification" / "interpreter_baseline_v1.3.json"
TARGET = ROOT / "verification" / "interpreter_baseline_v1.5.json"

REPRESENTATIVES = {
    "official-multidimensional": ("official-scientific-quantity", "value", 18.75, "standard_uncertainty"),
    "official-probability": ("official-discrete-distribution", "probabilities", [0.2, 0.5, 0.3], "probabilities"),
    "official-graphs": ("official-graph-network", "edges", [{"source":"ingest","target":"check","weight":4.0},{"source":"check","target":"publish","weight":1.5}], "directed"),
    "official-generative": ("official-hmm", "initial", [0.6, 0.4], "emission"),
    "official-dynamics": ("official-ode-system", "initial_state", [1.25, 0.0], "time_span"),
    "official-fields": ("official-diffusion-pde", "diffusivity", 0.045, "dirichlet_boundary"),
    "official-geometry": ("official-point-cloud", "points", [[0,0,0],[3,0,0],[0,4,0],[0,0,5]], "coordinate_unit"),
    "official-mechanics": ("official-elastic-material", "youngs_modulus", 72.0, "poissons_ratio"),
    "official-statistics": ("official-linear-model", "response", [2.0,4.1,6.0,8.2,10.0], "include_intercept"),
    "official-optimisation": ("official-nonlinear-optimisation", "initial", [0.5, -0.5], "bounds"),
    "official-electrical": ("official-diode", "temperature_kelvin", 300.0, "ideality_factor"),
    "official-reactions": ("official-compartment", "initial_amounts", [75.0, 0.0], "transfers"),
    "official-learning": ("official-supervised-study", "features", [[-3,0],[-1,2],[0,-2],[2,0],[3,1],[4,3]], "targets"),
}

PHRASES = {
    "value":"value", "standard_uncertainty":"standard uncertainty", "probabilities":"probabilities",
    "edges":"weighted edges", "directed":"choice between a directed and an undirected graph",
    "emission":"emission probabilities", "initial_state":"initial state vector", "time_span":"time interval",
    "diffusivity":"diffusivity", "dirichlet_boundary":"Dirichlet boundary values", "points":"point coordinates",
    "coordinate_unit":"coordinate unit", "youngs_modulus":"Young's modulus", "poissons_ratio":"Poisson ratio",
    "response":"response observations", "include_intercept":"intercept choice", "bounds":"variable bounds",
    "temperature_kelvin":"temperature", "ideality_factor":"ideality factor", "initial_amounts":"initial amounts",
    "transfers":"transfer rates", "features":"feature rows", "targets":"target labels",
}

INITIAL_PHRASES = {
    "official-hmm": "initial hidden-state probabilities",
    "official-nonlinear-optimisation": "initial decision-variable guess",
}


def _dump(document: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(document), sort_keys=False, allow_unicode=True, width=1000)


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _phrase(base_id: str, field: str) -> str:
    if field == "initial":
        return INITIAL_PHRASES.get(base_id, "initial values")
    return PHRASES.get(field, field.replace("_", " "))


def _properties_prose(base_id: str, properties: Mapping[str, Any], *, omit: str | None = None) -> str:
    parts=[]
    for key, value in properties.items():
        if key == omit:
            continue
        parts.append(f"{_phrase(base_id, key)} {_value(value)}")
    return "; ".join(parts)


def _one_object(case: Mapping[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    doc=yaml.safe_load(case["expected"]["model_source"])
    objects=doc.get("objects", {})
    if len(objects) != 1:
        raise ValueError(f"Representative {case['id']} must contain exactly one object")
    object_id=next(iter(objects))
    return doc, object_id, objects[object_id]


def _edit_case(base: Mapping[str, Any], family: str, field: str, new_value: Any) -> dict[str, Any]:
    source_doc, object_id, obj=_one_object(base)
    expected_doc=copy.deepcopy(source_doc)
    expected_doc["objects"][object_id]["properties"][field]=copy.deepcopy(new_value)
    phrase=_phrase(str(base["id"]), field)
    kind_tail=str(obj["kind"]).rsplit(".",1)[-1].replace("-"," ")
    return {
        "id": f"{base['id']}-edit",
        "family": family,
        "instruction": (
            f"The model already contains {object_id}, a {kind_tail}. Preserve every unrelated setting and "
            f"replace its {phrase} with {_value(new_value)}."
        ),
        "current_model_source": _dump(source_doc),
        "expected": {"terminal_status":"proposal","first_action":"propose_edits","model_source":_dump(expected_doc)},
        "notes": "Official-pack edit coverage; excluded from all interpreter training corpora.",
    }


def _clarification_case(base: Mapping[str, Any], family: str, missing_field: str) -> dict[str, Any]:
    doc, object_id, obj=_one_object(base)
    properties=dict(obj["properties"])
    missing=properties[missing_field]
    phrase=_phrase(str(base["id"]), missing_field)
    kind_tail=str(obj["kind"]).rsplit(".",1)[-1].replace("-"," ")
    instruction=(
        f"Create a {kind_tail} object named {object_id} in a model named {doc['name']}. "
        f"Use {_properties_prose(str(base['id']), properties, omit=missing_field)}. "
        f"I have not specified the {phrase}; ask me for it rather than guessing."
    )
    keyword = "directed" if missing_field == "directed" else phrase.split()[0].casefold()
    answer = "Make the graph directed." if missing_field == "directed" else f"Use {_value(missing)} for the {phrase}."
    return {
        "id": f"{base['id']}-clarify",
        "family": family,
        "instruction": instruction,
        "current_model_source": "",
        "expected": {
            "terminal_status":"needs_clarification",
            "first_action":"needs_clarification",
            "clarification_keywords":[keyword],
            "clarification_followup": {
                "answer": answer,
                "terminal_status":"proposal",
                "model_source": _dump(doc),
            },
        },
        "notes": "Official-pack clarification and continuation coverage; excluded from all interpreter training corpora.",
    }


def main() -> None:
    benchmark=json.loads(SOURCE.read_text(encoding="utf-8"))
    benchmark["schema_version"]="1.5"
    by_id={case["id"]:case for case in benchmark["cases"]}
    additions=[]
    for family,(case_id,edit_field,new_value,missing_field) in REPRESENTATIVES.items():
        base=by_id[case_id]
        additions.append(_edit_case(base,family,edit_field,new_value))
        additions.append(_clarification_case(base,family,missing_field))
    ids={case["id"] for case in benchmark["cases"]}
    if ids.intersection(case["id"] for case in additions):
        raise ValueError("v1.5 benchmark IDs collide with existing cases")
    benchmark["cases"].extend(additions)
    if len(benchmark["cases"]) != 140:
        raise ValueError("The v1.5 development benchmark must contain exactly 140 cases")
    benchmark["title"]="Model Laboratory v1.16.1 corrected AI-interpreter held-out development benchmark"
    benchmark["review"]={
        "status":"internal-curated",
        "provenance":(
            "Original 114 cases retained; 13 official-pack edit and 13 clarification-continuation cases added, "
            "with kind-specific scientific wording and deterministic validation. Not claimed to be independently expert-reviewed."
        ),
    }
    TARGET.write_text(json.dumps(benchmark,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
