from __future__ import annotations

import base64
import json
from pathlib import Path
import subprocess
import sys

import pytest

import desktop_engine
from model_lab import __version__
from model_lab.interpreter import (
    FROZEN_BASE_ARTIFACT_LOCK_SHA256,
    FROZEN_BASE_IDENTITY_VERIFICATION,
    GENERATION_OPTIONS,
    INTERPRETER_OUTPUT_SCHEMA,
    INTERPRETER_OUTPUT_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]


def _source(name: str = "quadratic.yaml") -> str:
    return (ROOT / "models" / name).read_text(encoding="utf-8")


def _interpreter_provider() -> dict[str, object]:
    return {
        "provider": "ollama",
        "endpoint": "http://127.0.0.1:11434",
        "runtime_version": "0.11.4",
        "model_tag": "qwen3:4b-instruct-2507-q4_K_M",
        "observed_model_digest": "sha256:0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0",
        "identity_verification": FROZEN_BASE_IDENTITY_VERIFICATION,
        "expected_base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "expected_quantization": "Q4_K_M",
        "generation_options": {
            **GENERATION_OPTIONS,
        },
        "model_role": "frozen_base",
        "registry_entry_sha256": None,
        "artifact_evidence": {
            "identity_verification": FROZEN_BASE_IDENTITY_VERIFICATION,
            "artifact_lock_sha256": FROZEN_BASE_ARTIFACT_LOCK_SHA256,
            "local_manifest_sha256": "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0",
            "model_blob_sha256": "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9",
            "verified_model_blob_size_bytes": 2497280480,
            "all_manifest_blobs_verified": True,
        },
    }


def _compiled_rename(current: str, name: str, instruction: str):
    prepared = desktop_engine.dispatch(
        {
            "action": "prepare_interpreter_context",
            "payload": {
                "instruction": instruction,
                "current_model_source": current,
            },
        }
    )
    context = prepared["context"]
    return desktop_engine.dispatch(
        {
            "action": "process_interpreter_output",
            "payload": {
                "instruction": instruction,
                "current_model_source": current,
                "context": context,
                "provider_identity": _interpreter_provider(),
                "output": {
                    "schema": INTERPRETER_OUTPUT_SCHEMA,
                    "schema_version": INTERPRETER_OUTPUT_SCHEMA_VERSION,
                    "action": "propose_edits",
                    "context_sha256": context["context_sha256"],
                    "context_requests": [],
                    "operations": [
                        {"op": "set", "path": ["name"], "value": name}
                    ],
                    "explanation": "Only the human-readable model name was changed.",
                    "warnings": [],
                    "clarification_question": "",
                },
            },
        }
    )


def test_health_exposes_versioned_isolated_engine_contract() -> None:
    result = desktop_engine.dispatch({"action": "health"})

    assert result["version"] == __version__ == "1.17.0"
    assert result["engine"] == "python-sidecar"
    assert result["protocol_version"] == 7
    assert result["process_mode"] == "persistent"
    assert result["cache"]["maximum_bytes"] == desktop_engine.MAX_CACHE_BYTES
    assert len(result["build_identity"]["source_tree_sha256"]) == 64
    assert result["interpreter"]["status"] == "deterministic_edit_compiler"
    assert result["interpreter"]["configured_model_tag"] == "qwen3:4b-instruct-2507-q4_K_M"


def test_example_model_is_available_through_the_engine_boundary() -> None:
    result = desktop_engine.dispatch({"action": "example_model"})

    assert result["model"]["name"] == "Quadratic example"
    assert "functions:" in result["source"]


def test_model_inspection_and_analysis_are_json_safe() -> None:
    model = desktop_engine.dispatch(
        {"action": "inspect_model", "payload": {"source": _source()}}
    )
    analysis = desktop_engine.dispatch(
        {"action": "analyse_model", "payload": {"source": _source()}}
    )

    assert model["name"] == "Quadratic example"
    assert model["source_sha256"]
    assert analysis["selected_visualisation"] == "2D function plot"
    assert analysis["stationary"]["available"] is True
    assert analysis["symbolic"]["entries"]
    assert analysis["figure"]["data"]
    json.dumps(analysis, allow_nan=False)


def test_later_official_pack_is_discovered_run_and_rendered_through_desktop_boundary() -> None:
    source = _source("mechanics-structure.yaml")
    inspected = desktop_engine.dispatch(
        {"action": "inspect_model", "payload": {"source": source}}
    )
    assert len(inspected["capabilities"]["official_packs"]) == 13
    mechanics = next(
        item for item in inspected["capabilities"]["official_packs"]
        if item["id"] == "org.modellab.pack.mechanics-structures-materials"
    )
    assert mechanics["applicable_capability_count"] == 3
    result = desktop_engine.dispatch(
        {
            "action": "run_capability",
            "payload": {
                "source": source,
                "capability_id": "org.modellab.mechanics.solve-truss-static",
                "settings": {"object_id": "roof-truss", "load_case": "asymmetric-snow"},
            },
        }
    )
    assert result["execution_performed"] is True
    assert result["run"]["capability_id"] == "org.modellab.mechanics.solve-truss-static"
    assert result["artifacts"][0]["artifact_type"] == "org.modellab.artifact.truss-static-result"
    assert result["figure"]["data"]
    json.dumps(result, allow_nan=False)


def test_v115_domain_pack_is_discovered_run_and_rendered_through_desktop_boundary() -> None:
    source = _source("chemical-biological.yaml")
    inspected = desktop_engine.dispatch(
        {"action": "inspect_model", "payload": {"source": source}}
    )
    chemistry = next(
        item for item in inspected["capabilities"]["official_packs"]
        if item["id"] == "org.modellab.pack.chemical-reaction-biological-systems"
    )
    assert chemistry["applicable_capability_count"] == 6
    result = desktop_engine.dispatch(
        {
            "action": "run_capability",
            "payload": {
                "source": source,
                "capability_id": "org.modellab.chemistry.simulate-reaction-network",
                "settings": {"object_id": "sequential-reaction", "samples": 31},
            },
        }
    )
    assert result["execution_performed"] is True
    assert result["artifacts"][0]["artifact_type"] == "org.modellab.artifact.reaction-trajectory"
    assert result["figure"]["data"]
    json.dumps(result, allow_nan=False)


def test_surface_analysis_produces_offline_three_dimensional_plot() -> None:
    result = desktop_engine.dispatch(
        {
            "action": "analyse_model",
            "payload": {
                "source": _source("surface.yaml"),
                "selected_visualisation": "3D surface",
                "evaluation_settings": {"points_per_axis_2d": 20},
            },
        }
    )

    assert result["selected_visualisation"] == "3D surface"
    assert any(trace["type"] == "surface" for trace in result["figure"]["data"])


def test_parameter_sweep_is_available_through_desktop_boundary() -> None:
    result = desktop_engine.dispatch(
        {
            "action": "run_sweep",
            "payload": {
                "source": _source(),
                "sweep": {
                    "parameter_name": "a",
                    "start": -1,
                    "end": 1,
                    "step_count": 5,
                    "selected_visualisation": "stationary-point counts",
                },
            },
        }
    )

    assert result["step_count"] == 5
    assert result["parameter_name"] == "a"
    assert result["figure"]["data"]


def test_inspecting_experiment_never_calls_reproduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = desktop_engine.dispatch(
        {"action": "prepare_experiment", "payload": {"source": _source()}}
    )

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("inspection must not execute reproduction")

    monkeypatch.setattr(desktop_engine, "reproduce_experiment", unexpected)
    inspected = desktop_engine.dispatch(
        {
            "action": "inspect_experiment",
            "payload": {
                "filename": "draft.mlab",
                "data_base64": prepared["draft_bundle_base64"],
            },
        }
    )

    assert inspected["execution_performed"] is False
    assert inspected["workload"]["within_budget"] is True
    assert inspected["recorded_results"]


def test_author_review_can_be_frozen_only_by_exact_digest() -> None:
    prepared = desktop_engine.dispatch(
        {"action": "prepare_experiment", "payload": {"source": _source()}}
    )
    with pytest.raises(Exception, match="does not match"):
        desktop_engine.dispatch(
            {
                "action": "finalize_experiment",
                "payload": {
                    "state_json": prepared["state_json"],
                    "approved_review_sha256": "0" * 64,
                },
            }
        )

    final = desktop_engine.dispatch(
        {
            "action": "finalize_experiment",
            "payload": {
                "state_json": prepared["state_json"],
                "approved_review_sha256": prepared["review_sha256"],
            },
        }
    )
    assert final["author_approved_for_publication"] is True
    assert base64.b64decode(final["bundle_base64"]).startswith(b"PK")


def test_prepared_state_can_be_committed_and_inspected_without_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = desktop_engine.dispatch(
        {"action": "prepare_experiment", "payload": {"source": _source()}}
    )

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("committing or inspecting committed state must not execute reproduction")

    monkeypatch.setattr(desktop_engine, "reproduce_experiment", unexpected)
    committed = desktop_engine.dispatch(
        {
            "action": "commit_experiment_state",
            "payload": {
                "state_json": prepared["state_json"],
                "expected_state_sha256": prepared["state_sha256"],
                "parent_commit_sha256": None,
            },
        }
    )

    assert committed["status"] == "COMMITTED"
    assert committed["experiment_state_sha256"] == prepared["state_sha256"]
    assert committed["execution_policy"]["live_state_authority"] is False
    assert len(committed["commit_sha256"]) == 64

    inspected = desktop_engine.dispatch(
        {
            "action": "inspect_committed_experiment",
            "payload": {
                "commit_json": committed["commit_json"],
                "expected_commit_sha256": committed["commit_sha256"],
            },
        }
    )
    assert inspected["status"] == "COMMITTED"
    assert inspected["commit_sha256"] == committed["commit_sha256"]
    assert inspected["commit_json"] == committed["commit_json"]
    assert inspected["model_source"] == _source()
    assert inspected["parameter_values"] == {"a": 1.0, "b": 0.0, "c": 0.0}
    assert inspected["evaluation_settings"]["points_1d"] == 1000
    assert inspected["reproduction_settings"]["relative_tolerance"] == pytest.approx(1e-8)
    assert inspected["interpreter_acceptances"] == []
    assert inspected["model"]["name"] == "Quadratic example"


def test_interpreter_context_short_circuits_unimplemented_scientific_composition() -> None:
    prepared = desktop_engine.dispatch(
        {
            "action": "prepare_interpreter_context",
            "payload": {
                "instruction": "Optimize a truss for minimum mass.",
                "current_model_source": "",
            },
        }
    )
    assert prepared["status"] == "unable"
    assert "Structural optimisation is not installed" in prepared["explanation"]
    assert prepared["execution_performed"] is False


def test_generic_run_reference_state_uses_same_commit_boundary() -> None:
    prepared = desktop_engine.dispatch(
        {
            "action": "prepare_run_experiment",
            "payload": {
                "source": _source(),
                "runs": [
                    {
                        "capability_id": "org.modellab.scalar.evaluate-curve",
                        "capability_version": "1.0",
                        "settings": {"points": 51},
                    }
                ],
            },
        }
    )
    committed = desktop_engine.dispatch(
        {
            "action": "commit_experiment_state",
            "payload": {
                "state_json": prepared["state_json"],
                "expected_state_sha256": prepared["state_sha256"],
            },
        }
    )

    assert committed["experiment_state_sha256"] == prepared["state_sha256"]
    inspected = desktop_engine.dispatch(
        {
            "action": "inspect_committed_experiment",
            "payload": {"commit_json": committed["commit_json"]},
        }
    )
    assert inspected["status"] == "COMMITTED"


def test_commit_rejects_stale_prepared_state_identity() -> None:
    prepared = desktop_engine.dispatch(
        {"action": "prepare_experiment", "payload": {"source": _source()}}
    )
    with pytest.raises(Exception, match="changed before commit"):
        desktop_engine.dispatch(
            {
                "action": "commit_experiment_state",
                "payload": {
                    "state_json": prepared["state_json"],
                    "expected_state_sha256": "0" * 64,
                },
            }
        )


def test_process_protocol_returns_bounded_structured_errors() -> None:
    response = json.loads(
        desktop_engine.handle_request(
            json.dumps({"id": 7, "action": "inspect_model", "payload": {"source": ""}}).encode()
        )
    )

    assert response["id"] == 7
    assert response["ok"] is False
    assert response["error"]["type"] == "DesktopEngineError"


def test_process_protocol_handles_multiple_requests_in_one_python_process() -> None:
    requests = "".join(
        json.dumps(request) + "\n"
        for request in (
            {"id": 1, "action": "health", "payload": {}},
            {"id": 2, "action": "example_model", "payload": {}},
            {"id": 3, "action": "health", "payload": {}},
        )
    )
    completed = subprocess.run(
        [sys.executable, str(ROOT / "desktop_engine.py")],
        cwd=ROOT,
        input=requests,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [response["id"] for response in responses] == [1, 2, 3]
    assert all(response["ok"] for response in responses)
    assert responses[0]["result"]["process_mode"] == "persistent"
    assert responses[1]["result"]["model"]["name"] == "Quadratic example"


def test_deterministic_desktop_responses_use_the_bounded_process_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    desktop_engine._RESPONSE_CACHE.clear()
    calls = 0
    original = desktop_engine._current_analysis_document

    def counted(payload: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return original(payload)  # type: ignore[arg-type]

    monkeypatch.setattr(desktop_engine, "_current_analysis_document", counted)
    request = {"action": "analyse_model", "payload": {"source": _source()}}

    first = desktop_engine.dispatch(request)
    second = desktop_engine.dispatch(request)

    assert first == second
    assert calls == 1
    assert desktop_engine._RESPONSE_CACHE.hits == 1


def test_interpreter_proposal_is_validated_without_execution_or_mutation() -> None:
    current = _source()
    result = _compiled_rename(
        current,
        "AI-reviewed example",
        "Rename this model for the lecture.",
    )

    assert result["execution_performed"] is False
    assert result["status"] == "proposal"
    assert result["model"]["name"] == "AI-reviewed example"
    assert result["proposal"]["requires_user_acceptance"] is True
    assert result["proposal"]["compiler"]["operation_count"] == 1
    assert "-name: Quadratic example" in result["diff"]


def test_interpreter_attachment_is_ingested_locally_and_resolved_by_identifier() -> None:
    ingested = desktop_engine.dispatch(
        {
            "action": "ingest_interpreter_attachment",
            "payload": {
                "name": "lecture-notes.md",
                "data_base64": base64.b64encode(b"Use a cubic response.\n").decode("ascii"),
            },
        }
    )
    attachment = ingested["attachment"]
    prepared = desktop_engine.dispatch(
        {
            "action": "prepare_interpreter_context",
            "payload": {
                "instruction": "Use the attached lecture notes.",
                "current_model_source": _source(),
                "attachment_ids": [attachment["attachment_id"]],
            },
        }
    )
    assert prepared["execution_performed"] is False
    assert prepared["context"]["schema_version"] == "1.3"
    assert prepared["context"]["attachments"]["documents"] == [attachment]
    assert "Use a cubic response" in prepared["context"]["attachments"]["included_chunks"][0]["text"]

    released = desktop_engine.dispatch(
        {
            "action": "release_interpreter_attachment",
            "payload": {"attachment_id": attachment["attachment_id"]},
        }
    )
    assert released["removed"] is True
    with pytest.raises(Exception, match="Reattach"):
        desktop_engine.dispatch(
            {
                "action": "prepare_interpreter_context",
                "payload": {
                    "instruction": "Use it.",
                    "current_model_source": _source(),
                    "attachment_ids": [attachment["attachment_id"]],
                },
            }
        )


def test_interpreter_acceptance_rejects_a_stale_editor_source() -> None:
    current = _source()
    validated = _compiled_rename(current, "Accepted example", "Rename this model.")
    proposal = validated["proposal"]

    with pytest.raises(Exception, match="editor source changed"):
        desktop_engine.dispatch(
            {
                "action": "accept_interpreter_proposal",
                "payload": {
                    "proposal": proposal,
                    "expected_proposal_sha256": proposal["proposal_sha256"],
                    "current_model_source": current + "\n",
                },
            }
        )


def test_interpreter_acceptance_returns_validated_model_and_provenance() -> None:
    current = _source()
    validated = _compiled_rename(current, "Accepted example", "Rename this model.")
    proposal = validated["proposal"]
    result = desktop_engine.dispatch(
        {
            "action": "accept_interpreter_proposal",
            "payload": {
                "proposal": proposal,
                "expected_proposal_sha256": proposal["proposal_sha256"],
                "current_model_source": current,
            },
        }
    )

    assert result["execution_performed"] is False
    assert result["model"]["name"] == "Accepted example"
    assert result["acceptance"]["proposal_sha256"] == proposal["proposal_sha256"]

    prepared = desktop_engine.dispatch(
        {
            "action": "prepare_experiment",
            "payload": {
                "source": result["source"],
                "interpreter_acceptances": [result["acceptance"]],
            },
        }
    )
    inspected = desktop_engine.dispatch(
        {
            "action": "inspect_experiment",
            "payload": {
                "filename": "interpreter-draft.mlab",
                "data_base64": prepared["draft_bundle_base64"],
            },
        }
    )
    assert inspected["interpreter_acceptances"] == [result["acceptance"]]

    finalized = desktop_engine.dispatch(
        {
            "action": "finalize_experiment",
            "payload": {
                "state_json": prepared["state_json"],
                "approved_review_sha256": prepared["review_sha256"],
            },
        }
    )
    reproduced = desktop_engine.dispatch(
        {
            "action": "reproduce_experiment",
            "payload": {
                "filename": "interpreter-published.mlab",
                "data_base64": finalized["bundle_base64"],
            },
        }
    )
    assert reproduced["report"]["status"] == "EXACT REPRODUCTION"
    assert reproduced["report"]["provenance"]["interpreter_acceptances"] == [
        result["acceptance"]
    ]
    assert "Accepted interpreter proposals: 1" in reproduced["report_text"]
