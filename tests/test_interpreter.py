from __future__ import annotations

from copy import deepcopy
import base64
import hashlib
import json

import pytest
import yaml

from model_lab.canonical import canonical_json_sha256
from model_lab.interpreter import (
    FROZEN_BASE_ARTIFACT_LOCK_SHA256,
    FROZEN_BASE_IDENTITY_VERIFICATION,
    GENERATION_OPTIONS,
    INTERPRETER_ACCEPTANCE_SCHEMA,
    INTERPRETER_CONTEXT_SCHEMA,
    INTERPRETER_OUTPUT_SCHEMA,
    INTERPRETER_OUTPUT_SCHEMA_VERSION,
    INTERPRETER_PROPOSAL_SCHEMA,
    MAX_CONTEXT_PACKAGE_BYTES,
    MAX_MODEL_SOURCE_BYTES,
    InterpreterProposalError,
    accept_interpreter_proposal,
    create_interpreter_context,
    create_interpreter_proposal,
    generation_options_for_request,
    process_interpreter_output,
    proposal_diff,
    validate_interpreter_acceptance_document,
    validate_interpreter_proposal_document,
    validate_raw_interpreter_output,
)
from model_lab.parser import parse_model_text
from model_lab.interpreter_attachments import ingest_attachment


CURRENT_SOURCE = """name: Original
variables:
  x:
    domain: [-2, 2]
functions:
  y: x**2
"""

INSTRUCTION = "Add a vertical-shift parameter named a."


def _provider(*, legacy: bool = False) -> dict[str, object]:
    options = dict(GENERATION_OPTIONS)
    if legacy:
        options.update({"num_ctx": 8192, "num_predict": 4096})
    common = {
        "provider": "ollama",
        "endpoint": "http://127.0.0.1:11434",
        "runtime_version": "0.11.4",
        "generation_options": options,
    }
    if legacy:
        return {
            **common,
            "model": "qwen3:4b-instruct-2507-q4_K_M",
            "model_digest": "sha256:" + "a" * 64,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "quantization": "Q4_K_M",
        }
    return {
        **common,
        "model_tag": "qwen3:4b-instruct-2507-q4_K_M",
        "observed_model_digest": "sha256:0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0",
        "identity_verification": FROZEN_BASE_IDENTITY_VERIFICATION,
        "expected_base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "expected_quantization": "Q4_K_M",
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


def _output(
    context: dict[str, object],
    *,
    operations: list[dict[str, object]] | None = None,
    action: str = "propose_edits",
    requests: list[list[str]] | None = None,
) -> dict[str, object]:
    return {
        "schema": INTERPRETER_OUTPUT_SCHEMA,
        "schema_version": INTERPRETER_OUTPUT_SCHEMA_VERSION,
        "action": action,
        "context_sha256": context["context_sha256"],
        "context_requests": requests or [],
        "operations": operations or [],
        "explanation": (
            "Adds a bounded vertical-shift parameter."
            if action != "unable"
            else "The requested transformation is not expressible safely."
        ),
        "warnings": [],
        "clarification_question": "",
    }


def _proposal() -> dict[str, object]:
    context = create_interpreter_context(
        instruction=INSTRUCTION,
        current_model_source=CURRENT_SOURCE,
    )
    return create_interpreter_proposal(
        raw_output=_output(
            context,
            operations=[
                {
                    "op": "set",
                    "path": ["parameters", "a"],
                    "value": {"default": 1, "domain": [-5, 5]},
                },
                {"op": "set", "path": ["functions", "y"], "value": "x**2 + a"},
                {"op": "set", "path": ["name"], "value": "Shifted quadratic"},
            ],
        ),
        context_document=context,
        provider_identity=_provider(),
        instruction=INSTRUCTION,
        current_model_source=CURRENT_SOURCE,
        proposal_id="d26c92a6-0e21-4d8b-8231-641904d64aa0",
        created_at_utc="2026-08-25T12:00:00+00:00",
    )


def test_context_is_checksum_bound_and_declares_actual_expression_language() -> None:
    context = create_interpreter_context(
        instruction=INSTRUCTION,
        current_model_source=CURRENT_SOURCE,
    )

    assert context["schema"] == INTERPRETER_CONTEXT_SCHEMA
    assert context["model"]["scope"] == "complete"
    assert context["contract"]["expression_language"]["functions"] == [
        "abs",
        "cos",
        "cosh",
        "exp",
        "log",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    ]
    schema_catalogue = context["contract"]["model"]["generated_schema_catalogue"]
    assert len(schema_catalogue["authoritative_schema_sha256"]) == 64
    assert schema_catalogue["sections"]["vector_functions"]["required"] == ["components"]
    assert "input_types" in schema_catalogue["definitions"]["ValueTypeSpec"]["properties"]
    assert len(json.dumps(context, ensure_ascii=False).encode("utf-8")) <= MAX_CONTEXT_PACKAGE_BYTES


def test_clarification_answer_is_visible_checksum_bound_and_recorded_in_provenance() -> None:
    instruction = "Create y=k*x on x [-2,2], but k has no default or range."
    first_context = create_interpreter_context(
        instruction=instruction, current_model_source=""
    )
    first_output = _output(first_context, action="needs_clarification")
    first_output["clarification_question"] = "What default and range should k use?"
    first = process_interpreter_output(
        raw_output=first_output,
        context_document=first_context,
        provider_identity=_provider(),
        instruction=instruction,
        current_model_source="",
    )
    assert first["status"] == "needs_clarification"
    lineage = [first["turn_evidence"]]
    history = [{"question": first["question"], "answer": "Default 1; range [-5,5]."}]
    second_context = create_interpreter_context(
        instruction=instruction,
        current_model_source="",
        clarification_history=history,
    )
    assert second_context["clarification_history"] == history
    proposal = create_interpreter_proposal(
        raw_output=_output(
            second_context,
            operations=[
                {"op": "set", "path": ["name"], "value": "Clarified line"},
                {"op": "set", "path": ["variables", "x"], "value": {"domain": [-2, 2]}},
                {"op": "set", "path": ["parameters", "k"], "value": {"default": 1, "domain": [-5, 5]}},
                {"op": "set", "path": ["functions", "y"], "value": "k*x"},
            ],
        ),
        context_document=second_context,
        provider_identity=_provider(),
        instruction=instruction,
        current_model_source="",
        clarification_history=history,
        conversation_id=first["conversation_id"],
        conversation_lineage=lineage,
    )
    assert proposal["request"]["clarification_turn_count"] == 1
    assert proposal["request"]["clarification_history_sha256"] == canonical_json_sha256(history)
    assert proposal["dialogue"]["turn_count"] == 1
    assert proposal["dialogue"]["turns"][0]["turn_sha256"] == lineage[0]["turn_sha256"]
    changed_provider = _provider()
    changed_provider["runtime_version"] = "0.11.5"
    with pytest.raises(InterpreterProposalError, match="changed during clarification"):
        create_interpreter_proposal(
            raw_output=_output(
                second_context,
                operations=[{"op": "set", "path": ["name"], "value": "Changed runtime"}],
            ),
            context_document=second_context,
            provider_identity=changed_provider,
            instruction=instruction,
            current_model_source="",
            clarification_history=history,
            conversation_id=first["conversation_id"],
            conversation_lineage=lineage,
        )
    acceptance = accept_interpreter_proposal(
        proposal, expected_proposal_sha256=proposal["proposal_sha256"]
    )
    assert acceptance["clarification_turn_count"] == 1
    with pytest.raises(InterpreterProposalError, match="does not match"):
        create_interpreter_proposal(
            raw_output=_output(second_context, operations=[{"op": "set", "path": ["name"], "value": "Wrong"}]),
            context_document=second_context,
            provider_identity=_provider(),
            instruction=instruction,
            current_model_source="",
            clarification_history=[{"question": first["question"], "answer": "A different answer"}],
        )


def test_adaptive_generation_profiles_preserve_large_request_acceptance() -> None:
    small_context = {"schema": "test", "value": "short"}
    assert generation_options_for_request(
        instruction="create x", context_document=small_context
    )["num_ctx"] == 32768
    medium = generation_options_for_request(
        instruction="m" * (50 * 1024), context_document=small_context
    )
    assert (medium["num_ctx"], medium["num_predict"]) == (65536, 8192)
    large_context = {"value": "c" * (30 * 1024)}
    large = generation_options_for_request(
        instruction="i" * (64 * 1024), context_document=large_context
    )
    assert (large["num_ctx"], large["num_predict"]) == (131072, 8192)


def test_typed_edits_compile_to_a_checksum_bound_validated_proposal() -> None:
    proposal = _proposal()
    validated = validate_interpreter_proposal_document(proposal)
    compiled = parse_model_text(validated["proposed_model"]["source"])

    assert validated["schema"] == INTERPRETER_PROPOSAL_SCHEMA
    assert validated["schema_version"] == "2.4"
    assert validated["requires_user_acceptance"] is True
    assert compiled.name == "Shifted quadratic"
    assert compiled.parameters["a"].default == 1
    assert compiled.functions["y"] == "x**2 + a"
    assert validated["compiler"]["operation_count"] == 3
    assert validated["request"]["instruction_sha256"] == hashlib.sha256(
        INSTRUCTION.encode("utf-8")
    ).hexdigest()
    assert INSTRUCTION not in str(validated["request"])


def test_model_output_cannot_contain_yaml_or_provider_metadata() -> None:
    context = create_interpreter_context(
        instruction=INSTRUCTION,
        current_model_source=CURRENT_SOURCE,
    )
    output = _output(context, operations=[])
    output["model_source"] = CURRENT_SOURCE
    with pytest.raises(InterpreterProposalError, match="unknown field"):
        validate_raw_interpreter_output(output)

    forged = _output(
        context,
        operations=[{"op": "set", "path": ["name"], "value": "New"}],
    )
    forged["provider"] = "remote"
    with pytest.raises(InterpreterProposalError, match="unknown field"):
        validate_raw_interpreter_output(forged)


def test_current_frozen_provider_rejects_forged_manifest_or_gguf_evidence() -> None:
    provider = _provider()
    provider["artifact_evidence"]["model_blob_sha256"] = "f" * 64
    context = create_interpreter_context(
        instruction=INSTRUCTION,
        current_model_source=CURRENT_SOURCE,
    )
    with pytest.raises(InterpreterProposalError, match="exact bundled manifest/GGUF lock"):
        create_interpreter_proposal(
            raw_output=_output(
                context,
                operations=[{"op": "set", "path": ["name"], "value": "Forged"}],
            ),
            context_document=context,
            provider_identity=provider,
            instruction=INSTRUCTION,
            current_model_source=CURRENT_SOURCE,
        )


def test_operation_order_does_not_change_compiled_source_or_program_identity() -> None:
    context = create_interpreter_context(
        instruction=INSTRUCTION,
        current_model_source=CURRENT_SOURCE,
    )
    operations = [
        {
            "op": "set",
            "path": ["parameters", "a"],
            "value": {"default": 1, "domain": [-5, 5]},
        },
        {"op": "set", "path": ["functions", "y"], "value": "x**2 + a"},
    ]
    first = create_interpreter_proposal(
        raw_output=_output(context, operations=operations),
        context_document=context,
        provider_identity=_provider(),
        instruction=INSTRUCTION,
        current_model_source=CURRENT_SOURCE,
    )
    second = create_interpreter_proposal(
        raw_output=_output(context, operations=list(reversed(operations))),
        context_document=context,
        provider_identity=_provider(),
        instruction=INSTRUCTION,
        current_model_source=CURRENT_SOURCE,
    )

    assert first["proposed_model"]["source"] == second["proposed_model"]["source"]
    assert first["compiler"]["edit_program_sha256"] == second["compiler"]["edit_program_sha256"]


def test_mapping_key_order_does_not_change_compiled_source_or_program_identity() -> None:
    context = create_interpreter_context(
        instruction=INSTRUCTION,
        current_model_source=CURRENT_SOURCE,
    )
    first_value = {"default": 1, "domain": [-5, 5], "description": "shift"}
    second_value = {"description": "shift", "domain": [-5, 5], "default": 1}
    first = create_interpreter_proposal(
        raw_output=_output(
            context,
            operations=[{"op": "set", "path": ["parameters", "a"], "value": first_value}],
        ),
        context_document=context,
        provider_identity=_provider(),
        instruction=INSTRUCTION,
        current_model_source=CURRENT_SOURCE,
    )
    second = create_interpreter_proposal(
        raw_output=_output(
            context,
            operations=[{"op": "set", "path": ["parameters", "a"], "value": second_value}],
        ),
        context_document=context,
        provider_identity=_provider(),
        instruction=INSTRUCTION,
        current_model_source=CURRENT_SOURCE,
    )

    assert first["proposed_model"]["source"] == second["proposed_model"]["source"]
    assert first["compiler"]["edit_program_sha256"] == second["compiler"]["edit_program_sha256"]


def test_overlapping_or_invalid_edit_programs_are_rejected() -> None:
    context = create_interpreter_context(
        instruction=INSTRUCTION,
        current_model_source=CURRENT_SOURCE,
    )
    with pytest.raises(InterpreterProposalError, match="overlapping"):
        validate_raw_interpreter_output(
            _output(
                context,
                operations=[
                    {"op": "set", "path": ["functions", "y"], "value": "x"},
                    {
                        "op": "set",
                        "path": ["functions", "y", "description"],
                        "value": "duplicate scope",
                    },
                ],
            )
        )
    with pytest.raises(InterpreterProposalError, match="null"):
        validate_raw_interpreter_output(
            _output(
                context,
                operations=[{"op": "remove", "path": ["functions", "y"], "value": "x"}],
            )
        )


def test_large_source_is_accepted_but_never_copied_into_the_llm_context() -> None:
    note = "scientific-note-" * 30000
    source = yaml.safe_dump(
        {
            "name": "Large model",
            "metadata": {"notes": note},
            "variables": {"x": {"domain": [-1, 1]}},
            "functions": {"y": "x"},
        },
        sort_keys=False,
        width=1000,
    )
    assert 400_000 < len(source.encode("utf-8")) < MAX_MODEL_SOURCE_BYTES
    context = create_interpreter_context(
        instruction="Rename the model to Large renamed model.",
        current_model_source=source,
    )
    encoded_context = json.dumps(context, ensure_ascii=False).encode("utf-8")

    assert context["model"]["scope"] == "partial"
    assert len(encoded_context) <= MAX_CONTEXT_PACKAGE_BYTES
    assert note[:1000] not in encoded_context.decode("utf-8")

    proposal = create_interpreter_proposal(
        raw_output=_output(
            context,
            operations=[
                {"op": "set", "path": ["name"], "value": "Large renamed model"}
            ],
        ),
        context_document=context,
        provider_identity=_provider(),
        instruction="Rename the model to Large renamed model.",
        current_model_source=source,
    )
    compiled = parse_model_text(proposal["proposed_model"]["source"])
    assert compiled.name == "Large renamed model"
    assert compiled.metadata.notes == note


def test_partial_context_requires_disclosure_before_existing_content_is_changed() -> None:
    note = "n" * 100_000
    source = yaml.safe_dump(
        {
            "name": "Partial model",
            "metadata": {"notes": note},
            "variables": {"x": {"domain": [-1, 1]}},
            "functions": {"y": "x"},
        },
        sort_keys=False,
    )
    instruction = "Change the output expression."
    context = create_interpreter_context(
        instruction=instruction,
        current_model_source=source,
    )
    edit = _output(
        context,
        operations=[{"op": "set", "path": ["functions", "y"], "value": "x**2"}],
    )
    with pytest.raises(InterpreterProposalError, match="not supplied"):
        create_interpreter_proposal(
            raw_output=edit,
            context_document=context,
            provider_identity=_provider(),
            instruction=instruction,
            current_model_source=source,
        )

    requested = process_interpreter_output(
        raw_output=_output(
            context,
            action="request_context",
            requests=[["functions", "y"]],
        ),
        context_document=context,
        provider_identity=_provider(),
        instruction=instruction,
        current_model_source=source,
    )
    assert requested["status"] == "context_required"
    expanded = requested["context"]
    proposal = create_interpreter_proposal(
        raw_output=_output(
            expanded,
            operations=[{"op": "set", "path": ["functions", "y"], "value": "x**2"}],
        ),
        context_document=expanded,
        provider_identity=_provider(),
        instruction=instruction,
        current_model_source=source,
    )
    assert parse_model_text(proposal["proposed_model"]["source"]).functions["y"] == "x**2"
    assert proposal["compiler"]["context_round"] == 1


def test_omitted_large_name_is_not_implicitly_visible_or_editable() -> None:
    hidden_name = "N" * 40_000
    source = yaml.safe_dump(
        {
            "name": hidden_name,
            "variables": {"x": {"domain": [-1, 1]}},
            "functions": {"y": "x"},
        },
        sort_keys=False,
    )
    instruction = "Rename the model."
    context = create_interpreter_context(
        instruction=instruction,
        current_model_source=source,
    )

    assert context["model"]["scope"] == "partial"
    assert ["name"] not in context["model"]["visible_paths"]
    assert hidden_name not in json.dumps(context, ensure_ascii=False)
    with pytest.raises(InterpreterProposalError, match="not supplied"):
        create_interpreter_proposal(
            raw_output=_output(
                context,
                operations=[{"op": "set", "path": ["name"], "value": "Visible name"}],
            ),
            context_document=context,
            provider_identity=_provider(),
            instruction=instruction,
            current_model_source=source,
        )


def test_round_trip_compiler_preserves_comments_quotes_and_untouched_layout() -> None:
    source = '''# model header\nname: "Styled model"  # preserve name style\nmetadata:\n  tags:\n    - "alpha"\n    - 'beta'\nvariables:\n  x:\n    domain: [-2, 2]  # inline domain\nfunctions:\n  y: "x**2"  # exact source style\n'''
    instruction = "Rename the model."
    context = create_interpreter_context(
        instruction=instruction,
        current_model_source=source,
    )
    proposal = create_interpreter_proposal(
        raw_output=_output(
            context,
            operations=[{"op": "set", "path": ["name"], "value": "Renamed model"}],
        ),
        context_document=context,
        provider_identity=_provider(),
        instruction=instruction,
        current_model_source=source,
    )
    compiled = proposal["proposed_model"]["source"]

    assert compiled.startswith("# model header\n")
    assert 'name: "Renamed model" # preserve name style\n' in compiled
    assert "  tags:\n    - \"alpha\"\n    - 'beta'\n" in compiled
    assert "    domain: [-2, 2]  # inline domain\n" in compiled
    assert '  y: "x**2"  # exact source style\n' in compiled


def test_disclosing_a_child_field_does_not_authorise_replacing_its_parent() -> None:
    source = yaml.safe_dump(
        {
            "name": "Partial model",
            "metadata": {"notes": "n" * 100_000},
            "variables": {"x": {"domain": [-1, 1]}},
            "functions": {
                "y": {"expression": "x", "description": "must remain undisclosed"}
            },
        },
        sort_keys=False,
    )
    instruction = "Change only the output expression."
    context = create_interpreter_context(
        instruction=instruction,
        current_model_source=source,
    )
    requested = process_interpreter_output(
        raw_output=_output(
            context,
            action="request_context",
            requests=[["functions", "y", "expression"]],
        ),
        context_document=context,
        provider_identity=_provider(),
        instruction=instruction,
        current_model_source=source,
    )
    expanded = requested["context"]

    with pytest.raises(InterpreterProposalError, match="not supplied"):
        create_interpreter_proposal(
            raw_output=_output(
                expanded,
                operations=[
                    {"op": "set", "path": ["functions", "y"], "value": "x**2"}
                ],
            ),
            context_document=expanded,
            provider_identity=_provider(),
            instruction=instruction,
            current_model_source=source,
        )

    proposal = create_interpreter_proposal(
        raw_output=_output(
            expanded,
            operations=[
                {
                    "op": "set",
                    "path": ["functions", "y", "expression"],
                    "value": "x**2",
                }
            ],
        ),
        context_document=expanded,
        provider_identity=_provider(),
        instruction=instruction,
        current_model_source=source,
    )
    compiled = yaml.safe_load(proposal["proposed_model"]["source"])
    assert compiled["functions"]["y"]["expression"] == "x**2"
    assert compiled["functions"]["y"]["description"] == "must remain undisclosed"


def test_empty_source_is_constructed_from_entries_without_whole_yaml_output() -> None:
    instruction = "Create a bounded hyperbolic tangent model."
    context = create_interpreter_context(instruction=instruction, current_model_source="")
    proposal = create_interpreter_proposal(
        raw_output=_output(
            context,
            operations=[
                {"op": "set", "path": ["name"], "value": "Tanh model"},
                {
                    "op": "set",
                    "path": ["variables", "x"],
                    "value": {"domain": [-5, 5]},
                },
                {"op": "set", "path": ["functions", "y"], "value": "tanh(x)"},
            ],
        ),
        context_document=context,
        provider_identity=_provider(),
        instruction=instruction,
        current_model_source="",
    )
    assert parse_model_text(proposal["proposed_model"]["source"]).functions["y"] == "tanh(x)"


def test_diff_is_stable_and_labels_current_and_compiled_sources() -> None:
    proposed = _proposal()["proposed_model"]["source"]
    difference = proposal_diff(CURRENT_SOURCE, proposed)
    assert difference.startswith("--- current/model.yaml\n+++ proposed/model.yaml\n")
    assert "+parameters:\n" in difference
    assert proposal_diff(CURRENT_SOURCE, proposed) == difference


def test_tampering_is_detected_before_acceptance() -> None:
    proposal = deepcopy(_proposal())
    proposal["compiler"]["operation_count"] = 2
    with pytest.raises(InterpreterProposalError, match="checksum"):
        validate_interpreter_proposal_document(proposal)


def test_acceptance_requires_exact_review_and_records_compiler_identity() -> None:
    proposal = _proposal()
    with pytest.raises(InterpreterProposalError, match="reviewed proposal checksum"):
        accept_interpreter_proposal(proposal, expected_proposal_sha256="0" * 64)

    acceptance = accept_interpreter_proposal(
        proposal,
        expected_proposal_sha256=proposal["proposal_sha256"],
        accepted_at_utc="2026-08-25T12:01:00+00:00",
    )
    validated = validate_interpreter_acceptance_document(acceptance)
    assert validated["schema"] == INTERPRETER_ACCEPTANCE_SCHEMA
    assert validated["schema_version"] == "1.5"
    assert validated["compiler"]["edit_program_sha256"] == proposal["compiler"]["edit_program_sha256"]
    assert validated["accepted_model_source_sha256"] == proposal["proposed_model"]["source_sha256"]
    assert "source" not in validated


def test_reference_attachments_are_bounded_and_checksum_bound_through_acceptance() -> None:
    attachment = ingest_attachment(
        "requirements.md",
        base64.b64encode(b"# Requested model\nRename the model to Referenced quadratic.\n").decode("ascii"),
    )
    plain = create_interpreter_context(
        instruction="Rename the model according to the attached requirements.",
        current_model_source=CURRENT_SOURCE,
    )
    context = create_interpreter_context(
        instruction="Rename the model according to the attached requirements.",
        current_model_source=CURRENT_SOURCE,
        attachments=[attachment],
    )
    assert plain["schema_version"] == "1.2"
    assert "attachments" not in plain
    assert context["schema_version"] == "1.3"
    assert context["attachments"]["documents"][0]["name"] == "requirements.md"
    assert context["attachments"]["included_chunks"][0]["text"].startswith("# Requested")

    instruction = "Rename the model according to the attached requirements."
    proposal = create_interpreter_proposal(
        raw_output=_output(
            context,
            operations=[
                {"op": "set", "path": ["name"], "value": "Referenced quadratic"}
            ],
        ),
        context_document=context,
        provider_identity=_provider(),
        instruction=instruction,
        current_model_source=CURRENT_SOURCE,
        attachments=[attachment],
    )
    evidence = proposal["request"]["attachments"]
    assert evidence["documents"][0]["attachment_id"] == attachment["attachment_id"]
    assert evidence["disclosed_chunks"][0]["text_sha256"] == context["attachments"]["included_chunks"][0]["text_sha256"]
    assert proposal["compiler"]["attachment_evidence_sha256"] == canonical_json_sha256(evidence)

    acceptance = accept_interpreter_proposal(
        proposal, expected_proposal_sha256=proposal["proposal_sha256"]
    )
    assert acceptance["attachments"] == evidence
    assert validate_interpreter_acceptance_document(acceptance)["attachments"] == evidence


def test_attachment_evidence_tampering_and_attachment_edit_paths_are_rejected() -> None:
    attachment = ingest_attachment(
        "notes.txt", base64.b64encode(b"Reference only.\n").decode("ascii")
    )
    instruction = "Use the attached reference to rename the model."
    context = create_interpreter_context(
        instruction=instruction,
        current_model_source=CURRENT_SOURCE,
        attachments=[attachment],
    )
    forbidden = _output(
        context,
        operations=[
            {
                "op": "set",
                "path": ["@attachment", attachment["attachment_id"], "0"],
                "value": "forged",
            }
        ],
    )
    with pytest.raises(InterpreterProposalError, match="unsupported model path"):
        validate_raw_interpreter_output(forbidden)

    proposal = create_interpreter_proposal(
        raw_output=_output(
            context,
            operations=[{"op": "set", "path": ["name"], "value": "Referenced"}],
        ),
        context_document=context,
        provider_identity=_provider(),
        instruction=instruction,
        current_model_source=CURRENT_SOURCE,
        attachments=[attachment],
    )
    forged = deepcopy(proposal)
    forged["request"]["attachments"]["documents"][0]["name"] = "forged.txt"
    forged["proposal_sha256"] = canonical_json_sha256(
        {key: value for key, value in forged.items() if key != "proposal_sha256"}
    )
    with pytest.raises(InterpreterProposalError, match="identifier does not match"):
        validate_interpreter_proposal_document(forged)


def test_stale_context_and_wrong_generation_profile_are_rejected() -> None:
    context = create_interpreter_context(
        instruction=INSTRUCTION,
        current_model_source=CURRENT_SOURCE,
    )
    stale = deepcopy(context)
    stale["round"] = 1
    with pytest.raises(InterpreterProposalError, match="does not match"):
        create_interpreter_proposal(
            raw_output=_output(
                context,
                operations=[{"op": "set", "path": ["name"], "value": "New"}],
            ),
            context_document=stale,
            provider_identity=_provider(),
            instruction=INSTRUCTION,
            current_model_source=CURRENT_SOURCE,
        )

    historical_shape = _provider()
    historical_shape.pop("model_role")
    historical_shape.pop("registry_entry_sha256")
    with pytest.raises(InterpreterProposalError, match="missing field"):
        create_interpreter_proposal(
            raw_output=_output(
                context,
                operations=[{"op": "set", "path": ["name"], "value": "New"}],
            ),
            context_document=context,
            provider_identity=historical_shape,
            instruction=INSTRUCTION,
            current_model_source=CURRENT_SOURCE,
        )

    provider = _provider()
    provider["generation_options"] = {**GENERATION_OPTIONS, "num_ctx": 8192}
    with pytest.raises(InterpreterProposalError, match="num_ctx"):
        create_interpreter_proposal(
            raw_output=_output(
                context,
                operations=[{"op": "set", "path": ["name"], "value": "New"}],
            ),
            context_document=context,
            provider_identity=provider,
            instruction=INSTRUCTION,
            current_model_source=CURRENT_SOURCE,
        )


def test_legacy_acceptance_schema_and_generation_profile_remain_loadable() -> None:
    legacy = {
        "schema": INTERPRETER_ACCEPTANCE_SCHEMA,
        "schema_version": "1.0",
        "proposal_id": "d26c92a6-0e21-4d8b-8231-641904d64aa0",
        "proposal_sha256": "1" * 64,
        "accepted_at_utc": "2026-08-25T12:01:00+00:00",
        "provider": _provider(legacy=True),
        "instruction_sha256": "2" * 64,
        "previous_model_source_sha256": "3" * 64,
        "accepted_model_source_sha256": "4" * 64,
        "accepted_model_ir_sha256": "5" * 64,
        "explanation": "Legacy whole-source proposal.",
        "warnings": [],
    }
    legacy["acceptance_sha256"] = canonical_json_sha256(legacy)

    validated = validate_interpreter_acceptance_document(legacy)
    assert validated["schema_version"] == "1.0"
    assert validated["provider"]["generation_options"]["num_ctx"] == 8192
    assert "compiler" not in validated


def test_user_clarification_is_distinct_from_hidden_context_request() -> None:
    context = create_interpreter_context(
        instruction="Create y=a*x but I have not chosen a default for a.",
        current_model_source="",
    )
    output = _output(context, action="needs_clarification")
    output["explanation"] = "An adjustable parameter requires user-supplied information."
    output["clarification_question"] = "What default value and allowed interval should parameter a use?"
    validated = validate_raw_interpreter_output(output)
    processed = process_interpreter_output(
        raw_output=validated,
        context_document=context,
        provider_identity=_provider(),
        instruction="Create y=a*x but I have not chosen a default for a.",
        current_model_source="",
    )
    assert processed["status"] == "needs_clarification"
    assert "default" in processed["question"].lower()
    assert processed["execution_performed"] is False

    invalid = dict(output)
    invalid["context_requests"] = [["functions", "y"]]
    with pytest.raises(InterpreterProposalError, match="needs_clarification forbids"):
        validate_raw_interpreter_output(invalid)
