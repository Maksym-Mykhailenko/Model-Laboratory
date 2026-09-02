from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_lab.interpreter_training import (
    _ARCHETYPE_SPLITS,
    _FAMILY_COUNTS,
    _benchmark_denylist,
    _candidate_for,
    _context_candidate,
    _context_followup,
    _clarification_candidate,
    _clarification_followup,
    _example_payload,
    _DEFAULT_BENCHMARK,
    BENCHMARK_MAX_TEXT_SIMILARITY,
    DEFAULT_EXAMPLE_COUNT,
    DEFAULT_REVIEW_QUEUE,
    DEFAULT_SEED,
    materialize_sft_record,
    select_review_queue,
    validate_corpus_examples,
)


def _payload_for(family: str, index: int = 0):
    instructions, semantics, official_objects = _benchmark_denylist(_DEFAULT_BENCHMARK)
    candidate_index = index
    while True:
        candidate = _candidate_for(family, candidate_index, DEFAULT_SEED)
        try:
            return _example_payload(
                candidate,
                ordinal=index,
                seed=DEFAULT_SEED,
                benchmark_instructions=instructions,
                benchmark_semantics=semantics, benchmark_official_objects=official_objects,
            )
        except ValueError as exc:
            if "too similar" in str(exc) or "duplicates held-out" in str(exc):
                candidate_index += 1
                continue
            raise


def test_frozen_stage6_distribution_totals_6300():
    assert sum(_FAMILY_COUNTS.values()) == DEFAULT_EXAMPLE_COUNT == 6300
    assert _FAMILY_COUNTS["context-negotiation"] % 2 == 0


def test_archetype_splits_are_disjoint_and_nonempty():
    for family, splits in _ARCHETYPE_SPLITS.items():
        train = set(splits["train"])
        validation = set(splits["validation"])
        assert train
        assert validation
        assert train.isdisjoint(validation), family


@pytest.mark.parametrize(
    "family",
    [
        "scalar-create",
        "multivariate-create",
        "roles",
        "derived-assumptions",
        "constraints",
        "vector-matrix",
        "edit",
        "metadata-units",
        "ambiguity-recording",
        "model-graph",
        "structured-assets",
        "clarification",
        "safety-unsupported",
    ],
)
def test_representative_family_target_survives_real_pipeline(family: str):
    payload = _payload_for(family)
    assert payload["family"] == family
    assert payload["held_out_screen"]["benchmark_semantic_collision"] is False
    assert payload["held_out_screen"]["benchmark_max_instruction_similarity"] < BENCHMARK_MAX_TEXT_SIMILARITY
    if payload["target"]["action"] == "propose_edits":
        assert payload["compiled"]["status"] == "proposal"
        assert len(payload["compiled"]["canonical_model_ir_sha256"]) == 64
        assert len(payload["compiled"]["scientific_semantics_sha256"]) == 64


def test_context_negotiation_is_a_grounded_two_round_pair():
    instructions, semantics, official_objects = _benchmark_denylist(_DEFAULT_BENCHMARK)
    first = _context_candidate(0, DEFAULT_SEED)
    second = _context_followup(first)
    p1 = _example_payload(first, ordinal=0, seed=DEFAULT_SEED,
                          benchmark_instructions=instructions, benchmark_semantics=semantics, benchmark_official_objects=official_objects)
    p2 = _example_payload(second, ordinal=1, seed=DEFAULT_SEED,
                          benchmark_instructions=instructions, benchmark_semantics=semantics, benchmark_official_objects=official_objects)
    assert p1["conversation"]["conversation_id"] == p2["conversation"]["conversation_id"]
    assert p1["conversation"]["turn_index"] == 0
    assert p2["conversation"]["turn_index"] == 1
    assert p1["target"]["action"] == "request_context"
    assert p2["target"]["action"] == "propose_edits"
    assert p1["target"]["context_requests"] == [["assumptions", "a69"]]
    assert p2["conversation"]["requested_paths"] == [["assumptions", "a69"]]
    assert p1["context_sha256"] != p2["context_sha256"]
    assert p2["compiled"]["status"] == "proposal"


def test_clarification_training_is_a_real_answer_then_proposal_pair():
    instructions, semantics, official_objects = _benchmark_denylist(_DEFAULT_BENCHMARK)
    first = _clarification_candidate(0, DEFAULT_SEED)
    second = _clarification_followup(first)
    p1 = _example_payload(first, ordinal=0, seed=DEFAULT_SEED,
                          benchmark_instructions=instructions, benchmark_semantics=semantics, benchmark_official_objects=official_objects)
    p2 = _example_payload(second, ordinal=1, seed=DEFAULT_SEED,
                          benchmark_instructions=instructions, benchmark_semantics=semantics, benchmark_official_objects=official_objects)
    assert p1["conversation"]["conversation_id"] == p2["conversation"]["conversation_id"]
    assert p1["target"]["action"] == "needs_clarification"
    assert p2["target"]["action"] == "propose_edits"
    assert p2["conversation"]["clarification_history"] == [{
        "question": first.clarification_question,
        "answer": first.clarification_answer,
    }]
    assert p2["compiled"]["status"] == "proposal"


def test_materialized_sft_uses_the_production_prompt_contract():
    payload = _payload_for("scalar-create")
    sft = materialize_sft_record(payload)
    assert [m["role"] for m in sft["messages"]] == ["system", "user", "assistant"]
    assistant = json.loads(sft["messages"][2]["content"])
    assert assistant == payload["target"]
    assert payload["conversation"]["instruction"] in sft["messages"][1]["content"]
    assert sft["metadata"]["content_sha256"] == payload["content_sha256"]


def test_review_queue_is_explicitly_pending_and_balanced_across_available_groups():
    examples = [_payload_for("scalar-create", i) for i in range(5)]
    examples += [_payload_for("clarification", i) for i in range(5)]
    queue = select_review_queue(examples, count=8)
    assert len(queue) == 8
    assert all(item["review"]["status"] == "pending-human-review" for item in queue)
    assert all(item["review"]["reviewer"] == "" and item["review"]["notes"] == "" for item in queue)
    assert all(item["source_content_sha256"] for item in queue)
    assert {item["family"] for item in queue} == {"scalar-create", "clarification"}


def test_validation_rejects_wrong_split_for_archetype():
    first = _payload_for("scalar-create", 0)
    bad = json.loads(json.dumps(first))
    bad["id"] = "synthetic-leak"
    bad["split"] = "validation"
    from model_lab.canonical import canonical_json_sha256
    bad_without_hash = {k: v for k, v in bad.items() if k != "content_sha256"}
    bad["content_sha256"] = canonical_json_sha256(bad_without_hash)
    with pytest.raises(ValueError, match="wrong split"):
        validate_corpus_examples([bad], deep=False)


def test_review_queue_default_is_several_hundred():
    assert DEFAULT_REVIEW_QUEUE == 725


def test_cached_create_contract_returns_fresh_mutation_safe_contexts():
    from model_lab.interpreter import create_interpreter_context
    c1 = create_interpreter_context(instruction="x in [-2,2], f=x**2", current_model_source="")
    expected_hash = c1["context_sha256"]
    c1["contract"]["output"]["actions"].append("corrupt-test")
    c2 = create_interpreter_context(instruction="x in [-2,2], f=x**2", current_model_source="")
    assert "corrupt-test" not in c2["contract"]["output"]["actions"]
    assert c2["context_sha256"] == expected_hash == "ddfd1a156c865fd029c8bcdcad7f6d237206db00b6229fa8bb2877a33fdae536"


def test_current_corpus_opening_language_is_not_dominated_by_a_small_template_set() -> None:
    import collections
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "training" / "interpreter_corpus_v1.5"
    records = []
    for split in ("train", "validation"):
        records.extend(
            json.loads(line)
            for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        )
    trigrams = collections.Counter(
        " ".join(record["conversation"]["instruction"].casefold().split()[:3])
        for record in records
    )
    top_eight = sum(count for _, count in trigrams.most_common(8))
    assert top_eight < len(records) * 0.45
    assert trigrams.most_common(1)[0][1] < len(records) * 0.08
