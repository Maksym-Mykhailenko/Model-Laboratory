from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_lab.canonical import canonical_model_ir_sha256
from model_lab.interpreter import INTERPRETER_OUTPUT_SCHEMA, INTERPRETER_OUTPUT_SCHEMA_VERSION
from model_lab.interpreter_baseline import (
    BaselineInfrastructureError,
    InferenceResult,
    EVALUATION_GENERATION_OPTIONS,
    OllamaIdentity,
    base_model_lock,
    baseline_contract,
    compare_model_semantics,
    few_shot_examples,
    load_benchmark,
    output_schema,
    run_baseline,
    scientific_semantics_sha256,
    system_prompt,
    _normalize_ollama_manifest_digest,
    user_prompt,
    validation_report,
)
from model_lab.parser import parse_model_text
from model_lab.validator import validate_model

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "verification" / "interpreter_baseline_v1.6.json"


def _identity() -> OllamaIdentity:
    return OllamaIdentity(
        "test-runtime",
        "sha256:0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0",
        2497280480,
        "sha256:0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0",
        "sha256:85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9",
        2497280480,
        262_144,
    )




def test_ollama_api_digest_normalization_accepts_documented_bare_hex_and_prefixed_form() -> None:
    digest = "0edcdef34593" + "a" * 52
    assert len(digest) == 64
    assert _normalize_ollama_manifest_digest(digest) == "sha256:" + digest
    assert _normalize_ollama_manifest_digest("sha256:" + digest) == "sha256:" + digest
    with pytest.raises(BaselineInfrastructureError):
        _normalize_ollama_manifest_digest("0edcdef34593")

def test_stage5_assets_are_versioned_and_frozen_to_exact_base_artifacts() -> None:
    contract = baseline_contract()
    schema = output_schema()
    prompt = system_prompt()
    lock = base_model_lock()

    assert contract["adapters_permitted"] is False
    assert contract["expected_base_model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert contract["expected_quantization"] == "Q4_K_M"
    assert contract["production_generation_options"]["temperature"] == 0.7
    assert [item["num_ctx"] for item in contract["production_generation_profiles"]] == [
        32768,
        65536,
        131072,
    ]
    assert contract["production_profile_selection"] == "smallest conservative request-size fit"
    assert contract["evaluation_generation_options"] == EVALUATION_GENERATION_OPTIONS
    assert EVALUATION_GENERATION_OPTIONS["temperature"] == 0.0
    assert contract["upstream_revision"] == "abcc171021d4f320b2e7f47c6f0deca67ded870c"
    assert len(contract["system_prompt_sha256"]) == 64
    assert len(contract["user_prompt_template_sha256"]) == 64
    assert len(contract["output_schema_sha256"]) == 64
    assert len(contract["base_model_lock_sha256"]) == 64
    assert lock["upstream"]["tokenizer"]["sha256"] == "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    model_layer = next(
        layer
        for layer in lock["ollama"]["layers"]
        if layer["media_type"] == "application/vnd.ollama.image.model"
    )
    assert model_layer["sha256"] == "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
    assert lock["ollama"]["registry_manifest_sha256"] == "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
    assert len(lock["upstream"]["weight_files"]) == 3
    assert schema["properties"]["schema"]["const"] == INTERPRETER_OUTPUT_SCHEMA
    assert schema["properties"]["schema_version"]["const"] == INTERPRETER_OUTPUT_SCHEMA_VERSION
    assert schema["additionalProperties"] is False
    assert "needs_clarification" in schema["properties"]["action"]["enum"]
    assert "Example A" in prompt and "Example D" in prompt


def test_shared_user_prompt_template_is_deterministic() -> None:
    context = {"context_sha256": "a" * 64, "mode": "create"}
    rendered = user_prompt("Create f=x.", context)
    assert rendered.count("Create f=x.") == 1
    assert '"context_sha256":"' + "a" * 64 + '"' in rendered
    assert rendered.endswith("END MODEL CONTEXT JSON")


def test_held_out_benchmark_is_large_curated_disjoint_and_training_excluded() -> None:
    benchmark = load_benchmark(BENCHMARK)

    assert len(benchmark["cases"]) == 150
    assert len(benchmark["family_counts"]) == 28
    assert benchmark["family_counts"]["scientific-capability-boundary"] == 10
    assert benchmark["family_counts"]["clarification"] == 8
    assert benchmark["family_counts"]["context-negotiation"] == 4
    assert benchmark["family_counts"]["model-graph"] == 5
    assert benchmark["training_exclusion"] is True
    assert benchmark["review"]["status"] == "internal-curated"
    assert "not claimed" in benchmark["review"]["provenance"].casefold()
    demonstrations = {" ".join(x.casefold().split()) for x in few_shot_examples()}
    assert all(" ".join(case["instruction"].casefold().split()) not in demonstrations for case in benchmark["cases"])
    assert all(case["expected"]["first_action"] is not None for case in benchmark["cases"])
    assert len(benchmark["benchmark_sha256"]) == 64


def test_semantic_comparator_handles_algebraic_equivalence_but_preserves_graph_source() -> None:
    left = validate_model(parse_model_text("""name: A
variables:
  x: {domain: [-2, 2]}
  y: {domain: [-2, 2]}
parameters:
  a: {default: 1, domain: [-2, 2]}
functions:
  f: x + y + (x-a)**2
"""))
    equivalent = validate_model(parse_model_text("""name: B
variables:
  y: {domain: [-2, 2]}
  x: {domain: [-2, 2]}
parameters:
  a: {default: 1, domain: [-2, 2]}
functions:
  f: (a-x)**2 + y + x
"""))
    assert compare_model_semantics(left, equivalent) == (True, [])
    assert scientific_semantics_sha256(left) == scientific_semantics_sha256(equivalent)

    expanded = validate_model(parse_model_text("""name: Expanded
variables:
  x: {domain: [-2, 2]}
functions:
  f: x**2 + 2*x + 1
"""))
    factored = validate_model(parse_model_text("""name: Factored
variables:
  x: {domain: [-2, 2]}
functions:
  f: (x+1)**2
"""))
    assert compare_model_semantics(expanded, factored) == (True, [])

    trig = validate_model(parse_model_text("""name: Trig
variables:
  x: {domain: [-2, 2]}
functions:
  f: sin(x)**2 + cos(x)**2
"""))
    one = validate_model(parse_model_text("""name: One
variables:
  x: {domain: [-2, 2]}
functions:
  f: '1'
"""))
    assert compare_model_semantics(trig, one) == (True, [])

    scaled_constraint = validate_model(parse_model_text("""name: C1
variables:
  x: {domain: [-4, 4]}
functions:
  f: x
constraints:
  c: {left: 2*x, relation: '==', right: 4}
"""))
    simple_constraint = validate_model(parse_model_text("""name: C2
variables:
  x: {domain: [-4, 4]}
functions:
  f: x
constraints:
  c: {left: x, relation: '==', right: 2}
"""))
    assert compare_model_semantics(scaled_constraint, simple_constraint) == (True, [])

    graph_a = validate_model(parse_model_text("""name: G
objects:
  a: {kind: org.example.node}
  b: {kind: org.example.node}
  c: {kind: org.example.node}
relationships:
  r: {kind: org.example.edge, source: a, target: c}
"""))
    graph_b = validate_model(parse_model_text("""name: G
objects:
  a: {kind: org.example.node}
  b: {kind: org.example.node}
  c: {kind: org.example.node}
relationships:
  r: {kind: org.example.edge, source: b, target: c}
"""))
    equivalent_graph, differences = compare_model_semantics(graph_a, graph_b)
    assert equivalent_graph is False
    assert any(".source" in item for item in differences)
    assert scientific_semantics_sha256(graph_a) != scientific_semantics_sha256(graph_b)


def test_validate_only_report_never_fabricates_a_model_score() -> None:
    report = validation_report(load_benchmark(BENCHMARK))
    assert report["status"] == "benchmark_validated_model_not_run"
    assert "metrics" not in report
    assert report["benchmark"]["case_count"] == 150
    assert report["benchmark"]["training_exclusion"] is True
    assert "implementation_identity" in report


class _PerfectOneCaseClient:
    def identity(self) -> OllamaIdentity:
        return _identity()

    def infer(self, *, instruction: str, context: dict[str, object]):
        return (
            {
                "schema": INTERPRETER_OUTPUT_SCHEMA,
                "schema_version": INTERPRETER_OUTPUT_SCHEMA_VERSION,
                "action": "propose_edits",
                "context_sha256": context["context_sha256"],
                "context_requests": [],
                "operations": [
                    {"op": "set", "path": ["name"], "value": "A harmless different title"},
                    {"op": "set", "path": ["variables", "x"], "value": {"domain": [-5, 5]}},
                    {"op": "set", "path": ["functions", "y"], "value": "x ** 2"},
                ],
                "explanation": "Creates the requested scalar model.",
                "warnings": [],
                "clarification_question": "",
            },
            {"prompt_tokens": 1, "generated_tokens": 1},
        )


def _one_case_benchmark() -> dict[str, object]:
    expected_source = """name: Gold title
variables:
  x: {domain: [-5, 5]}
functions:
  y: x**2
"""
    expected_model = validate_model(parse_model_text(expected_source))
    return {
        "schema": "model-laboratory-interpreter-benchmark",
        "schema_version": "1.2",
        "benchmark_sha256": "b" * 64,
        "review": {"status": "internal-curated", "provenance": "test"},
        "training_exclusion": True,
        "family_counts": {"scalar-create": 1},
        "cases": [{
            "id": "one",
            "family": "scalar-create",
            "instruction": "Construct y as the square of x over the interval from -5 to 5.",
            "current_model_source": "",
            "expected": {
                "terminal_status": "proposal",
                "first_action": "propose_edits",
                "model_source": expected_source,
                "canonical_model_ir_sha256": canonical_model_ir_sha256(expected_model),
                "scientific_semantics_sha256": scientific_semantics_sha256(expected_model),
                "clarification_keywords": [],
            },
            "notes": "",
        }],
    }


def test_runner_scores_semantics_separately_from_canonical_identity_and_keeps_raw_evidence() -> None:
    report = run_baseline(_one_case_benchmark(), client=_PerfectOneCaseClient())
    assert report["campaign_status"] == "valid"
    assert report["metrics"]["scientific_semantic_accuracy"]["rate"] == 1.0
    assert report["metrics"]["canonical_ir_exact_for_proposals"]["rate"] == 0.0
    assert report["metrics"]["first_action_accuracy"]["rate"] == 1.0
    round0 = report["cases"][0]["rounds"][0]
    assert round0["raw_content"]
    assert round0["parsed_output"]["action"] == "propose_edits"
    assert round0["validated_output"]["action"] == "propose_edits"
    assert isinstance(round0["context_document"], dict)
    assert len(round0["user_prompt_sha256"]) == 64
    assert report["implementation_identity"]["scorer_version"] == "1.4"


class _MalformedOutputClient:
    def identity(self) -> OllamaIdentity:
        return _identity()

    def infer(self, *, instruction: str, context: dict[str, object]) -> InferenceResult:
        return InferenceResult("not json", None, {"prompt_tokens": 1}, "JSONDecodeError: test")


def test_no_output_cannot_receive_first_action_credit() -> None:
    report = run_baseline(_one_case_benchmark(), client=_MalformedOutputClient())
    assert report["metrics"]["raw_schema_compliance"]["rate"] == 0.0
    assert report["metrics"]["first_action_accuracy"]["rate"] == 0.0
    assert report["metrics"]["terminal_status_accuracy"]["rate"] == 0.0


class _InfrastructureFailureClient:
    def identity(self) -> OllamaIdentity:
        return _identity()

    def infer(self, *, instruction: str, context: dict[str, object]):
        raise BaselineInfrastructureError("transport down")


def test_infrastructure_failure_invalidates_campaign_instead_of_lowering_model_accuracy() -> None:
    with pytest.raises(BaselineInfrastructureError, match="transport down"):
        run_baseline(_one_case_benchmark(), client=_InfrastructureFailureClient())


def test_limited_run_recomputes_family_counts() -> None:
    benchmark = load_benchmark(BENCHMARK)

    class UnableClient:
        def identity(self): return _identity()
        def infer(self, *, instruction, context):
            return ({
                "schema": INTERPRETER_OUTPUT_SCHEMA,
                "schema_version": INTERPRETER_OUTPUT_SCHEMA_VERSION,
                "action": "unable",
                "context_sha256": context["context_sha256"],
                "context_requests": [],
                "operations": [],
                "explanation": "Cannot safely represent this request.",
                "warnings": [],
                "clarification_question": "",
            }, {})

    report = run_baseline(benchmark, client=UnableClient(), limit=2)
    assert report["benchmark"]["case_count"] == 2
    assert report["benchmark"]["family_counts"] == {"scalar-create": 2}
    assert report["benchmark"]["limited_run"] is True


def test_benchmark_rejects_unknown_expected_fields(tmp_path: Path) -> None:
    raw = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    raw["cases"][0]["expected"]["typo"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_benchmark(path)


def test_benchmark_rejects_near_duplicate_few_shot_instruction(tmp_path: Path) -> None:
    raw = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    raw["cases"][0]["instruction"] = "Create response=t**4 - 2*t using t from -3 through 3."
    path = tmp_path / "contaminated.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="too similar"):
        load_benchmark(path)


class _InvalidGraphProposalClient:
    def identity(self) -> OllamaIdentity:
        return _identity()

    def infer(self, *, instruction: str, context: dict[str, object]):
        return ({
            "schema": INTERPRETER_OUTPUT_SCHEMA,
            "schema_version": INTERPRETER_OUTPUT_SCHEMA_VERSION,
            "action": "propose_edits",
            "context_sha256": context["context_sha256"],
            "context_requests": [],
            "operations": [
                {"op": "set", "path": ["objects", "node_a"], "value": {"kind": "NotNamespaced"}},
            ],
            "explanation": "Attempts to add an invalid graph object.",
            "warnings": [],
            "clarification_question": "",
        }, {"prompt_tokens": 1, "generated_tokens": 1})


def test_invalid_compiled_model_is_scored_as_model_failure_not_campaign_crash() -> None:
    report = run_baseline(_one_case_benchmark(), client=_InvalidGraphProposalClient())
    case = report["cases"][0]
    assert report["campaign_status"] == "valid"
    assert case["raw_schema_valid"] is True
    assert case["first_action_correct"] is True
    assert case["terminal_status"] == "model_failure"
    assert case["terminal_status_correct"] is False
    assert case["semantic_equivalent"] is False
    assert "InterpreterProposalError" in case["error"]
    assert "Model Graph object 'node_a'" in case["error"]


class _CountingPerfectClient(_PerfectOneCaseClient):
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, *, instruction: str, context: dict[str, object]):
        self.calls += 1
        return super().infer(instruction=instruction, context=context)


def test_live_campaign_checkpoint_resumes_completed_prefix(tmp_path: Path) -> None:
    benchmark = json.loads(json.dumps(_one_case_benchmark()))
    second = json.loads(json.dumps(benchmark["cases"][0]))
    second["id"] = "two"
    benchmark["cases"].append(second)
    benchmark["family_counts"] = {"scalar-create": 2}
    checkpoint = tmp_path / "baseline.checkpoint.json"

    first_client = _CountingPerfectClient()
    first = run_baseline(
        benchmark,
        client=first_client,
        limit=1,
        checkpoint_path=checkpoint,
    )
    assert first_client.calls == 1
    assert first["benchmark"]["case_count"] == 1
    assert checkpoint.exists()

    resumed_client = _CountingPerfectClient()
    resumed = run_baseline(
        benchmark,
        client=resumed_client,
        limit=2,
        checkpoint_path=checkpoint,
    )
    assert resumed_client.calls == 1
    assert resumed["benchmark"]["case_count"] == 2
    assert [item["id"] for item in resumed["cases"]] == ["one", "two"]


def test_stage5_evaluation_profile_is_bounded_and_checkpointed() -> None:
    from model_lab.interpreter_baseline import (
        BASELINE_PROTOCOL_VERSION,
        EVALUATION_GENERATION_OPTIONS,
        REPORT_SCHEMA_VERSION,
    )
    assert BASELINE_PROTOCOL_VERSION == "1.4"
    assert REPORT_SCHEMA_VERSION == "1.4"
    assert EVALUATION_GENERATION_OPTIONS["temperature"] == 0.0
    assert EVALUATION_GENERATION_OPTIONS["num_ctx"] == 32768
    assert EVALUATION_GENERATION_OPTIONS["num_predict"] == 4096


class _ClarificationThenBadContinuationClient:
    def identity(self) -> OllamaIdentity:
        return _identity()

    def infer(self, *, instruction: str, context: dict[str, object]):
        del instruction
        if not context.get("clarification_history"):
            raw = {
                "schema": INTERPRETER_OUTPUT_SCHEMA,
                "schema_version": INTERPRETER_OUTPUT_SCHEMA_VERSION,
                "action": "needs_clarification",
                "context_sha256": context["context_sha256"],
                "context_requests": [],
                "operations": [],
                "explanation": "The allowed range for k is missing.",
                "warnings": [],
                "clarification_question": "What allowed range should parameter k use?",
            }
        else:
            raw = {
                "schema": INTERPRETER_OUTPUT_SCHEMA,
                "schema_version": INTERPRETER_OUTPUT_SCHEMA_VERSION,
                "action": "propose_edits",
                "context_sha256": context["context_sha256"],
                "context_requests": [],
                "operations": [
                    {"op": "set", "path": ["name"], "value": "Wrong continuation"},
                    {"op": "set", "path": ["variables", "x"], "value": {"domain": [-2, 2]}},
                    {"op": "set", "path": ["parameters", "k"], "value": {"default": 1, "domain": [0, 2]}},
                    {"op": "set", "path": ["functions", "y"], "value": "k*x"},
                ],
                "explanation": "Uses the answer, but incorrectly.",
                "warnings": [],
                "clarification_question": "",
            }
        return raw, {"prompt_tokens": 1, "generated_tokens": 1}


def _one_clarification_benchmark() -> dict[str, object]:
    expected_source = """name: Clarified
variables:
  x: {domain: [-2, 2]}
parameters:
  k: {default: 1, domain: [0, 3]}
functions:
  y: k*x
"""
    expected_model = validate_model(parse_model_text(expected_source))
    followup = {
        "answer": "Use default 1 and allowed range [0, 3].",
        "terminal_status": "proposal",
        "model_source": expected_source,
        "canonical_model_ir_sha256": canonical_model_ir_sha256(expected_model),
        "scientific_semantics_sha256": scientific_semantics_sha256(expected_model),
    }
    return {
        "schema": "model-laboratory-interpreter-benchmark",
        "schema_version": "1.2",
        "benchmark_sha256": "c" * 64,
        "review": {"status": "internal-curated", "provenance": "test"},
        "training_exclusion": True,
        "family_counts": {"clarification": 1},
        "cases": [{
            "id": "clarification-two-turn",
            "family": "clarification",
            "instruction": "Create y=k*x with x on [-2,2]; k is adjustable but I did not specify its allowed range.",
            "current_model_source": "",
            "expected": {
                "terminal_status": "needs_clarification",
                "first_action": "needs_clarification",
                "model_source": None,
                "canonical_model_ir_sha256": None,
                "scientific_semantics_sha256": None,
                "clarification_keywords": ["range", "k"],
                "clarification_followup": followup,
            },
            "notes": "test",
        }],
    }


def test_clarification_gate_requires_using_the_held_out_answer() -> None:
    report = run_baseline(
        _one_clarification_benchmark(), client=_ClarificationThenBadContinuationClient()
    )
    assert report["metrics"]["clarification_question_accuracy"]["rate"] == 1.0
    assert report["metrics"]["clarification_continuation_accuracy"]["rate"] == 0.0
    assert report["metrics"]["clarification_accuracy"]["rate"] == 0.0
    case = report["cases"][0]
    assert case["terminal_status"] == "needs_clarification"
    assert case["clarification_followup_terminal_status"] == "proposal"
    assert [round_["phase"] for round_ in case["rounds"]] == ["initial", "clarification_followup"]


def test_context_negotiation_first_action_is_measured_separately_from_final_semantics() -> None:
    benchmark = _one_case_benchmark()
    benchmark["family_counts"] = {"context-negotiation": 1}
    benchmark["cases"][0]["family"] = "context-negotiation"
    benchmark["cases"][0]["expected"]["first_action"] = "request_context"
    report = run_baseline(benchmark, client=_PerfectOneCaseClient())
    assert report["metrics"]["scientific_semantic_accuracy"]["rate"] == 1.0
    assert report["metrics"]["context_negotiation_first_action_accuracy"]["rate"] == 0.0
