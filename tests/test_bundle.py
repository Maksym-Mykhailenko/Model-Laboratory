from __future__ import annotations

from io import BytesIO
from pathlib import Path
import base64
import hashlib
import json
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from model_lab.analysis import run_parameter_sweep, run_stationary_point_analysis
from model_lab.bundle import (
    MLAB_FORMAT_NAME,
    MLAB_FORMAT_VERSION,
    MlabBundleError,
    canonical_experiment_document,
    create_mlab_bundle,
    load_mlab_bundle,
)
from model_lab.canonical import canonical_model_ir_payload
from model_lab.capabilities import AnalysisVisualisation, ModelVisualisation
from model_lab.evaluator import evaluate_single_variable_function
from model_lab.experiment import (
    EvaluationSettings,
    StationaryPointSettings,
    SweepConfiguration,
    create_experiment_state,
)
from model_lab.interpreter import (
    FROZEN_BASE_ARTIFACT_LOCK_SHA256,
    FROZEN_BASE_IDENTITY_VERIFICATION,
    GENERATION_OPTIONS,
    INTERPRETER_OUTPUT_SCHEMA,
    INTERPRETER_OUTPUT_SCHEMA_VERSION,
    accept_interpreter_proposal,
    create_interpreter_context,
    create_interpreter_proposal,
)
from model_lab.parser import parse_model_text
from model_lab.validator import validate_model


MODEL_TEXT = """name: Bundle example
variables:
  x:
    domain:
    - -3
    - 3
parameters:
  a:
    default: 1
    domain:
    - -2
    - 2
functions:
  y: x**2 + a*x
"""


def _experiment(interpreter_acceptances=()):
    model = validate_model(parse_model_text(MODEL_TEXT))
    parameters = {"a": 1.25}
    evaluation_settings = EvaluationSettings(points_1d=129, points_per_axis_2d=45)
    stationary_settings = StationaryPointSettings(
        samples_1d=801,
        seeds_per_axis_2d=7,
        root_tolerance=1e-10,
    )
    evaluation = evaluate_single_variable_function(
        model, parameters, points=evaluation_settings.points_1d
    )
    stationary = run_stationary_point_analysis(
        model,
        parameters,
        samples_1d=stationary_settings.samples_1d,
        seeds_per_axis_2d=stationary_settings.seeds_per_axis_2d,
        root_tolerance=stationary_settings.root_tolerance,
    )
    sweep = run_parameter_sweep(
        model,
        "a",
        start=-2,
        end=2,
        step_count=7,
        fixed_parameter_values={},
        samples_1d=stationary_settings.samples_1d,
        seeds_per_axis_2d=stationary_settings.seeds_per_axis_2d,
        root_tolerance=stationary_settings.root_tolerance,
    )
    sweep_configuration = SweepConfiguration(
        parameter_name="a",
        start=-2,
        end=2,
        step_count=7,
        selected_visualisation=AnalysisVisualisation.SWEEP_POSITIONS.value,
    )
    state = create_experiment_state(
        model_source=MODEL_TEXT,
        model=model,
        parameter_values=parameters,
        selected_model_visualisation=ModelVisualisation.TWO_D_FUNCTION_PLOT,
        evaluation=evaluation,
        stationary_result=stationary,
        evaluation_settings=evaluation_settings,
        stationary_settings=stationary_settings,
        sweep_configuration=sweep_configuration,
        sweep_result=sweep,
        interpreter_acceptances=tuple(interpreter_acceptances),
    )
    return state, model, evaluation, stationary, sweep


def _accepted_interpreter_provenance() -> dict[str, object]:
    provider = {
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
    instruction = "Construct the bundle example."
    previous = MODEL_TEXT.replace("name: Bundle example", "name: Empty draft")
    context = create_interpreter_context(
        instruction=instruction,
        current_model_source=previous,
    )
    proposal = create_interpreter_proposal(
        raw_output={
            "schema": INTERPRETER_OUTPUT_SCHEMA,
            "schema_version": INTERPRETER_OUTPUT_SCHEMA_VERSION,
            "action": "propose_edits",
            "context_sha256": context["context_sha256"],
            "context_requests": [],
            "operations": [
                {"op": "set", "path": ["name"], "value": "Bundle example"}
            ],
            "explanation": "Constructed the requested bounded quadratic model.",
            "warnings": [],
            "clarification_question": "",
        },
        context_document=context,
        provider_identity=provider,
        instruction=instruction,
        current_model_source=previous,
        proposal_id="b03342ba-a56c-492a-8619-437713060bd8",
        created_at_utc="2026-08-25T12:00:00+00:00",
    )
    return accept_interpreter_proposal(
        proposal,
        expected_proposal_sha256=proposal["proposal_sha256"],
        accepted_at_utc="2026-08-25T12:01:00+00:00",
    )


def _bundle_bytes():
    state, model, evaluation, stationary, sweep = _experiment()
    data = create_mlab_bundle(
        state=state,
        model=model,
        evaluation=evaluation,
        stationary_result=stationary,
        sweep_result=sweep,
    )
    return data, state, model, evaluation, stationary, sweep


def _zip_members(data: bytes) -> dict[str, bytes]:
    with ZipFile(BytesIO(data), "r") as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _rebuild_zip(members: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(members):
            info = ZipInfo(name)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, members[name])
    return buffer.getvalue()


def _refresh_manifest(members: dict[str, bytes]) -> None:
    manifest = json.loads(members["manifest.json"])
    for entry in manifest["members"]:
        raw = members[entry["path"]]
        entry["size"] = len(raw)
        entry["sha256"] = hashlib.sha256(raw).hexdigest()
    members["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def test_mlab_round_trip_reconstructs_exact_experiment_state_without_execution(monkeypatch) -> None:
    data, state, model, *_ = _bundle_bytes()

    def forbidden(*args, **kwargs):  # pragma: no cover - only called on regression
        raise AssertionError("bundle loading must not execute an analysis")

    monkeypatch.setattr("model_lab.evaluator.evaluate_single_variable_function", forbidden)
    monkeypatch.setattr("model_lab.analysis.run_stationary_point_analysis", forbidden)
    monkeypatch.setattr("model_lab.analysis.run_parameter_sweep", forbidden)

    loaded = load_mlab_bundle(data)

    assert loaded.state.payload_dict() == state.payload_dict()
    assert loaded.state.state_sha256 == state.state_sha256
    assert canonical_model_ir_payload(loaded.model) == canonical_model_ir_payload(model)


def test_mlab_serialisation_is_deterministic_for_identical_state_and_results() -> None:
    state, model, evaluation, stationary, sweep = _experiment()
    first = create_mlab_bundle(
        state=state,
        model=model,
        evaluation=evaluation,
        stationary_result=stationary,
        sweep_result=sweep,
    )
    second = create_mlab_bundle(
        state=state,
        model=model,
        evaluation=evaluation,
        stationary_result=stationary,
        sweep_result=sweep,
    )
    assert first == second


def test_current_bundle_cannot_be_relabelled_as_format_1_2() -> None:
    data, *_ = _bundle_bytes()
    members = _zip_members(data)
    manifest = json.loads(members["manifest.json"])
    manifest["format_version"] = "1.2"
    members["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    with pytest.raises(MlabBundleError, match="format version does not match"):
        load_mlab_bundle(_rebuild_zip(members))


def test_optional_member_must_be_covered_by_manifest_integrity_records() -> None:
    data, *_ = _bundle_bytes()
    members = _zip_members(data)
    members["metadata/authoring.json"] = b"{}\n"

    with pytest.raises(MlabBundleError, match="exact container member set"):
        load_mlab_bundle(_rebuild_zip(members))


def test_manifest_describes_every_member_with_correct_sha256_and_size() -> None:
    data, *_ = _bundle_bytes()
    members = _zip_members(data)
    manifest = json.loads(members["manifest.json"])

    assert manifest["format"] == MLAB_FORMAT_NAME
    assert manifest["format_version"] == MLAB_FORMAT_VERSION
    described = {item["path"]: item for item in manifest["members"]}
    assert set(described) == set(members) - {"manifest.json"}
    for path, item in described.items():
        assert item["size"] == len(members[path])
        assert item["sha256"] == hashlib.sha256(members[path]).hexdigest()


def test_member_tampering_is_detected_by_manifest_checksum() -> None:
    data, *_ = _bundle_bytes()
    members = _zip_members(data)
    members["model.yaml"] += b"\n# modified\n"

    with pytest.raises(MlabBundleError, match="size|checksum"):
        load_mlab_bundle(_rebuild_zip(members))


def test_semantic_model_ir_mismatch_is_detected_even_with_refreshed_manifest() -> None:
    data, *_ = _bundle_bytes()
    members = _zip_members(data)
    model_ir = json.loads(members["model_ir.json"])
    model_ir["name"] = "Different canonical model"
    members["model_ir.json"] = (
        json.dumps(model_ir, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _refresh_manifest(members)

    with pytest.raises(MlabBundleError, match="canonical Model IR"):
        load_mlab_bundle(_rebuild_zip(members))


def test_canonical_experiment_representation_is_stable_after_bundle_round_trip() -> None:
    data, state, model, *_ = _bundle_bytes()
    before = canonical_experiment_document(state, model)
    loaded = load_mlab_bundle(data)
    after = canonical_experiment_document(loaded.state, loaded.model)
    assert before == after


def test_reference_document_defines_portable_representation_for_current_results() -> None:
    data, *_ = _bundle_bytes()
    loaded = load_mlab_bundle(data)
    document = loaded.reference_document
    policies = document["storage_policy"]
    references = {item["name"]: item["reference"] for item in document["references"]}

    assert set(policies) == {"evaluation", "stationary_points", "parameter_sweep", "symbolic"}
    assert policies["evaluation"]["array_encoding"].startswith("zlib+base64")
    assert policies["evaluation"]["full_array_limit_bytes"] == 8 * 1024 * 1024
    assert policies["evaluation"]["maximum_canonical_samples"] == 4096
    assert policies["stationary_points"]["comparison"].startswith("semantic point matching")
    assert policies["parameter_sweep"]["representation"].startswith("full step-wise")
    assert policies["symbolic"]["comparison"] == "exact fingerprint"
    assert set(references) == {"evaluation", "stationary_points", "parameter_sweep"}
    assert references["evaluation"]["kind"] == "FunctionEvaluation"
    assert references["stationary_points"]["kind"] == "StationaryPointAnalysisResult"
    assert references["parameter_sweep"]["kind"] == "ParameterSweepResult"


def test_bundle_records_canonical_validated_model_and_explicit_configuration() -> None:
    data, state, model, *_ = _bundle_bytes()
    loaded = load_mlab_bundle(data)
    experiment = loaded.experiment_document

    assert experiment["model"]["source_sha256"] == state.model_sha256
    assert experiment["parameter_state"] == [{"name": "a", "value": 1.25}]
    assert experiment["numerical_settings"]["random_seed"] is None
    assert experiment["analyses"]
    assert experiment["visualisation"]["model"] == ModelVisualisation.TWO_D_FUNCTION_PLOT.value
    assert loaded.manifest["experiment_id"] == experiment["experiment_id"]
    assert loaded.model.name == model.name


def test_bundle_preserves_checksum_bound_interpreter_acceptance_provenance() -> None:
    acceptance = _accepted_interpreter_provenance()
    state, model, evaluation, stationary, sweep = _experiment((acceptance,))
    data = create_mlab_bundle(
        state=state,
        model=model,
        evaluation=evaluation,
        stationary_result=stationary,
        sweep_result=sweep,
    )

    loaded = load_mlab_bundle(data)

    assert loaded.state.interpreter_acceptances == (acceptance,)
    assert loaded.experiment_document["interpreter_acceptances"] == [acceptance]
    assert loaded.provenance_document["interpreter_acceptances"] == [acceptance]
    assert loaded.experiment_document["schema_version"] == "2.0"
    assert loaded.provenance_document["schema_version"] == "2.0"


def test_bundle_rejects_unexpected_members() -> None:
    data, *_ = _bundle_bytes()
    members = _zip_members(data)
    members["unexpected.txt"] = b"not allowed"

    with pytest.raises(MlabBundleError, match="unexpected"):
        load_mlab_bundle(_rebuild_zip(members))


def test_bundle_creation_rejects_result_not_belonging_to_state() -> None:
    state, model, _, stationary, sweep = _experiment()
    wrong_evaluation = evaluate_single_variable_function(model, {"a": 1.0}, points=129)

    with pytest.raises(MlabBundleError, match="Evaluation result does not match"):
        create_mlab_bundle(
            state=state,
            model=model,
            evaluation=wrong_evaluation,
            stationary_result=stationary,
            sweep_result=sweep,
        )


def test_semantically_inconsistent_provenance_is_rejected_even_with_refreshed_manifest() -> None:
    data, *_ = _bundle_bytes()
    members = _zip_members(data)
    provenance = json.loads(members["provenance.json"])
    provenance["results"][0]["operation"] = "invented operation"
    members["provenance.json"] = (
        json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _refresh_manifest(members)

    with pytest.raises(MlabBundleError, match="provenance.json"):
        load_mlab_bundle(_rebuild_zip(members))


def test_new_bundle_uses_model_laboratory_expression_ast_not_sympy_representation() -> None:
    data, *_ = _bundle_bytes()
    members = _zip_members(data)
    model_ir = json.loads(members["model_ir.json"])

    assert model_ir["schema_version"] == "3.0"
    expression = model_ir["functions"][0]["expression"]
    assert expression["schema"] == "model-laboratory-expression-ast"
    assert expression["schema_version"] == "1.2"
    assert "sympy_srepr" not in members["model_ir.json"].decode("utf-8")


def test_real_v1_0_bundle_migrates_without_requiring_current_sympy_srepr() -> None:
    fixture = Path(__file__).parent / "fixtures" / "legacy-v1.0-expression-format.mlab"
    loaded = load_mlab_bundle(fixture.read_bytes())

    assert loaded.manifest["format_version"] == "1.0"
    assert loaded.model.name == "Legacy AST migration fixture"
    # The in-memory Model IR is current even though the persisted canonical IR was legacy.
    assert canonical_model_ir_payload(loaded.model)["schema_version"] == "3.0"


def test_legacy_v1_0_bundle_accepts_changed_sympy_printer_representation_when_legacy_hashes_are_consistent() -> None:
    from model_lab.canonical import canonical_json_sha256

    fixture = Path(__file__).parent / "fixtures" / "legacy-v1.0-expression-format.mlab"
    members = _zip_members(fixture.read_bytes())
    model_ir = json.loads(members["model_ir.json"])

    # Simulate a future/other SymPy release producing different internal/printer strings
    # while the supplied mathematical model itself remains unchanged.
    model_ir["functions"][0]["expression"]["sympy_srepr"] = "FutureSymPyRepresentation(...)"
    model_ir["functions"][0]["expression"]["normalised"] = "future-printer-output"
    members["model_ir.json"] = (
        json.dumps(model_ir, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    experiment = json.loads(members["experiment.json"])
    experiment["model"]["canonical_ir_sha256"] = canonical_json_sha256(model_ir)
    members["experiment.json"] = (
        json.dumps(experiment, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _refresh_manifest(members)

    loaded = load_mlab_bundle(_rebuild_zip(members))
    assert loaded.model.name == "Legacy AST migration fixture"


def test_legacy_v1_0_bundle_still_rejects_changes_to_model_laboratory_owned_ir_fields() -> None:
    fixture = Path(__file__).parent / "fixtures" / "legacy-v1.0-expression-format.mlab"
    members = _zip_members(fixture.read_bytes())
    model_ir = json.loads(members["model_ir.json"])
    model_ir["functions"][0]["dependencies"] = ["x"]  # drops declared dependency on a
    members["model_ir.json"] = (
        json.dumps(model_ir, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _refresh_manifest(members)

    with pytest.raises(MlabBundleError, match="Legacy model_ir.json"):
        load_mlab_bundle(_rebuild_zip(members))


def test_real_v1_1_bundle_migrates_expression_ast_1_0_to_compact_1_1() -> None:
    fixture = Path(__file__).parent / "fixtures" / "legacy-v1.1-expression-ast.mlab"
    loaded = load_mlab_bundle(fixture.read_bytes())

    assert loaded.manifest["format_version"] == "1.1"
    payload = canonical_model_ir_payload(loaded.model)
    assert payload["schema_version"] == "3.0"
    expression = payload["functions"][0]["expression"]
    assert expression["schema_version"] == "1.2"
    # The legacy 1.2300e-4 token is migrated without fixed-decimal expansion.
    real = expression["root"]["right"]
    assert real == {"type": "real", "coefficient": "123", "exponent": -6}


def test_real_v1_4_bundle_preserves_sinh_as_a_historical_identifier() -> None:
    fixture = Path(__file__).parent / "fixtures" / "legacy-v1.4-sinh-identifier.mlab.b64"
    loaded = load_mlab_bundle(base64.b64decode(fixture.read_text(encoding="ascii")))

    assert loaded.manifest["format_version"] == "1.4"
    assert loaded.model.parameter("sinh").default == 1.0
    assert "sinh" in loaded.model.functions[0].dependencies


def test_v1_1_bundle_with_context_rounded_real_literal_still_opens() -> None:
    from model_lab.canonical import canonical_json_sha256

    fixture = Path(__file__).parent / "fixtures" / "legacy-v1.1-expression-ast.mlab"
    members = _zip_members(fixture.read_bytes())
    model_ir = json.loads(members["model_ir.json"])
    # Simulate the v1.4 Decimal-context bug: the exact source remains 1.2300e-4,
    # while the persisted AST real digits were rounded by a different process context.
    model_ir["functions"][0]["expression"]["root"]["right"]["value"] = "0.00012"
    members["model_ir.json"] = (
        json.dumps(model_ir, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    experiment = json.loads(members["experiment.json"])
    experiment["model"]["canonical_ir_sha256"] = canonical_json_sha256(model_ir)
    members["experiment.json"] = (
        json.dumps(experiment, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _refresh_manifest(members)

    loaded = load_mlab_bundle(_rebuild_zip(members))
    assert loaded.manifest["format_version"] == "1.1"
    assert loaded.model.name == "AST 1.0 migration fixture"
