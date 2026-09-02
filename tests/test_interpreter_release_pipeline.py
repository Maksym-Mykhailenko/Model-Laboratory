from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from model_lab import __version__
from model_lab.canonical import canonical_json_sha256
from model_lab.interpreter_candidate import compare_candidate_to_baseline
from model_lab.interpreter_export import (
    InterpreterExportError,
    create_export_plan,
    validate_export_receipt,
)
from model_lab.interpreter_finetuning import (
    FINETUNING_REPORT_SCHEMA,
    FINETUNING_REPORT_VERSION,
    FineTuningError,
    validate_training_report,
)
from model_lab.interpreter_registry import (
    InterpreterRegistryError,
    empty_registry,
    load_registry,
    promote_candidate,
    register_evaluated_candidate,
    register_training_candidate,
    rollback_interpreter_model,
    runtime_selection,
)


def _write_forged_minimal_training_candidate(root: Path, *, status: str = "candidate_adapter_unpromoted") -> dict:
    root.mkdir(parents=True, exist_ok=True)
    artifact = root / "adapter_model.safetensors"
    artifact.write_bytes(b"adapter")
    report = {
        "schema": FINETUNING_REPORT_SCHEMA,
        "schema_version": FINETUNING_REPORT_VERSION,
        "status": status,
        "promotion_performed": False,
        "output_files": [{
            "path": artifact.name,
            "size": artifact.stat().st_size,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }],
    }
    report["report_sha256"] = canonical_json_sha256(report)
    (root / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
    return report


def _write_export_receipt(root: Path, training_sha: str) -> dict:
    inventory = {}
    for key, name, raw in (
        ("f16_gguf", "candidate-f16.gguf", b"f16"),
        ("quantized_gguf", "candidate-q4_k_m.gguf", b"q4"),
        ("modelfile", "Modelfile", b"FROM candidate\n"),
    ):
        (root / name).write_bytes(raw)
        inventory[key] = {
            "path": name,
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    receipt = {
        "schema": "model-laboratory-interpreter-export-receipt",
        "schema_version": "1.1",
        "laboratory_version": __version__,
        "candidate_id": "candidate-0123456789abcdef0123",
        "created_at_utc": "2026-08-27T00:00:00+00:00",
        "candidate_tag": "modellab-qwen3-candidate:one",
        "quantization": "Q4_K_M",
        "training_report_sha256": training_sha,
        "training_status": "candidate_adapter_unpromoted",
        "base_snapshot_receipt_sha256": "b" * 64,
        "artifacts": inventory,
        "tools": {
            "converter_sha256": "d" * 64,
            "quantizer_sha256": "e" * 64,
            "merge_device": "cpu",
        },
        "commands": [{}, {}, {}],
        "ollama": {"runtime_version": "0.11", "model_digest": "c" * 64, "model_size_bytes": 2},
        "promotion_performed": False,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    path = root / "export_receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt


def _training(sha: str = "a" * 64, *, experimental: bool = False) -> dict:
    return {
        "status": "candidate_adapter_experimental" if experimental else "candidate_adapter_unpromoted",
        "report_sha256": sha,
        "base_model": {"receipt_sha256": "b" * 64},
    }


def _metrics(rate: float) -> dict:
    names = (
        "raw_schema_compliance",
        "first_action_accuracy",
        "terminal_status_accuracy",
        "scientific_semantic_accuracy",
        "canonical_ir_exact_for_proposals",
        "expected_proposal_compiler_acceptance",
        "unsupported_accuracy",
        "clarification_question_accuracy",
        "clarification_continuation_accuracy",
        "clarification_accuracy",
        "context_negotiation_first_action_accuracy",
    )
    passed = round(rate * 10)
    result = {
        name: {"rate": rate, "passed": passed, "total": 10}
        for name in names
    }
    result["by_family"] = {
        "clarification": {
            "semantic_equivalent_rate": rate,
            "semantic_equivalent": passed,
        },
        "context-negotiation": {
            "semantic_equivalent_rate": rate,
            "semantic_equivalent": passed,
        },
    }
    return result


def test_minimal_self_hashed_training_report_is_rejected(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    _write_forged_minimal_training_candidate(adapter)
    with pytest.raises(FineTuningError, match="incomplete|unknown fields"):
        validate_training_report(adapter, base_model_directory=tmp_path / "base")


def test_export_refuses_experimental_training_state_even_if_validator_regresses(tmp_path: Path, monkeypatch) -> None:
    from model_lab import interpreter_export as export_module

    monkeypatch.setattr(export_module, "load_qlora_config", lambda value: {})
    monkeypatch.setattr(
        export_module,
        "validate_training_report",
        lambda *args, **kwargs: _training(experimental=True),
    )
    with pytest.raises(InterpreterExportError, match="Experimental adapters"):
        create_export_plan(
            adapter_directory=tmp_path / "adapter",
            base_model_directory=tmp_path / "base",
            output_directory=tmp_path / "out",
            candidate_tag="candidate:test",
            convert_hf_to_gguf=tmp_path / "convert.py",
            llama_quantize=tmp_path / "quantize",
        )


def test_export_receipt_still_binds_exported_artifact_bytes(tmp_path: Path) -> None:
    export = tmp_path / "export"
    export.mkdir()
    receipt = _write_export_receipt(export, "a" * 64)
    assert validate_export_receipt(export / "export_receipt.json")["receipt_sha256"] == receipt["receipt_sha256"]
    (export / "candidate-q4_k_m.gguf").write_bytes(b"tampered")
    with pytest.raises(InterpreterExportError, match="differs"):
        validate_export_receipt(export / "export_receipt.json")


def test_candidate_policy_has_clarification_continuation_and_context_first_action_gates() -> None:
    baseline_metrics = _metrics(0.8)
    for name in (
        "raw_schema_compliance",
        "expected_proposal_compiler_acceptance",
        "unsupported_accuracy",
    ):
        baseline_metrics[name].update({"rate": 1.0, "passed": 10})
    baseline = {"metrics": baseline_metrics}
    candidate = json.loads(json.dumps(baseline_metrics))
    candidate["scientific_semantic_accuracy"].update({"rate": 0.9, "passed": 9})
    assert compare_candidate_to_baseline(
        baseline,
        candidate,
        training_status="candidate_adapter_unpromoted",
        evaluation_purpose="promotion",
    )["eligible_for_promotion"] is True
    development = compare_candidate_to_baseline(
        baseline,
        candidate,
        training_status="candidate_adapter_unpromoted",
        evaluation_purpose="development",
    )
    assert development["eligible_for_promotion"] is False
    assert next(
        gate for gate in development["gates"]
        if gate["gate"] == "evaluation_purpose:one_shot_promotion"
    )["passed"] is False

    candidate = json.loads(json.dumps(baseline_metrics))
    candidate["scientific_semantic_accuracy"].update({"rate": 0.9, "passed": 9})
    candidate["clarification_continuation_accuracy"].update({"rate": 0.7, "passed": 7})
    result = compare_candidate_to_baseline(
        baseline, candidate,
        training_status="candidate_adapter_unpromoted",
        evaluation_purpose="promotion",
    )
    assert result["eligible_for_promotion"] is False
    assert next(g for g in result["gates"] if g["gate"] == "metric:clarification_continuation_accuracy")["passed"] is False

    candidate = json.loads(json.dumps(baseline_metrics))
    candidate["scientific_semantic_accuracy"].update({"rate": 0.9, "passed": 9})
    candidate["context_negotiation_first_action_accuracy"].update({"rate": 0.7, "passed": 7})
    result = compare_candidate_to_baseline(
        baseline, candidate,
        training_status="candidate_adapter_unpromoted",
        evaluation_purpose="promotion",
    )
    assert result["eligible_for_promotion"] is False
    assert next(g for g in result["gates"] if g["gate"] == "metric:context_negotiation_first_action_accuracy")["passed"] is False


def test_registry_records_experimental_and_evaluated_lifecycle_states(tmp_path: Path, monkeypatch) -> None:
    from model_lab import interpreter_registry as registry_module

    path = tmp_path / "registry.json"
    training = _training("a" * 64, experimental=True)
    monkeypatch.setattr(registry_module, "validate_training_report", lambda *args, **kwargs: training)
    registered = register_training_candidate(
        registry_path=path,
        adapter_directory=tmp_path / "adapter-experimental",
        base_model_directory=tmp_path / "base",
    )
    assert registered["entries"][0]["lifecycle_state"] == "experimental"
    assert registered["entries"][0]["training_status"] == "candidate_adapter_experimental"

    training = _training("c" * 64, experimental=False)
    receipt = {
        "candidate_id": "candidate-0123456789abcdef0123",
        "candidate_tag": "modellab-qwen3-candidate:evaluated",
        "training_report_sha256": training["report_sha256"],
        "training_status": training["status"],
        "receipt_sha256": "f" * 64,
        "artifacts": {"quantized_gguf": {"sha256": "1" * 64}},
    }
    report = {
        "report_sha256": "d" * 64,
        "candidate": {
            "candidate_id": receipt["candidate_id"],
            "candidate_tag": receipt["candidate_tag"],
            "training_report_sha256": training["report_sha256"],
            "training_status": training["status"],
            "export_receipt_sha256": receipt["receipt_sha256"],
            "quantized_gguf_sha256": "1" * 64,
        },
        "runtime_identity": {"observed_model_digest": "2" * 64, "model_size_bytes": 123},
        "evaluation": {"purpose": "development", "promotion_campaign_id": None},
        "benchmark": {"benchmark_sha256": "3" * 64},
        "metrics": _metrics(0.8),
        "comparison": {"eligible_for_promotion": False},
    }
    monkeypatch.setattr(registry_module, "validate_training_report", lambda *args, **kwargs: training)
    monkeypatch.setattr(
        registry_module, "validate_candidate_evaluation_report",
        lambda value, **kwargs: report,
    )
    monkeypatch.setattr(registry_module, "validate_export_receipt", lambda value, verify_artifacts=True: receipt)
    monkeypatch.setattr(registry_module, "baseline_gate", lambda value, **kwargs: (True, "valid", {"metrics": _metrics(0.8)}))
    monkeypatch.setattr(registry_module, "compare_candidate_to_baseline", lambda base, metrics, **kwargs: report["comparison"])
    evaluated = register_evaluated_candidate(
        registry_path=path,
        adapter_directory=tmp_path / "adapter-evaluated",
        base_model_directory=tmp_path / "base",
        candidate_evaluation_report=tmp_path / "candidate.json",
        export_receipt=tmp_path / "receipt.json",
        evaluation_baseline_report=tmp_path / "base-evaluation.json",
        benchmark_path=tmp_path / "development-benchmark.json",
    )
    states = {entry["training_report_sha256"]: entry["lifecycle_state"] for entry in evaluated["entries"]}
    assert states["a" * 64] == "experimental"
    assert states["c" * 64] == "evaluated"
    evaluated_entry = next(entry for entry in evaluated["entries"] if entry["training_report_sha256"] == "c" * 64)
    assert evaluated_entry["candidate_evaluation_report_sha256"] == "d" * 64
    assert evaluated_entry["eligible_for_promotion"] is False


def test_promotion_has_final_non_experimental_training_state_guard(tmp_path: Path, monkeypatch) -> None:
    from model_lab import interpreter_registry as registry_module

    initial = empty_registry()
    monkeypatch.setattr(
        registry_module,
        "validate_training_report",
        lambda *args, **kwargs: _training(experimental=True),
    )
    monkeypatch.setattr(
        registry_module, "validate_candidate_evaluation_report",
        lambda value, **kwargs: {"report_sha256": "d" * 64},
    )
    monkeypatch.setattr(registry_module, "validate_export_receipt", lambda value, verify_artifacts=True: {})
    with pytest.raises(InterpreterRegistryError, match="candidate_adapter_unpromoted"):
        promote_candidate(
            registry_path=tmp_path / "registry.json",
            candidate_evaluation_report=tmp_path / "candidate.json",
            export_receipt=tmp_path / "receipt.json",
            adapter_directory=tmp_path / "adapter",
            base_model_directory=tmp_path / "base",
            evaluation_baseline_report=tmp_path / "base-evaluation.json",
            benchmark_path=tmp_path / "promotion-benchmark.json",
            promotion_seal=tmp_path / "promotion-seal.json",
            expected_candidate_report_sha256="d" * 64,
            expected_registry_sha256=initial["registry_sha256"],
            verify_runtime=False,
        )


def test_registry_promotion_uses_evaluated_state_and_exposes_training_and_evaluation_identity(tmp_path: Path, monkeypatch) -> None:
    from model_lab import interpreter_registry as registry_module

    path = tmp_path / "registry.json"
    training = _training("a" * 64)
    candidate_metrics = _metrics(1.0)
    comparison = {
        "policy_version": "1.1",
        "policy": "test",
        "gates": [{"passed": True}],
        "eligible_for_promotion": True,
    }
    report = {
        "report_sha256": "d" * 64,
        "comparison": comparison,
        "metrics": candidate_metrics,
        "baseline": {"report_sha256": "e" * 64},
        "candidate": {
            "candidate_id": "candidate-0123456789abcdef0123",
            "candidate_tag": "modellab-qwen3-candidate:one",
            "training_report_sha256": training["report_sha256"],
            "training_status": training["status"],
            "export_receipt_sha256": "f" * 64,
            "quantized_gguf_sha256": "1" * 64,
        },
        "runtime_identity": {"observed_model_digest": "2" * 64, "model_size_bytes": 123},
        "evaluation": {
            "purpose": "promotion",
            "promotion_campaign_id": "7b79d7c3-1482-4bf7-a61e-c999fa9eb7cf",
            "promotion_seal_sha256": "4" * 64,
        },
        "benchmark": {"benchmark_sha256": "3" * 64},
    }
    receipt = {
        "candidate_id": report["candidate"]["candidate_id"],
        "candidate_tag": report["candidate"]["candidate_tag"],
        "training_report_sha256": training["report_sha256"],
        "training_status": training["status"],
        "receipt_sha256": "f" * 64,
        "artifacts": {"quantized_gguf": {"sha256": "1" * 64}},
        "ollama": {"model_digest": "2" * 64, "model_size_bytes": 123},
    }
    monkeypatch.setattr(registry_module, "validate_training_report", lambda *args, **kwargs: training)
    monkeypatch.setattr(
        registry_module, "validate_candidate_evaluation_report",
        lambda value, **kwargs: report,
    )
    monkeypatch.setattr(registry_module, "validate_export_receipt", lambda value, verify_artifacts=True: receipt)
    monkeypatch.setattr(
        registry_module,
        "baseline_gate",
        lambda value, **kwargs: (True, "valid", {"report_sha256": "e" * 64, "metrics": _metrics(0.5)}),
    )
    monkeypatch.setattr(registry_module, "compare_candidate_to_baseline", lambda base, metrics, **kwargs: comparison)

    evaluated = register_evaluated_candidate(
        registry_path=path,
        adapter_directory=tmp_path / "adapter",
        base_model_directory=tmp_path / "base",
        candidate_evaluation_report=tmp_path / "candidate.json",
        export_receipt=tmp_path / "receipt.json",
        evaluation_baseline_report=tmp_path / "base-evaluation.json",
        benchmark_path=tmp_path / "promotion-benchmark.json",
        promotion_seal=tmp_path / "promotion-seal.json",
    )
    assert evaluated["entries"][0]["lifecycle_state"] == "evaluated"

    promoted = promote_candidate(
        registry_path=path,
        candidate_evaluation_report=tmp_path / "candidate.json",
        export_receipt=tmp_path / "receipt.json",
        adapter_directory=tmp_path / "adapter",
        base_model_directory=tmp_path / "base",
        evaluation_baseline_report=tmp_path / "base-evaluation.json",
        benchmark_path=tmp_path / "promotion-benchmark.json",
        promotion_seal=tmp_path / "promotion-seal.json",
        expected_candidate_report_sha256="d" * 64,
        expected_registry_sha256=evaluated["registry_sha256"],
        verify_runtime=False,
    )
    selection = runtime_selection(promoted)
    assert selection["model_role"] == "approved_candidate"
    assert selection["training_report_sha256"] == "a" * 64
    assert selection["candidate_evaluation_report_sha256"] == "d" * 64
    assert selection["candidate_lifecycle_state"] == "approved"

    rolled_back = rollback_interpreter_model(
        registry_path=path,
        target_entry_id="frozen_base",
        expected_registry_sha256=promoted["registry_sha256"],
    )
    base_selection = runtime_selection(rolled_back)
    assert base_selection["model_role"] == "frozen_base"
    assert base_selection["base_model_lock_sha256"] == "1b73f807ea8c2a53088ae5e38b9b376f3cd9acd8213bf7c881af728018c7f38e"
    assert base_selection["expected_manifest_sha256"] == "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
    assert base_selection["expected_model_blob_sha256"] == "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
    assert base_selection["identity_verification"] == "exact_local_manifest_and_all_blobs_sha256"
    assert load_registry(path) == rolled_back
