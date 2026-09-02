from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_lab.canonical import canonical_json_sha256
import model_lab.interpreter_finetuning as finetuning_module
from model_lab.interpreter_finetuning import (
    CorpusAudit,
    FineTuningError,
    SNAPSHOT_RECEIPT,
    _configure_deterministic_environment,
    _tokenize_records,
    _validate_corrected_target,
    baseline_gate,
    preflight_report,
    effective_examples,
    load_qlora_config,
    load_training_environment_lock,
    validate_training_environment_receipt,
    verify_base_snapshot,
    write_snapshot_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


class _PrefixTokenizer:
    @staticmethod
    def apply_chat_template(messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        rendered = ""
        for message in messages:
            role = message["role"]
            rendered += f"<{role}>" + message["content"] + f"</{role}>"
        if add_generation_prompt:
            rendered += "<assistant>"
        else:
            rendered = rendered.replace("<assistant>", "<assistant>", 1)
        return list(rendered.encode("utf-8"))


def _first_training_record() -> dict:
    path = ROOT / "training" / "interpreter_corpus_v1.5" / "train.jsonl"
    with path.open(encoding="utf-8") as stream:
        return json.loads(stream.readline())


def test_frozen_qlora_configuration_is_context_safe_and_completion_only():
    config = load_qlora_config()
    assert config["base_model"]["revision"] == "abcc171021d4f320b2e7f47c6f0deca67ded870c"
    assert config["quantization"] == {
        "method": "qlora-nf4",
        "load_in_4bit": True,
        "quant_type": "nf4",
        "double_quantization": True,
        "compute_dtype": "auto-bf16-or-fp16",
    }
    assert config["lora"]["target_modules"] == "all-linear"
    assert config["sequence"] == {
        "maximum_tokens": 16384,
        "truncate": False,
        "assistant_completion_only": True,
        "packing": False,
    }
    environment = load_training_environment_lock(
        ROOT / "training" / "interpreter_training_environment_v1.2.json"
    )
    assert config["_training_environment_lock_sha256"] == environment["_sha256"]
    assert environment["direct_packages"] == {
        "torch": "2.8.0",
        "transformers": "5.15.1",
        "accelerate": "1.12.0",
        "peft": "0.20.0",
        "bitsandbytes": "0.50.1",
        "huggingface-hub": "1.5.0",
        "safetensors": "0.8.0",
        "packaging": "26.0",
    }
    assert environment["determinism"]["torch_deterministic_algorithms"] is True
    assert environment["determinism"]["transformers_full_determinism"] is True
    assert environment["determinism"]["cuda_tf32"] is False
    assert environment["python"] == {
        "implementation": "CPython",
        "minimum_version": "3.10",
        "maximum_exclusive_version": "3.14",
        "hash_seed": "0",
    }
    assert environment["environment_receipt"][
        "require_complete_distribution_inventory"
    ] is True
    assert config["_training_environment_receipt"] is None


def test_deterministic_training_environment_rejects_a_conflicting_cublas_setting(
    monkeypatch,
) -> None:
    config = load_qlora_config()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(FineTuningError, match="conflicts"):
        _configure_deterministic_environment(config)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    _configure_deterministic_environment(config)
    assert config["_training_environment"]["runtime"]["cublas_workspace_config"] == ":4096:8"


def test_environment_receipt_binds_complete_distribution_and_cuda_identity(
    tmp_path: Path,
) -> None:
    lock = load_training_environment_lock(
        ROOT / "training" / "interpreter_training_environment_v1.2.json"
    )
    distributions = [
        {
            "name": name,
            "version": version,
            "record_sha256": "a" * 64,
            "content_sha256": "b" * 64,
            "file_count": 1,
        }
        for name, version in sorted(lock["direct_packages"].items())
    ]
    receipt = {
        "schema": "model-laboratory-interpreter-training-environment-receipt",
        "schema_version": "1.0",
        "created_at_utc": "2026-08-27T00:00:00+00:00",
        "policy_sha256": lock["_sha256"],
        "python": {
            "implementation": "CPython",
            "version": "3.12.10",
            "cache_tag": "cpython-312",
            "executable_sha256": "c" * 64,
        },
        "platform": {
            "system": "Windows",
            "release": "11",
            "version": "test-build",
            "machine": "AMD64",
        },
        "distributions": distributions,
        "distribution_inventory_sha256": canonical_json_sha256(distributions),
        "numerical_runtime": {
            "torch_version": "2.8.0",
            "cuda_version": "12.8",
            "cudnn_version": 91002,
            "nvidia_driver_version": "test",
            "gpu_name": "Test GPU",
            "gpu_total_memory_bytes": 16 * 1024**3,
            "gpu_compute_capability": [8, 9],
        },
        "deterministic_process": {
            "pythonhashseed": "0",
            "cublas_workspace_config": ":4096:8",
        },
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    path = tmp_path / "environment-receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert validate_training_environment_receipt(
        path, lock, verify_current=False
    )["receipt_sha256"] == receipt["receipt_sha256"]

    receipt["distributions"][0]["version"] = "forged"
    receipt["distribution_inventory_sha256"] = canonical_json_sha256(
        receipt["distributions"]
    )
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(FineTuningError, match="Frozen direct package differs"):
        validate_training_environment_receipt(path, lock, verify_current=False)


def test_supported_python_range_is_distinct_from_exact_receipt_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    lock = load_training_environment_lock(
        ROOT / "training" / "interpreter_training_environment_v1.2.json"
    )
    distributions = [
        {
            "name": name,
            "version": version,
            "record_sha256": "a" * 64,
            "content_sha256": "b" * 64,
            "file_count": 1,
        }
        for name, version in sorted(lock["direct_packages"].items())
    ]
    projection = {
        "python": {
            "implementation": "CPython",
            "version": "3.13.15",
            "cache_tag": "cpython-313",
            "executable_sha256": "c" * 64,
        },
        "platform": {
            "system": "Windows",
            "release": "11",
            "version": "test-build",
            "machine": "AMD64",
        },
        "distributions": distributions,
        "distribution_inventory_sha256": canonical_json_sha256(distributions),
        "numerical_runtime": {
            "torch_version": "2.8.0",
            "cuda_version": "12.8",
            "cudnn_version": 91002,
            "nvidia_driver_version": "test",
            "gpu_name": "Test GPU",
            "gpu_total_memory_bytes": 16 * 1024**3,
            "gpu_compute_capability": [8, 9],
        },
        "deterministic_process": {
            "pythonhashseed": "0",
            "cublas_workspace_config": ":4096:8",
        },
    }
    receipt = {
        "schema": "model-laboratory-interpreter-training-environment-receipt",
        "schema_version": "1.0",
        "created_at_utc": "2026-08-27T00:00:00+00:00",
        "policy_sha256": lock["_sha256"],
        **projection,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    path = tmp_path / "environment-receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")

    assert validate_training_environment_receipt(
        path, lock, verify_current=False
    )["python"]["version"] == "3.13.15"

    changed = json.loads(json.dumps(projection))
    changed["python"]["version"] = "3.13.16"
    monkeypatch.setattr(
        finetuning_module,
        "_environment_projection",
        lambda *, hash_payloads: changed,
    )
    with pytest.raises(FineTuningError, match="differs from its frozen receipt: python"):
        validate_training_environment_receipt(path, lock, verify_current=True)


def test_python_outside_supported_training_range_is_rejected(tmp_path: Path) -> None:
    lock = load_training_environment_lock(
        ROOT / "training" / "interpreter_training_environment_v1.2.json"
    )
    distributions = [
        {
            "name": name,
            "version": version,
            "record_sha256": "a" * 64,
            "content_sha256": "b" * 64,
            "file_count": 1,
        }
        for name, version in sorted(lock["direct_packages"].items())
    ]
    receipt = {
        "schema": "model-laboratory-interpreter-training-environment-receipt",
        "schema_version": "1.0",
        "created_at_utc": "2026-08-27T00:00:00+00:00",
        "policy_sha256": lock["_sha256"],
        "python": {
            "implementation": "CPython",
            "version": "3.14.0",
            "cache_tag": "cpython-314",
            "executable_sha256": "c" * 64,
        },
        "platform": {
            "system": "Windows",
            "release": "11",
            "version": "test-build",
            "machine": "AMD64",
        },
        "distributions": distributions,
        "distribution_inventory_sha256": canonical_json_sha256(distributions),
        "numerical_runtime": {
            "torch_version": "2.8.0",
            "cuda_version": "12.8",
            "cudnn_version": 91002,
            "nvidia_driver_version": "test",
            "gpu_name": "Test GPU",
            "gpu_total_memory_bytes": 16 * 1024**3,
            "gpu_compute_capability": [8, 9],
        },
        "deterministic_process": {
            "pythonhashseed": "0",
            "cublas_workspace_config": ":4096:8",
        },
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    path = tmp_path / "environment-receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(FineTuningError, match="outside the supported training range"):
        validate_training_environment_receipt(path, lock, verify_current=False)


def test_current_validation_only_report_does_not_satisfy_stage5_gate():
    valid, reason, report = baseline_gate()
    assert valid is False
    assert "live 150-case baseline has not been run" in reason
    assert report["status"] == "benchmark_validated_model_not_run"


def test_chat_tokenization_masks_prompt_and_never_truncates():
    record = _first_training_record()
    encoded, profile = _tokenize_records([record], _PrefixTokenizer(), 100_000)
    labels = encoded[0]["labels"]
    assert profile["records"] == 1
    assert -100 in labels
    first_completion = labels.index(next(value for value in labels if value != -100))
    assert all(value == -100 for value in labels[:first_completion])
    assert all(value != -100 for value in labels[first_completion:])
    with pytest.raises(FineTuningError, match="truncation is forbidden"):
        _tokenize_records([record], _PrefixTokenizer(), 10)


def test_review_decisions_are_applied_without_mutating_source_records():
    source = {"id": "one", "target": {"action": "unable"}}
    corrected = {"action": "needs_clarification"}
    review = {
        "id": "one",
        "review": {
            "status": "corrected",
            "reviewer": "Researcher",
            "reviewed_at_utc": "2026-08-26T00:00:00+00:00",
            "notes": "Correction",
            "corrected_target": corrected,
        },
    }
    corpus = CorpusAudit(
        ROOT,
        {},
        (source,),
        (),
        (review,),
        {"pending-human-review": 0, "accepted": 0, "corrected": 1, "rejected": 0},
        True,
        {},
        (),
    )
    result = effective_examples(corpus, "train", allow_unreviewed=False)
    assert result[0]["target"] == corrected
    assert source["target"] == {"action": "unable"}


def test_corrected_clarification_turn_reconstructs_its_dialogue_context() -> None:
    corpus = ROOT / "training" / "interpreter_corpus_v1.5"
    record = next(
        item
        for path in (corpus / "train.jsonl", corpus / "validation.jsonl")
        for item in map(json.loads, path.read_text(encoding="utf-8").splitlines())
        if item["conversation"].get("clarification_history")
    )
    _validate_corrected_target(record, record["target"])


def test_snapshot_receipt_binds_every_downloaded_member(tmp_path: Path):
    for filename in (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ):
        (tmp_path / filename).write_bytes(filename.encode("ascii"))
    receipt = write_snapshot_receipt(
        tmp_path,
        repository="Qwen/Qwen3-4B-Instruct-2507",
        revision="abcc171021d4f320b2e7f47c6f0deca67ded870c",
    )
    assert (tmp_path / SNAPSHOT_RECEIPT).is_file()
    without_sha = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert receipt["receipt_sha256"] == canonical_json_sha256(without_sha)
    (tmp_path / "config.json").write_text("changed", encoding="utf-8")
    with pytest.raises(FineTuningError, match="differs from its receipt"):
        verify_base_snapshot(tmp_path, load_qlora_config())


def test_identical_custom_qlora_config_is_automatically_experimental(tmp_path: Path) -> None:
    source = ROOT / "training" / "interpreter_qlora_v1.7.json"
    custom = tmp_path / "custom.json"
    custom.write_bytes(source.read_bytes())
    config = load_qlora_config(custom)
    assert config["_sha256"] == load_qlora_config()["_sha256"]
    assert config["_official_frozen_config"] is False


def test_stage7_gate_rejects_self_hashed_fabricated_baseline(tmp_path: Path) -> None:
    from model_lab.interpreter_baseline import load_benchmark, validation_report

    benchmark = load_benchmark(ROOT / "verification" / "interpreter_baseline_v1.6.json")
    report = validation_report(benchmark)
    report.pop("status", None)
    report.pop("reason", None)
    report["campaign_status"] = "valid"
    report["runtime_identity"] = {}
    report["benchmark"].update(
        {
            "schema": benchmark["schema"],
            "schema_version": benchmark["schema_version"],
            "benchmark_sha256": "0" * 64,
            "case_count": 150,
            "limited_run": False,
        }
    )
    report["cases"] = [{} for _ in range(150)]
    report["metrics"] = {
        name: {"passed": 150, "total": 150, "rate": 1.0}
        for name in (
            "raw_schema_compliance",
            "first_action_accuracy",
            "terminal_status_accuracy",
            "scientific_semantic_accuracy",
        )
    }
    report["report_sha256"] = canonical_json_sha256(report)
    path = tmp_path / "fake-baseline.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    valid, reason, _ = baseline_gate(path)
    assert valid is False
    assert "exact supplied full benchmark" in reason or "runtime identity" in reason


def test_qlora_config_pins_immutable_stage6_splits() -> None:
    config = load_qlora_config()
    corpus = config["corpus"]
    assert corpus["content_sha256"] == "bfee076b5501a75311f0d85247a3109a7e8d66910a11ba3dfb59ee72931df28b"
    assert corpus["train_sha256"] == "5f396dec5eceb14f4c7aa9cf1cc9c58c92e8ac8a3bef7a2508191e44e0f01576"
    assert corpus["validation_sha256"] == "8e6b05ce1b669fd39897889b93e7794ce04786210d5cbeb0a6b6a4078fdc43eb"


def test_shallow_preflight_reports_bound_deep_evidence_without_claiming_deep_run(monkeypatch):
    corpus_dir = ROOT / "training" / "interpreter_corpus_v1.5"
    fake = CorpusAudit(
        directory=corpus_dir,
        manifest={},
        train=tuple({} for _ in range(5039)),
        validation=tuple({} for _ in range(1261)),
        review=tuple({} for _ in range(725)),
        review_counts={"pending-human-review": 725, "accepted": 0, "corrected": 0, "rejected": 0},
        review_complete=False,
        held_out_exclusion={},
        blockers=("human review incomplete: 725 of 725 queue records remain pending",),
    )
    monkeypatch.setattr(finetuning_module, "audit_corpus", lambda config, deep, workers: fake)
    monkeypatch.setattr(finetuning_module, "dependency_status", lambda: {name: True for name in ("torch", "transformers", "accelerate", "peft", "bitsandbytes", "huggingface_hub", "safetensors", "packaging")})
    monkeypatch.setattr(finetuning_module, "dependency_details", lambda: {})
    monkeypatch.setattr(finetuning_module, "hardware_status", lambda: {"cuda_available": True, "gpu_memory_bytes": 16 * 1024**3})
    monkeypatch.setattr(finetuning_module, "_training_runtime_contract_errors", lambda config: [])
    report = preflight_report(deep_corpus=False)
    assert report["corpus_validation_depth"] == "shallow"
    assert any("authoritative 6,300-record deep validation is PASS" in item for item in report["blockers"])
    evidence = report["deep_validation_evidence"]
    assert evidence["source"] == "verification/interpreter_corpus_v1.5_validation.json"
    assert evidence["status"] == "PASS"
    assert evidence["deep"] is True
    assert evidence["bound_to_current_corpus"] is True
    assert isinstance(evidence["report_sha256"], str) and len(evidence["report_sha256"]) == 64
    assert evidence["corpus_manifest_file_sha256"] == report["corpus"]["manifest_sha256"]
    assert report["baseline"]["reason"].endswith("live 150-case baseline has not been run")


def test_preflight_distinguishes_missing_from_version_incompatible_dependencies(monkeypatch):
    corpus_dir = ROOT / "training" / "interpreter_corpus_v1.5"
    fake = CorpusAudit(
        directory=corpus_dir,
        manifest={},
        train=tuple({} for _ in range(5039)),
        validation=tuple({} for _ in range(1261)),
        review=tuple({} for _ in range(725)),
        review_counts={"pending-human-review": 725, "accepted": 0, "corrected": 0, "rejected": 0},
        review_complete=False,
        held_out_exclusion={},
        blockers=("human review incomplete: 725 of 725 queue records remain pending",),
    )
    monkeypatch.setattr(finetuning_module, "audit_corpus", lambda config, deep, workers: fake)
    monkeypatch.setattr(finetuning_module, "baseline_gate", lambda path: (True, "ok", {"status": "PASS"}))
    monkeypatch.setattr(finetuning_module, "dependency_details", lambda: {
        "missing_pkg": {"installed": False, "importable": False, "compatible": False},
        "old_pkg": {"installed": True, "importable": True, "compatible": False},
        "good_pkg": {"installed": True, "importable": True, "compatible": True},
    })
    monkeypatch.setattr(finetuning_module, "hardware_status", lambda: {"cuda_available": True, "gpu_memory_bytes": 16 * 1024**3})
    monkeypatch.setattr(finetuning_module, "_training_runtime_contract_errors", lambda config: [])
    report = preflight_report(deep_corpus=True, allow_unreviewed=True)
    assert any("missing or non-importable optional training dependencies: missing_pkg" in item for item in report["blockers"])
    assert any("installed optional training dependencies have incompatible versions: old_pkg" in item for item in report["blockers"])
    assert all("missing optional training dependencies" not in item for item in report["blockers"])
