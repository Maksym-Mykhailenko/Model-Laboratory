"""Reproducible Stage-7 QLoRA preparation and training for the local interpreter.

The desktop application does not import the optional ML training stack.  This module keeps
corpus/baseline/review gates and artifact identity in ordinary Python, then imports PyTorch,
Transformers, bitsandbytes and PEFT only after a user explicitly starts training.
"""
from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
import uuid

from . import __version__
from .build_identity import git_commit_or_build_id, source_tree_sha256
from .canonical import canonical_json_sha256
from .interpreter import (
    GENERATION_OPTIONS,
    FROZEN_BASE_ARTIFACT_LOCK_SHA256,
    FROZEN_BASE_IDENTITY_VERIFICATION,
    FROZEN_BASE_MANIFEST_SHA256,
    FROZEN_BASE_MODEL_BLOB_SHA256,
    FROZEN_BASE_MODEL_BLOB_SIZE_BYTES,
    LOCAL_BASE_MODEL,
    _create_dialogue_turn_evidence,
    create_interpreter_context,
    generation_options_for_request,
    process_interpreter_output,
)
from .interpreter_training import (
    load_jsonl,
    materialize_sft_record,
    validate_corpus_examples,
    validate_corpus_examples_parallel,
    validate_held_out_exclusion,
    validate_review_queue,
)


FINETUNING_CONFIG_SCHEMA = "model-laboratory-interpreter-qlora-config"
FINETUNING_CONFIG_VERSION = "1.7"
PREVIOUS_FINETUNING_CONFIG_VERSION = "1.6"
LEGACY_FINETUNING_CONFIG_VERSIONS = {"1.0", "1.1", "1.2", "1.3", "1.4", "1.5"}
ENVIRONMENT_LOCK_CONFIG_VERSIONS = {"1.2", "1.3", "1.4", "1.5", "1.6", "1.7"}
ENVIRONMENT_RECEIPT_CONFIG_VERSIONS = {"1.3", "1.4", "1.5", "1.6", "1.7"}
FINETUNING_REPORT_SCHEMA = "model-laboratory-interpreter-qlora-report"
FINETUNING_REPORT_VERSION = "1.4"
TRAINING_ENVIRONMENT_LOCK_SCHEMA = "model-laboratory-interpreter-training-environment-lock"
TRAINING_ENVIRONMENT_LOCK_VERSION = "1.2"
LEGACY_TRAINING_ENVIRONMENT_LOCK_VERSIONS = {"1.1"}
TRAINING_ENVIRONMENT_RECEIPT_SCHEMA = (
    "model-laboratory-interpreter-training-environment-receipt"
)
TRAINING_ENVIRONMENT_RECEIPT_VERSION = "1.0"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = _PROJECT_ROOT / "training" / "interpreter_qlora_v1.7.json"
DEFAULT_DEVELOPMENT_BASELINE_REPORT = (
    _PROJECT_ROOT / "verification" / "interpreter_baseline_report.json"
)
DEFAULT_DEVELOPMENT_BENCHMARK = (
    _PROJECT_ROOT / "verification" / "interpreter_baseline_v1.6.json"
)
# Backwards-compatible names now explicitly designate the repeatable development campaign.
DEFAULT_BASELINE_REPORT = DEFAULT_DEVELOPMENT_BASELINE_REPORT
DEFAULT_BENCHMARK = DEFAULT_DEVELOPMENT_BENCHMARK
FROZEN_UPSTREAM_REVISION = "abcc171021d4f320b2e7f47c6f0deca67ded870c"
OFFICIAL_CONFIG_SHA256 = "47eddd1b58b8951d596575e9e3ea88c2143020d5832dc2bdfc822def882b208f"
OFFICIAL_TRAINING_ENVIRONMENT_LOCK_SHA256 = "483c28888ecccfc04053726697101e610acbb8c0f8624d60534709c16d68fc77"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_PYTHON_RELEASE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:\.(0|[1-9][0-9]*))?$")
SNAPSHOT_RECEIPT = ".model_laboratory_snapshot.json"


class FineTuningError(RuntimeError):
    """A gate, identity, dependency, hardware, or training contract failed."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FineTuningError(f"Cannot read JSON document '{path}': {exc}") from exc
    if not isinstance(value, dict):
        raise FineTuningError(f"JSON document '{path}' must contain an object.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any], *, sort_keys: bool = False) -> None:
    text = json.dumps(
        dict(value), indent=2, sort_keys=sort_keys, ensure_ascii=False, allow_nan=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()


def _exact_keys(value: Mapping[str, Any], required: set[str], field: str) -> None:
    if set(value) != required:
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unknown " + ", ".join(extra))
        raise FineTuningError(f"{field} has invalid fields ({'; '.join(detail)}).")


def _python_release_tuple(value: object, field: str) -> tuple[int, int, int]:
    match = _PYTHON_RELEASE.fullmatch(str(value))
    if match is None:
        raise FineTuningError(f"{field} must be a stable numeric Python release.")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def _python_version_supported(version: object, contract: Mapping[str, Any]) -> bool:
    observed = _python_release_tuple(version, "Python version")
    if "exact_version" in contract:
        return observed == _python_release_tuple(
            contract["exact_version"], "Exact Python version"
        )
    minimum = _python_release_tuple(
        contract.get("minimum_version"), "Minimum Python version"
    )
    maximum = _python_release_tuple(
        contract.get("maximum_exclusive_version"),
        "Maximum-exclusive Python version",
    )
    return minimum <= observed < maximum


def load_training_environment_lock(path: str | Path) -> dict[str, Any]:
    """Load the policy that an immutable pre-training environment receipt must satisfy."""
    lock_path = Path(path).resolve()
    value = _json(lock_path)
    _exact_keys(
        value,
        {
            "schema", "schema_version", "python", "direct_packages", "platform",
            "runtime", "environment_receipt", "determinism",
        },
        "Training environment lock",
    )
    lock_version = value.get("schema_version")
    if (
        value.get("schema") != TRAINING_ENVIRONMENT_LOCK_SCHEMA
        or lock_version
        not in {TRAINING_ENVIRONMENT_LOCK_VERSION, *LEGACY_TRAINING_ENVIRONMENT_LOCK_VERSIONS}
    ):
        raise FineTuningError("Unsupported training environment lock schema.")
    python_contract = value.get("python")
    packages = value.get("direct_packages")
    platform_contract = value.get("platform")
    runtime = value.get("runtime")
    receipt_contract = value.get("environment_receipt")
    determinism = value.get("determinism")
    if not all(
        isinstance(item, Mapping)
        for item in (
            python_contract, packages, platform_contract, runtime, receipt_contract,
            determinism,
        )
    ):
        raise FineTuningError("Training environment lock sections must be objects.")
    _exact_keys(
        python_contract,
        (
            {"implementation", "minimum_version", "maximum_exclusive_version", "hash_seed"}
            if lock_version == TRAINING_ENVIRONMENT_LOCK_VERSION
            else {"implementation", "exact_version", "hash_seed"}
        ),
        "Training environment Python contract",
    )
    _exact_keys(
        platform_contract,
        {"allowed_systems", "allowed_machines"},
        "Training environment platform contract",
    )
    _exact_keys(
        runtime,
        {
            "cuda_required", "minimum_gpu_memory_bytes", "cublas_workspace_config",
            "freeze_cuda_version", "freeze_cudnn_version", "freeze_gpu_identity",
        },
        "Training environment runtime contract",
    )
    _exact_keys(
        receipt_contract,
        {
            "schema", "schema_version", "require_complete_distribution_inventory",
            "require_distribution_record_hashes", "require_distribution_content_hashes",
            "require_python_executable_sha256", "must_precede_training",
        },
        "Training environment receipt contract",
    )
    _exact_keys(
        determinism,
        {
            "seed_source", "torch_deterministic_algorithms",
            "transformers_full_determinism", "cuda_tf32", "cudnn_tf32",
            "cudnn_benchmark", "dataloader_workers",
        },
        "Training determinism contract",
    )
    expected_packages = {
        "torch": "2.8.0",
        "transformers": "5.15.1",
        "accelerate": "1.12.0",
        "peft": "0.20.0",
        "bitsandbytes": "0.50.1",
        "huggingface-hub": "1.5.0",
        "safetensors": "0.8.0",
        "packaging": "26.0",
    }
    if dict(packages) != expected_packages:
        raise FineTuningError("Training environment direct-package versions differ from v1.1.")
    python_policy_valid = (
        python_contract.get("implementation") == "CPython"
        and python_contract.get("hash_seed") == "0"
        and (
            (
                lock_version == TRAINING_ENVIRONMENT_LOCK_VERSION
                and python_contract.get("minimum_version") == "3.10"
                and python_contract.get("maximum_exclusive_version") == "3.14"
            )
            or (
                lock_version in LEGACY_TRAINING_ENVIRONMENT_LOCK_VERSIONS
                and python_contract.get("exact_version") == "3.12.10"
            )
        )
    )
    if (
        not python_policy_valid
        or platform_contract.get("allowed_systems") != ["Windows", "Linux"]
        or platform_contract.get("allowed_machines") != ["AMD64", "x86_64"]
        or runtime.get("cuda_required") is not True
        or runtime.get("minimum_gpu_memory_bytes") != 12 * 1024**3
        or runtime.get("cublas_workspace_config") != ":4096:8"
        or runtime.get("freeze_cuda_version") is not True
        or runtime.get("freeze_cudnn_version") is not True
        or runtime.get("freeze_gpu_identity") is not True
        or dict(receipt_contract) != {
            "schema": TRAINING_ENVIRONMENT_RECEIPT_SCHEMA,
            "schema_version": TRAINING_ENVIRONMENT_RECEIPT_VERSION,
            "require_complete_distribution_inventory": True,
            "require_distribution_record_hashes": True,
            "require_distribution_content_hashes": True,
            "require_python_executable_sha256": True,
            "must_precede_training": True,
        }
        or dict(determinism) != {
            "seed_source": "qlora-config",
            "torch_deterministic_algorithms": True,
            "transformers_full_determinism": True,
            "cuda_tf32": False,
            "cudnn_tf32": False,
            "cudnn_benchmark": False,
            "dataloader_workers": 0,
        }
    ):
        raise FineTuningError("Training environment runtime/determinism policy is unsupported.")
    value["_path"] = str(lock_path)
    value["_sha256"] = _sha256_file(lock_path)
    return value


def _normalised_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _distribution_inventory(*, hash_payloads: bool) -> list[dict[str, Any]]:
    """Return a complete deterministic inventory of the active Python environment.

    Wheel ``RECORD`` files bind installation layout and upstream wheel hashes.  The additional
    content hash streams every installed payload, so modifications after installation are also
    detected without retaining file bytes in memory.
    """
    inventory: list[dict[str, Any]] = []
    names: set[str] = set()
    environment_root = Path(sys.prefix).resolve()
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise FineTuningError("An installed distribution has no canonical Name metadata.")
        name = _normalised_distribution_name(raw_name)
        if name in names:
            raise FineTuningError(f"Duplicate installed distribution identity: {name}.")
        names.add(name)
        record_text = distribution.read_text("RECORD")
        if not isinstance(record_text, str) or not record_text.strip():
            raise FineTuningError(
                f"Installed distribution {name} has no wheel RECORD and cannot be frozen exactly."
            )
        record_sha = hashlib.sha256(record_text.encode("utf-8")).hexdigest()
        rows = list(csv.reader(record_text.splitlines()))
        content = hashlib.sha256()
        file_count = 0
        for row in sorted(rows, key=lambda item: item[0] if item else ""):
            if not row or not row[0]:
                continue
            located = Path(distribution.locate_file(row[0])).resolve()
            if not located.is_relative_to(environment_root):
                raise FineTuningError(
                    f"Installed distribution {name} RECORD escapes the active Python environment."
                )
            if not located.is_file():
                # Bytecode/cache members may legally disappear. Their absence is itself bound.
                content.update(b"missing\0")
                content.update(row[0].encode("utf-8"))
                content.update(b"\0")
                continue
            file_count += 1
            content.update(b"file\0")
            content.update(row[0].encode("utf-8"))
            content.update(b"\0")
            declared_hash = row[1] if len(row) > 1 else ""
            if hash_payloads and declared_hash:
                member_sha = hashlib.sha256()
                with located.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        member_sha.update(block)
                content.update(member_sha.hexdigest().encode("ascii"))
            else:
                # RECORD deliberately omits hashes for generated bytecode and RECORD itself.
                # Bind the declared marker/path, not mutable interpreter caches.
                content.update(declared_hash.encode("utf-8"))
            content.update(b"\0")
        inventory.append(
            {
                "name": name,
                "version": distribution.version,
                "record_sha256": record_sha,
                "content_sha256": content.hexdigest(),
                "file_count": file_count,
            }
        )
    return sorted(inventory, key=lambda item: item["name"])


def _nvidia_driver_version() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    values = sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})
    return ",".join(values) if values else "unavailable"


def _numerical_runtime_identity() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise FineTuningError("PyTorch must be installed before freezing the training environment.") from exc
    if not torch.cuda.is_available():
        raise FineTuningError("A CUDA GPU must be available before freezing the training environment.")
    properties = torch.cuda.get_device_properties(0)
    return {
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
        "nvidia_driver_version": _nvidia_driver_version(),
        "gpu_name": str(properties.name),
        "gpu_total_memory_bytes": int(properties.total_memory),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
    }


def _environment_projection(*, hash_payloads: bool) -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    inventory = _distribution_inventory(hash_payloads=hash_payloads)
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": str(sys.implementation.cache_tag),
            "executable_sha256": _sha256_file(executable),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "distributions": inventory,
        "distribution_inventory_sha256": canonical_json_sha256(inventory),
        "numerical_runtime": _numerical_runtime_identity(),
        "deterministic_process": {
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        },
    }


def _validate_environment_projection(
    projection: Mapping[str, Any], lock: Mapping[str, Any]
) -> None:
    python_identity = projection.get("python")
    platform_identity = projection.get("platform")
    process = projection.get("deterministic_process")
    runtime = projection.get("numerical_runtime")
    distributions = projection.get("distributions")
    if not all(
        isinstance(item, Mapping)
        for item in (python_identity, platform_identity, process, runtime)
    ) or not isinstance(distributions, list):
        raise FineTuningError("Training environment receipt projection is incomplete.")
    _exact_keys(
        python_identity,
        {"implementation", "version", "cache_tag", "executable_sha256"},
        "Training receipt Python identity",
    )
    _exact_keys(
        platform_identity,
        {"system", "release", "version", "machine"},
        "Training receipt platform identity",
    )
    _exact_keys(
        process,
        {"pythonhashseed", "cublas_workspace_config"},
        "Training receipt deterministic process",
    )
    _exact_keys(
        runtime,
        {
            "torch_version", "cuda_version", "cudnn_version",
            "nvidia_driver_version", "gpu_name", "gpu_total_memory_bytes",
            "gpu_compute_capability",
        },
        "Training receipt numerical runtime",
    )
    if (
        _SHA256.fullmatch(str(python_identity.get("executable_sha256", ""))) is None
        or not isinstance(python_identity.get("cache_tag"), str)
        or not python_identity["cache_tag"]
    ):
        raise FineTuningError("Training receipt Python executable identity is invalid.")
    if python_identity.get("implementation") != lock["python"]["implementation"]:
        raise FineTuningError("Python implementation differs from the environment policy.")
    if not _python_version_supported(python_identity.get("version"), lock["python"]):
        if "exact_version" in lock["python"]:
            requirement = str(lock["python"]["exact_version"])
        else:
            requirement = (
                f">={lock['python']['minimum_version']},"
                f"<{lock['python']['maximum_exclusive_version']}"
            )
        raise FineTuningError(
            f"Python version is outside the supported training range ({requirement})."
        )
    if (
        process.get("pythonhashseed") != lock["python"]["hash_seed"]
        or process.get("cublas_workspace_config")
        != lock["runtime"]["cublas_workspace_config"]
    ):
        raise FineTuningError(
            "Deterministic process settings differ from the environment policy."
        )
    if (
        platform_identity.get("system") not in lock["platform"]["allowed_systems"]
        or platform_identity.get("machine") not in lock["platform"]["allowed_machines"]
    ):
        raise FineTuningError("Platform is outside the frozen training policy.")
    by_name = {
        item.get("name"): item
        for item in distributions
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    if len(by_name) != len(distributions):
        raise FineTuningError("Training distribution inventory has duplicate or malformed entries.")
    if projection.get("distribution_inventory_sha256") != canonical_json_sha256(
        distributions
    ):
        raise FineTuningError("Training distribution inventory projection is inconsistent.")
    for name, version in lock["direct_packages"].items():
        item = by_name.get(_normalised_distribution_name(name))
        if not isinstance(item, Mapping) or item.get("version") != version:
            raise FineTuningError(f"Frozen direct package differs: {name}=={version} is required.")
    for item in distributions:
        if not isinstance(item, Mapping) or set(item) != {
            "name", "version", "record_sha256", "content_sha256", "file_count"
        }:
            raise FineTuningError("Training distribution receipt entry is malformed.")
        if (
            not isinstance(item.get("version"), str)
            or _SHA256.fullmatch(str(item.get("record_sha256", ""))) is None
            or _SHA256.fullmatch(str(item.get("content_sha256", ""))) is None
            or isinstance(item.get("file_count"), bool)
            or not isinstance(item.get("file_count"), int)
            or item["file_count"] <= 0
        ):
            raise FineTuningError("Training distribution receipt identity is incomplete.")
    capability = runtime.get("gpu_compute_capability")
    if (
        runtime.get("cuda_version") in {None, "None", ""}
        or isinstance(runtime.get("cudnn_version"), bool)
        or not isinstance(runtime.get("cudnn_version"), int)
        or int(runtime["cudnn_version"]) <= 0
        or not isinstance(runtime.get("gpu_name"), str)
        or not runtime["gpu_name"]
        or not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in capability)
    ):
        raise FineTuningError("CUDA/cuDNN identity could not be frozen.")
    if int(runtime.get("gpu_total_memory_bytes", 0)) < int(
        lock["runtime"]["minimum_gpu_memory_bytes"]
    ):
        raise FineTuningError("The frozen GPU has less memory than the training policy requires.")


def freeze_training_environment_receipt(
    lock_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze the complete current environment before any empirical training begins."""
    lock = load_training_environment_lock(lock_path)
    projection = _environment_projection(hash_payloads=True)
    _validate_environment_projection(projection, lock)
    receipt: dict[str, Any] = {
        "schema": TRAINING_ENVIRONMENT_RECEIPT_SCHEMA,
        "schema_version": TRAINING_ENVIRONMENT_RECEIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy_sha256": lock["_sha256"],
        **projection,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    _atomic_json(Path(output_path).resolve(), receipt, sort_keys=True)
    return receipt


def validate_training_environment_receipt(
    path: str | Path,
    lock: Mapping[str, Any],
    *,
    verify_current: bool,
) -> dict[str, Any]:
    receipt = _json(Path(path).resolve())
    required = {
        "schema", "schema_version", "created_at_utc", "policy_sha256", "python",
        "platform", "distributions", "distribution_inventory_sha256",
        "numerical_runtime", "deterministic_process", "receipt_sha256",
    }
    _exact_keys(receipt, required, "Training environment receipt")
    if (
        receipt.get("schema") != TRAINING_ENVIRONMENT_RECEIPT_SCHEMA
        or receipt.get("schema_version") != TRAINING_ENVIRONMENT_RECEIPT_VERSION
        or receipt.get("policy_sha256") != lock.get("_sha256")
    ):
        raise FineTuningError("Training environment receipt does not match the frozen policy.")
    declared = receipt.get("receipt_sha256")
    without_sha = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    if (
        not isinstance(declared, str)
        or _SHA256.fullmatch(declared) is None
        or declared != canonical_json_sha256(without_sha)
    ):
        raise FineTuningError("Training environment receipt checksum is invalid.")
    try:
        created = datetime.fromisoformat(str(receipt.get("created_at_utc")))
    except (TypeError, ValueError) as exc:
        raise FineTuningError("Training environment receipt timestamp is malformed.") from exc
    if created.tzinfo is None:
        raise FineTuningError("Training environment receipt timestamp must include a UTC offset.")
    inventory = receipt.get("distributions")
    if not isinstance(inventory, list) or receipt.get(
        "distribution_inventory_sha256"
    ) != canonical_json_sha256(inventory):
        raise FineTuningError("Training environment distribution inventory checksum is invalid.")
    _validate_environment_projection(receipt, lock)
    if verify_current:
        current = _environment_projection(hash_payloads=True)
        _validate_environment_projection(current, lock)
        for field in (
            "python", "platform", "distributions", "distribution_inventory_sha256",
            "numerical_runtime", "deterministic_process",
        ):
            if current[field] != receipt[field]:
                raise FineTuningError(
                    f"Current training environment differs from its frozen receipt: {field}."
                )
    return receipt


def load_qlora_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and strictly validate the frozen Stage-7 configuration."""
    config_path = Path(path).resolve()
    value = _json(config_path)
    version = value.get("schema_version")
    required_config_keys = {
        "schema", "schema_version", "seed", "base_model", "corpus",
        "quantization", "lora", "sequence", "optimization",
    }
    if version in ENVIRONMENT_LOCK_CONFIG_VERSIONS:
        required_config_keys.add("training_environment_lock")
    if version in ENVIRONMENT_RECEIPT_CONFIG_VERSIONS:
        required_config_keys.add("training_environment_receipt")
    _exact_keys(value, required_config_keys, "QLoRA configuration")
    if value["schema"] != FINETUNING_CONFIG_SCHEMA or version not in {
        *LEGACY_FINETUNING_CONFIG_VERSIONS,
        PREVIOUS_FINETUNING_CONFIG_VERSION,
        FINETUNING_CONFIG_VERSION,
    }:
        raise FineTuningError("Unsupported QLoRA configuration schema.")
    if isinstance(value["seed"], bool) or not isinstance(value["seed"], int):
        raise FineTuningError("QLoRA seed must be an integer.")
    base = value["base_model"]
    corpus = value["corpus"]
    quant = value["quantization"]
    lora = value["lora"]
    sequence = value["sequence"]
    optimization = value["optimization"]
    if not all(isinstance(item, Mapping) for item in (base, corpus, quant, lora, sequence, optimization)):
        raise FineTuningError("QLoRA configuration sections must be objects.")
    _exact_keys(base, {"repository", "revision", "lock_file"}, "base_model")
    _exact_keys(
        corpus,
        {
            "directory", "require_completed_human_review", "require_valid_full_baseline",
            "content_sha256", "train_sha256", "validation_sha256",
        },
        "corpus",
    )
    _exact_keys(
        quant,
        {"method", "load_in_4bit", "quant_type", "double_quantization", "compute_dtype"},
        "quantization",
    )
    _exact_keys(lora, {"rank", "alpha", "dropout", "target_modules", "bias"}, "lora")
    _exact_keys(
        sequence,
        {"maximum_tokens", "truncate", "assistant_completion_only", "packing"},
        "sequence",
    )
    _exact_keys(
        optimization,
        {
            "epochs", "learning_rate", "micro_batch_size", "gradient_accumulation_steps",
            "optimizer", "scheduler", "warmup_ratio", "weight_decay",
            "maximum_gradient_norm", "gradient_checkpointing", "evaluation_strategy",
            "save_strategy",
        },
        "optimization",
    )
    if (
        base["repository"] != LOCAL_BASE_MODEL
        or base["revision"] != FROZEN_UPSTREAM_REVISION
        or not isinstance(base["revision"], str)
        or not _GIT_COMMIT.fullmatch(base["revision"])
    ):
        raise FineTuningError("QLoRA base model must use the exact frozen Qwen repository and revision.")
    if not all(
        isinstance(corpus[field], bool)
        for field in ("require_completed_human_review", "require_valid_full_baseline")
    ) or not all(
        corpus[field]
        for field in ("require_completed_human_review", "require_valid_full_baseline")
    ):
        raise FineTuningError("QLoRA corpus review and baseline gates must both be enabled.")
    for field in ("content_sha256", "train_sha256", "validation_sha256"):
        if not isinstance(corpus[field], str) or _SHA256.fullmatch(corpus[field]) is None:
            raise FineTuningError(f"corpus.{field} must be a frozen SHA-256 digest.")
    if quant != {
        "method": "qlora-nf4",
        "load_in_4bit": True,
        "quant_type": "nf4",
        "double_quantization": True,
        "compute_dtype": "auto-bf16-or-fp16",
    }:
        raise FineTuningError("Only the frozen NF4 double-quantized QLoRA method is supported.")
    if (
        isinstance(lora["rank"], bool)
        or not isinstance(lora["rank"], int)
        or lora["rank"] < 1
        or isinstance(lora["alpha"], bool)
        or not isinstance(lora["alpha"], int)
        or lora["alpha"] < 1
        or isinstance(lora["dropout"], bool)
        or not isinstance(lora["dropout"], (int, float))
        or not math.isfinite(float(lora["dropout"]))
        or not 0 <= float(lora["dropout"]) < 1
        or lora["target_modules"] != "all-linear"
        or lora["bias"] != "none"
    ):
        raise FineTuningError("QLoRA adapter configuration is invalid.")
    if (
        isinstance(sequence["maximum_tokens"], bool)
        or not isinstance(sequence["maximum_tokens"], int)
        or sequence["maximum_tokens"] < 1024
        or sequence["truncate"] is not False
        or sequence["assistant_completion_only"] is not True
        or sequence["packing"] is not False
    ):
        raise FineTuningError("Training must use untruncated, unpacked, completion-only sequences.")
    for field in ("epochs", "learning_rate", "warmup_ratio", "weight_decay", "maximum_gradient_norm"):
        candidate = optimization[field]
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)) or not math.isfinite(float(candidate)):
            raise FineTuningError(f"optimization.{field} must be finite numeric data.")
    if (
        float(optimization["epochs"]) <= 0
        or float(optimization["learning_rate"]) <= 0
        or not 0 <= float(optimization["warmup_ratio"]) < 1
        or float(optimization["weight_decay"]) < 0
        or float(optimization["maximum_gradient_norm"]) <= 0
    ):
        raise FineTuningError("QLoRA numeric optimisation settings are outside safe bounds.")
    for field in ("micro_batch_size", "gradient_accumulation_steps"):
        candidate = optimization[field]
        if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < 1:
            raise FineTuningError(f"optimization.{field} must be a positive integer.")
    if (
        optimization["optimizer"] != "paged_adamw_8bit"
        or optimization["scheduler"] != "cosine"
        or optimization["evaluation_strategy"] != "epoch"
        or optimization["save_strategy"] != "epoch"
    ):
        raise FineTuningError("QLoRA optimiser/scheduler/checkpoint policy differs from v1.0.")
    value["_path"] = str(config_path)
    value["_sha256"] = _sha256_file(config_path)
    if version in ENVIRONMENT_RECEIPT_CONFIG_VERSIONS:
        environment_path = _resolve_project_path(
            value,
            value["training_environment_lock"],
            "training_environment_lock",
        )
        environment = load_training_environment_lock(environment_path)
        if (
            version == FINETUNING_CONFIG_VERSION
            and environment["_sha256"] != OFFICIAL_TRAINING_ENVIRONMENT_LOCK_SHA256
        ):
            raise FineTuningError(
                "The official QLoRA configuration references a modified training environment lock."
            )
        value["_training_environment"] = environment
        value["_training_environment_lock_sha256"] = environment["_sha256"]
        receipt_path = _resolve_project_path(
            value,
            value["training_environment_receipt"],
            "training_environment_receipt",
        )
        value["_training_environment_receipt_path"] = str(receipt_path)
        value["_training_environment_receipt"] = (
            validate_training_environment_receipt(
                receipt_path, environment, verify_current=False
            )
            if receipt_path.is_file()
            else None
        )
        value["_training_environment_receipt_sha256"] = (
            value["_training_environment_receipt"]["receipt_sha256"]
            if value["_training_environment_receipt"] is not None
            else None
        )
    else:
        value["_training_environment"] = None
        value["_training_environment_lock_sha256"] = None
        value["_training_environment_receipt_path"] = None
        value["_training_environment_receipt"] = None
        value["_training_environment_receipt_sha256"] = None
    value["_official_frozen_config"] = (
        config_path == DEFAULT_CONFIG.resolve()
        and value["_sha256"] == OFFICIAL_CONFIG_SHA256
        and value["_training_environment_lock_sha256"]
        == OFFICIAL_TRAINING_ENVIRONMENT_LOCK_SHA256
    )
    return value


def _resolve_project_path(config: Mapping[str, Any], value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FineTuningError(f"{field} must be a project-relative path.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise FineTuningError(f"{field} must stay inside the project tree.")
    return (_PROJECT_ROOT / path).resolve()


def verify_base_snapshot(snapshot: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the exact tokenizer and upstream safetensor shards before quantized loading."""
    directory = Path(snapshot).resolve()
    receipt_path = directory / SNAPSHOT_RECEIPT
    receipt = _json(receipt_path)
    if receipt.get("schema") != "model-laboratory-huggingface-snapshot" or receipt.get("schema_version") != "1.0":
        raise FineTuningError(
            f"{SNAPSHOT_RECEIPT} is missing or has an unsupported schema; use the pinned download command."
        )
    declared_receipt_sha = receipt.get("receipt_sha256")
    receipt_without_sha = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if (
        not isinstance(declared_receipt_sha, str)
        or declared_receipt_sha != canonical_json_sha256(receipt_without_sha)
    ):
        raise FineTuningError("The base-model snapshot receipt does not verify.")
    if (
        receipt.get("repository") != config["base_model"]["repository"]
        or receipt.get("revision") != config["base_model"]["revision"]
    ):
        raise FineTuningError("The downloaded snapshot receipt names a different base model.")
    receipt_files = receipt.get("files")
    if not isinstance(receipt_files, list) or not receipt_files:
        raise FineTuningError("The downloaded snapshot receipt has no file identities.")
    expected_receipt_files: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(receipt_files):
        if not isinstance(item, Mapping) or set(item) != {"path", "size", "sha256"}:
            raise FineTuningError(f"Snapshot receipt file {index + 1} is malformed.")
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in expected_receipt_files
        ):
            raise FineTuningError("Snapshot receipt contains an unsafe or duplicate path.")
        if (
            isinstance(item.get("size"), bool)
            or not isinstance(item.get("size"), int)
            or item["size"] < 0
            or not isinstance(item.get("sha256"), str)
            or _SHA256.fullmatch(item["sha256"]) is None
        ):
            raise FineTuningError("Snapshot receipt contains an invalid file identity.")
        expected_receipt_files[relative] = item
    observed_paths = {
        path.relative_to(directory).as_posix(): path
        for path in directory.rglob("*")
        if path.is_file()
        and path.name != SNAPSHOT_RECEIPT
        and not any(part.startswith(".") for part in path.relative_to(directory).parts)
    }
    if set(observed_paths) != set(expected_receipt_files):
        raise FineTuningError("Base-model snapshot membership differs from its download receipt.")
    required_snapshot_members = {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    if set(observed_paths) != required_snapshot_members:
        raise FineTuningError(
            "Base-model snapshot membership differs from the exact Stage-7 downloader contract."
        )
    for relative, path in observed_paths.items():
        identity = expected_receipt_files[relative]
        if (
            identity.get("size") != path.stat().st_size
            or identity.get("sha256") != _sha256_file(path)
        ):
            raise FineTuningError(f"Base-model snapshot file differs from its receipt: {relative}.")
    # Bind the small configuration files semantically as well as through the download receipt.
    # The full weights/tokenizer bytes are independently pinned below.
    model_config = _json(directory / "config.json")
    expected_config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "max_position_embeddings": 262144,
        "torch_dtype": "bfloat16",
    }
    if any(model_config.get(key) != value for key, value in expected_config.items()):
        raise FineTuningError("Frozen Qwen config.json differs from the Stage-7 architecture contract.")
    generation_config = _json(directory / "generation_config.json")
    for key, value in {
        "bos_token_id": 151643, "pad_token_id": 151643,
        "temperature": 0.7, "top_k": 20, "top_p": 0.8,
    }.items():
        if generation_config.get(key) != value:
            raise FineTuningError("Frozen Qwen generation_config.json differs from the Stage-7 contract.")
    if generation_config.get("eos_token_id") != [151645, 151643]:
        raise FineTuningError("Frozen Qwen generation_config.json has unexpected EOS tokens.")
    tokenizer_config = _json(directory / "tokenizer_config.json")
    chat_template = tokenizer_config.get("chat_template")
    if (
        not isinstance(chat_template, str)
        or "<|im_start|>" not in chat_template
        or "<|im_end|>" not in chat_template
        or "<think>" in chat_template
        or "reasoning_content" in chat_template
    ):
        raise FineTuningError("Frozen tokenizer_config.json is not the non-thinking Qwen Instruct template.")
    index = _json(directory / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or set(weight_map.values()) != {
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    }:
        raise FineTuningError("Frozen model index does not reference exactly the three locked shards.")

    lock_path = _resolve_project_path(config, config["base_model"]["lock_file"], "base_model.lock_file")
    lock = _json(lock_path)
    upstream = lock.get("upstream")
    if not isinstance(upstream, Mapping):
        raise FineTuningError("Base-model lock has no upstream identity.")
    if upstream.get("repository") != config["base_model"]["repository"] or upstream.get("revision") != config["base_model"]["revision"]:
        raise FineTuningError("Base-model lock does not match the QLoRA configuration.")
    expected: list[Mapping[str, Any]] = []
    tokenizer = upstream.get("tokenizer")
    weights = upstream.get("weight_files")
    if not isinstance(tokenizer, Mapping) or not isinstance(weights, list) or not all(isinstance(item, Mapping) for item in weights):
        raise FineTuningError("Base-model lock contains malformed file identities.")
    expected.append(tokenizer)
    expected.extend(weights)
    files: list[dict[str, Any]] = []
    for item in expected:
        filename = item.get("filename")
        digest = item.get("sha256")
        if not isinstance(filename, str) or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise FineTuningError("Base-model lock contains an invalid file digest.")
        path = directory / filename
        if not path.is_file():
            raise FineTuningError(f"Frozen base-model file is missing: {filename}.")
        observed = _sha256_file(path)
        if observed != digest:
            raise FineTuningError(f"Frozen base-model file checksum differs: {filename}.")
        files.append({"filename": filename, "sha256": observed, "size": path.stat().st_size})
    return {
        "repository": upstream["repository"],
        "revision": upstream["revision"],
        "lock_sha256": _sha256_file(lock_path),
        "receipt_sha256": declared_receipt_sha,
        "snapshot": str(directory),
        "files": files,
    }


def write_snapshot_receipt(
    snapshot: str | Path, *, repository: str, revision: str
) -> dict[str, Any]:
    """Bind every downloaded, non-cache model member to one pinned Hub revision."""
    directory = Path(snapshot).resolve()
    files = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory)
        if path.name == SNAPSHOT_RECEIPT or any(part.startswith(".") for part in relative.parts):
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    receipt: dict[str, Any] = {
        "schema": "model-laboratory-huggingface-snapshot",
        "schema_version": "1.0",
        "repository": repository,
        "revision": revision,
        "files": files,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    _atomic_json(directory / SNAPSHOT_RECEIPT, receipt, sort_keys=True)
    return receipt


def _review_decisions(review: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Mapping[str, Any]], dict[str, int], list[str]]:
    decisions: dict[str, Mapping[str, Any]] = {}
    counts = {"pending-human-review": 0, "accepted": 0, "corrected": 0, "rejected": 0}
    problems: list[str] = []
    for item in review:
        identifier = item.get("id")
        metadata = item.get("review")
        if not isinstance(identifier, str) or not isinstance(metadata, Mapping):
            problems.append("review queue contains a malformed decision")
            continue
        status = metadata.get("status")
        if status not in counts:
            problems.append(f"{identifier}: invalid review status")
            continue
        counts[str(status)] += 1
        if identifier in decisions:
            problems.append(f"{identifier}: duplicate review decision")
        decisions[identifier] = item
        if status == "pending-human-review":
            continue
        reviewer = metadata.get("reviewer")
        reviewed_at = metadata.get("reviewed_at_utc")
        notes = metadata.get("notes")
        corrected = metadata.get("corrected_target")
        if not isinstance(reviewer, str) or not reviewer.strip():
            problems.append(f"{identifier}: completed review has no reviewer")
        if not isinstance(reviewed_at, str) or not reviewed_at:
            problems.append(f"{identifier}: completed review has no timestamp")
        else:
            try:
                parsed = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    raise ValueError
            except ValueError:
                problems.append(f"{identifier}: review timestamp is not timezone-aware ISO-8601")
        if not isinstance(notes, str):
            problems.append(f"{identifier}: review notes must be text")
        if status == "corrected" and not isinstance(corrected, Mapping):
            problems.append(f"{identifier}: corrected review has no corrected target")
        if status != "corrected" and corrected is not None:
            problems.append(f"{identifier}: only corrected reviews may contain corrected_target")
        if status in {"corrected", "rejected"} and isinstance(notes, str) and not notes.strip():
            problems.append(f"{identifier}: corrected/rejected review requires a reason")
    return decisions, counts, problems


def _provider_identity(generation_options: Mapping[str, int | float]) -> dict[str, Any]:
    return {
        "provider": "ollama",
        "endpoint": "http://127.0.0.1:11434",
        "runtime_version": "stage7-corpus-validation",
        "model_tag": "qwen3:4b-instruct-2507-q4_K_M",
        "observed_model_digest": FROZEN_BASE_MANIFEST_SHA256,
        "identity_verification": FROZEN_BASE_IDENTITY_VERIFICATION,
        "expected_base_model": LOCAL_BASE_MODEL,
        "expected_quantization": "Q4_K_M",
        "generation_options": dict(generation_options),
        "model_role": "frozen_base",
        "registry_entry_sha256": None,
        "artifact_evidence": {
            "identity_verification": FROZEN_BASE_IDENTITY_VERIFICATION,
            "artifact_lock_sha256": FROZEN_BASE_ARTIFACT_LOCK_SHA256,
            "local_manifest_sha256": FROZEN_BASE_MANIFEST_SHA256,
            "model_blob_sha256": FROZEN_BASE_MODEL_BLOB_SHA256,
            "verified_model_blob_size_bytes": FROZEN_BASE_MODEL_BLOB_SIZE_BYTES,
            "all_manifest_blobs_verified": True,
        },
    }


def _validate_corrected_target(record: Mapping[str, Any], target: Mapping[str, Any]) -> None:
    conversation = record.get("conversation")
    if not isinstance(conversation, Mapping):
        raise FineTuningError("Corrected review record has malformed conversation data.")
    instruction = conversation.get("instruction")
    source = conversation.get("current_model_source")
    requested = conversation.get("requested_paths")
    round_index = conversation.get("round")
    clarification_history = conversation.get("clarification_history", [])
    if not isinstance(instruction, str) or not isinstance(source, str) or not isinstance(requested, list) or not isinstance(round_index, int) or not isinstance(clarification_history, list):
        raise FineTuningError("Corrected review record has malformed context inputs.")
    context = create_interpreter_context(
        instruction=instruction,
        current_model_source=source,
        requested_paths=requested,
        round_index=round_index,
        clarification_history=clarification_history,
    )
    generation_options = generation_options_for_request(
        instruction=instruction,
        context_document=context,
    )
    provider = _provider_identity(generation_options)
    conversation_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            "model-laboratory-review:" + str(conversation.get("conversation_id", "unknown")),
        )
    )
    lineage: list[dict[str, Any]] = []
    for index, turn in enumerate(clarification_history):
        lineage.append(
            _create_dialogue_turn_evidence(
                conversation_id=conversation_id,
                turn_index=index,
                parent_turn_sha256=lineage[-1]["turn_sha256"] if lineage else None,
                context_sha256=hashlib.sha256(
                    f"review:{conversation_id}:{index}".encode("utf-8")
                ).hexdigest(),
                raw_output={"action": "needs_clarification", "question": turn["question"]},
                provider_identity=provider,
                question=turn["question"],
            )
        )
    try:
        process_interpreter_output(
            raw_output=target,
            context_document=context,
            provider_identity=provider,
            instruction=instruction,
            current_model_source=source,
            clarification_history=clarification_history,
            conversation_id=conversation_id,
            conversation_lineage=lineage,
            expected_generation_options=generation_options,
        )
    except Exception as exc:
        raise FineTuningError(f"Corrected review target does not pass the production compiler: {exc}") from exc


def validate_review_records(
    corpus_directory: str | Path, review: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    """Validate prospective human-review data before it replaces the authoritative queue."""
    directory = Path(corpus_directory).resolve()
    examples = [
        *load_jsonl(directory / "train.jsonl"),
        *load_jsonl(directory / "validation.jsonl"),
    ]
    manifest = _json(directory / "manifest.json")
    validate_review_queue(
        review, examples, expected_count=manifest.get("review", {}).get("queue_count")
    )
    decisions, counts, problems = _review_decisions(review)
    by_id = {str(item.get("id")): item for item in examples}
    for identifier, decision in decisions.items():
        if decision["review"]["status"] != "corrected":
            continue
        source = by_id.get(identifier)
        if source is None:
            problems.append(f"{identifier}: corrected review has no source record")
            continue
        try:
            _validate_corrected_target(source, decision["review"]["corrected_target"])
        except FineTuningError as exc:
            problems.append(f"{identifier}: {exc}")
    if problems:
        raise FineTuningError("; ".join(problems[:20]))
    return counts


@dataclass(frozen=True)
class CorpusAudit:
    directory: Path
    manifest: Mapping[str, Any]
    train: tuple[Mapping[str, Any], ...]
    validation: tuple[Mapping[str, Any], ...]
    review: tuple[Mapping[str, Any], ...]
    review_counts: Mapping[str, int]
    review_complete: bool
    held_out_exclusion: Mapping[str, Any]
    blockers: tuple[str, ...]


def audit_corpus(
    config: Mapping[str, Any],
    *,
    deep: bool = False,
    workers: int | None = None,
) -> CorpusAudit:
    directory = _resolve_project_path(config, config["corpus"]["directory"], "corpus.directory")
    manifest_path = directory / "manifest.json"
    manifest = _json(manifest_path)
    train_path = directory / "train.jsonl"
    validation_path = directory / "validation.jsonl"
    review_path = directory / "review_queue.jsonl"
    if _sha256_file(train_path) != config["corpus"]["train_sha256"]:
        raise FineTuningError("Frozen training split differs from the Stage-7 QLoRA contract.")
    if _sha256_file(validation_path) != config["corpus"]["validation_sha256"]:
        raise FineTuningError("Frozen validation split differs from the Stage-7 QLoRA contract.")
    summary = manifest.get("summary")
    if not isinstance(summary, Mapping) or summary.get("content_sha256") != config["corpus"]["content_sha256"]:
        raise FineTuningError("Corpus scientific-content identity differs from the Stage-7 QLoRA contract.")
    train = tuple(load_jsonl(train_path))
    validation = tuple(load_jsonl(validation_path))
    review = tuple(load_jsonl(review_path))
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise FineTuningError("Corpus manifest has no file identity map.")
    for filename, records in (
        ("train.jsonl", train), ("validation.jsonl", validation), ("review_queue.jsonl", review)
    ):
        expected = files.get(filename)
        path = directory / filename
        if not isinstance(expected, Mapping):
            raise FineTuningError(f"Corpus manifest does not identify {filename}.")
        if expected.get("records") != len(records) or expected.get("bytes") != path.stat().st_size or expected.get("sha256") != _sha256_file(path):
            raise FineTuningError(f"Corpus member identity differs from manifest: {filename}.")
    examples = (*train, *validation)
    recomputed = (
        validate_corpus_examples_parallel(examples, workers=workers)
        if deep
        else validate_corpus_examples(examples, deep=False)
    )
    if recomputed != manifest.get("summary"):
        raise FineTuningError("Corpus summary does not match a fresh validation pass.")
    validate_review_queue(review, examples, expected_count=manifest.get("review", {}).get("queue_count"))
    decisions, counts, problems = _review_decisions(review)
    by_id = {str(item.get("id")): item for item in examples}
    for identifier, decision in decisions.items():
        status = decision["review"]["status"]
        if status == "corrected":
            source = by_id.get(identifier)
            if source is None:
                problems.append(f"{identifier}: corrected review has no source record")
            else:
                try:
                    _validate_corrected_target(source, decision["review"]["corrected_target"])
                except FineTuningError as exc:
                    problems.append(f"{identifier}: {exc}")
    held_out_exclusion = validate_held_out_exclusion(
        examples, benchmark_path=DEFAULT_BENCHMARK
    )
    complete = counts["pending-human-review"] == 0 and not problems
    blockers = list(problems)
    if config["corpus"]["require_completed_human_review"] and not complete:
        blockers.append(
            f"human review incomplete: {counts['pending-human-review']} of {len(review)} queue records remain pending"
        )
    return CorpusAudit(
        directory, manifest, train, validation, review, counts, complete,
        held_out_exclusion, tuple(blockers)
    )


def baseline_gate(
    path: str | Path = DEFAULT_BASELINE_REPORT,
    *,
    benchmark_path: str | Path = DEFAULT_DEVELOPMENT_BENCHMARK,
) -> tuple[bool, str, Mapping[str, Any]]:
    """Accept only a full baseline whose recorded inference evidence replays exactly.

    A report self-hash alone is not scientific evidence.  This gate binds the report to the
    explicitly supplied held-out benchmark, current evaluator/compiler/validator identity, frozen Ollama
    artifact, and then replays every recorded raw model response through the deterministic
    Stage-5 scorer.  Metrics are recomputed from those replayed cases before acceptance.
    """
    from .interpreter_baseline import (
        REPORT_SCHEMA,
        REPORT_SCHEMA_VERSION,
        EVALUATION_GENERATION_OPTIONS,
        InferenceResult,
        OllamaIdentity,
        _build_report,
        _normalize_ollama_manifest_digest,
        _score_case,
        _source_identity,
        baseline_contract,
        base_model_lock,
        load_benchmark,
    )

    report_path = Path(path).resolve()
    report = _json(report_path)

    def reject(reason: str) -> tuple[bool, str, Mapping[str, Any]]:
        return False, "no valid full untouched-model baseline: " + reason, report

    if report.get("status") == "benchmark_validated_model_not_run" and report.get("campaign_status") is None:
        return reject("live 150-case baseline has not been run")

    declared_sha = report.get("report_sha256")
    without_sha = {key: value for key, value in report.items() if key != "report_sha256"}
    if (
        not isinstance(declared_sha, str)
        or _SHA256.fullmatch(declared_sha) is None
        or declared_sha != canonical_json_sha256(without_sha)
    ):
        return reject("baseline report integrity hash is absent or invalid")
    if report.get("schema") != REPORT_SCHEMA or report.get("schema_version") != REPORT_SCHEMA_VERSION:
        return reject("baseline report schema does not match the current evaluator")
    if report.get("campaign_status") != "valid":
        return reject("baseline campaign is not a valid completed live run")
    if report.get("baseline_contract") != baseline_contract():
        return reject("baseline report uses a different prompt/model/evaluation contract")
    if report.get("implementation_identity") != _source_identity():
        return reject("baseline report was produced by different scientific evaluator/compiler code")

    try:
        benchmark = load_benchmark(benchmark_path)
    except Exception as exc:
        raise FineTuningError(f"Cannot validate the supplied evaluation benchmark: {exc}") from exc
    reported_benchmark = report.get("benchmark")
    if not isinstance(reported_benchmark, Mapping):
        return reject("baseline report has no benchmark identity")
    if (
        reported_benchmark.get("benchmark_sha256") != benchmark["benchmark_sha256"]
        or reported_benchmark.get("schema") != benchmark["schema"]
        or reported_benchmark.get("schema_version") != benchmark["schema_version"]
        or reported_benchmark.get("case_count") != len(benchmark["cases"])
        or reported_benchmark.get("family_counts") != benchmark["family_counts"]
        or reported_benchmark.get("training_exclusion") is not True
        or reported_benchmark.get("limited_run") is not False
    ):
        return reject("baseline report does not identify the exact supplied full benchmark")

    runtime = report.get("runtime_identity")
    if not isinstance(runtime, Mapping):
        return reject("baseline report has no verified runtime identity")
    required_runtime = {
        "runtime", "runtime_version", "observed_model_digest", "model_size_bytes",
        "local_manifest_sha256", "model_blob_digest", "verified_model_blob_size_bytes",
        "model_context_length", "identity_verification",
    }
    if set(runtime) != required_runtime:
        return reject("baseline runtime identity record is incomplete or has unknown fields")
    if runtime.get("runtime") != "ollama" or runtime.get("identity_verification") != "exact_local_manifest_and_all_blobs_sha256":
        return reject("baseline runtime identity was not produced by the strict frozen-artifact verifier")
    try:
        normalized_digest = _normalize_ollama_manifest_digest(str(runtime["observed_model_digest"]))
    except Exception:
        return reject("baseline runtime model digest is malformed")
    lock = base_model_lock()["ollama"]
    if normalized_digest != "sha256:" + lock["registry_manifest_sha256"]:
        return reject("baseline runtime manifest is not the frozen published Ollama artifact")
    if runtime.get("local_manifest_sha256") != normalized_digest:
        return reject("baseline local manifest hash does not match the recorded Ollama digest")
    expected_blob = "sha256:" + next(
        item["sha256"]
        for item in lock["layers"]
        if item["media_type"] == "application/vnd.ollama.image.model"
    )
    if runtime.get("model_blob_digest") != expected_blob:
        return reject("baseline runtime GGUF digest does not match the frozen Q4_K_M blob")
    for field in ("model_size_bytes", "verified_model_blob_size_bytes", "model_context_length"):
        if isinstance(runtime.get(field), bool) or not isinstance(runtime.get(field), int) or int(runtime[field]) <= 0:
            return reject(f"baseline runtime identity has invalid {field}")
    if int(runtime["model_context_length"]) < int(EVALUATION_GENERATION_OPTIONS["num_ctx"]):
        return reject("baseline runtime does not support the frozen evaluation context length")

    identity = OllamaIdentity(
        runtime_version=str(runtime["runtime_version"]),
        model_digest=str(runtime["observed_model_digest"]),
        model_size_bytes=int(runtime["model_size_bytes"]),
        local_manifest_sha256=str(runtime["local_manifest_sha256"]),
        model_blob_digest=str(runtime["model_blob_digest"]),
        verified_model_blob_size_bytes=int(runtime["verified_model_blob_size_bytes"]),
        model_context_length=int(runtime["model_context_length"]),
    )

    stored_cases = report.get("cases")
    if not isinstance(stored_cases, list) or len(stored_cases) != len(benchmark["cases"]):
        return reject(
            "baseline report does not contain every per-case evidence record from the supplied benchmark"
        )

    replayed_cases: list[dict[str, Any]] = []
    for expected_case, stored in zip(benchmark["cases"], stored_cases, strict=True):
        if not isinstance(stored, Mapping) or stored.get("id") != expected_case["id"]:
            return reject("baseline case order/identity differs from the frozen benchmark")
        rounds = stored.get("rounds")
        if not isinstance(rounds, list) or not rounds:
            return reject(f"baseline case {expected_case['id']} has no raw inference evidence")

        class _ReplayClient:
            def __init__(self, evidence: Sequence[object]) -> None:
                self._evidence = list(evidence)
                self._index = 0

            def infer(self, *, instruction: str, context: Mapping[str, Any]) -> InferenceResult:
                del instruction, context
                if self._index >= len(self._evidence):
                    raise FineTuningError("recorded baseline evidence ended before scorer replay completed")
                item = self._evidence[self._index]
                self._index += 1
                if not isinstance(item, Mapping):
                    raise FineTuningError("recorded baseline round is malformed")
                raw_content = item.get("raw_content")
                telemetry = item.get("telemetry")
                if not isinstance(raw_content, str) or not isinstance(telemetry, Mapping):
                    raise FineTuningError("recorded baseline round lacks raw content or telemetry")
                if hashlib.sha256(raw_content.encode("utf-8")).hexdigest() != item.get("raw_content_sha256"):
                    raise FineTuningError("recorded baseline raw-content hash does not verify")
                parsed_output = item.get("parsed_output")
                if parsed_output is not None:
                    try:
                        reparsed = json.loads(raw_content)
                    except json.JSONDecodeError as exc:
                        raise FineTuningError(
                            "recorded baseline parsed output exists but raw content is not JSON"
                        ) from exc
                    if reparsed != parsed_output:
                        raise FineTuningError(
                            "recorded baseline parsed output does not match its raw JSON evidence"
                        )
                return InferenceResult(
                    raw_content=raw_content,
                    parsed_output=parsed_output,
                    telemetry=dict(telemetry),
                    parse_error=str(item.get("model_output_error") or ""),
                )

        client = _ReplayClient(rounds)
        try:
            replayed = _score_case(expected_case, client, identity)
        except Exception as exc:
            return reject(f"baseline case {expected_case['id']} does not replay through the deterministic scorer: {exc}")
        if client._index != len(rounds):
            return reject(f"baseline case {expected_case['id']} contains unused/fabricated inference rounds")
        if replayed != dict(stored):
            return reject(f"baseline case {expected_case['id']} evidence does not reproduce its stored score")
        replayed_cases.append(replayed)

    rebuilt = _build_report(benchmark, identity=identity, results=replayed_cases, limit=None)
    if rebuilt.get("metrics") != report.get("metrics"):
        return reject("baseline aggregate metrics do not recompute from the per-case evidence")
    if rebuilt.get("benchmark") != report.get("benchmark"):
        return reject("baseline benchmark summary does not recompute from the per-case evidence")

    return True, "valid full untouched-model baseline with exact replayable evidence", report


_TRAINING_DEPENDENCIES: dict[str, tuple[str, str]] = {
    "torch": ("torch", "2.8.0"),
    "transformers": ("transformers", "5.15.1"),
    "accelerate": ("accelerate", "1.12.0"),
    "peft": ("peft", "0.20.0"),
    "bitsandbytes": ("bitsandbytes", "0.50.1"),
    "huggingface_hub": ("huggingface-hub", "1.5.0"),
    "safetensors": ("safetensors", "0.8.0"),
    "packaging": ("packaging", "26.0"),
}


def dependency_details() -> dict[str, dict[str, Any]]:
    """Import and exact-version-check every optional Stage-7 dependency."""
    result: dict[str, dict[str, Any]] = {}
    for module_name, (distribution, expected_version) in _TRAINING_DEPENDENCIES.items():
        record: dict[str, Any] = {
            "distribution": distribution,
            "required": f"=={expected_version}",
            "installed": False,
            "importable": False,
            "compatible": False,
            "version": None,
            "error": "",
        }
        try:
            version = importlib.metadata.version(distribution)
            record["installed"] = True
            record["version"] = version
            record["compatible"] = version == expected_version
            importlib.import_module(module_name)
            record["importable"] = True
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        result[module_name] = record
    return result


def dependency_status() -> dict[str, bool]:
    return {
        name: bool(item["installed"] and item["importable"] and item["compatible"])
        for name, item in dependency_details().items()
    }


def hardware_status() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda_available": False,
        "gpu": None,
        "gpu_memory_bytes": 0,
    }
    try:
        import torch
    except ImportError:
        result["reason"] = "PyTorch is not installed in this environment."
        return result
    result["torch_version"] = torch.__version__
    if not torch.cuda.is_available():
        result["reason"] = "No CUDA-capable NVIDIA GPU is available; CPU QLoRA is intentionally refused."
        return result
    properties = torch.cuda.get_device_properties(0)
    result.update(
        {
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "gpu": properties.name,
            "gpu_memory_bytes": int(properties.total_memory),
            "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "cudnn_version": torch.backends.cudnn.version(),
        }
    )
    return result


def _training_environment_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    lock = config.get("_training_environment")
    receipt = config.get("_training_environment_receipt")
    if not isinstance(lock, Mapping) or not isinstance(receipt, Mapping):
        raise FineTuningError(
            "Current Stage-7 training requires the policy and immutable environment receipt."
        )
    return {
        "lock_sha256": config["_training_environment_lock_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt_created_at_utc": receipt["created_at_utc"],
        "python": dict(receipt["python"]),
        "platform": dict(receipt["platform"]),
        "distribution_inventory_sha256": receipt["distribution_inventory_sha256"],
        "direct_packages": dict(lock["direct_packages"]),
        "numerical_runtime": dict(receipt["numerical_runtime"]),
        "deterministic_process": dict(receipt["deterministic_process"]),
        "determinism": dict(lock["determinism"]),
    }


def _configure_deterministic_environment(config: Mapping[str, Any]) -> None:
    lock = config.get("_training_environment")
    if not isinstance(lock, Mapping):
        raise FineTuningError(
            "Current Stage-7 training requires the exact versioned environment lock."
        )
    required = str(lock["runtime"]["cublas_workspace_config"])
    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed not in {None, required}:
        raise FineTuningError(
            "CUBLAS_WORKSPACE_CONFIG conflicts with the frozen determinism contract."
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = required


def _training_runtime_contract_errors(config: Mapping[str, Any]) -> list[str]:
    lock = config.get("_training_environment")
    if not isinstance(lock, Mapping):
        return ["the exact training environment lock is absent"]
    errors: list[str] = []
    if platform.python_implementation() != lock["python"]["implementation"]:
        errors.append("Python implementation differs from the training environment lock")
    try:
        supported_python = _python_version_supported(
            platform.python_version(), lock["python"]
        )
    except FineTuningError as exc:
        errors.append(str(exc))
    else:
        if not supported_python:
            if "exact_version" in lock["python"]:
                requirement = str(lock["python"]["exact_version"])
            else:
                requirement = (
                    f">={lock['python']['minimum_version']},"
                    f"<{lock['python']['maximum_exclusive_version']}"
                )
            errors.append(
                f"Python version is outside the supported training range ({requirement})"
            )
    if os.environ.get("PYTHONHASHSEED") != lock["python"]["hash_seed"]:
        errors.append("PYTHONHASHSEED was not fixed before interpreter startup")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != lock["runtime"][
        "cublas_workspace_config"
    ]:
        errors.append("CUBLAS_WORKSPACE_CONFIG is not the locked deterministic value")
    receipt_path = config.get("_training_environment_receipt_path")
    if not isinstance(receipt_path, str) or not Path(receipt_path).is_file():
        errors.append(
            "immutable training environment receipt is missing; run "
            "scripts/freeze_interpreter_training_environment.py first"
        )
    else:
        try:
            validate_training_environment_receipt(
                receipt_path, lock, verify_current=True
            )
        except FineTuningError as exc:
            errors.append(str(exc))
    return errors


def preflight_report(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    baseline_report: str | Path = DEFAULT_BASELINE_REPORT,
    deep_corpus: bool = True,
    corpus_workers: int | None = None,
    allow_unreviewed: bool = False,
    allow_missing_baseline: bool = False,
) -> dict[str, Any]:
    config = load_qlora_config(config_path)
    _configure_deterministic_environment(config)
    corpus = audit_corpus(config, deep=deep_corpus, workers=corpus_workers)
    baseline_valid, baseline_reason, baseline = baseline_gate(baseline_report)
    dependency_evidence = dependency_details()
    dependencies = {
        name: bool(item["installed"] and item["importable"] and item["compatible"])
        for name, item in dependency_evidence.items()
    }
    hardware = hardware_status()
    blockers = [
        item
        for item in corpus.blockers
        if not (allow_unreviewed and item.startswith("human review incomplete:"))
    ]
    blockers.extend(_training_runtime_contract_errors(config))
    deep_validation_evidence: dict[str, Any] = {
        "source": "current-preflight" if deep_corpus else "verification/interpreter_corpus_v1.5_validation.json",
        "status": "PASS" if deep_corpus else "UNAVAILABLE",
        "deep": bool(deep_corpus),
        "report_sha256": None,
        "corpus_manifest_file_sha256": _sha256_file(corpus.directory / "manifest.json"),
        "bound_to_current_corpus": bool(deep_corpus),
    }
    if not deep_corpus:
        deep_evidence = Path(__file__).resolve().parents[1] / "verification" / "interpreter_corpus_v1.5_validation.json"
        deep_note = "current preflight used shallow corpus validation"
        try:
            evidence = _json(deep_evidence)
            report_sha = _sha256_file(deep_evidence)
            bound = (
                evidence.get("status") == "PASS"
                and evidence.get("deep") is True
                and evidence.get("corpus_manifest_file_sha256") == _sha256_file(corpus.directory / "manifest.json")
            )
            deep_validation_evidence = {
                "source": "verification/interpreter_corpus_v1.5_validation.json",
                "status": str(evidence.get("status", "unknown")),
                "deep": bool(evidence.get("deep") is True),
                "report_sha256": report_sha,
                "corpus_manifest_file_sha256": evidence.get("corpus_manifest_file_sha256"),
                "bound_to_current_corpus": bool(bound),
            }
            if bound:
                deep_note += "; separately bound authoritative 6,300-record deep validation is PASS"
        except Exception as exc:
            deep_validation_evidence["error"] = f"{type(exc).__name__}: {exc}"
        blockers.append(deep_note)
    if (
        config["corpus"]["require_valid_full_baseline"]
        and not baseline_valid
        and not allow_missing_baseline
    ):
        blockers.append(baseline_reason)
    unavailable = sorted(
        name for name, item in dependency_evidence.items()
        if not bool(item.get("installed")) or not bool(item.get("importable"))
    )
    incompatible = sorted(
        name for name, item in dependency_evidence.items()
        if bool(item.get("installed")) and bool(item.get("importable")) and not bool(item.get("compatible"))
    )
    if unavailable:
        blockers.append("missing or non-importable optional training dependencies: " + ", ".join(unavailable))
    if incompatible:
        blockers.append("installed optional training dependencies have incompatible versions: " + ", ".join(incompatible))
    if not hardware["cuda_available"]:
        blockers.append(str(hardware.get("reason", "CUDA GPU unavailable")))
    elif int(hardware["gpu_memory_bytes"]) < 12 * 1024**3:
        blockers.append("less than 12 GiB GPU memory is unsafe for this untruncated QLoRA profile")
    return {
        "schema": "model-laboratory-interpreter-qlora-preflight",
        "schema_version": "1.1",
        "laboratory_version": __version__,
        "status": "READY" if not blockers else "BLOCKED",
        "experimental_overrides": {
            "allow_unreviewed": allow_unreviewed,
            "allow_missing_baseline": allow_missing_baseline,
        },
        "config_sha256": config["_sha256"],
        "training_environment_lock_sha256": config[
            "_training_environment_lock_sha256"
        ],
        "training_environment_receipt_sha256": config[
            "_training_environment_receipt_sha256"
        ],
        "official_frozen_config": bool(config.get("_official_frozen_config")),
        "corpus_validation_depth": "deep" if deep_corpus else "shallow",
        "deep_validation_evidence": deep_validation_evidence,
        "corpus": {
            "manifest_sha256": _sha256_file(corpus.directory / "manifest.json"),
            "train_records": len(corpus.train),
            "validation_records": len(corpus.validation),
            "review_records": len(corpus.review),
            "review_counts": dict(corpus.review_counts),
            "review_complete": corpus.review_complete,
        },
        "baseline": {
            "valid": baseline_valid,
            "reason": baseline_reason,
            "report_sha256": _sha256_file(Path(baseline_report).resolve()),
            "status": baseline.get("campaign_status", baseline.get("status", "unknown")),
        },
        "dependencies": dependencies,
        "dependency_details": dependency_evidence,
        "training_environment": (
            _training_environment_evidence(config)
            if isinstance(config.get("_training_environment_receipt"), Mapping)
            else {
                "lock_sha256": config["_training_environment_lock_sha256"],
                "receipt_sha256": None,
                "status": "not-frozen",
            }
        ),
        "hardware": hardware,
        "blockers": blockers,
    }


def effective_examples(
    corpus: CorpusAudit,
    split: str,
    *,
    allow_unreviewed: bool,
) -> list[dict[str, Any]]:
    """Apply accepted/corrected/rejected review decisions without mutating Stage-6 data."""
    if split not in {"train", "validation"}:
        raise FineTuningError("Training split must be train or validation.")
    if not corpus.review_complete and not allow_unreviewed:
        raise FineTuningError("The human review gate is incomplete.")
    decisions, _, _ = _review_decisions(corpus.review)
    source = corpus.train if split == "train" else corpus.validation
    result: list[dict[str, Any]] = []
    for item in source:
        copy = deepcopy(dict(item))
        decision = decisions.get(str(item["id"]))
        if decision is not None:
            status = decision["review"]["status"]
            if status == "rejected":
                continue
            if status == "corrected":
                copy["target"] = deepcopy(decision["review"]["corrected_target"])
            copy["review_decision"] = status
        else:
            copy["review_decision"] = "not-in-review-sample"
        result.append(copy)
    return result


def _tokenize_record(
    item: Mapping[str, Any], tokenizer: Any, maximum: int
) -> dict[str, list[int]]:
    messages = materialize_sft_record(item)["messages"]
    prompt_ids = list(
        tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=True)
    )
    full_ids = list(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise FineTuningError("Tokenizer chat template does not preserve the prompt prefix.")
    if len(full_ids) > maximum:
        raise FineTuningError(
            f"Record {item['id']} requires {len(full_ids)} tokens, exceeding the frozen "
            f"{maximum}-token limit; truncation is forbidden."
        )
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if all(value == -100 for value in labels):
        raise FineTuningError(f"Record {item['id']} contains no assistant completion tokens.")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def _token_profile(
    records: Sequence[Mapping[str, Any]], tokenizer: Any, maximum: int
) -> dict[str, Any]:
    if not records:
        raise FineTuningError("A training/evaluation split cannot be empty.")
    lengths: list[int] = []
    longest_id = ""
    longest = -1
    for item in records:
        length = len(_tokenize_record(item, tokenizer, maximum)["input_ids"])
        lengths.append(length)
        if length > longest:
            longest = length
            longest_id = str(item["id"])
    ordered = sorted(lengths)
    percentile = lambda q: ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]
    return {
        "records": len(records),
        "minimum_tokens": ordered[0],
        "median_tokens": percentile(0.5),
        "p95_tokens": percentile(0.95),
        "maximum_tokens": ordered[-1],
        "longest_record_id": longest_id,
    }


def _tokenize_records(
    records: Sequence[Mapping[str, Any]], tokenizer: Any, maximum: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Small-test helper; production training uses the lazy dataset below."""
    tokenized = [_tokenize_record(item, tokenizer, maximum) for item in records]
    return tokenized, _token_profile(records, tokenizer, maximum)


class _LazyCompletionDataset:
    """Random-access dataset that does not materialize the entire token corpus in RAM."""

    def __init__(self, records: Sequence[Mapping[str, Any]], tokenizer: Any, maximum: int):
        self._records = records
        self._tokenizer = tokenizer
        self._maximum = maximum

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return _tokenize_record(self._records[index], self._tokenizer, self._maximum)


def _package_versions() -> dict[str, str]:
    names = (
        "torch", "transformers", "accelerate", "peft", "bitsandbytes",
        "huggingface-hub", "safetensors", "packaging",
    )
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _complete_python_environment() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if isinstance(name, str) and name:
            packages[_normalised_distribution_name(name)] = distribution.version
    return dict(sorted(packages.items()))


def _output_hashes(directory: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.name == "training_report.json" or any(
            part.startswith("checkpoint-") for part in path.parts
        ):
            continue
        result.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return result


def _training_execution_contract(
    config: Mapping[str, Any],
    *,
    train_records: int,
    validation_records: int,
) -> dict[str, Any]:
    """Recompute the full-run shape from frozen config and reviewed corpus counts."""
    optimization = config["optimization"]
    micro_batch = int(optimization["micro_batch_size"])
    accumulation = int(optimization["gradient_accumulation_steps"])
    epochs = float(optimization["epochs"])
    batches_per_epoch = math.ceil(train_records / micro_batch)
    optimizer_steps_per_epoch = max(1, math.ceil(batches_per_epoch / accumulation))
    expected_optimizer_steps = math.ceil(epochs * optimizer_steps_per_epoch)
    return {
        "train_records": train_records,
        "validation_records": validation_records,
        "requested_epochs": epochs,
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "expected_optimizer_steps": expected_optimizer_steps,
    }


def validate_training_report(
    adapter_directory: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    baseline_report: str | Path = DEFAULT_BASELINE_REPORT,
    base_model_directory: str | Path | None = None,
    require_promotable: bool = False,
) -> dict[str, Any]:
    """Revalidate a Stage-7 adapter against evidence outside its self-hashed report.

    The report checksum and adapter inventory prove only internal consistency.  A release
    boundary must additionally reconstruct the frozen config, reviewed corpus, untouched
    baseline, base snapshot, implementation identity, and full-run shape from their current
    authoritative sources.  No caller may treat a self-authored report as its own authority.
    """
    if base_model_directory is None:
        raise FineTuningError(
            "Training-report validation requires the frozen base-model directory for independent revalidation."
        )
    directory = Path(adapter_directory).resolve()
    report_path = directory / "training_report.json"
    report = _json(report_path)
    required_top = {
        "schema", "schema_version", "laboratory_version", "status",
        "promotion_performed", "limited_training", "started_at_utc",
        "completed_at_utc", "config_sha256", "official_frozen_config", "seed",
        "implementation_identity", "base_model", "corpus", "baseline",
        "training_execution", "training_environment", "hardware", "packages", "python_environment",
        "training_metrics", "evaluation_metrics", "output_files", "next_gate",
        "report_sha256",
    }
    if set(report) != required_top:
        raise FineTuningError("The QLoRA training report is incomplete or contains unknown fields.")
    if report.get("schema") != FINETUNING_REPORT_SCHEMA or report.get(
        "schema_version"
    ) != FINETUNING_REPORT_VERSION:
        raise FineTuningError("Unsupported QLoRA training report schema.")
    if report.get("laboratory_version") != __version__:
        raise FineTuningError("The QLoRA training report was produced by a different Model Laboratory release.")
    status = report.get("status")
    if status not in {
        "candidate_adapter_experimental",
        "candidate_adapter_unpromoted",
    } or report.get("promotion_performed") is not False:
        raise FineTuningError("The training report is not an unpromoted adapter candidate.")
    declared = report.get("report_sha256")
    without_sha = {key: value for key, value in report.items() if key != "report_sha256"}
    if (
        not isinstance(declared, str)
        or not _SHA256.fullmatch(declared)
        or declared != canonical_json_sha256(without_sha)
    ):
        raise FineTuningError("The QLoRA training report checksum is invalid.")

    # Adapter bytes are checked independently of every other scientific gate.
    files = report.get("output_files")
    if not isinstance(files, list) or not files:
        raise FineTuningError("The QLoRA training report contains no adapter artifacts.")
    expected_paths: set[str] = set()
    for index, item in enumerate(files):
        if not isinstance(item, Mapping) or set(item) != {"path", "size", "sha256"}:
            raise FineTuningError(f"Adapter output_files[{index}] is malformed.")
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in expected_paths
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise FineTuningError("Adapter output paths must be unique safe relative paths.")
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise FineTuningError(f"Declared adapter artifact is missing or unsafe: {relative}.")
        size = item.get("size")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or path.stat().st_size != size
        ):
            raise FineTuningError(f"Adapter artifact size differs from its report: {relative}.")
        digest = item.get("sha256")
        if (
            not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
            or _sha256_file(path) != digest
        ):
            raise FineTuningError(f"Adapter artifact differs from its report: {relative}.")
        expected_paths.add(relative)
    if {item["path"] for item in _output_hashes(directory)} != expected_paths:
        raise FineTuningError("The adapter directory contains unreported output artifacts.")
    required_adapter_files = {"adapter_config.json", "adapter_model.safetensors"}
    if not required_adapter_files.issubset(expected_paths):
        raise FineTuningError(
            "The adapter inventory is not a PEFT LoRA candidate: adapter_config.json and "
            "adapter_model.safetensors are both required."
        )
    adapter_config = _json(directory / "adapter_config.json")
    if (
        not isinstance(adapter_config, Mapping)
        or str(adapter_config.get("peft_type", "")).upper() != "LORA"
        or str(adapter_config.get("task_type", "")).upper() != "CAUSAL_LM"
    ):
        raise FineTuningError("The adapter_config.json does not identify a causal-language-model LoRA adapter.")

    # Reconstruct every release-relevant identity from authoritative inputs.
    config = load_qlora_config(config_path)
    if report.get("config_sha256") != config["_sha256"]:
        raise FineTuningError("Training report does not match the currently validated QLoRA config.")
    if report.get("official_frozen_config") is not bool(config.get("_official_frozen_config")):
        raise FineTuningError("Training report misstates whether the frozen official config was used.")
    if report.get("seed") != int(config["seed"]):
        raise FineTuningError("Training report seed differs from the frozen QLoRA config.")
    reported_environment = report.get("training_environment")
    if not isinstance(reported_environment, Mapping) or set(reported_environment) != {
        "lock_sha256", "receipt_sha256", "receipt_created_at_utc", "python",
        "platform", "distribution_inventory_sha256", "direct_packages",
        "numerical_runtime", "deterministic_process", "determinism",
    }:
        raise FineTuningError("Training report environment evidence is incomplete.")
    locked_environment = config.get("_training_environment")
    receipt = config.get("_training_environment_receipt")
    if not isinstance(locked_environment, Mapping) or not isinstance(receipt, Mapping):
        raise FineTuningError(
            "Training report cannot be validated without its immutable environment receipt."
        )
    expected_environment = _training_environment_evidence(config)
    if dict(reported_environment) != expected_environment:
        raise FineTuningError("Training report differs from its frozen environment receipt.")
    try:
        receipt_created = datetime.fromisoformat(str(receipt["created_at_utc"]))
        training_started = datetime.fromisoformat(str(report["started_at_utc"]))
    except (TypeError, ValueError) as exc:
        raise FineTuningError("Training/environment timestamps are malformed.") from exc
    if receipt_created.tzinfo is None or training_started.tzinfo is None:
        raise FineTuningError("Training/environment timestamps must include UTC offsets.")
    if receipt_created > training_started:
        raise FineTuningError("Training environment receipt was created after training began.")
    expected_packages = dict(locked_environment["direct_packages"])
    reported_packages = report.get("packages")
    if not isinstance(reported_packages, Mapping) or dict(reported_packages) != expected_packages:
        raise FineTuningError("Training report package versions differ from the frozen environment lock.")
    receipt_versions = {
        item["name"]: item["version"] for item in receipt["distributions"]
    }
    if report.get("python_environment") != receipt_versions:
        raise FineTuningError(
            "Training report complete distribution inventory differs from the environment receipt."
        )
    lora = config["lora"]
    if (
        adapter_config.get("r") != int(lora["rank"])
        or adapter_config.get("lora_alpha") != int(lora["alpha"])
        or float(adapter_config.get("lora_dropout", -1.0)) != float(lora["dropout"])
        or str(adapter_config.get("bias", "")).lower() != str(lora["bias"]).lower()
    ):
        raise FineTuningError("The saved PEFT adapter configuration differs from the frozen QLoRA contract.")

    implementation = report.get("implementation_identity")
    current_implementation = {
        "git_commit_or_build_id": git_commit_or_build_id(),
        "source_tree_sha256": source_tree_sha256(),
        "training_source_sha256": _sha256_file(Path(__file__).resolve()),
    }
    if implementation != current_implementation:
        raise FineTuningError("Training report was produced by a different Stage-7 implementation identity.")

    base = verify_base_snapshot(base_model_directory, config)
    reported_base = report.get("base_model")
    if not isinstance(reported_base, Mapping):
        raise FineTuningError("Training report has no frozen base-model identity.")
    base_identity_fields = {"repository", "revision", "lock_sha256", "receipt_sha256", "files"}
    if (
        any(reported_base.get(key) != base.get(key) for key in base_identity_fields)
        or set(reported_base) != base_identity_fields | {"snapshot"}
        or not isinstance(reported_base.get("snapshot"), str)
    ):
        raise FineTuningError("Training report does not match the independently verified frozen base snapshot.")

    corpus = audit_corpus(config, deep=True)
    reported_corpus = report.get("corpus")
    required_corpus = {
        "manifest_sha256", "review_queue_sha256", "review_complete",
        "review_counts", "held_out_exclusion", "train_profile", "validation_profile",
    }
    if not isinstance(reported_corpus, Mapping) or set(reported_corpus) != required_corpus:
        raise FineTuningError("Training report corpus evidence is incomplete.")
    if reported_corpus.get("manifest_sha256") != _sha256_file(corpus.directory / "manifest.json"):
        raise FineTuningError("Training report corpus manifest identity is stale or false.")
    if reported_corpus.get("review_queue_sha256") != _sha256_file(corpus.directory / "review_queue.jsonl"):
        raise FineTuningError("Training report human-review queue identity is stale or false.")
    if reported_corpus.get("review_complete") is not corpus.review_complete:
        raise FineTuningError("Training report misstates human-review completion.")
    if reported_corpus.get("review_counts") != dict(corpus.review_counts):
        raise FineTuningError("Training report human-review counts do not revalidate.")
    if reported_corpus.get("held_out_exclusion") != dict(corpus.held_out_exclusion):
        raise FineTuningError("Training report held-out benchmark exclusion does not revalidate.")

    baseline_valid, baseline_reason, baseline = baseline_gate(baseline_report)
    reported_baseline = report.get("baseline")
    if not isinstance(reported_baseline, Mapping) or set(reported_baseline) != {
        "valid_full_baseline", "report_sha256", "internal_report_sha256"
    }:
        raise FineTuningError("Training report baseline evidence is incomplete.")
    baseline_path = Path(baseline_report).resolve()
    baseline_file_sha = _sha256_file(baseline_path)
    if reported_baseline.get("report_sha256") != baseline_file_sha:
        raise FineTuningError("Training report is bound to a different baseline-report file.")
    if reported_baseline.get("internal_report_sha256") != baseline.get("report_sha256"):
        raise FineTuningError("Training report baseline self-identity does not revalidate.")
    if reported_baseline.get("valid_full_baseline") is not baseline_valid:
        raise FineTuningError("Training report misstates untouched-baseline validity.")

    full_train = effective_examples(corpus, "train", allow_unreviewed=True)
    full_validation = effective_examples(corpus, "validation", allow_unreviewed=True)
    execution = report.get("training_execution")
    execution_keys = {
        "train_records", "validation_records", "requested_epochs", "micro_batch_size",
        "gradient_accumulation_steps", "optimizer_steps_per_epoch",
        "expected_optimizer_steps", "trainer_global_step", "trainer_max_steps",
        "trainer_epoch", "resumed_from_checkpoint", "maximum_train_examples",
        "maximum_validation_examples",
    }
    if not isinstance(execution, Mapping) or set(execution) != execution_keys:
        raise FineTuningError("Training report execution evidence is incomplete.")

    def validated_limit(name: str, full_count: int) -> int | None:
        value = execution.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= full_count:
            raise FineTuningError(f"Training report {name} is invalid.")
        return value

    train_limit = validated_limit("maximum_train_examples", len(full_train))
    validation_limit = validated_limit("maximum_validation_examples", len(full_validation))
    actual_train_count = len(full_train) if train_limit is None else min(len(full_train), train_limit)
    actual_validation_count = (
        len(full_validation) if validation_limit is None else min(len(full_validation), validation_limit)
    )
    contract = _training_execution_contract(
        config, train_records=actual_train_count, validation_records=actual_validation_count
    )
    for key, value in contract.items():
        if execution.get(key) != value:
            raise FineTuningError(f"Training report execution contract does not revalidate: {key}.")
    if not isinstance(execution.get("resumed_from_checkpoint"), bool):
        raise FineTuningError("Training report resumed_from_checkpoint flag is malformed.")

    train_profile = reported_corpus.get("train_profile")
    validation_profile = reported_corpus.get("validation_profile")
    if (
        not isinstance(train_profile, Mapping)
        or train_profile.get("records") != actual_train_count
        or not isinstance(validation_profile, Mapping)
        or validation_profile.get("records") != actual_validation_count
    ):
        raise FineTuningError("Training report token profiles do not match the declared execution shape.")
    global_step = execution.get("trainer_global_step")
    max_steps = execution.get("trainer_max_steps")
    trainer_epoch = execution.get("trainer_epoch")
    completed_training = (
        isinstance(global_step, int)
        and not isinstance(global_step, bool)
        and isinstance(max_steps, int)
        and not isinstance(max_steps, bool)
        and global_step == max_steps == contract["expected_optimizer_steps"]
        and isinstance(trainer_epoch, (int, float))
        and not isinstance(trainer_epoch, bool)
        and float(trainer_epoch) >= float(contract["requested_epochs"])
    )
    if not completed_training:
        raise FineTuningError("Training report does not prove completion of its declared training run.")
    full_shape_ok = train_limit is None and validation_limit is None

    limited_training = report.get("limited_training")
    if not isinstance(limited_training, bool):
        raise FineTuningError("Training report limited_training flag is malformed.")
    if limited_training is not (train_limit is not None or validation_limit is not None):
        raise FineTuningError("Training report limited_training flag disagrees with execution limits.")
    independently_experimental = (
        not bool(config.get("_official_frozen_config"))
        or not corpus.review_complete
        or not baseline_valid
        or limited_training
        or not full_shape_ok
    )
    expected_status = (
        "candidate_adapter_experimental"
        if independently_experimental
        else "candidate_adapter_unpromoted"
    )
    if status != expected_status:
        reason = baseline_reason if not baseline_valid else "independent Stage-7 evidence differs"
        raise FineTuningError(
            f"Training report lifecycle state is inconsistent with revalidated evidence: {reason}."
        )
    if require_promotable and status != "candidate_adapter_unpromoted":
        raise FineTuningError(
            "Only candidate_adapter_unpromoted may enter production promotion; experimental adapters are permanently non-promotable."
        )
    return deepcopy(report)

def run_qlora(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    baseline_report: str | Path = DEFAULT_BASELINE_REPORT,
    model_directory: str | Path,
    output_directory: str | Path,
    allow_unreviewed: bool = False,
    allow_missing_baseline: bool = False,
    maximum_train_examples: int | None = None,
    maximum_validation_examples: int | None = None,
    resume_from_checkpoint: str | bool | None = None,
) -> dict[str, Any]:
    """Train one frozen QLoRA candidate; never promote it into production automatically."""
    config = load_qlora_config(config_path)
    _configure_deterministic_environment(config)
    runtime_errors = _training_runtime_contract_errors(config)
    if runtime_errors:
        raise FineTuningError("; ".join(runtime_errors) + ".")
    corpus = audit_corpus(config, deep=True)
    baseline_valid, baseline_reason, baseline_evidence = baseline_gate(baseline_report)
    if not baseline_valid and not allow_missing_baseline:
        raise FineTuningError(baseline_reason + ".")
    if not corpus.review_complete and not allow_unreviewed:
        raise FineTuningError("Human review is incomplete; use the review workflow before Stage 7.")
    missing = [name for name, present in dependency_status().items() if not present]
    if missing:
        raise FineTuningError("Install requirements-training.txt; missing: " + ", ".join(missing) + ".")
    hardware = hardware_status()
    if not hardware["cuda_available"]:
        raise FineTuningError(str(hardware.get("reason", "CUDA GPU unavailable")))
    if int(hardware["gpu_memory_bytes"]) < 12 * 1024**3:
        raise FineTuningError("This profile requires at least 12 GiB CUDA GPU memory.")

    # Optional imports begin only after every inexpensive scientific gate has passed.
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    base_identity = verify_base_snapshot(model_directory, config)
    model_root = Path(model_directory).resolve()
    output = Path(output_directory).resolve()
    if output == model_root or output.is_relative_to(model_root) or model_root.is_relative_to(output):
        raise FineTuningError("The adapter output and frozen base-model snapshot must be separate trees.")
    if output.exists() and any(output.iterdir()) and resume_from_checkpoint is None:
        raise FineTuningError("The adapter output directory must be empty for a new run.")
    if resume_from_checkpoint not in (None, False, True):
        checkpoint = Path(str(resume_from_checkpoint)).resolve()
        if not checkpoint.is_dir() or not checkpoint.is_relative_to(output):
            raise FineTuningError("A resume checkpoint must be an existing directory inside output_dir.")
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode("error")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_root), trust_remote_code=False, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_records = effective_examples(corpus, "train", allow_unreviewed=allow_unreviewed)
    validation_records = effective_examples(corpus, "validation", allow_unreviewed=allow_unreviewed)
    if maximum_train_examples is not None:
        if maximum_train_examples < 1:
            raise FineTuningError("maximum_train_examples must be positive.")
        train_records = train_records[:maximum_train_examples]
    if maximum_validation_examples is not None:
        if maximum_validation_examples < 1:
            raise FineTuningError("maximum_validation_examples must be positive.")
        validation_records = validation_records[:maximum_validation_examples]
    maximum_tokens = int(config["sequence"]["maximum_tokens"])
    train_profile = _token_profile(train_records, tokenizer, maximum_tokens)
    validation_profile = _token_profile(validation_records, tokenizer, maximum_tokens)
    train_dataset = _LazyCompletionDataset(train_records, tokenizer, maximum_tokens)
    validation_dataset = _LazyCompletionDataset(
        validation_records, tokenizer, maximum_tokens
    )

    use_bf16 = bool(torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_root),
        trust_remote_code=False,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=compute_dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=bool(config["optimization"]["gradient_checkpointing"])
    )
    lora = config["lora"]
    peft_config = LoraConfig(
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    optimization = config["optimization"]
    arguments = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=float(optimization["epochs"]),
        per_device_train_batch_size=int(optimization["micro_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(optimization["gradient_accumulation_steps"]),
        learning_rate=float(optimization["learning_rate"]),
        lr_scheduler_type=str(optimization["scheduler"]),
        warmup_ratio=float(optimization["warmup_ratio"]),
        weight_decay=float(optimization["weight_decay"]),
        max_grad_norm=float(optimization["maximum_gradient_norm"]),
        optim=str(optimization["optimizer"]),
        eval_strategy=str(optimization["evaluation_strategy"]),
        save_strategy=str(optimization["save_strategy"]),
        save_total_limit=2,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=bool(optimization["gradient_checkpointing"]),
        report_to="none",
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        full_determinism=True,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ),
        processing_class=tokenizer,
    )
    started = datetime.now(timezone.utc)
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    evaluation = trainer.evaluate()
    completed = datetime.now(timezone.utc)
    limited_training = (
        maximum_train_examples is not None or maximum_validation_examples is not None
    )
    experimental = (
        not corpus.review_complete
        or not baseline_valid
        or limited_training
        or not bool(config.get("_official_frozen_config"))
    )
    execution_contract = _training_execution_contract(
        config, train_records=len(train_records), validation_records=len(validation_records)
    )
    trainer_global_step = int(getattr(trainer.state, "global_step", -1))
    trainer_max_steps = int(getattr(trainer.state, "max_steps", -1))
    trainer_epoch_value = getattr(trainer.state, "epoch", None)
    trainer_epoch = float(trainer_epoch_value) if trainer_epoch_value is not None else -1.0
    if (
        trainer_global_step != execution_contract["expected_optimizer_steps"]
        or trainer_max_steps != execution_contract["expected_optimizer_steps"]
        or trainer_epoch < float(execution_contract["requested_epochs"])
    ):
        raise FineTuningError(
            "Trainer completed without matching the independently computed Stage-7 full-run step contract."
        )

    report: dict[str, Any] = {
        "schema": FINETUNING_REPORT_SCHEMA,
        "schema_version": FINETUNING_REPORT_VERSION,
        "laboratory_version": __version__,
        "status": "candidate_adapter_experimental" if experimental else "candidate_adapter_unpromoted",
        "promotion_performed": False,
        "limited_training": limited_training,
        "started_at_utc": started.isoformat(timespec="seconds"),
        "completed_at_utc": completed.isoformat(timespec="seconds"),
        "config_sha256": config["_sha256"],
        "official_frozen_config": bool(config.get("_official_frozen_config")),
        "seed": seed,
        "implementation_identity": {
            "git_commit_or_build_id": git_commit_or_build_id(),
            "source_tree_sha256": source_tree_sha256(),
            "training_source_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "base_model": base_identity,
        "corpus": {
            "manifest_sha256": _sha256_file(corpus.directory / "manifest.json"),
            "review_queue_sha256": _sha256_file(corpus.directory / "review_queue.jsonl"),
            "review_complete": corpus.review_complete,
            "review_counts": dict(corpus.review_counts),
            "held_out_exclusion": dict(corpus.held_out_exclusion),
            "train_profile": train_profile,
            "validation_profile": validation_profile,
        },
        "baseline": {
            "valid_full_baseline": baseline_valid,
            "report_sha256": _sha256_file(Path(baseline_report).resolve()),
            "internal_report_sha256": baseline_evidence.get("report_sha256"),
        },
        "training_execution": {
            **execution_contract,
            "trainer_global_step": trainer_global_step,
            "trainer_max_steps": trainer_max_steps,
            "trainer_epoch": trainer_epoch,
            "resumed_from_checkpoint": resume_from_checkpoint not in (None, False),
            "maximum_train_examples": maximum_train_examples,
            "maximum_validation_examples": maximum_validation_examples,
        },
        "training_environment": _training_environment_evidence(config),
        "hardware": hardware,
        "packages": _package_versions(),
        "python_environment": _complete_python_environment(),
        "training_metrics": dict(result.metrics),
        "evaluation_metrics": dict(evaluation),
        "output_files": [],
        "next_gate": "Run the same frozen Stage-5 benchmark against the exported adapter and compare every safety/scientific metric before production promotion.",
    }
    report["output_files"] = _output_hashes(output)
    report["report_sha256"] = canonical_json_sha256(report)
    _atomic_json(output / "training_report.json", report)
    return report


def update_review_manifest(corpus_directory: str | Path) -> dict[str, Any]:
    """Rebind a human-edited review queue into the Stage-6 manifest."""
    directory = Path(corpus_directory).resolve()
    manifest_path = directory / "manifest.json"
    manifest = _json(manifest_path)
    review_path = directory / "review_queue.jsonl"
    review = load_jsonl(review_path)
    _, counts, problems = _review_decisions(review)
    if problems:
        raise FineTuningError("; ".join(problems[:20]))
    complete = counts["pending-human-review"] == 0
    manifest["review"]["status"] = "completed-human-review" if complete else "human-review-in-progress"
    manifest["review"]["decision_counts"] = counts
    manifest["review"]["completed"] = complete
    manifest["files"]["review_queue.jsonl"] = {
        "records": len(review),
        "bytes": review_path.stat().st_size,
        "sha256": _sha256_file(review_path),
    }
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    _atomic_json(manifest_path, manifest, sort_keys=True)
    return manifest
