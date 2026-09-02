"""Fail-closed export of a Stage-7 QLoRA adapter into an Ollama candidate.

Export never promotes a model.  It verifies the adapter report and frozen base snapshot,
merges the adapter, converts the merged Hugging Face model to GGUF, quantizes it to Q4_K_M,
creates a distinct Ollama tag, and records a checksum-bound receipt for later evaluation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

from . import __version__
from .canonical import canonical_json_sha256
from .interpreter import LOCAL_ENDPOINT, LOCAL_MODEL, LOCAL_QUANTIZATION
from .interpreter_finetuning import (
    DEFAULT_BASELINE_REPORT,
    DEFAULT_CONFIG,
    FineTuningError,
    load_qlora_config,
    validate_training_report,
    verify_base_snapshot,
)


EXPORT_RECEIPT_SCHEMA = "model-laboratory-interpreter-export-receipt"
EXPORT_RECEIPT_VERSION = "1.1"
EXPORT_PLAN_SCHEMA = "model-laboratory-interpreter-export-plan"
EXPORT_PLAN_VERSION = "1.1"
DEFAULT_QUANTIZATION = "Q4_K_M"
MAX_TOOL_OUTPUT_BYTES = 2 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODEL_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")


class InterpreterExportError(RuntimeError):
    """An adapter/export identity, tool, or Ollama import gate failed."""


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return digest.hexdigest(), size


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
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
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()


def _safe_tag(value: str) -> str:
    if not isinstance(value, str) or _MODEL_TAG.fullmatch(value) is None:
        raise InterpreterExportError("Candidate model tag is malformed.")
    if value == LOCAL_MODEL:
        raise InterpreterExportError("A candidate export must not overwrite the frozen base tag.")
    return value


def _tool(path: str | Path, field: str) -> Path:
    candidate = Path(path).resolve()
    if not candidate.is_file():
        raise InterpreterExportError(f"{field} is not an existing file: {candidate}.")
    return candidate


def _run(command: Sequence[str], *, timeout_seconds: int = 6 * 60 * 60) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [str(item) for item in command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise InterpreterExportError(f"Export tool failed to run: {command[0]}: {exc}") from exc
    output = completed.stdout[: MAX_TOOL_OUTPUT_BYTES]
    if completed.returncode != 0:
        tail = output[-8192:].decode("utf-8", errors="replace")
        raise InterpreterExportError(
            f"Export tool returned exit code {completed.returncode}: {command[0]}\n{tail}"
        )
    return {
        "argv": [str(item) for item in command],
        "exit_code": completed.returncode,
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "output_bytes_recorded": len(output),
        "output_truncated": len(completed.stdout) > len(output),
    }


def _ollama_json(path: str) -> Mapping[str, Any]:
    opener = urlrequest.build_opener(urlrequest.ProxyHandler({}))
    try:
        with opener.open(LOCAL_ENDPOINT + path, timeout=10) as response:
            raw = response.read(MAX_TOOL_OUTPUT_BYTES + 1)
    except (urlerror.URLError, TimeoutError, OSError) as exc:
        raise InterpreterExportError(f"Ollama is not reachable at the loopback endpoint: {exc}") from exc
    if len(raw) > MAX_TOOL_OUTPUT_BYTES:
        raise InterpreterExportError("Ollama identity response exceeded its safety limit.")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InterpreterExportError("Ollama returned malformed identity JSON.") from exc
    if not isinstance(value, Mapping):
        raise InterpreterExportError("Ollama identity response must be an object.")
    return value


def _ollama_identity(model_tag: str) -> dict[str, Any]:
    version = _ollama_json("/api/version").get("version")
    models = _ollama_json("/api/tags").get("models")
    if not isinstance(version, str) or not version or not isinstance(models, list):
        raise InterpreterExportError("Ollama returned an incomplete runtime identity.")
    match = next(
        (
            item
            for item in models
            if isinstance(item, Mapping)
            and (item.get("name") == model_tag or item.get("model") == model_tag)
        ),
        None,
    )
    if not isinstance(match, Mapping):
        raise InterpreterExportError("Ollama did not install the requested candidate tag.")
    digest = match.get("digest")
    size = match.get("size")
    if not isinstance(digest, str) or not digest or len(digest) > 256:
        raise InterpreterExportError("Ollama returned an invalid candidate digest.")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise InterpreterExportError("Ollama returned an invalid candidate size.")
    return {"runtime_version": version, "model_digest": digest, "model_size_bytes": size}


def create_export_plan(
    *,
    adapter_directory: str | Path,
    base_model_directory: str | Path,
    output_directory: str | Path,
    candidate_tag: str,
    convert_hf_to_gguf: str | Path,
    llama_quantize: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    baseline_report: str | Path = DEFAULT_BASELINE_REPORT,
) -> dict[str, Any]:
    """Validate all existing inputs and return the exact, non-executing export plan."""
    tag = _safe_tag(candidate_tag)
    try:
        config = load_qlora_config(config_path)
        training = validate_training_report(
            adapter_directory,
            config_path=config_path,
            baseline_report=baseline_report,
            base_model_directory=base_model_directory,
            require_promotable=False,
        )
        if training.get("status") != "candidate_adapter_unpromoted":
            raise InterpreterExportError(
                "Experimental adapters cannot enter the production export/evaluation pipeline."
            )
        base = verify_base_snapshot(base_model_directory, config)
    except FineTuningError as exc:
        raise InterpreterExportError(str(exc)) from exc
    converter = _tool(convert_hf_to_gguf, "convert_hf_to_gguf")
    quantizer = _tool(llama_quantize, "llama_quantize")
    output = Path(output_directory).resolve()
    adapter = Path(adapter_directory).resolve()
    base_root = Path(base_model_directory).resolve()
    if output in {adapter, base_root} or output.is_relative_to(adapter) or output.is_relative_to(base_root):
        raise InterpreterExportError("Export output must be separate from base and adapter trees.")
    merged = output / "merged-hf"
    f16 = output / "candidate-f16.gguf"
    quantized = output / "candidate-q4_k_m.gguf"
    return {
        "schema": EXPORT_PLAN_SCHEMA,
        "schema_version": EXPORT_PLAN_VERSION,
        "laboratory_version": __version__,
        "candidate_tag": tag,
        "adapter_directory": str(adapter),
        "training_report_sha256": training["report_sha256"],
        "training_status": training["status"],
        "base_model_directory": str(base_root),
        "base_snapshot_receipt_sha256": base["receipt_sha256"],
        "output_directory": str(output),
        "merged_model_directory": str(merged),
        "f16_gguf": str(f16),
        "quantized_gguf": str(quantized),
        "quantization": DEFAULT_QUANTIZATION,
        "converter": str(converter),
        "converter_sha256": _sha256_file(converter)[0],
        "quantizer": str(quantizer),
        "quantizer_sha256": _sha256_file(quantizer)[0],
        "promotion_performed": False,
    }


def _merge_adapter(
    *, adapter: Path, base: Path, destination: Path, device: str
) -> None:
    if device not in {"cpu", "cuda"}:
        raise InterpreterExportError("Merge device must be cpu or cuda.")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise InterpreterExportError(
            "Adapter export requires the Stage-7 Transformers, PyTorch, and PEFT dependencies."
        ) from exc
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(base), local_files_only=True, trust_remote_code=False, torch_dtype=dtype,
        device_map={"": 0} if device == "cuda" else {"": "cpu"}, low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(model, str(adapter), is_trainable=False).merge_and_unload()
    destination.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(str(destination), safe_serialization=True, max_shard_size="4GB")
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    tokenizer.save_pretrained(str(destination))


def export_interpreter_candidate(
    *,
    adapter_directory: str | Path,
    base_model_directory: str | Path,
    output_directory: str | Path,
    candidate_tag: str,
    convert_hf_to_gguf: str | Path,
    llama_quantize: str | Path,
    ollama_executable: str | Path = "ollama",
    config_path: str | Path = DEFAULT_CONFIG,
    baseline_report: str | Path = DEFAULT_BASELINE_REPORT,
    merge_device: str = "cpu",
) -> dict[str, Any]:
    """Execute the complete merge/convert/quantize/import pipeline and write its receipt."""
    plan = create_export_plan(
        adapter_directory=adapter_directory,
        base_model_directory=base_model_directory,
        output_directory=output_directory,
        candidate_tag=candidate_tag,
        convert_hf_to_gguf=convert_hf_to_gguf,
        llama_quantize=llama_quantize,
        config_path=config_path,
        baseline_report=baseline_report,
    )
    output = Path(plan["output_directory"])
    if output.exists() and any(output.iterdir()):
        raise InterpreterExportError("Export output directory must be empty.")
    output.mkdir(parents=True, exist_ok=True)
    merged = Path(plan["merged_model_directory"])
    f16 = Path(plan["f16_gguf"])
    quantized = Path(plan["quantized_gguf"])
    _merge_adapter(
        adapter=Path(plan["adapter_directory"]),
        base=Path(plan["base_model_directory"]),
        destination=merged,
        device=merge_device,
    )
    commands = [
        _run(
            [
                sys.executable,
                plan["converter"],
                str(merged),
                "--outfile",
                str(f16),
                "--outtype",
                "f16",
            ]
        ),
        _run([plan["quantizer"], str(f16), str(quantized), DEFAULT_QUANTIZATION]),
    ]
    if not f16.is_file() or not quantized.is_file():
        raise InterpreterExportError("GGUF conversion did not produce both required artifacts.")
    modelfile = output / "Modelfile"
    modelfile.write_text(
        f'FROM "{quantized.as_posix()}"\nPARAMETER num_ctx 131072\n',
        encoding="utf-8",
        newline="\n",
    )
    commands.append(
        _run([str(ollama_executable), "create", plan["candidate_tag"], "-f", str(modelfile)])
    )
    identity = _ollama_identity(plan["candidate_tag"])
    f16_sha, f16_size = _sha256_file(f16)
    q4_sha, q4_size = _sha256_file(quantized)
    model_file_sha, model_file_size = _sha256_file(modelfile)
    seed = canonical_json_sha256(
        {
            "training_report_sha256": plan["training_report_sha256"],
            "gguf_sha256": q4_sha,
            "model_digest": identity["model_digest"],
        }
    )
    receipt: dict[str, Any] = {
        "schema": EXPORT_RECEIPT_SCHEMA,
        "schema_version": EXPORT_RECEIPT_VERSION,
        "laboratory_version": __version__,
        "candidate_id": "candidate-" + seed[:20],
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate_tag": plan["candidate_tag"],
        "quantization": DEFAULT_QUANTIZATION,
        "training_report_sha256": plan["training_report_sha256"],
        "training_status": plan["training_status"],
        "base_snapshot_receipt_sha256": plan["base_snapshot_receipt_sha256"],
        "artifacts": {
            "f16_gguf": {"path": f16.name, "size": f16_size, "sha256": f16_sha},
            "quantized_gguf": {"path": quantized.name, "size": q4_size, "sha256": q4_sha},
            "modelfile": {"path": modelfile.name, "size": model_file_size, "sha256": model_file_sha},
        },
        "tools": {
            "converter_sha256": plan["converter_sha256"],
            "quantizer_sha256": plan["quantizer_sha256"],
            "merge_device": merge_device,
        },
        "commands": commands,
        "ollama": identity,
        "promotion_performed": False,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    _atomic_json(output / "export_receipt.json", receipt)
    return receipt


def validate_export_receipt(
    value: str | Path | Mapping[str, Any], *, verify_artifacts: bool = True
) -> dict[str, Any]:
    """Validate receipt integrity and, when possible, every exported local artifact."""
    root: Path | None = None
    if isinstance(value, (str, Path)):
        path = Path(value).resolve()
        root = path.parent
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InterpreterExportError(f"Cannot read export receipt: {exc}") from exc
    else:
        raw = dict(value)
    if not isinstance(raw, Mapping):
        raise InterpreterExportError("Export receipt must be a JSON object.")
    required = {
        "schema", "schema_version", "laboratory_version", "candidate_id",
        "created_at_utc", "candidate_tag", "quantization",
        "training_report_sha256", "training_status", "base_snapshot_receipt_sha256", "artifacts",
        "tools", "commands", "ollama", "promotion_performed", "receipt_sha256",
    }
    if set(raw) != required:
        raise InterpreterExportError("Export receipt has missing or unknown fields.")
    if raw.get("schema") != EXPORT_RECEIPT_SCHEMA or raw.get("schema_version") != EXPORT_RECEIPT_VERSION:
        raise InterpreterExportError("Unsupported export receipt schema.")
    declared = raw.get("receipt_sha256")
    without_sha = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    if not isinstance(declared, str) or _SHA256.fullmatch(declared) is None or declared != canonical_json_sha256(without_sha):
        raise InterpreterExportError("Export receipt checksum is invalid.")
    _safe_tag(str(raw.get("candidate_tag", "")))
    if raw.get("quantization") != LOCAL_QUANTIZATION or raw.get("promotion_performed") is not False:
        raise InterpreterExportError("Export receipt has an invalid quantization/promotion state.")
    if raw.get("laboratory_version") != __version__:
        raise InterpreterExportError("Export receipt was produced by a different laboratory version.")
    if not isinstance(raw.get("candidate_id"), str) or not raw["candidate_id"].startswith("candidate-"):
        raise InterpreterExportError("Export receipt candidate identity is invalid.")
    for field in ("training_report_sha256", "base_snapshot_receipt_sha256"):
        if not isinstance(raw.get(field), str) or _SHA256.fullmatch(raw[field]) is None:
            raise InterpreterExportError(f"Export receipt {field} is invalid.")
    if raw.get("training_status") not in {
        "candidate_adapter_experimental", "candidate_adapter_unpromoted"
    }:
        raise InterpreterExportError("Export receipt training lifecycle state is invalid.")
    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"f16_gguf", "quantized_gguf", "modelfile"}:
        raise InterpreterExportError("Export receipt artifact inventory is incomplete.")
    for name, item in artifacts.items():
        if not isinstance(item, Mapping) or set(item) != {"path", "size", "sha256"}:
            raise InterpreterExportError(f"Export artifact {name} is malformed.")
        relative = item.get("path")
        size = item.get("size")
        digest = item.get("sha256")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise InterpreterExportError(f"Export artifact {name} has an unsafe path.")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise InterpreterExportError(f"Export artifact {name} has an invalid size.")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise InterpreterExportError(f"Export artifact {name} has an invalid SHA-256.")
        if verify_artifacts:
            if root is None:
                raise InterpreterExportError("Artifact verification requires a receipt file path.")
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise InterpreterExportError(f"Export artifact is missing: {relative}.")
            observed, observed_size = _sha256_file(path)
            if observed != digest or observed_size != size:
                raise InterpreterExportError(f"Export artifact differs from its receipt: {relative}.")
    ollama = raw.get("ollama")
    if not isinstance(ollama, Mapping) or set(ollama) != {"runtime_version", "model_digest", "model_size_bytes"}:
        raise InterpreterExportError("Export receipt Ollama identity is malformed.")
    if not isinstance(ollama.get("model_digest"), str) or not ollama["model_digest"]:
        raise InterpreterExportError("Export receipt has no Ollama model digest.")
    tools = raw.get("tools")
    if not isinstance(tools, Mapping) or set(tools) != {
        "converter_sha256", "quantizer_sha256", "merge_device"
    }:
        raise InterpreterExportError("Export receipt tool identity is malformed.")
    if any(
        not isinstance(tools.get(field), str) or _SHA256.fullmatch(tools[field]) is None
        for field in ("converter_sha256", "quantizer_sha256")
    ) or tools.get("merge_device") not in {"cpu", "cuda"}:
        raise InterpreterExportError("Export receipt tool identity is invalid.")
    commands = raw.get("commands")
    if not isinstance(commands, list) or len(commands) != 3 or not all(
        isinstance(item, Mapping) for item in commands
    ):
        raise InterpreterExportError("Export receipt command evidence is incomplete.")
    return dict(raw)
