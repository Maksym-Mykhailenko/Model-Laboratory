"""Held-out evaluation and comparison for exported interpreter candidates."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import uuid
from typing import Any, Callable, Mapping, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

from .canonical import canonical_json_sha256
from .interpreter import (
    LOCAL_BASE_MODEL,
    LOCAL_ENDPOINT,
    LOCAL_QUANTIZATION,
)
from .interpreter_baseline import (
    EVALUATION_GENERATION_OPTIONS,
    InferenceResult,
    _build_report,
    _score_case,
    _source_identity,
    load_benchmark,
    output_schema,
    system_prompt,
    user_prompt,
)
from .interpreter_export import validate_export_receipt
from .interpreter_finetuning import (
    DEFAULT_BASELINE_REPORT,
    DEFAULT_CONFIG,
    FineTuningError,
    baseline_gate,
    validate_training_report,
)
from .interpreter_promotion import (
    PromotionEvidenceError,
    complete_promotion_benchmark,
    default_promotion_ledger_path,
    reserve_promotion_benchmark,
    validate_promotion_consumption,
    validate_promotion_seal,
)


CANDIDATE_REPORT_SCHEMA = "model-laboratory-interpreter-candidate-evaluation-report"
CANDIDATE_REPORT_VERSION = "1.3"
CANDIDATE_CHECKPOINT_SCHEMA = "model-laboratory-interpreter-candidate-checkpoint"
CANDIDATE_CHECKPOINT_VERSION = "1.0"
PROMOTION_POLICY_VERSION = "1.2"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CORE_METRICS = (
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
_BENEFIT_METRICS = (
    "first_action_accuracy",
    "terminal_status_accuracy",
    "scientific_semantic_accuracy",
    "clarification_continuation_accuracy",
    "context_negotiation_first_action_accuracy",
)
_ABSOLUTE_FLOORS = {
    "raw_schema_compliance": 1.0,
    "expected_proposal_compiler_acceptance": 1.0,
    "unsupported_accuracy": 1.0,
    "scientific_semantic_accuracy": 0.80,
    "clarification_continuation_accuracy": 0.75,
    "context_negotiation_first_action_accuracy": 0.75,
}


class CandidateEvaluationError(RuntimeError):
    """Candidate identity, evidence, comparison, or transport is invalid."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _normalized_digest(value: str) -> str:
    text = value.strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    if _SHA256.fullmatch(text) is None:
        raise CandidateEvaluationError("Candidate Ollama digest is malformed.")
    return text


@dataclass(frozen=True)
class CandidateIdentity:
    runtime_version: str
    model_tag: str
    model_digest: str
    model_size_bytes: int
    export_receipt_sha256: str
    local_manifest_sha256: str = ""
    model_blob_sha256: str = ""
    verified_model_blob_size_bytes: int = 0

    def provider_identity(
        self, generation_options: Mapping[str, int | float] | None = None
    ) -> dict[str, Any]:
        return {
            "provider": "ollama",
            "endpoint": LOCAL_ENDPOINT,
            "runtime_version": self.runtime_version,
            "model_tag": self.model_tag,
            "observed_model_digest": self.model_digest,
            "identity_verification": "export_receipt_exact_manifest_and_model_blob_sha256",
            "expected_base_model": LOCAL_BASE_MODEL,
            "expected_quantization": LOCAL_QUANTIZATION,
            "generation_options": dict(generation_options or EVALUATION_GENERATION_OPTIONS),
            "model_role": "evaluation_candidate",
            "registry_entry_sha256": self.export_receipt_sha256,
            "artifact_evidence": {
                "identity_verification": "export_receipt_exact_manifest_and_model_blob_sha256",
                "artifact_lock_sha256": self.export_receipt_sha256,
                "local_manifest_sha256": self.local_manifest_sha256,
                "model_blob_sha256": self.model_blob_sha256,
                "verified_model_blob_size_bytes": self.verified_model_blob_size_bytes,
                "all_manifest_blobs_verified": True,
            },
        }

    def report_payload(self) -> dict[str, Any]:
        return {
            "runtime": "ollama",
            "runtime_version": self.runtime_version,
            "model_tag": self.model_tag,
            "observed_model_digest": self.model_digest,
            "model_size_bytes": self.model_size_bytes,
            "export_receipt_sha256": self.export_receipt_sha256,
            "local_manifest_sha256": self.local_manifest_sha256,
            "model_blob_sha256": self.model_blob_sha256,
            "verified_model_blob_size_bytes": self.verified_model_blob_size_bytes,
            "all_manifest_blobs_verified": True,
            "identity_verification": "exact_export_receipt_manifest_and_all_local_blobs_sha256",
        }


def _ollama_model_roots() -> list[Path]:
    roots: list[Path] = []
    configured = os.environ.get("OLLAMA_MODELS", "").strip()
    if configured:
        roots.append(Path(configured))
    roots.extend(
        [Path.home() / ".ollama" / "models", Path("/usr/share/ollama/.ollama/models")]
    )
    result: list[Path] = []
    for root in roots:
        if root not in result:
            result.append(root)
    return result


def _manifest_relative_path(model_tag: str) -> Path:
    if "\\" in model_tag or ":" not in model_tag:
        raise CandidateEvaluationError("Candidate Ollama tag has an unsafe path form.")
    name, tag = model_tag.rsplit(":", 1)
    components = name.split("/")
    safe = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    if not tag or safe.fullmatch(tag) is None or any(
        safe.fullmatch(component) is None or component in {".", ".."}
        for component in components
    ):
        raise CandidateEvaluationError("Candidate Ollama tag has unsafe components.")
    explicit_registry = len(components) > 1 and (
        "." in components[0] or components[0] == "localhost"
    )
    registry = components.pop(0) if explicit_registry else "registry.ollama.ai"
    if len(components) == 1:
        components.insert(0, "library")
    return Path("manifests", registry, *components, tag)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return digest.hexdigest(), size


def _verify_blob(root: Path, digest: str, declared_size: int) -> int:
    normalised = _normalized_digest(digest)
    path = root / "blobs" / ("sha256-" + normalised)
    if not path.is_file():
        raise CandidateEvaluationError(f"Required candidate Ollama blob is missing: {normalised[:12]}…")
    observed, size = _sha256_file(path)
    if observed != normalised or size != declared_size:
        raise CandidateEvaluationError("Candidate Ollama blob differs from its manifest identity.")
    return size


def _verify_candidate_manifest(
    *, model_tag: str, observed_digest: str, expected_model_blob_sha256: str
) -> tuple[str, int]:
    expected_manifest_sha = _normalized_digest(observed_digest)
    relative = _manifest_relative_path(model_tag)
    located = next(
        ((root, root / relative) for root in _ollama_model_roots() if (root / relative).is_file()),
        None,
    )
    if located is None:
        raise CandidateEvaluationError(
            "Cannot verify the candidate: its local Ollama manifest was not found. Set OLLAMA_MODELS when using a non-default directory."
        )
    root, path = located
    raw = path.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if manifest_sha != expected_manifest_sha:
        raise CandidateEvaluationError(
            "Ollama /api/tags digest differs from the candidate manifest bytes."
        )
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CandidateEvaluationError("Candidate Ollama manifest is malformed JSON.") from exc
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("layers"), list):
        raise CandidateEvaluationError("Candidate Ollama manifest is incomplete.")
    model_size = 0
    model_layers = 0
    for layer in manifest["layers"]:
        if not isinstance(layer, Mapping):
            raise CandidateEvaluationError("Candidate Ollama manifest contains a malformed layer.")
        media_type, digest, size = layer.get("mediaType"), layer.get("digest"), layer.get("size")
        if (
            not isinstance(media_type, str)
            or not isinstance(digest, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise CandidateEvaluationError("Candidate Ollama manifest layer identity is malformed.")
        verified_size = _verify_blob(root, digest, size)
        if media_type == "application/vnd.ollama.image.model":
            model_layers += 1
            if _normalized_digest(digest) != expected_model_blob_sha256:
                raise CandidateEvaluationError(
                    "Installed candidate GGUF differs from the exact exported artifact."
                )
            model_size = verified_size
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise CandidateEvaluationError("Candidate Ollama manifest has no config blob.")
    config_digest, config_size = config.get("digest"), config.get("size")
    if (
        not isinstance(config_digest, str)
        or isinstance(config_size, bool)
        or not isinstance(config_size, int)
        or config_size <= 0
    ):
        raise CandidateEvaluationError("Candidate Ollama config identity is malformed.")
    _verify_blob(root, config_digest, config_size)
    if model_layers != 1 or model_size <= 0:
        raise CandidateEvaluationError("Candidate Ollama manifest must contain exactly one model blob.")
    return manifest_sha, model_size


class OllamaCandidateClient:
    """Loopback-only client bound to one exact export receipt and candidate tag."""

    def __init__(self, receipt: Mapping[str, Any], *, timeout_seconds: int = 1200) -> None:
        self.receipt = dict(receipt)
        self.model_tag = str(receipt["candidate_tag"])
        self.timeout_seconds = timeout_seconds
        self._opener = urlrequest.build_opener(urlrequest.ProxyHandler({}))

    def _json(self, method: str, path: str, payload: object | None = None, *, timeout: int | None = None) -> Any:
        body = None if payload is None else _canonical_bytes(payload)
        request = urlrequest.Request(
            LOCAL_ENDPOINT + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        try:
            with self._opener.open(request, timeout=timeout or self.timeout_seconds) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except (urlerror.URLError, TimeoutError, OSError) as exc:
            raise CandidateEvaluationError(f"Ollama candidate request failed: {exc}") from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise CandidateEvaluationError("Ollama candidate response exceeded its safety limit.")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CandidateEvaluationError("Ollama candidate response is malformed JSON.") from exc

    def identity(self) -> CandidateIdentity:
        version = self._json("GET", "/api/version", timeout=5)
        tags = self._json("GET", "/api/tags", timeout=5)
        runtime_version = version.get("version") if isinstance(version, Mapping) else None
        models = tags.get("models") if isinstance(tags, Mapping) else None
        if not isinstance(runtime_version, str) or not runtime_version or not isinstance(models, list):
            raise CandidateEvaluationError("Ollama returned an invalid candidate identity response.")
        match = next(
            (
                item for item in models
                if isinstance(item, Mapping)
                and (item.get("name") == self.model_tag or item.get("model") == self.model_tag)
            ),
            None,
        )
        if not isinstance(match, Mapping):
            raise CandidateEvaluationError(f"Exported candidate tag is not installed: {self.model_tag}.")
        digest = match.get("digest")
        expected_digest = self.receipt["ollama"]["model_digest"]
        if not isinstance(digest, str) or _normalized_digest(digest) != _normalized_digest(str(expected_digest)):
            raise CandidateEvaluationError(
                "Installed candidate tag digest differs from the exact export receipt."
            )
        size = match.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise CandidateEvaluationError("Ollama returned an invalid candidate size.")
        expected_model_blob_sha = str(
            self.receipt["artifacts"]["quantized_gguf"]["sha256"]
        )
        if _SHA256.fullmatch(expected_model_blob_sha) is None:
            raise CandidateEvaluationError("Export receipt has no exact candidate GGUF SHA-256.")
        manifest_sha, verified_model_size = _verify_candidate_manifest(
            model_tag=self.model_tag,
            observed_digest=digest,
            expected_model_blob_sha256=expected_model_blob_sha,
        )
        expected_gguf_size = self.receipt["artifacts"]["quantized_gguf"]["size"]
        if verified_model_size != expected_gguf_size:
            raise CandidateEvaluationError(
                "Installed candidate GGUF size differs from the export receipt."
            )
        return CandidateIdentity(
            runtime_version=runtime_version,
            model_tag=self.model_tag,
            model_digest=digest,
            model_size_bytes=size,
            export_receipt_sha256=str(self.receipt["receipt_sha256"]),
            local_manifest_sha256=manifest_sha,
            model_blob_sha256=expected_model_blob_sha,
            verified_model_blob_size_bytes=verified_model_size,
        )

    def infer(self, *, instruction: str, context: Mapping[str, Any]) -> InferenceResult:
        payload = {
            "model": self.model_tag,
            "messages": [
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": user_prompt(instruction, context)},
            ],
            "stream": False,
            "format": output_schema(),
            "keep_alive": "10m",
            "options": dict(EVALUATION_GENERATION_OPTIONS),
        }
        started = time.monotonic()
        response = self._json("POST", "/api/chat", payload)
        elapsed = time.monotonic() - started
        if not isinstance(response, Mapping):
            raise CandidateEvaluationError("Ollama candidate chat response must be an object.")
        message = response.get("message")
        if response.get("done") is not True or response.get("model") != self.model_tag or not isinstance(message, Mapping):
            raise CandidateEvaluationError("Ollama did not complete inference with the exact candidate tag.")
        content = message.get("content")
        if not isinstance(content, str):
            raise CandidateEvaluationError("Ollama candidate returned no interpreter output.")
        telemetry = {
            "elapsed_seconds": elapsed,
            "done_reason": response.get("done_reason", ""),
            "total_duration_nanoseconds": response.get("total_duration", 0),
            "load_duration_nanoseconds": response.get("load_duration", 0),
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "generated_tokens": response.get("eval_count", 0),
        }
        try:
            return InferenceResult(content, json.loads(content), telemetry)
        except json.JSONDecodeError as exc:
            return InferenceResult(content, None, telemetry, f"JSONDecodeError: {exc}")


def compare_candidate_to_baseline(
    baseline_report: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    *,
    training_status: str | None = None,
    evaluation_purpose: str = "promotion",
) -> dict[str, Any]:
    """Apply the frozen non-regression policy to aggregate and family results."""
    baseline_metrics = baseline_report.get("metrics")
    if not isinstance(baseline_metrics, Mapping):
        raise CandidateEvaluationError("Baseline report has no metrics.")
    gates: list[dict[str, Any]] = []
    for name in _CORE_METRICS:
        base = baseline_metrics.get(name)
        candidate = candidate_metrics.get(name)
        if not isinstance(base, Mapping) or not isinstance(candidate, Mapping):
            raise CandidateEvaluationError(f"Comparison metric is missing: {name}.")
        base_rate = base.get("rate")
        candidate_rate = candidate.get("rate")
        base_passed, candidate_passed = base.get("passed"), candidate.get("passed")
        base_total, candidate_total = base.get("total"), candidate.get("total")
        if base_rate is None and candidate_rate is None and base_total == candidate_total:
            passed = True
            delta = None
            passed_delta = 0
        elif (
            isinstance(base_rate, (int, float))
            and isinstance(candidate_rate, (int, float))
            and isinstance(base_passed, int)
            and not isinstance(base_passed, bool)
            and isinstance(candidate_passed, int)
            and not isinstance(candidate_passed, bool)
            and base_total == candidate_total
        ):
            delta = float(candidate_rate) - float(base_rate)
            passed_delta = candidate_passed - base_passed
            passed = passed_delta >= 0
        else:
            passed = False
            delta = None
            passed_delta = None
        gates.append(
            {
                "gate": f"metric:{name}",
                "baseline_rate": base_rate,
                "candidate_rate": candidate_rate,
                "delta": delta,
                "passed_case_delta": passed_delta,
                "passed": passed,
            }
        )
    baseline_families = baseline_metrics.get("by_family")
    candidate_families = candidate_metrics.get("by_family")
    if not isinstance(baseline_families, Mapping) or not isinstance(candidate_families, Mapping):
        raise CandidateEvaluationError("Comparison family metrics are missing.")
    if set(baseline_families) != set(candidate_families):
        raise CandidateEvaluationError("Candidate and baseline family sets differ.")
    for family in sorted(baseline_families):
        base_rate = baseline_families[family].get("semantic_equivalent_rate")
        candidate_rate = candidate_families[family].get("semantic_equivalent_rate")
        passed = isinstance(base_rate, (int, float)) and isinstance(candidate_rate, (int, float)) and float(candidate_rate) >= float(base_rate)
        gates.append(
            {
                "gate": f"family:{family}:scientific_semantic_accuracy",
                "baseline_rate": base_rate,
                "candidate_rate": candidate_rate,
                "delta": float(candidate_rate) - float(base_rate) if passed or (isinstance(base_rate, (int, float)) and isinstance(candidate_rate, (int, float))) else None,
                "passed_case_delta": (
                    candidate_families[family].get("semantic_equivalent")
                    - baseline_families[family].get("semantic_equivalent")
                    if isinstance(candidate_families[family].get("semantic_equivalent"), int)
                    and isinstance(baseline_families[family].get("semantic_equivalent"), int)
                    else None
                ),
                "passed": passed,
            }
        )
    for name, floor in _ABSOLUTE_FLOORS.items():
        candidate = candidate_metrics.get(name)
        rate = candidate.get("rate") if isinstance(candidate, Mapping) else None
        gates.append(
            {
                "gate": f"absolute_floor:{name}",
                "baseline_rate": floor,
                "candidate_rate": rate,
                "delta": float(rate) - floor if isinstance(rate, (int, float)) else None,
                "passed_case_delta": None,
                "passed": isinstance(rate, (int, float)) and float(rate) >= floor,
            }
        )
    improvements = []
    for name in _BENEFIT_METRICS:
        base = baseline_metrics.get(name)
        candidate = candidate_metrics.get(name)
        if isinstance(base, Mapping) and isinstance(candidate, Mapping):
            base_passed, candidate_passed = base.get("passed"), candidate.get("passed")
            if (
                isinstance(base_passed, int)
                and not isinstance(base_passed, bool)
                and isinstance(candidate_passed, int)
                and not isinstance(candidate_passed, bool)
                and candidate.get("total") == base.get("total")
                and candidate_passed > base_passed
            ):
                improvements.append(
                    {"metric": name, "passed_case_delta": candidate_passed - base_passed}
                )
    gates.append(
        {
            "gate": "minimum_scientific_benefit:one_additional_correct_case",
            "baseline_rate": None,
            "candidate_rate": None,
            "delta": None,
            "passed_case_delta": sum(item["passed_case_delta"] for item in improvements),
            "passed": bool(improvements),
        }
    )
    if training_status is not None:
        if training_status not in {
            "candidate_adapter_experimental", "candidate_adapter_unpromoted"
        }:
            raise CandidateEvaluationError("Candidate training lifecycle state is invalid.")
        gates.append(
            {
                "gate": "training_state:candidate_adapter_unpromoted",
                "baseline_rate": None,
                "candidate_rate": None,
                "delta": None,
                "passed_case_delta": None,
                "passed": training_status == "candidate_adapter_unpromoted",
            }
        )
    if evaluation_purpose not in {"development", "promotion"}:
        raise CandidateEvaluationError("Evaluation purpose must be development or promotion.")
    gates.append(
        {
            "gate": "evaluation_purpose:one_shot_promotion",
            "baseline_rate": None,
            "candidate_rate": None,
            "delta": None,
            "passed_case_delta": None,
            "passed": evaluation_purpose == "promotion",
        }
    )
    return {
        "policy_version": PROMOTION_POLICY_VERSION,
        "policy": "all aggregate/family passed-case counts must not regress; absolute safety/scientific floors must pass; at least one predeclared scientific benefit metric must gain a correct case; production promotion requires non-experimental training",
        "absolute_floors": dict(_ABSOLUTE_FLOORS),
        "benefit_metrics": list(_BENEFIT_METRICS),
        "observed_improvements": improvements,
        "evaluation_purpose": evaluation_purpose,
        "gates": gates,
        "eligible_for_promotion": all(item["passed"] for item in gates),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as stream:
            json.dump(dict(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()


def run_candidate_evaluation(
    *,
    benchmark_path: str | Path | None = None,
    baseline_report_path: str | Path | None = None,
    training_baseline_report_path: str | Path = DEFAULT_BASELINE_REPORT,
    config_path: str | Path = DEFAULT_CONFIG,
    adapter_directory: str | Path,
    base_model_directory: str | Path,
    export_receipt_path: str | Path,
    client: object | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    progress: Callable[[int, int, Mapping[str, Any]], None] | None = None,
    evaluation_purpose: str = "development",
    promotion_campaign_id: str | None = None,
    promotion_seal_path: str | Path | None = None,
    promotion_ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate an exact candidate against an explicitly selected, matching base campaign."""
    if evaluation_purpose not in {"development", "promotion"}:
        raise CandidateEvaluationError("Evaluation purpose must be development or promotion.")
    if benchmark_path is None or baseline_report_path is None:
        raise CandidateEvaluationError(
            "Candidate evaluation requires explicit benchmark and untouched-base report paths."
        )
    benchmark_path = Path(benchmark_path).resolve()
    baseline_report_path = Path(baseline_report_path).resolve()
    seal: dict[str, Any] | None = None
    reservation: dict[str, Any] | None = None
    ledger_path: Path | None = None
    if evaluation_purpose == "promotion":
        try:
            campaign_id = str(uuid.UUID(str(promotion_campaign_id)))
        except (ValueError, AttributeError) as exc:
            raise CandidateEvaluationError(
                "A one-shot promotion evaluation requires an explicit UUID campaign ID."
            ) from exc
        if promotion_seal_path is None:
            raise CandidateEvaluationError(
                "Promotion evaluation requires a candidate-bound private-benchmark seal."
            )
        ledger_path = (
            Path(promotion_ledger_path).resolve()
            if promotion_ledger_path is not None
            else default_promotion_ledger_path()
        )
    elif (
        promotion_campaign_id is not None
        or promotion_seal_path is not None
        or promotion_ledger_path is not None
    ):
        raise CandidateEvaluationError(
            "Development evaluation must not declare promotion seal, campaign, or ledger evidence."
        )
    else:
        campaign_id = None
    valid, reason, baseline = baseline_gate(
        baseline_report_path, benchmark_path=benchmark_path
    )
    if not valid:
        raise CandidateEvaluationError(reason + ".")
    try:
        training = validate_training_report(
            adapter_directory,
            config_path=config_path,
            baseline_report=training_baseline_report_path,
            base_model_directory=base_model_directory,
            require_promotable=False,
        )
        receipt = validate_export_receipt(export_receipt_path, verify_artifacts=True)
    except (FineTuningError, RuntimeError) as exc:
        raise CandidateEvaluationError(str(exc)) from exc
    if receipt["training_report_sha256"] != training["report_sha256"]:
        raise CandidateEvaluationError("Export receipt belongs to a different training report.")
    if receipt.get("training_status") != training["status"]:
        raise CandidateEvaluationError("Export receipt misstates the adapter training lifecycle state.")
    if evaluation_purpose == "promotion":
        try:
            seal = validate_promotion_seal(
                promotion_seal_path,  # type: ignore[arg-type]
                benchmark_path=benchmark_path,
                baseline_report_path=baseline_report_path,
                export_receipt_path=export_receipt_path,
            )
            if seal["campaign_id"] != campaign_id:
                raise CandidateEvaluationError(
                    "Promotion campaign ID differs from the candidate-bound seal."
                )
            reservation = reserve_promotion_benchmark(ledger_path, seal)  # type: ignore[arg-type]
        except PromotionEvidenceError as exc:
            raise CandidateEvaluationError(str(exc)) from exc
    benchmark = load_benchmark(benchmark_path)
    reported_benchmark = baseline.get("benchmark")
    if not isinstance(reported_benchmark, Mapping) or reported_benchmark.get("benchmark_sha256") != benchmark["benchmark_sha256"]:
        raise CandidateEvaluationError("Candidate benchmark differs from the valid base-model baseline.")
    client = client or OllamaCandidateClient(receipt)
    try:
        identity = client.identity()  # type: ignore[attr-defined]
    except CandidateEvaluationError:
        raise
    except Exception as exc:
        raise CandidateEvaluationError(f"Candidate runtime identity verification failed: {exc}") from exc
    if not isinstance(identity, CandidateIdentity):
        raise CandidateEvaluationError("Candidate client returned an invalid identity record.")
    cases = list(benchmark["cases"])
    results: list[dict[str, Any]] = []
    checkpoint = Path(checkpoint_path).resolve() if checkpoint_path is not None else None
    checkpoint_contract = canonical_json_sha256(
        {
            "benchmark_sha256": benchmark["benchmark_sha256"],
            "baseline_report_sha256": baseline["report_sha256"],
            "training_report_sha256": training["report_sha256"],
            "export_receipt_sha256": receipt["receipt_sha256"],
            "evaluation_purpose": evaluation_purpose,
            "promotion_seal_sha256": seal["seal_sha256"] if seal else None,
            "promotion_reservation_sha256": (
                reservation["reservation_sha256"] if reservation else None
            ),
            "implementation_identity": _source_identity(),
            "runtime_identity": identity.report_payload(),
        }
    )
    if checkpoint is not None and resume and checkpoint.exists():
        try:
            stored = json.loads(checkpoint.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CandidateEvaluationError(f"Candidate checkpoint is unreadable: {exc}") from exc
        if not isinstance(stored, Mapping) or stored.get("schema") != CANDIDATE_CHECKPOINT_SCHEMA or stored.get("schema_version") != CANDIDATE_CHECKPOINT_VERSION or stored.get("contract_sha256") != checkpoint_contract:
            raise CandidateEvaluationError("Candidate checkpoint belongs to a different evaluation contract.")
        stored_results = stored.get("results")
        if not isinstance(stored_results, list) or [item.get("id") for item in stored_results if isinstance(item, Mapping)] != [case["id"] for case in cases[: len(stored_results)]]:
            raise CandidateEvaluationError("Candidate checkpoint is not a valid benchmark prefix.")
        results = [dict(item) for item in stored_results]
    for case in cases[len(results):]:
        result = _score_case(case, client, identity)  # type: ignore[arg-type]
        results.append(result)
        if checkpoint is not None:
            _atomic_json(
                checkpoint,
                {
                    "schema": CANDIDATE_CHECKPOINT_SCHEMA,
                    "schema_version": CANDIDATE_CHECKPOINT_VERSION,
                    "contract_sha256": checkpoint_contract,
                    "completed_count": len(results),
                    "results": results,
                },
            )
        if progress is not None:
            progress(len(results), len(cases), result)
    aggregate = _build_report(benchmark, identity=identity, results=results, limit=None)  # type: ignore[arg-type]
    comparison = compare_candidate_to_baseline(
        baseline,
        aggregate["metrics"],
        training_status=training["status"],
        evaluation_purpose=evaluation_purpose,
    )
    report: dict[str, Any] = {
        "schema": CANDIDATE_REPORT_SCHEMA,
        "schema_version": CANDIDATE_REPORT_VERSION,
        "campaign_status": "valid",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "implementation_identity": _source_identity(),
        "runtime_identity": identity.report_payload(),
        "evaluation": {
            "purpose": evaluation_purpose,
            "promotion_campaign_id": campaign_id,
            "promotion_seal_sha256": seal["seal_sha256"] if seal else None,
            "candidate_freeze_sha256": (
                seal["candidate"]["candidate_freeze_sha256"] if seal else None
            ),
            "promotion_reservation_sha256": (
                reservation["reservation_sha256"] if reservation else None
            ),
            "benchmark_consumption": (
                "one-shot-sequestered" if evaluation_purpose == "promotion" else "repeatable-development"
            ),
            "benchmark_disclosure": (
                "private-external-not-shipped"
                if evaluation_purpose == "promotion"
                else "public-repeatable-development"
            ),
        },
        "candidate": {
            "candidate_id": receipt["candidate_id"],
            "candidate_tag": receipt["candidate_tag"],
            "training_status": training["status"],
            "training_report_sha256": training["report_sha256"],
            "export_receipt_sha256": receipt["receipt_sha256"],
            "quantized_gguf_sha256": receipt["artifacts"]["quantized_gguf"]["sha256"],
        },
        "baseline": {
            "report_sha256": baseline["report_sha256"],
            "benchmark_sha256": benchmark["benchmark_sha256"],
        },
        "benchmark": aggregate["benchmark"],
        "metrics": aggregate["metrics"],
        "comparison": comparison,
        "cases": aggregate["cases"],
        "promotion_performed": False,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def finalize_candidate_evaluation_report(
    report: Mapping[str, Any],
    output_path: str | Path,
    *,
    promotion_ledger_path: str | Path | None = None,
) -> None:
    """Durably write the report before marking a private benchmark consumed."""
    declared = report.get("report_sha256")
    without_sha = {key: item for key, item in report.items() if key != "report_sha256"}
    if (
        not isinstance(declared, str)
        or _SHA256.fullmatch(declared) is None
        or declared != canonical_json_sha256(without_sha)
    ):
        raise CandidateEvaluationError("Candidate evaluation report checksum is invalid.")
    _atomic_json(Path(output_path).resolve(), report)
    evaluation = report.get("evaluation")
    if isinstance(evaluation, Mapping) and evaluation.get("purpose") == "promotion":
        reservation = evaluation.get("promotion_reservation_sha256")
        if not isinstance(reservation, str) or _SHA256.fullmatch(reservation) is None:
            raise CandidateEvaluationError("Promotion report lacks its reservation identity.")
        ledger = (
            Path(promotion_ledger_path).resolve()
            if promotion_ledger_path is not None
            else default_promotion_ledger_path()
        )
        try:
            complete_promotion_benchmark(
                ledger,
                reservation_sha256=reservation,
                candidate_report_sha256=declared,
            )
        except PromotionEvidenceError as exc:
            raise CandidateEvaluationError(str(exc)) from exc


def validate_candidate_evaluation_report(
    value: str | Path | Mapping[str, Any],
    *,
    benchmark_path: str | Path,
    baseline_report_path: str | Path,
    export_receipt_path: str | Path | None = None,
    promotion_seal_path: str | Path | None = None,
    promotion_ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        try:
            raw = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CandidateEvaluationError(f"Cannot read candidate evaluation report: {exc}") from exc
    else:
        raw = dict(value)
    if not isinstance(raw, Mapping) or raw.get("schema") != CANDIDATE_REPORT_SCHEMA or raw.get("schema_version") != CANDIDATE_REPORT_VERSION:
        raise CandidateEvaluationError("Unsupported candidate evaluation report schema.")
    required = {
        "schema", "schema_version", "campaign_status", "created_at_utc",
        "implementation_identity", "runtime_identity", "evaluation", "candidate", "baseline",
        "benchmark", "metrics", "comparison", "cases", "promotion_performed",
        "report_sha256",
    }
    if set(raw) != required:
        raise CandidateEvaluationError("Candidate evaluation report has missing or unknown fields.")
    declared = raw.get("report_sha256")
    without_sha = {key: item for key, item in raw.items() if key != "report_sha256"}
    if not isinstance(declared, str) or _SHA256.fullmatch(declared) is None or declared != canonical_json_sha256(without_sha):
        raise CandidateEvaluationError("Candidate evaluation report checksum is invalid.")
    if raw.get("campaign_status") != "valid" or raw.get("promotion_performed") is not False:
        raise CandidateEvaluationError("Candidate evaluation campaign is incomplete or already mutated.")
    if raw.get("implementation_identity") != _source_identity():
        raise CandidateEvaluationError(
            "Candidate evaluation was produced by different evaluator/compiler code."
        )
    evaluation = raw.get("evaluation")
    if not isinstance(evaluation, Mapping) or set(evaluation) != {
        "purpose", "promotion_campaign_id", "promotion_seal_sha256",
        "candidate_freeze_sha256", "promotion_reservation_sha256",
        "benchmark_consumption", "benchmark_disclosure",
    }:
        raise CandidateEvaluationError("Candidate evaluation purpose evidence is malformed.")
    purpose = evaluation.get("purpose")
    if purpose == "promotion":
        try:
            uuid.UUID(str(evaluation.get("promotion_campaign_id")))
        except (ValueError, AttributeError) as exc:
            raise CandidateEvaluationError("Promotion campaign ID must be a UUID.") from exc
        if evaluation.get("benchmark_consumption") != "one-shot-sequestered":
            raise CandidateEvaluationError("Promotion benchmark consumption policy is invalid.")
        if evaluation.get("benchmark_disclosure") != "private-external-not-shipped":
            raise CandidateEvaluationError("Promotion benchmark disclosure policy is invalid.")
        for field in (
            "promotion_seal_sha256", "candidate_freeze_sha256",
            "promotion_reservation_sha256",
        ):
            if (
                not isinstance(evaluation.get(field), str)
                or _SHA256.fullmatch(evaluation[field]) is None
            ):
                raise CandidateEvaluationError(f"Promotion {field} evidence is invalid.")
        if (
            export_receipt_path is None
            or promotion_seal_path is None
            or promotion_ledger_path is None
        ):
            raise CandidateEvaluationError(
                "Promotion report validation requires its export receipt, benchmark seal, and consumption ledger."
            )
    elif purpose == "development":
        if (
            evaluation.get("promotion_campaign_id") is not None
            or evaluation.get("promotion_seal_sha256") is not None
            or evaluation.get("candidate_freeze_sha256") is not None
            or evaluation.get("promotion_reservation_sha256") is not None
            or evaluation.get("benchmark_consumption") != "repeatable-development"
            or evaluation.get("benchmark_disclosure")
            != "public-repeatable-development"
        ):
            raise CandidateEvaluationError("Development evaluation purpose evidence is invalid.")
    else:
        raise CandidateEvaluationError("Candidate evaluation purpose is unsupported.")
    candidate = raw.get("candidate")
    runtime = raw.get("runtime_identity")
    if not isinstance(candidate, Mapping) or not isinstance(runtime, Mapping):
        raise CandidateEvaluationError("Candidate evaluation lacks candidate/runtime identity.")
    if set(candidate) != {
        "candidate_id", "candidate_tag", "training_status", "training_report_sha256",
        "export_receipt_sha256", "quantized_gguf_sha256",
    } or not isinstance(candidate.get("candidate_id"), str) or not candidate["candidate_id"].startswith("candidate-"):
        raise CandidateEvaluationError("Candidate evaluation artifact identity is malformed.")
    for field in ("training_report_sha256", "export_receipt_sha256", "quantized_gguf_sha256"):
        if not isinstance(candidate.get(field), str) or _SHA256.fullmatch(candidate[field]) is None:
            raise CandidateEvaluationError(f"Candidate evaluation {field} is invalid.")
    if candidate.get("training_status") not in {
        "candidate_adapter_experimental", "candidate_adapter_unpromoted"
    }:
        raise CandidateEvaluationError("Candidate evaluation training lifecycle state is invalid.")
    baseline = raw.get("baseline")
    if not isinstance(baseline, Mapping) or set(baseline) != {
        "report_sha256", "benchmark_sha256"
    } or any(
        not isinstance(baseline.get(field), str) or _SHA256.fullmatch(baseline[field]) is None
        for field in ("report_sha256", "benchmark_sha256")
    ):
        raise CandidateEvaluationError("Candidate evaluation baseline identity is malformed.")
    valid_baseline, baseline_reason, baseline_evidence = baseline_gate(
        baseline_report_path, benchmark_path=benchmark_path
    )
    if not valid_baseline:
        raise CandidateEvaluationError(baseline_reason + ".")
    if (
        baseline.get("report_sha256") != baseline_evidence.get("report_sha256")
        or baseline.get("benchmark_sha256")
        != baseline_evidence.get("benchmark", {}).get("benchmark_sha256")
    ):
        raise CandidateEvaluationError(
            "Candidate report is bound to a different untouched-base evaluation."
        )
    if purpose == "promotion":
        try:
            seal = validate_promotion_seal(
                promotion_seal_path,  # type: ignore[arg-type]
                benchmark_path=benchmark_path,
                baseline_report_path=baseline_report_path,
                export_receipt_path=export_receipt_path,  # type: ignore[arg-type]
            )
            if (
                seal["campaign_id"] != evaluation.get("promotion_campaign_id")
                or seal["seal_sha256"] != evaluation.get("promotion_seal_sha256")
                or seal["candidate"]["candidate_freeze_sha256"]
                != evaluation.get("candidate_freeze_sha256")
            ):
                raise CandidateEvaluationError(
                    "Promotion report differs from its candidate-bound benchmark seal."
                )
            validate_promotion_consumption(
                promotion_ledger_path,  # type: ignore[arg-type]
                reservation_sha256=evaluation["promotion_reservation_sha256"],
                candidate_report_sha256=declared,
            )
        except PromotionEvidenceError as exc:
            raise CandidateEvaluationError(str(exc)) from exc
    if not isinstance(raw.get("metrics"), Mapping):
        raise CandidateEvaluationError("Candidate evaluation metrics are missing.")
    required_runtime = {
        "runtime", "runtime_version", "model_tag", "observed_model_digest",
        "model_size_bytes", "export_receipt_sha256", "identity_verification",
        "local_manifest_sha256", "model_blob_sha256", "verified_model_blob_size_bytes",
        "all_manifest_blobs_verified",
    }
    if set(runtime) != required_runtime or runtime.get("runtime") != "ollama" or runtime.get(
        "identity_verification"
    ) != "exact_export_receipt_manifest_and_all_local_blobs_sha256":
        raise CandidateEvaluationError("Candidate runtime identity is incomplete or unverified.")
    if (
        runtime.get("model_tag") != candidate.get("candidate_tag")
        or runtime.get("export_receipt_sha256") != candidate.get("export_receipt_sha256")
    ):
        raise CandidateEvaluationError("Candidate report runtime and artifact identities differ.")
    model_size = runtime.get("model_size_bytes")
    if isinstance(model_size, bool) or not isinstance(model_size, int) or model_size <= 0:
        raise CandidateEvaluationError("Candidate runtime size is invalid.")
    for field in ("local_manifest_sha256", "model_blob_sha256"):
        if not isinstance(runtime.get(field), str) or _SHA256.fullmatch(runtime[field]) is None:
            raise CandidateEvaluationError(f"Candidate runtime {field} is invalid.")
    verified_blob_size = runtime.get("verified_model_blob_size_bytes")
    if isinstance(verified_blob_size, bool) or not isinstance(verified_blob_size, int) or verified_blob_size <= 0:
        raise CandidateEvaluationError("Candidate verified model-blob size is invalid.")
    if runtime.get("all_manifest_blobs_verified") is not True:
        raise CandidateEvaluationError("Candidate report does not prove verification of every manifest blob.")
    if _normalized_digest(str(runtime.get("observed_model_digest", ""))) != runtime["local_manifest_sha256"]:
        raise CandidateEvaluationError("Candidate manifest bytes differ from the observed Ollama digest.")
    if runtime["model_blob_sha256"] != candidate.get("quantized_gguf_sha256"):
        raise CandidateEvaluationError("Candidate runtime GGUF differs from the exported candidate artifact.")
    identity = CandidateIdentity(
        runtime_version=str(runtime.get("runtime_version", "")),
        model_tag=str(runtime.get("model_tag", "")),
        model_digest=str(runtime.get("observed_model_digest", "")),
        model_size_bytes=model_size,
        export_receipt_sha256=str(runtime.get("export_receipt_sha256", "")),
        local_manifest_sha256=str(runtime.get("local_manifest_sha256", "")),
        model_blob_sha256=str(runtime.get("model_blob_sha256", "")),
        verified_model_blob_size_bytes=verified_blob_size,
    )
    _normalized_digest(identity.model_digest)
    comparison = raw.get("comparison")
    if not isinstance(comparison, Mapping) or comparison.get("policy_version") != PROMOTION_POLICY_VERSION:
        raise CandidateEvaluationError("Candidate evaluation promotion policy is missing or unsupported.")
    if comparison.get("evaluation_purpose") != purpose:
        raise CandidateEvaluationError("Candidate comparison and evaluation purposes differ.")
    if comparison.get("absolute_floors") != _ABSOLUTE_FLOORS or comparison.get(
        "benefit_metrics"
    ) != list(_BENEFIT_METRICS):
        raise CandidateEvaluationError("Candidate comparison policy constants were modified.")
    gates = comparison.get("gates")
    expected_eligibility = all(
        isinstance(item, Mapping) and item.get("passed") is True for item in gates
    ) if isinstance(gates, list) else False
    if (
        not isinstance(gates, list)
        or not gates
        or comparison.get("eligible_for_promotion") != expected_eligibility
    ):
        raise CandidateEvaluationError("Candidate evaluation gate summary is inconsistent.")
    cases = raw.get("cases")
    benchmark = raw.get("benchmark")
    if not isinstance(cases, list) or not isinstance(benchmark, Mapping) or len(cases) != benchmark.get("case_count"):
        raise CandidateEvaluationError("Candidate evaluation report lacks complete case evidence.")
    fixed_benchmark = load_benchmark(benchmark_path)
    if (
        benchmark.get("benchmark_sha256") != fixed_benchmark["benchmark_sha256"]
        or len(cases) != len(fixed_benchmark["cases"])
        or benchmark.get("limited_run") is not False
    ):
        raise CandidateEvaluationError("Candidate report is not the complete supplied benchmark.")
    replayed: list[dict[str, Any]] = []
    for expected_case, stored in zip(fixed_benchmark["cases"], cases, strict=True):
        if not isinstance(stored, Mapping) or stored.get("id") != expected_case["id"]:
            raise CandidateEvaluationError("Candidate case order/identity differs from the benchmark.")
        rounds = stored.get("rounds")
        if not isinstance(rounds, list) or not rounds:
            raise CandidateEvaluationError(f"Candidate case {expected_case['id']} has no raw evidence.")

        class _ReplayClient:
            def __init__(self, evidence: Sequence[object]) -> None:
                self.evidence = list(evidence)
                self.index = 0

            def infer(self, *, instruction: str, context: Mapping[str, Any]) -> InferenceResult:
                del instruction, context
                if self.index >= len(self.evidence):
                    raise CandidateEvaluationError("Candidate evidence ended before replay completed.")
                item = self.evidence[self.index]
                self.index += 1
                if not isinstance(item, Mapping):
                    raise CandidateEvaluationError("Candidate inference evidence is malformed.")
                content, telemetry = item.get("raw_content"), item.get("telemetry")
                if not isinstance(content, str) or not isinstance(telemetry, Mapping):
                    raise CandidateEvaluationError("Candidate evidence lacks raw output or telemetry.")
                if hashlib.sha256(content.encode("utf-8")).hexdigest() != item.get("raw_content_sha256"):
                    raise CandidateEvaluationError("Candidate raw-output hash is invalid.")
                parsed = item.get("parsed_output")
                if parsed is not None:
                    try:
                        if json.loads(content) != parsed:
                            raise CandidateEvaluationError("Candidate parsed output differs from raw JSON.")
                    except json.JSONDecodeError as exc:
                        raise CandidateEvaluationError("Candidate raw output is not JSON.") from exc
                return InferenceResult(
                    raw_content=content,
                    parsed_output=parsed,
                    telemetry=dict(telemetry),
                    parse_error=str(item.get("model_output_error") or ""),
                )

        replay_client = _ReplayClient(rounds)
        replayed_case = _score_case(expected_case, replay_client, identity)  # type: ignore[arg-type]
        if replay_client.index != len(rounds) or replayed_case != dict(stored):
            raise CandidateEvaluationError(
                f"Candidate case {expected_case['id']} does not replay exactly."
            )
        replayed.append(replayed_case)
    rebuilt = _build_report(fixed_benchmark, identity=identity, results=replayed, limit=None)  # type: ignore[arg-type]
    if rebuilt["metrics"] != raw.get("metrics") or rebuilt["benchmark"] != benchmark:
        raise CandidateEvaluationError("Candidate aggregate results do not recompute from evidence.")
    recomputed_comparison = compare_candidate_to_baseline(
        baseline_evidence,
        rebuilt["metrics"],
        training_status=str(candidate.get("training_status")),
        evaluation_purpose=str(purpose),
    )
    if recomputed_comparison != comparison:
        raise CandidateEvaluationError(
            "Candidate comparison does not recompute from the supplied untouched-base evidence."
        )
    return dict(raw)
