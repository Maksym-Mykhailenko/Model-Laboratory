"""Reproducible Stage-5 baseline evaluation for the local Qwen interpreter.

The baseline is an evaluation campaign, not training code.  It binds the benchmark to a
versioned prompt/schema/compiler contract, verifies the exact frozen base-model artifact
before inference, preserves every evaluation round, separates infrastructure failures from
model failures, and scores typed scientific semantics independently of source formatting.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import difflib
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
import time
import uuid
from typing import Any, Callable, Mapping, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

import sympy as sp

from . import __version__
from .build_identity import git_commit_or_build_id, source_tree_sha256
from .canonical import canonical_json_sha256, canonical_model_ir_sha256
from .interpreter import (
    EVALUATION_GENERATION_OPTIONS,
    GENERATION_OPTIONS,
    GENERATION_PROFILES,
    INTERPRETER_OUTPUT_SCHEMA,
    INTERPRETER_OUTPUT_SCHEMA_VERSION,
    LOCAL_BASE_MODEL,
    LOCAL_ENDPOINT,
    LOCAL_MODEL,
    LOCAL_QUANTIZATION,
    MAX_CONTEXT_ROUNDS,
    InterpreterProposalError,
    create_interpreter_context,
    process_interpreter_output,
    validate_raw_interpreter_output,
)
from .model import ModelIR
from .parser import ModelParseError, parse_model_text
from .validator import ModelValidationError, validate_model

BASELINE_PROTOCOL = "model-laboratory-interpreter-baseline"
BASELINE_PROTOCOL_VERSION = "1.4"
BENCHMARK_SCHEMA = "model-laboratory-interpreter-benchmark"
BENCHMARK_SCHEMA_VERSION = "1.6"
PREVIOUS_BENCHMARK_SCHEMA_VERSIONS = {"1.5", "1.4"}
REPORT_SCHEMA = "model-laboratory-interpreter-baseline-report"
REPORT_SCHEMA_VERSION = "1.4"
PROMPT_TEMPLATE_VERSION = "1.3"
SCORER_VERSION = "1.4"
CHECKPOINT_SCHEMA = "model-laboratory-interpreter-baseline-checkpoint"
CHECKPOINT_SCHEMA_VERSION = "1.0"

_ASSET_DIR = Path(__file__).with_name("baseline")
_SYSTEM_PROMPT_PATH = _ASSET_DIR / "system_prompt_v1.3.txt"
_OUTPUT_SCHEMA_PATH = _ASSET_DIR / "output_schema_v1.2.json"
_USER_PROMPT_TEMPLATE_PATH = _ASSET_DIR / "user_prompt_template_v1.0.txt"
_MODEL_LOCK_PATH = _ASSET_DIR / "base_model_lock_v1.1.json"
_FEW_SHOT_PATH = _ASSET_DIR / "few_shot_examples_v1.2.json"

_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_REPORTED_DIFFERENCES = 64
_MAX_FEW_SHOT_SIMILARITY = 0.68
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_SHA256_PREFIX = re.compile(r"^sha256:[0-9a-f]{12,64}$")


class BaselineInfrastructureError(RuntimeError):
    """A runtime/identity/transport failure that invalidates an evaluation campaign."""


@dataclass(frozen=True)
class InferenceResult:
    raw_content: str
    parsed_output: object | None
    telemetry: dict[str, Any]
    parse_error: str = ""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return value


def system_prompt() -> str:
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def output_schema() -> dict[str, Any]:
    value = _load_json_object(_OUTPUT_SCHEMA_PATH)
    props = value.get("properties")
    if not isinstance(props, Mapping):
        raise ValueError("Interpreter output schema asset has no properties object.")
    if props.get("schema", {}).get("const") != INTERPRETER_OUTPUT_SCHEMA:
        raise ValueError("Interpreter output schema name does not match the compiler.")
    if props.get("schema_version", {}).get("const") != INTERPRETER_OUTPUT_SCHEMA_VERSION:
        raise ValueError("Interpreter output schema version does not match the compiler.")
    if value.get("additionalProperties") is not False:
        raise ValueError("Interpreter output schema must reject unknown fields.")
    return value


def _require_hex_digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_HEX.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest.")
    return value


def base_model_lock() -> dict[str, Any]:
    value = _load_json_object(_MODEL_LOCK_PATH)
    if set(value) != {"schema", "schema_version", "upstream", "ollama"}:
        raise ValueError("Base-model lock contains unexpected fields.")
    if value.get("schema") != "model-laboratory-base-model-lock" or value.get("schema_version") != "1.1":
        raise ValueError("Unsupported base-model lock asset.")

    upstream = value.get("upstream")
    if not isinstance(upstream, Mapping) or set(upstream) != {
        "repository", "revision", "license_spdx", "copyright", "tokenizer", "weight_files"
    }:
        raise ValueError("Base-model upstream lock is malformed.")
    if upstream.get("repository") != LOCAL_BASE_MODEL:
        raise ValueError("Base-model lock repository does not match the interpreter configuration.")
    revision = upstream.get("revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Base-model revision must be a full Git commit digest.")
    if upstream.get("license_spdx") != "Apache-2.0":
        raise ValueError("Base-model lock has an unexpected licence identifier.")
    tokenizer = upstream.get("tokenizer")
    if not isinstance(tokenizer, Mapping) or set(tokenizer) != {"filename", "sha256"}:
        raise ValueError("Tokenizer lock is malformed.")
    if tokenizer.get("filename") != "tokenizer.json":
        raise ValueError("Tokenizer lock must identify tokenizer.json.")
    _require_hex_digest(tokenizer.get("sha256"), "tokenizer.sha256")
    weights = upstream.get("weight_files")
    if not isinstance(weights, list) or len(weights) != 3:
        raise ValueError("Base-model lock must contain exactly three upstream weight shards.")
    expected_weight_names = {f"model-0000{i}-of-00003.safetensors" for i in range(1, 4)}
    observed_weight_names: set[str] = set()
    for index, item in enumerate(weights):
        if not isinstance(item, Mapping) or set(item) != {"filename", "sha256"}:
            raise ValueError(f"Weight lock entry {index + 1} is malformed.")
        filename = item.get("filename")
        if not isinstance(filename, str):
            raise ValueError("Weight filenames must be strings.")
        observed_weight_names.add(filename)
        _require_hex_digest(item.get("sha256"), f"weight_files[{index}].sha256")
    if observed_weight_names != expected_weight_names:
        raise ValueError("Base-model lock weight shard names are incomplete or unexpected.")

    ollama = value.get("ollama")
    expected_ollama_keys = {
        "model_tag", "quantization", "registry_manifest_sha256", "config", "layers"
    }
    if not isinstance(ollama, Mapping) or set(ollama) != expected_ollama_keys:
        raise ValueError("Ollama artifact lock is malformed.")
    if ollama.get("model_tag") != LOCAL_MODEL or ollama.get("quantization") != LOCAL_QUANTIZATION:
        raise ValueError("Ollama artifact lock does not match the configured model tag/quantization.")
    _require_hex_digest(ollama.get("registry_manifest_sha256"), "ollama.registry_manifest_sha256")
    config = ollama.get("config")
    if not isinstance(config, Mapping) or set(config) != {"media_type", "sha256", "size_bytes"}:
        raise ValueError("Ollama config lock is malformed.")
    if not isinstance(config.get("media_type"), str) or not config["media_type"]:
        raise ValueError("Ollama config media type is malformed.")
    _require_hex_digest(config.get("sha256"), "ollama.config.sha256")
    if isinstance(config.get("size_bytes"), bool) or not isinstance(config.get("size_bytes"), int) or config["size_bytes"] <= 0:
        raise ValueError("Ollama config size is malformed.")
    layers = ollama.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("Ollama layer locks are missing.")
    media_types: set[str] = set()
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping) or set(layer) != {"media_type", "sha256", "size_bytes"}:
            raise ValueError(f"Ollama layer lock {index} is malformed.")
        media_type = layer.get("media_type")
        if not isinstance(media_type, str) or not media_type or media_type in media_types:
            raise ValueError("Ollama layer media types must be non-empty and unique.")
        media_types.add(media_type)
        _require_hex_digest(layer.get("sha256"), f"ollama.layers[{index}].sha256")
        if isinstance(layer.get("size_bytes"), bool) or not isinstance(layer.get("size_bytes"), int) or layer["size_bytes"] <= 0:
            raise ValueError(f"Ollama layer lock {index} has an invalid size.")
    if "application/vnd.ollama.image.model" not in media_types:
        raise ValueError("Ollama layer lock has no model artifact.")
    return value


def few_shot_examples() -> tuple[str, ...]:
    value = _load_json_object(_FEW_SHOT_PATH)
    items = value.get("instructions")
    if not isinstance(items, list) or not items or not all(isinstance(item, str) and item.strip() for item in items):
        raise ValueError("Few-shot example registry is invalid.")
    return tuple(item.strip() for item in items)


def user_prompt(instruction: str, context: Mapping[str, Any]) -> str:
    template = _USER_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    if template.count("{{INSTRUCTION}}") != 1 or template.count("{{CONTEXT_JSON}}") != 1:
        raise ValueError("User prompt template must contain each placeholder exactly once.")
    context_json = json.dumps(context, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return template.replace("{{INSTRUCTION}}", instruction).replace("{{CONTEXT_JSON}}", context_json).rstrip("\n")


def baseline_contract() -> dict[str, Any]:
    lock = base_model_lock()
    return {
        "schema": BASELINE_PROTOCOL,
        "schema_version": BASELINE_PROTOCOL_VERSION,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt_sha256": _sha256_bytes(_SYSTEM_PROMPT_PATH.read_bytes()),
        "user_prompt_template_sha256": _sha256_bytes(_USER_PROMPT_TEMPLATE_PATH.read_bytes()),
        "output_schema_sha256": _sha256_bytes(_OUTPUT_SCHEMA_PATH.read_bytes()),
        "base_model_lock_sha256": _sha256_bytes(_MODEL_LOCK_PATH.read_bytes()),
        "few_shot_registry_sha256": _sha256_bytes(_FEW_SHOT_PATH.read_bytes()),
        "expected_base_model": LOCAL_BASE_MODEL,
        "model_tag": LOCAL_MODEL,
        "expected_quantization": LOCAL_QUANTIZATION,
        "upstream_revision": lock["upstream"]["revision"],
        "endpoint": LOCAL_ENDPOINT,
        "production_generation_options": dict(GENERATION_OPTIONS),
        "production_generation_profiles": [dict(item) for item in GENERATION_PROFILES],
        "production_profile_selection": "smallest conservative request-size fit",
        "evaluation_generation_options": dict(EVALUATION_GENERATION_OPTIONS),
        "evaluation_sampling": "greedy deterministic profile with fixed seed=0",
        "adapters_permitted": False,
        "scorer_version": SCORER_VERSION,
    }


# ---------- typed semantic comparison ----------

def _metadata_payload(value: object) -> dict[str, Any]:
    return {
        "label": getattr(value, "label", None),
        "description": getattr(value, "description", None),
        "unit": getattr(value, "unit", None),
        "tags": sorted(getattr(value, "tags", ())),
    }


def _domain_payload(value: object) -> dict[str, Any]:
    return {
        "lower": float(getattr(value, "lower")),
        "upper": float(getattr(value, "upper")),
        "lower_inclusive": bool(getattr(value, "lower_inclusive")),
        "upper_inclusive": bool(getattr(value, "upper_inclusive")),
    }


def _normalized_expression(expr: sp.Expr) -> str:
    """Deterministic evaluator normal form; never persisted as canonical Model IR.

    Baseline accuracy should not depend on harmless algebraic spelling.  ``simplify`` plus
    ``trigsimp`` handles ordinary polynomial/rational/trigonometric equivalence while the
    authoritative expression parser still determines what mathematics is representable.
    This is an evaluation representation only; it never changes Model IR fingerprints.
    """
    simplified = sp.trigsimp(sp.simplify(expr))
    normalized = sp.cancel(sp.together(sp.expand(simplified)))
    return sp.srepr(normalized)


def _constraint_semantics(item: object) -> dict[str, Any]:
    left = getattr(item, "left")
    right = getattr(item, "right")
    relation = getattr(item, "relation")
    relation_value = relation.value if hasattr(relation, "value") else str(relation)
    delta = sp.trigsimp(sp.simplify(left - right))
    if relation_value in {">", ">="}:
        delta = -delta
        relation_value = "<" if relation_value == ">" else "<="

    # Multiplying both sides of an equality by any non-zero numeric scalar, or an inequality
    # by a positive scalar, does not change the declared relation.
    coeff, rest = sp.factor(delta).as_coeff_Mul()
    if relation_value == "==" and coeff != 0:
        delta = rest
    elif relation_value in {"<", "<="} and bool(coeff.is_positive):
        delta = rest

    if relation_value == "==":
        a = _normalized_expression(delta)
        b = _normalized_expression(-delta)
        delta_repr = min(a, b)
    else:
        delta_repr = _normalized_expression(delta)
    return {
        "name": getattr(item, "name"),
        "relation": relation_value,
        "delta": delta_repr,
        "metadata": _metadata_payload(getattr(item, "metadata")),
    }


def scientific_semantics_payload(model: ModelIR) -> dict[str, Any]:
    """Typed evaluation payload that ignores authoring provenance/title but not science.

    Native mathematical objects are represented by their typed fields and symbolically
    normalized expressions.  Core Model-Graph mirrors are omitted because those same native
    objects are already represented here.  Extension graph objects/relationships/assets are
    retained exactly, including relationship ``source`` and ``target`` endpoints.
    """
    def expr_item(item: object) -> dict[str, Any]:
        return {
            "name": getattr(item, "name"),
            "expression": _normalized_expression(getattr(item, "expression")),
            "metadata": _metadata_payload(getattr(item, "metadata")),
        }

    extension_objects = sorted(
        (item.payload() for item in model.graph.objects if not item.kind.startswith("org.modellab.core.")),
        key=lambda item: item["id"],
    )
    extension_relationships = sorted(
        (item.payload() for item in model.graph.relationships if not item.kind.startswith("org.modellab.core.")),
        key=lambda item: item["id"],
    )
    assets = sorted((item.payload() for item in model.graph.assets), key=lambda item: item["id"])
    return {
        "variables": sorted(
            ({"name": x.name, "domain": _domain_payload(x.domain), "initial_value": x.initial_value, "metadata": _metadata_payload(x.metadata)} for x in model.variables),
            key=lambda item: item["name"],
        ),
        "parameters": sorted(
            ({"name": x.name, "default": x.default, "domain": _domain_payload(x.domain), "metadata": _metadata_payload(x.metadata)} for x in model.parameters),
            key=lambda item: item["name"],
        ),
        "constants": sorted(
            ({"name": x.name, "value": x.value, "metadata": _metadata_payload(x.metadata)} for x in model.constants),
            key=lambda item: item["name"],
        ),
        "derived_quantities": sorted((expr_item(x) for x in model.derived_quantities), key=lambda item: item["name"]),
        "functions": sorted((expr_item(x) for x in model.functions), key=lambda item: item["name"]),
        "vector_functions": sorted(
            ({"name": x.name, "components": [expr_item(c) for c in x.components], "metadata": _metadata_payload(x.metadata)} for x in model.vector_functions),
            key=lambda item: item["name"],
        ),
        "matrix_functions": sorted(
            ({"name": x.name, "entries": [[expr_item(c) for c in row] for row in x.entries], "row_labels": list(x.row_labels), "column_labels": list(x.column_labels), "metadata": _metadata_payload(x.metadata)} for x in model.matrix_functions),
            key=lambda item: item["name"],
        ),
        "constraints": sorted((_constraint_semantics(x) for x in model.constraints), key=lambda item: item["name"]),
        "assumptions": sorted(
            ({"name": x.name, "statement": x.statement, "affects": sorted(x.affects)} for x in model.assumptions),
            key=lambda item: item["name"],
        ),
        "ambiguities": sorted(
            ({"name": x.name, "statement": x.statement, "options": sorted(x.options), "resolution": x.resolution, "blocking": x.blocking, "affects": sorted(x.affects)} for x in model.ambiguities),
            key=lambda item: item["name"],
        ),
        "extension_graph": {
            "objects": extension_objects,
            "relationships": extension_relationships,
            "assets": assets,
        },
    }


def scientific_semantics_sha256(model: ModelIR) -> str:
    """Stable hash of the typed evaluator payload (not a persisted scientific fingerprint)."""
    return canonical_json_sha256(scientific_semantics_payload(model))


def compare_model_semantics(expected: ModelIR, actual: ModelIR) -> tuple[bool, list[str]]:
    """Compare typed scientific semantics and return bounded human-readable differences."""
    left = scientific_semantics_payload(expected)
    right = scientific_semantics_payload(actual)
    if left == right:
        return True, []

    differences: list[str] = []

    def walk(a: object, b: object, path: str) -> None:
        if len(differences) >= _MAX_REPORTED_DIFFERENCES or a == b:
            return
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for key in sorted(set(a) | set(b)):
                if key not in a:
                    differences.append(f"{path}.{key}: unexpected")
                elif key not in b:
                    differences.append(f"{path}.{key}: missing")
                else:
                    walk(a[key], b[key], f"{path}.{key}")
                if len(differences) >= _MAX_REPORTED_DIFFERENCES:
                    return
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                differences.append(f"{path}: length {len(b)} != expected {len(a)}")
                return
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{index}]")
                if len(differences) >= _MAX_REPORTED_DIFFERENCES:
                    return
            return
        differences.append(f"{path}: {b!r} != expected {a!r}")

    walk(left, right, "model")
    return False, differences


# ---------- benchmark validation ----------

def _normalized_instruction(value: str) -> str:
    return " ".join(value.casefold().split())


def _instruction_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(
        None, _normalized_instruction(left), _normalized_instruction(right), autojunk=False
    ).ratio()


def _validate_benchmark_case(case: object, seen: set[str], few_shot: set[str]) -> dict[str, Any]:
    if not isinstance(case, Mapping):
        raise ValueError("Every benchmark case must be an object.")
    allowed = {"id", "family", "instruction", "current_model_source", "expected", "notes"}
    required = {"id", "family", "instruction", "current_model_source", "expected"}
    missing = required - set(case)
    extra = set(case) - allowed
    if missing or extra:
        raise ValueError(f"Benchmark case has invalid fields; missing={sorted(missing)}, extra={sorted(extra)}")
    case_id, family, instruction, source = (case[key] for key in ("id", "family", "instruction", "current_model_source"))
    if not all(isinstance(item, str) for item in (case_id, family, instruction, source)):
        raise ValueError("Benchmark id, family, instruction, and current_model_source must be strings.")
    if not case_id.strip() or case_id in seen:
        raise ValueError("Benchmark case IDs must be unique non-empty strings.")
    if not family.strip() or not instruction.strip():
        raise ValueError(f"Benchmark case {case_id} requires non-empty family and instruction.")
    normalized_instruction = _normalized_instruction(instruction)
    if normalized_instruction in few_shot:
        raise ValueError(f"Benchmark case {case_id} duplicates a production few-shot instruction.")
    for demonstration in few_shot_examples():
        similarity = _instruction_similarity(instruction, demonstration)
        if similarity >= _MAX_FEW_SHOT_SIMILARITY:
            raise ValueError(
                f"Benchmark case {case_id} is too similar to a production few-shot instruction "
                f"({similarity:.3f} >= {_MAX_FEW_SHOT_SIMILARITY:.2f})."
            )
    seen.add(case_id)
    if source.strip():
        validate_model(parse_model_text(source))

    expected = case["expected"]
    if not isinstance(expected, Mapping):
        raise ValueError(f"Benchmark case {case_id} expected must be an object.")
    expected_allowed = {
        "terminal_status",
        "first_action",
        "model_source",
        "clarification_keywords",
        "clarification_followup",
    }
    extra_expected = set(expected) - expected_allowed
    if extra_expected:
        raise ValueError(f"Benchmark case {case_id} expected has unknown fields: {sorted(extra_expected)}")
    terminal = expected.get("terminal_status")
    if terminal not in {"proposal", "needs_clarification", "unable"}:
        raise ValueError(f"Benchmark case {case_id} has unsupported terminal_status.")
    first_action = expected.get("first_action")
    if first_action not in {"request_context", "propose_edits", "needs_clarification", "unable"}:
        raise ValueError(f"Benchmark case {case_id} requires an explicit supported first_action.")
    if terminal == "proposal" and first_action not in {"request_context", "propose_edits"}:
        raise ValueError(f"Proposal benchmark case {case_id} has impossible first_action.")
    if terminal != "proposal" and first_action != terminal:
        raise ValueError(f"Terminal {terminal} benchmark case {case_id} must use matching first_action.")

    expected_source = expected.get("model_source")
    keywords = expected.get("clarification_keywords", [])
    followup = expected.get("clarification_followup")
    if terminal == "proposal":
        if not isinstance(expected_source, str) or not expected_source.strip():
            raise ValueError(f"Proposal benchmark case {case_id} requires expected model_source.")
        if keywords:
            raise ValueError(f"Proposal benchmark case {case_id} cannot declare clarification_keywords.")
        expected_model = validate_model(parse_model_text(expected_source))
        expected_ir = canonical_model_ir_sha256(expected_model)
        expected_semantics = scientific_semantics_sha256(expected_model)
        stored_source = expected_source
    else:
        if expected_source is not None:
            raise ValueError(f"Non-proposal benchmark case {case_id} must not include model_source.")
        expected_ir = expected_semantics = stored_source = None
    if terminal == "needs_clarification":
        if not isinstance(keywords, list) or not keywords or not all(isinstance(x, str) and x.strip() for x in keywords):
            raise ValueError(f"Clarification benchmark case {case_id} requires non-empty clarification_keywords.")
        keywords = [x.strip().casefold() for x in keywords]
        if not isinstance(followup, Mapping) or set(followup) != {
            "answer", "terminal_status", "model_source"
        }:
            raise ValueError(
                f"Clarification benchmark case {case_id} requires one exact held-out continuation."
            )
        answer = followup.get("answer")
        followup_terminal = followup.get("terminal_status")
        followup_source = followup.get("model_source")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Clarification benchmark case {case_id} requires a non-empty held-out answer.")
        if followup_terminal != "proposal":
            raise ValueError(
                f"Clarification benchmark case {case_id} continuation must terminate in a proposal."
            )
        if not isinstance(followup_source, str) or not followup_source.strip():
            raise ValueError(
                f"Clarification benchmark case {case_id} continuation requires expected model_source."
            )
        followup_model = validate_model(parse_model_text(followup_source))
        validated_followup = {
            "answer": answer.strip(),
            "terminal_status": "proposal",
            "model_source": followup_source,
            "canonical_model_ir_sha256": canonical_model_ir_sha256(followup_model),
            "scientific_semantics_sha256": scientific_semantics_sha256(followup_model),
        }
    elif keywords:
        raise ValueError(f"Benchmark case {case_id} cannot declare clarification_keywords.")
    elif followup is not None:
        raise ValueError(f"Only clarification benchmark cases may declare clarification_followup.")
    else:
        validated_followup = None

    return {
        "id": case_id,
        "family": family,
        "instruction": instruction,
        "current_model_source": source,
        "expected": {
            "terminal_status": terminal,
            "first_action": first_action,
            "model_source": stored_source,
            "canonical_model_ir_sha256": expected_ir,
            "scientific_semantics_sha256": expected_semantics,
            "clarification_keywords": keywords,
            "clarification_followup": validated_followup,
        },
        "notes": str(case.get("notes", "")),
    }


def load_benchmark(path: str | Path) -> dict[str, Any]:
    benchmark_path = Path(path).resolve()
    stat = benchmark_path.stat()
    # Benchmark validation deliberately runs the real model parser/validator and can be
    # expensive for large held-out sets. Cache immutable-by-file-identity validation
    # results, but return a deep copy so callers cannot mutate the cached campaign.
    return copy.deepcopy(_load_benchmark_cached(str(benchmark_path), stat.st_mtime_ns, stat.st_size))


@lru_cache(maxsize=16)
def _load_benchmark_cached(path: str, mtime_ns: int, size: int) -> dict[str, Any]:
    del mtime_ns, size  # They are part of the cache key; content is read below.
    benchmark_path = Path(path)
    raw = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Benchmark must be a JSON object.")
    allowed = {"schema", "schema_version", "title", "review", "training_exclusion", "cases"}
    extra = set(raw) - allowed
    if extra:
        raise ValueError(f"Benchmark has unknown top-level fields: {sorted(extra)}")
    if raw.get("schema") != BENCHMARK_SCHEMA or raw.get("schema_version") not in {BENCHMARK_SCHEMA_VERSION, *PREVIOUS_BENCHMARK_SCHEMA_VERSIONS}:
        raise ValueError("Unsupported interpreter benchmark schema.")
    if raw.get("training_exclusion") is not True:
        raise ValueError("Baseline benchmark must explicitly be excluded from training data.")
    review = raw.get("review")
    if not isinstance(review, Mapping) or set(review) != {"status", "provenance"}:
        raise ValueError("Benchmark review metadata must contain exactly status and provenance.")
    if review.get("status") not in {"internal-curated", "independent-expert-reviewed"}:
        raise ValueError("Benchmark review status is unsupported.")
    if not isinstance(review.get("provenance"), str) or not review["provenance"].strip():
        raise ValueError("Benchmark review provenance must be explicit.")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Interpreter benchmark must contain at least one case.")
    few_shot = {_normalized_instruction(item) for item in few_shot_examples()}
    seen: set[str] = set()
    validated = [_validate_benchmark_case(case, seen, few_shot) for case in cases]
    family_counts: dict[str, int] = {}
    for case in validated:
        family_counts[case["family"]] = family_counts.get(case["family"], 0) + 1
    return {
        "schema": BENCHMARK_SCHEMA,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "title": str(raw.get("title", "Model Laboratory interpreter baseline")),
        "review": dict(review),
        "training_exclusion": True,
        "cases": validated,
        "family_counts": dict(sorted(family_counts.items())),
        "benchmark_sha256": _sha256_bytes(benchmark_path.read_bytes()),
    }


# ---------- frozen local runtime identity ----------
@dataclass(frozen=True)
class OllamaIdentity:
    runtime_version: str
    model_digest: str
    model_size_bytes: int
    local_manifest_sha256: str
    model_blob_digest: str
    verified_model_blob_size_bytes: int = 0
    model_context_length: int = 0

    def provider_identity(
        self, generation_options: Mapping[str, int | float] | None = None
    ) -> dict[str, Any]:
        options = dict(generation_options or GENERATION_OPTIONS)
        manifest_sha = _normalize_ollama_manifest_digest(self.local_manifest_sha256)[7:]
        model_blob_sha = _normalize_ollama_manifest_digest(self.model_blob_digest)[7:]
        return {
            "provider": "ollama",
            "endpoint": LOCAL_ENDPOINT,
            "runtime_version": self.runtime_version,
            "model_tag": LOCAL_MODEL,
            "observed_model_digest": self.model_digest,
            "identity_verification": "exact_local_manifest_and_all_blobs_sha256",
            "expected_base_model": LOCAL_BASE_MODEL,
            "expected_quantization": LOCAL_QUANTIZATION,
            "generation_options": options,
            "model_role": "frozen_base",
            "registry_entry_sha256": None,
            "artifact_evidence": {
                "identity_verification": "exact_local_manifest_and_all_blobs_sha256",
                "artifact_lock_sha256": _sha256_bytes(_MODEL_LOCK_PATH.read_bytes()),
                "local_manifest_sha256": manifest_sha,
                "model_blob_sha256": model_blob_sha,
                "verified_model_blob_size_bytes": self.verified_model_blob_size_bytes,
                "all_manifest_blobs_verified": True,
            },
        }

    def report_payload(self) -> dict[str, Any]:
        return {
            "runtime": "ollama",
            "runtime_version": self.runtime_version,
            "observed_model_digest": self.model_digest,
            "model_size_bytes": self.model_size_bytes,
            "local_manifest_sha256": self.local_manifest_sha256,
            "model_blob_digest": self.model_blob_digest,
            "verified_model_blob_size_bytes": self.verified_model_blob_size_bytes,
            "model_context_length": self.model_context_length,
            "identity_verification": "exact_local_manifest_and_all_blobs_sha256",
        }


def _candidate_ollama_manifests() -> list[tuple[Path, Path]]:
    roots: list[Path] = []
    configured = os.environ.get("OLLAMA_MODELS", "").strip()
    if configured:
        roots.append(Path(configured))
    roots.extend([Path.home() / ".ollama" / "models", Path("/usr/share/ollama/.ollama/models")])
    relative = Path("manifests/registry.ollama.ai/library/qwen3/4b-instruct-2507-q4_K_M")
    result: list[tuple[Path, Path]] = []
    for root in roots:
        candidate = root / relative
        pair = (root, candidate)
        if pair not in result:
            result.append(pair)
    return result


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _verify_blob_file(root: Path, digest: str) -> int:
    if not isinstance(digest, str) or not digest.startswith("sha256:") or _SHA256_HEX.fullmatch(digest[7:]) is None:
        raise BaselineInfrastructureError("The Ollama manifest contains a malformed blob digest.")
    path = root / "blobs" / ("sha256-" + digest[7:])
    if not path.is_file():
        raise BaselineInfrastructureError(f"Required local Ollama blob is missing: {digest[:19]}…")
    observed, size = _sha256_file(path)
    if observed != digest[7:]:
        raise BaselineInfrastructureError(f"Local Ollama blob failed SHA-256 verification: {digest[:19]}…")
    return size


def _normalize_ollama_manifest_digest(observed_digest: str) -> str:
    """Canonicalize Ollama API digests to the OCI-style ``sha256:<hex>`` form.

    Ollama's ``GET /api/tags`` response documents ``digest`` as a bare
    64-character hexadecimal SHA-256 value, while local manifests and layer
    identities use the ``sha256:`` prefix.  Accept exactly those two spellings
    and normalize before any identity comparison.
    """
    if not isinstance(observed_digest, str):
        raise BaselineInfrastructureError("Ollama returned a malformed model digest.")
    value = observed_digest.strip().lower()
    if value.startswith("sha256:"):
        hex_digest = value[7:]
    else:
        hex_digest = value
    if _SHA256_HEX.fullmatch(hex_digest) is None:
        raise BaselineInfrastructureError("Ollama returned a malformed model digest.")
    return "sha256:" + hex_digest


def _verify_local_ollama_manifest(observed_digest: str) -> tuple[str, str, int]:
    lock = base_model_lock()["ollama"]
    observed_manifest_digest = _normalize_ollama_manifest_digest(observed_digest)
    if observed_manifest_digest != "sha256:" + lock["registry_manifest_sha256"]:
        raise BaselineInfrastructureError(
            "Installed Ollama tag digest does not match the frozen published manifest identity."
        )
    located = next(((root, path) for root, path in _candidate_ollama_manifests() if path.is_file()), None)
    if located is None:
        raise BaselineInfrastructureError(
            "Cannot verify the frozen model: the local Ollama manifest file was not found. "
            "Set OLLAMA_MODELS if Ollama uses a non-default model directory."
        )
    root, path = located
    raw = path.read_bytes()
    manifest_sha = "sha256:" + _sha256_bytes(raw)
    if manifest_sha != observed_manifest_digest:
        raise BaselineInfrastructureError(
            "Ollama /api/tags digest does not match the SHA-256 of the local manifest bytes."
        )
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BaselineInfrastructureError("The local Ollama manifest is malformed JSON.") from exc
    if not isinstance(manifest, Mapping):
        raise BaselineInfrastructureError("The local Ollama manifest must be an object.")
    layers = manifest.get("layers")
    if not isinstance(layers, list):
        raise BaselineInfrastructureError("The local Ollama manifest has no layer list.")

    by_type: dict[str, tuple[str, int]] = {}
    for layer in layers:
        if not isinstance(layer, Mapping):
            raise BaselineInfrastructureError("The local Ollama manifest contains a malformed layer.")
        media_type, digest, size = layer.get("mediaType"), layer.get("digest"), layer.get("size")
        if (
            not isinstance(media_type, str)
            or not isinstance(digest, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise BaselineInfrastructureError("The local Ollama manifest contains a malformed layer identity.")
        if media_type in by_type:
            raise BaselineInfrastructureError(f"The local Ollama manifest duplicates layer type {media_type}.")
        by_type[media_type] = (digest, size)

    expected_layers = {item["media_type"]: item for item in lock["layers"]}
    if set(by_type) != set(expected_layers):
        raise BaselineInfrastructureError(
            "Installed Ollama manifest has missing or extra layers (including possible adapters/system layers)."
        )
    model_digest = by_type["application/vnd.ollama.image.model"][0]
    expected_model_digest = "sha256:" + expected_layers[
        "application/vnd.ollama.image.model"
    ]["sha256"]
    if model_digest != expected_model_digest:
        raise BaselineInfrastructureError("Installed Ollama GGUF model blob is not the frozen baseline artifact.")
    for media_type, expected in expected_layers.items():
        digest, size = by_type[media_type]
        if digest != "sha256:" + expected["sha256"] or size != expected["size_bytes"]:
            raise BaselineInfrastructureError(f"Installed Ollama {media_type} layer does not match the frozen baseline.")

    # Verify every local layer byte-for-byte against the full digest in the manifest.  The
    # model layer is additionally matched against the full independently frozen GGUF SHA-256.
    verified_model_size = 0
    for media_type, (digest, declared_size) in by_type.items():
        size = _verify_blob_file(root, digest)
        if size != declared_size:
            raise BaselineInfrastructureError(
                f"Installed Ollama {media_type} blob size differs from its manifest."
            )
        if media_type == "application/vnd.ollama.image.model":
            verified_model_size = size
    config = manifest.get("config")
    expected_config = lock["config"]
    if (
        not isinstance(config, Mapping)
        or config.get("mediaType") != expected_config["media_type"]
        or config.get("digest") != "sha256:" + expected_config["sha256"]
        or config.get("size") != expected_config["size_bytes"]
    ):
        raise BaselineInfrastructureError("Installed Ollama config blob differs from the frozen lock.")
    if _verify_blob_file(root, config["digest"]) != expected_config["size_bytes"]:
        raise BaselineInfrastructureError("Installed Ollama config blob size differs from its manifest.")
    return manifest_sha, model_digest, verified_model_size


class OllamaBaselineClient:
    """Loopback-only client that refuses to score an unverified model artifact."""

    def __init__(self, endpoint: str = LOCAL_ENDPOINT, *, timeout_seconds: int = 1200) -> None:
        if endpoint != LOCAL_ENDPOINT:
            raise ValueError("Baseline evaluation is fixed to the application's loopback endpoint.")
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self._opener = urlrequest.build_opener(urlrequest.ProxyHandler({}))

    def _json(self, method: str, path: str, payload: object | None = None, *, timeout: int | None = None) -> Any:
        body = None if payload is None else _canonical_bytes(payload)
        req = urlrequest.Request(self.endpoint + path, data=body, method=method, headers={"Content-Type": "application/json"} if body is not None else {})
        try:
            with self._opener.open(req, timeout=timeout or self.timeout_seconds) as response:
                data = response.read(_MAX_RESPONSE_BYTES + 1)
        except (urlerror.URLError, TimeoutError, OSError) as exc:
            raise BaselineInfrastructureError(f"Ollama request failed: {exc}") from exc
        if len(data) > _MAX_RESPONSE_BYTES:
            raise BaselineInfrastructureError("Ollama response exceeded the baseline safety limit.")
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            raise BaselineInfrastructureError("Ollama service returned malformed JSON.") from exc

    def identity(self) -> OllamaIdentity:
        version = self._json("GET", "/api/version", timeout=5)
        tags = self._json("GET", "/api/tags", timeout=5)
        runtime_version = version.get("version") if isinstance(version, Mapping) else None
        models = tags.get("models") if isinstance(tags, Mapping) else None
        if not isinstance(runtime_version, str) or not isinstance(models, list):
            raise BaselineInfrastructureError("Ollama returned an invalid identity response.")
        match = next((m for m in models if isinstance(m, Mapping) and (m.get("name") == LOCAL_MODEL or m.get("model") == LOCAL_MODEL)), None)
        if match is None:
            raise BaselineInfrastructureError(f"Configured baseline model is not installed: {LOCAL_MODEL}")
        digest = match.get("digest")
        size = match.get("size", 0)
        if not isinstance(digest, str) or not digest:
            raise BaselineInfrastructureError("Ollama returned no model digest.")
        if isinstance(size, bool) or not isinstance(size, int):
            size = 0
        manifest_sha, model_blob_digest, verified_model_size = _verify_local_ollama_manifest(digest)
        show = self._json("POST", "/api/show", {"model": LOCAL_MODEL}, timeout=10)
        details = show.get("details") if isinstance(show, Mapping) else None
        model_info = show.get("model_info") if isinstance(show, Mapping) else None
        if not isinstance(details, Mapping) or details.get("quantization_level") != LOCAL_QUANTIZATION:
            raise BaselineInfrastructureError("Ollama model details do not confirm the frozen quantization.")
        if not isinstance(model_info, Mapping) or model_info.get("general.architecture") != "qwen3":
            raise BaselineInfrastructureError("Ollama model metadata does not confirm Qwen3 architecture.")
        context_length = model_info.get("qwen3.context_length", 0)
        if isinstance(context_length, bool) or not isinstance(context_length, int) or context_length < int(EVALUATION_GENERATION_OPTIONS["num_ctx"]):
            raise BaselineInfrastructureError("Installed model does not support the frozen evaluation context length.")
        return OllamaIdentity(
            runtime_version, digest, size, manifest_sha, model_blob_digest,
            verified_model_size, context_length
        )

    def infer(self, *, instruction: str, context: Mapping[str, Any]) -> InferenceResult:
        payload = {
            "model": LOCAL_MODEL,
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
            raise BaselineInfrastructureError("Ollama chat response must be an object.")
        message = response.get("message")
        if response.get("done") is not True or response.get("model") != LOCAL_MODEL or not isinstance(message, Mapping):
            raise BaselineInfrastructureError("Ollama did not complete the request with the frozen model tag.")
        content = message.get("content")
        if not isinstance(content, str):
            raise BaselineInfrastructureError("Ollama returned no interpreter content.")
        telemetry = {
            "elapsed_seconds": elapsed,
            "done_reason": response.get("done_reason", ""),
            "total_duration_nanoseconds": response.get("total_duration", 0),
            "load_duration_nanoseconds": response.get("load_duration", 0),
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "generated_tokens": response.get("eval_count", 0),
        }
        try:
            parsed = json.loads(content)
            return InferenceResult(content, parsed, telemetry)
        except json.JSONDecodeError as exc:
            return InferenceResult(content, None, telemetry, f"JSONDecodeError: {exc}")


# ---------- scoring ----------
def _coerce_inference_result(value: object) -> InferenceResult:
    """Compatibility adapter for small deterministic test clients."""
    if isinstance(value, InferenceResult):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        raw, telemetry = value
        if not isinstance(telemetry, Mapping):
            raise TypeError("Test client telemetry must be a mapping.")
        text = json.dumps(raw, sort_keys=True, ensure_ascii=False) if raw is not None else ""
        return InferenceResult(text, raw, dict(telemetry))
    raise TypeError("Baseline client infer() must return InferenceResult or (raw, telemetry).")


def _score_case(case: Mapping[str, Any], client: object, identity: OllamaIdentity) -> dict[str, Any]:
    """Score one held-out case, including clarification continuation when required.

    Clarification is deliberately a two-stage contract.  The first stage scores whether the
    model asks the right question.  A held-out answer is then supplied through the normal
    clarification-history channel and the second stage must compile the intended proposal.
    This prevents a model from passing merely by learning to ask for clarification.
    """
    instruction = case["instruction"]
    source = case["current_model_source"]
    expected = case["expected"]
    actions: list[str] = []
    followup_actions: list[str] = []
    round_records: list[dict[str, Any]] = []
    raw_schema_valid = True
    terminal_status = "model_failure"
    proposed_ir: str | None = None
    proposed_semantics: str | None = None
    semantic_equivalent = False
    semantic_differences: list[str] = []
    clarification_question = ""
    clarification_question_correct = False
    clarification_continuation_expected = expected["terminal_status"] == "needs_clarification"
    clarification_continuation_correct = False
    clarification_followup_terminal_status: str | None = None
    clarification_followup_ir: str | None = None
    clarification_followup_semantics: str | None = None
    clarification_followup_differences: list[str] = []
    error_message = ""
    conversation_id = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"model-laboratory-benchmark:{case['id']}")
    )
    conversation_lineage: list[dict[str, Any]] = []

    def run_phase(
        *,
        clarification_history: Sequence[Mapping[str, str]],
        phase: str,
    ) -> tuple[str, str | None, str | None, str, str | None, str]:
        nonlocal raw_schema_valid
        context = create_interpreter_context(
            instruction=instruction,
            current_model_source=source,
            clarification_history=clarification_history,
        )
        phase_actions = actions if phase == "initial" else followup_actions
        phase_ir: str | None = None
        phase_semantics: str | None = None
        phase_question = ""
        phase_source: str | None = None
        phase_error = ""
        phase_status = "model_failure"

        for context_round in range(MAX_CONTEXT_ROUNDS):
            context_snapshot = json.loads(json.dumps(context))
            try:
                inference = _coerce_inference_result(
                    client.infer(instruction=instruction, context=context)  # type: ignore[attr-defined]
                )
            except BaselineInfrastructureError:
                raise
            except (urlerror.URLError, TimeoutError, OSError) as exc:
                raise BaselineInfrastructureError(f"Inference transport failed: {exc}") from exc

            record: dict[str, Any] = {
                "round": len(round_records),
                "phase": phase,
                "context_round": context_round,
                "context_sha256": context.get("context_sha256"),
                "context_document": context_snapshot,
                "user_prompt_sha256": _sha256_text(user_prompt(instruction, context)),
                "raw_content": inference.raw_content,
                "raw_content_sha256": _sha256_text(inference.raw_content),
                "parsed_output": inference.parsed_output,
                "validated_output": None,
                "telemetry": inference.telemetry,
                "model_output_error": inference.parse_error,
            }
            round_records.append(record)
            if inference.parsed_output is None:
                raw_schema_valid = False
                phase_error = inference.parse_error or "Model output was not JSON."
                break
            try:
                validated_raw = validate_raw_interpreter_output(inference.parsed_output)
                record["validated_output"] = validated_raw
            except InterpreterProposalError as exc:
                raw_schema_valid = False
                phase_error = f"{type(exc).__name__}: {exc}"
                break
            phase_actions.append(validated_raw["action"])
            try:
                processed = process_interpreter_output(
                    raw_output=validated_raw,
                    context_document=context,
                    provider_identity=identity.provider_identity(EVALUATION_GENERATION_OPTIONS),
                    instruction=instruction,
                    current_model_source=source,
                    clarification_history=clarification_history,
                    conversation_id=conversation_id,
                    conversation_lineage=conversation_lineage,
                    expected_generation_options=EVALUATION_GENERATION_OPTIONS,
                )
            except InterpreterProposalError as exc:
                phase_error = f"{type(exc).__name__}: {exc}"
                break
            status = processed["status"]
            if status == "context_required":
                context = processed["context"]
                continue
            if status == "proposal":
                phase_status = "proposal"
                phase_ir = processed["proposal"]["proposed_model"]["canonical_model_ir_sha256"]
                phase_source = processed["proposal"]["proposed_model"]["source"]
                proposed_model = validate_model(parse_model_text(phase_source))
                phase_semantics = scientific_semantics_sha256(proposed_model)
            elif status == "needs_clarification":
                phase_status = "needs_clarification"
                phase_question = processed["question"]
                conversation_lineage.append(dict(processed["turn_evidence"]))
            elif status == "unable":
                phase_status = "unable"
            else:
                phase_error = f"Unexpected interpreter status: {status}"
            break
        else:
            phase_error = "Interpreter exhausted the maximum context rounds."
        return phase_status, phase_ir, phase_semantics, phase_question, phase_source, phase_error

    terminal_status, proposed_ir, proposed_semantics, clarification_question, proposed_source, error_message = run_phase(
        clarification_history=(), phase="initial"
    )

    expected_first = expected["first_action"]
    first_action_correct = bool(actions) and actions[0] == expected_first
    terminal_correct = terminal_status == expected["terminal_status"]

    if expected["terminal_status"] == "proposal" and terminal_status == "proposal":
        expected_model = validate_model(parse_model_text(expected["model_source"]))
        if isinstance(proposed_source, str):
            proposed_model = validate_model(parse_model_text(proposed_source))
            semantic_equivalent, semantic_differences = compare_model_semantics(expected_model, proposed_model)
    elif expected["terminal_status"] == "unable":
        semantic_equivalent = terminal_status == "unable"
    elif expected["terminal_status"] == "needs_clarification":
        question_fold = " ".join(re.findall(r"[a-z0-9]+", clarification_question.casefold()))
        clarification_question_correct = terminal_status == "needs_clarification" and all(
            " ".join(re.findall(r"[a-z0-9]+", keyword.casefold())) in question_fold
            for keyword in expected["clarification_keywords"]
        )
        followup = expected.get("clarification_followup")
        if terminal_status == "needs_clarification" and isinstance(followup, Mapping):
            history = [{"question": clarification_question, "answer": followup["answer"]}]
            (
                clarification_followup_terminal_status,
                clarification_followup_ir,
                clarification_followup_semantics,
                _followup_question,
                followup_source,
                followup_error,
            ) = run_phase(clarification_history=history, phase="clarification_followup")
            if followup_error and not error_message:
                error_message = followup_error
            if clarification_followup_terminal_status == followup["terminal_status"] == "proposal":
                expected_followup_model = validate_model(parse_model_text(followup["model_source"]))
                if isinstance(followup_source, str):
                    observed_followup_model = validate_model(parse_model_text(followup_source))
                    clarification_continuation_correct, clarification_followup_differences = compare_model_semantics(
                        expected_followup_model, observed_followup_model
                    )
        semantic_equivalent = clarification_question_correct and clarification_continuation_correct

    canonical_exact = bool(
        terminal_status == "proposal" and proposed_ir == expected["canonical_model_ir_sha256"]
    )
    expected_followup = expected.get("clarification_followup")

    return {
        "id": case["id"],
        "family": case["family"],
        "raw_schema_valid": raw_schema_valid,
        "actions": actions,
        "first_action_correct": first_action_correct,
        "terminal_status": terminal_status,
        "terminal_status_correct": terminal_correct,
        "semantic_equivalent": semantic_equivalent,
        "semantic_differences": semantic_differences,
        "canonical_ir_exact": canonical_exact,
        "clarification_question": clarification_question,
        "clarification_question_correct": clarification_question_correct,
        "clarification_continuation_expected": clarification_continuation_expected,
        "clarification_continuation_correct": clarification_continuation_correct,
        "clarification_followup_actions": followup_actions,
        "clarification_followup_terminal_status": clarification_followup_terminal_status,
        "clarification_followup_semantic_differences": clarification_followup_differences,
        "expected_clarification_followup_canonical_model_ir_sha256": (
            expected_followup.get("canonical_model_ir_sha256") if isinstance(expected_followup, Mapping) else None
        ),
        "observed_clarification_followup_canonical_model_ir_sha256": clarification_followup_ir,
        "expected_clarification_followup_scientific_semantics_sha256": (
            expected_followup.get("scientific_semantics_sha256") if isinstance(expected_followup, Mapping) else None
        ),
        "observed_clarification_followup_scientific_semantics_sha256": clarification_followup_semantics,
        "expected_terminal_status": expected["terminal_status"],
        "expected_first_action": expected_first,
        "expected_canonical_model_ir_sha256": expected["canonical_model_ir_sha256"],
        "observed_canonical_model_ir_sha256": proposed_ir,
        "expected_scientific_semantics_sha256": expected["scientific_semantics_sha256"],
        "observed_scientific_semantics_sha256": proposed_semantics,
        "rounds": round_records,
        "error": error_message,
    }

def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _source_identity() -> dict[str, Any]:
    try:
        tree = source_tree_sha256()
    except Exception as exc:
        tree = f"unavailable:{type(exc).__name__}"
    package = Path(__file__).parent
    scientific_dependencies = {
        name: _sha256_bytes((package / name).read_bytes())
        for name in (
            "interpreter.py",
            "validator.py",
            "parser.py",
            "model_graph.py",
            "model.py",
            "canonical.py",
            "expression.py",
        )
    }
    return {
        "laboratory_version": __version__,
        "source_tree_sha256": tree,
        "git_commit_or_build_id": git_commit_or_build_id(),
        "evaluator_source_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "interpreter_compiler_source_sha256": scientific_dependencies["interpreter.py"],
        "scientific_dependency_sha256": scientific_dependencies,
        "scorer_version": SCORER_VERSION,
    }


def _build_report(
    benchmark: Mapping[str, Any],
    *,
    identity: OllamaIdentity,
    results: Sequence[Mapping[str, Any]],
    limit: int | None,
) -> dict[str, Any]:
    """Build a deterministic report from already-scored cases."""
    results = [dict(item) for item in results]
    total = len(results)
    proposal_expected = sum(item["expected_terminal_status"] == "proposal" for item in results)
    unable_expected = sum(item["expected_terminal_status"] == "unable" for item in results)
    clarification_expected = sum(
        item["expected_terminal_status"] == "needs_clarification" for item in results
    )

    def count(field: str) -> int:
        return sum(bool(item[field]) for item in results)

    family_metrics: dict[str, dict[str, Any]] = {}
    family_counts: dict[str, int] = {}
    for family in sorted({item["family"] for item in results}):
        members = [item for item in results if item["family"] == family]
        family_counts[family] = len(members)
        passed = sum(bool(item["semantic_equivalent"]) for item in members)
        family_metrics[family] = {
            "cases": len(members),
            "semantic_equivalent": passed,
            "semantic_equivalent_rate": _ratio(passed, len(members)),
        }

    unable_correct = sum(
        item["expected_terminal_status"] == "unable" and item["terminal_status"] == "unable"
        for item in results
    )
    clarification_question_correct = sum(
        item["expected_terminal_status"] == "needs_clarification"
        and item["clarification_question_correct"]
        for item in results
    )
    clarification_continuation_correct = sum(
        item["expected_terminal_status"] == "needs_clarification"
        and item["clarification_continuation_correct"]
        for item in results
    )
    clarification_correct = sum(
        item["expected_terminal_status"] == "needs_clarification" and item["semantic_equivalent"]
        for item in results
    )
    context_negotiation_members = [
        item for item in results if item["family"] == "context-negotiation"
    ]
    context_negotiation_first_action_correct = sum(
        bool(item["first_action_correct"]) for item in context_negotiation_members
    )
    proposal_compiled = sum(
        item["expected_terminal_status"] == "proposal" and item["terminal_status"] == "proposal"
        for item in results
    )
    canonical_ok = sum(
        item["expected_terminal_status"] == "proposal" and item["canonical_ir_exact"]
        for item in results
    )

    report = {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "campaign_status": "valid",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline_contract": baseline_contract(),
        "implementation_identity": _source_identity(),
        "runtime_identity": identity.report_payload(),
        "benchmark": {
            "schema": benchmark["schema"],
            "schema_version": benchmark["schema_version"],
            "benchmark_sha256": benchmark["benchmark_sha256"],
            "review": benchmark["review"],
            "training_exclusion": True,
            "case_count": total,
            "family_counts": family_counts,
            "limited_run": limit is not None,
        },
        "metrics": {
            "raw_schema_compliance": {
                "passed": count("raw_schema_valid"),
                "total": total,
                "rate": _ratio(count("raw_schema_valid"), total),
            },
            "first_action_accuracy": {
                "passed": count("first_action_correct"),
                "total": total,
                "rate": _ratio(count("first_action_correct"), total),
            },
            "terminal_status_accuracy": {
                "passed": count("terminal_status_correct"),
                "total": total,
                "rate": _ratio(count("terminal_status_correct"), total),
            },
            "scientific_semantic_accuracy": {
                "passed": count("semantic_equivalent"),
                "total": total,
                "rate": _ratio(count("semantic_equivalent"), total),
            },
            "canonical_ir_exact_for_proposals": {
                "passed": canonical_ok,
                "total": proposal_expected,
                "rate": _ratio(canonical_ok, proposal_expected),
            },
            "expected_proposal_compiler_acceptance": {
                "passed": proposal_compiled,
                "total": proposal_expected,
                "rate": _ratio(proposal_compiled, proposal_expected),
            },
            "unsupported_accuracy": {
                "passed": unable_correct,
                "total": unable_expected,
                "rate": _ratio(unable_correct, unable_expected),
            },
            "clarification_question_accuracy": {
                "passed": clarification_question_correct,
                "total": clarification_expected,
                "rate": _ratio(clarification_question_correct, clarification_expected),
            },
            "clarification_continuation_accuracy": {
                "passed": clarification_continuation_correct,
                "total": clarification_expected,
                "rate": _ratio(clarification_continuation_correct, clarification_expected),
            },
            "clarification_accuracy": {
                "passed": clarification_correct,
                "total": clarification_expected,
                "rate": _ratio(clarification_correct, clarification_expected),
            },
            "context_negotiation_first_action_accuracy": {
                "passed": context_negotiation_first_action_correct,
                "total": len(context_negotiation_members),
                "rate": _ratio(
                    context_negotiation_first_action_correct, len(context_negotiation_members)
                ),
            },
            "by_family": family_metrics,
        },
        "cases": results,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def _checkpoint_contract_sha256() -> str:
    return canonical_json_sha256(baseline_contract())


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temp, path)
    except OSError as exc:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise BaselineInfrastructureError(f"Could not write baseline checkpoint: {exc}") from exc


def _checkpoint_payload(
    benchmark: Mapping[str, Any],
    *,
    identity: OllamaIdentity,
    cases: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "benchmark_sha256": benchmark["benchmark_sha256"],
        "baseline_contract_sha256": _checkpoint_contract_sha256(),
        "implementation_identity": _source_identity(),
        "runtime_identity": identity.report_payload(),
        "selected_case_ids": [case["id"] for case in cases],
        "completed_case_ids": [item["id"] for item in results],
        "completed_count": len(results),
        "results": [dict(item) for item in results],
    }


def _load_checkpoint_results(
    path: Path,
    benchmark: Mapping[str, Any],
    *,
    identity: OllamaIdentity,
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineInfrastructureError(
            f"Existing baseline checkpoint cannot be read safely: {exc}. "
            "Use --restart to discard it."
        ) from exc
    if not isinstance(value, Mapping):
        raise BaselineInfrastructureError("Existing baseline checkpoint is malformed; use --restart.")
    if value.get("schema") != CHECKPOINT_SCHEMA or value.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise BaselineInfrastructureError("Existing baseline checkpoint has an incompatible schema; use --restart.")
    if value.get("benchmark_sha256") != benchmark["benchmark_sha256"]:
        raise BaselineInfrastructureError("Existing checkpoint belongs to a different benchmark; use --restart.")
    if value.get("baseline_contract_sha256") != _checkpoint_contract_sha256():
        raise BaselineInfrastructureError("Existing checkpoint belongs to a different baseline contract; use --restart.")
    if value.get("implementation_identity") != _source_identity():
        raise BaselineInfrastructureError("Existing checkpoint was produced by different evaluator/compiler code; use --restart.")
    if value.get("runtime_identity") != identity.report_payload():
        raise BaselineInfrastructureError("Existing checkpoint was produced by a different frozen model/runtime identity; use --restart.")
    results = value.get("results")
    if not isinstance(results, list) or not all(isinstance(item, Mapping) for item in results):
        raise BaselineInfrastructureError("Existing checkpoint has malformed case results; use --restart.")
    completed_ids = [str(item.get("id", "")) for item in results]
    selected_ids = [str(case["id"]) for case in cases]
    if completed_ids != selected_ids[: len(completed_ids)]:
        raise BaselineInfrastructureError("Existing checkpoint is not a valid prefix of this campaign; use --restart.")
    if len(completed_ids) > len(selected_ids):
        raise BaselineInfrastructureError("Existing checkpoint contains more cases than this campaign; use --restart.")
    return [dict(item) for item in results]


def run_baseline(
    benchmark: Mapping[str, Any],
    *,
    client: object | None = None,
    limit: int | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = True,
    progress: Callable[[int, int, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the frozen local Qwen artifact with fail-closed identity and resumable case scoring.

    Model-output/schema/compiler failures are ordinary scored model failures. Runtime identity,
    transport, and checkpoint-integrity failures invalidate the campaign. When ``checkpoint_path``
    is supplied, every completed case is atomically persisted and a later identical run resumes
    from the exact completed prefix.
    """
    client = client or OllamaBaselineClient()
    try:
        identity = client.identity()  # type: ignore[attr-defined]
    except BaselineInfrastructureError:
        raise
    except Exception as exc:
        raise BaselineInfrastructureError(f"Runtime identity verification failed: {exc}") from exc
    if not isinstance(identity, OllamaIdentity):
        raise BaselineInfrastructureError("Baseline client returned an invalid runtime identity record.")

    cases = list(benchmark["cases"])
    if limit is not None:
        if limit <= 0:
            raise ValueError("Baseline limit must be positive.")
        cases = cases[:limit]

    results: list[dict[str, Any]] = []
    if checkpoint_path is not None and resume and checkpoint_path.exists():
        results = _load_checkpoint_results(
            checkpoint_path, benchmark, identity=identity, cases=cases
        )

    for case in cases[len(results) :]:
        result = _score_case(case, client, identity)
        results.append(result)
        if checkpoint_path is not None:
            _atomic_write_json(
                checkpoint_path,
                _checkpoint_payload(
                    benchmark, identity=identity, cases=cases, results=results
                ),
            )
        if progress is not None:
            progress(len(results), len(cases), result)

    return _build_report(benchmark, identity=identity, results=results, limit=limit)


def validation_report(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all fixed assets without fabricating an empirical model score."""
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "benchmark_validated_model_not_run",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline_contract": baseline_contract(),
        "implementation_identity": _source_identity(),
        "benchmark": {
            "benchmark_sha256": benchmark["benchmark_sha256"],
            "review": benchmark["review"],
            "training_exclusion": True,
            "case_count": len(benchmark["cases"]),
            "family_counts": benchmark["family_counts"],
        },
        "reason": "A live score is intentionally absent until the exact frozen Ollama artifact passes local manifest/blob verification and inference is run.",
    }
