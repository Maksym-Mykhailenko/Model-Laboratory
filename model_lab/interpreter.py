"""Deterministic trust boundary and compiler for local interpreter output.

The language model never emits YAML and never mutates experiment state. A trusted local
context builder exposes a bounded view of the current validated model. Qwen may request
more exact paths or return a small typed edit program. This module validates that program,
applies it to the complete source document with deterministic rules, serialises the result,
and sends the ordinary Model Laboratory parser and mathematical validator the compiled YAML.

No function in this module executes an analysis or accepts a proposal on the user's behalf.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import difflib
import hashlib
from io import StringIO
import json
import math
import re
from collections.abc import MutableMapping
from typing import Any, Mapping, Sequence
import uuid

import yaml
from ruamel.yaml import YAML as RoundTripYAML

from .canonical import canonical_json_sha256, canonical_model_ir_sha256
from .expression import FUNCTION_ARITIES, NAMED_CONSTANTS
from .interpreter_attachments import (
    ATTACHMENT_CHUNK_BYTES,
    MAX_ACTIVE_ATTACHMENTS,
    InterpreterAttachmentError,
    attachment_chunks,
    attachment_manifest,
    validate_attachment_manifests,
    validate_attachment_record,
)
from .parser import ModelParseError, parse_model_data, parse_model_text
from .schema import (
    AmbiguitySpec,
    AssumptionSpec,
    ConstantSpec,
    ConstraintSpec,
    DerivedQuantitySpec,
    FunctionSpec,
    MatrixFunctionSpec,
    ModelMetadataSpec,
    ModelObjectSpec,
    ModelRelationshipSpec,
    ModelSpec,
    ParameterSpec,
    StructuredAssetSpec,
    VariableSpec,
    VectorFunctionSpec,
)
from .validator import ModelValidationError, validate_model


INTERPRETER_CONTEXT_SCHEMA = "model-laboratory-interpreter-context"
INTERPRETER_CONTEXT_SCHEMA_VERSION = "1.3"
PREVIOUS_INTERPRETER_CONTEXT_SCHEMA_VERSION = "1.2"
OLDER_INTERPRETER_CONTEXT_SCHEMA_VERSION = "1.1"
LEGACY_INTERPRETER_CONTEXT_SCHEMA_VERSION = "1.0"
INTERPRETER_OUTPUT_SCHEMA = "model-laboratory-interpreter-output"
INTERPRETER_OUTPUT_SCHEMA_VERSION = "1.2"
PREVIOUS_INTERPRETER_OUTPUT_SCHEMA_VERSION = "1.1"
LEGACY_INTERPRETER_OUTPUT_SCHEMA_VERSION = "1.0"
INTERPRETER_COMPILER_SCHEMA = "model-laboratory-interpreter-edit-compiler"
INTERPRETER_COMPILER_SCHEMA_VERSION = "1.4"
PREVIOUS_INTERPRETER_COMPILER_SCHEMA_VERSION = "1.3"
LEGACY_INTERPRETER_COMPILER_SCHEMA_VERSION = "1.2"
OLDER_INTERPRETER_COMPILER_SCHEMA_VERSION = "1.1"
INITIAL_INTERPRETER_COMPILER_SCHEMA_VERSION = "1.0"
INTERPRETER_PROPOSAL_SCHEMA = "model-laboratory-interpreter-proposal"
INTERPRETER_PROPOSAL_SCHEMA_VERSION = "2.4"
PREVIOUS_INTERPRETER_PROPOSAL_SCHEMA_VERSION = "2.3"
LEGACY_INTERPRETER_PROPOSAL_SCHEMA_VERSION = "2.2"
OLDER_INTERPRETER_PROPOSAL_SCHEMA_VERSION = "2.1"
INTERPRETER_ACCEPTANCE_SCHEMA = "model-laboratory-interpreter-acceptance"
INTERPRETER_ACCEPTANCE_SCHEMA_VERSION = "1.5"
PREVIOUS_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION = "1.4"
LEGACY_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION = "1.3"
OLDER_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION = "1.2"
EARLIER_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION = "1.1"
INITIAL_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION = "1.0"
INTERPRETER_DIALOGUE_TURN_SCHEMA = "model-laboratory-interpreter-dialogue-turn"
INTERPRETER_DIALOGUE_TURN_SCHEMA_VERSION = "1.0"

LOCAL_PROVIDER = "ollama"
LOCAL_ENDPOINT = "http://127.0.0.1:11434"
LOCAL_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
LOCAL_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LOCAL_QUANTIZATION = "Q4_K_M"
FROZEN_BASE_ARTIFACT_LOCK_SHA256 = "1b73f807ea8c2a53088ae5e38b9b376f3cd9acd8213bf7c881af728018c7f38e"
FROZEN_BASE_MANIFEST_SHA256 = "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
FROZEN_BASE_MODEL_BLOB_SHA256 = "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
FROZEN_BASE_MODEL_BLOB_SIZE_BYTES = 2497280480
FROZEN_BASE_IDENTITY_VERIFICATION = "exact_local_manifest_and_all_blobs_sha256"
CANDIDATE_IDENTITY_VERIFICATIONS = {
    "evaluation_candidate": "export_receipt_exact_manifest_and_model_blob_sha256",
    "approved_candidate": "approved_registry_exact_manifest_and_model_blob_sha256",
}

# Source and instruction acceptance limits remain independent of the LLM context. Only a
# bounded deterministic context package is sent to Qwen; the full source stays in Python.
MAX_MODEL_SOURCE_BYTES = 512 * 1024
MAX_INSTRUCTION_BYTES = 64 * 1024
MAX_CONTEXT_PACKAGE_BYTES = 32 * 1024
MAX_EDIT_PROGRAM_BYTES = 128 * 1024
MAX_EXPLANATION_BYTES = 16 * 1024
MAX_WARNING_BYTES = 2 * 1024
MAX_WARNINGS = 32
MAX_CONTEXT_REQUESTS = 32
MAX_ATTACHMENT_CONTEXT_REQUESTS = 4
MAX_CONTEXT_ROUNDS = 6
MAX_CLARIFICATION_TURNS = 8
MAX_CLARIFICATION_QUESTION_BYTES = 8 * 1024
MAX_CLARIFICATION_ANSWER_BYTES = 8 * 1024
MAX_CLARIFICATION_HISTORY_BYTES = 16 * 1024
MAX_OPERATIONS = 128
MAX_PATH_COMPONENT_BYTES = 128
MAX_JSON_DEPTH = 12

# Compact, word-bounded routing vocabulary for the complete official catalogue.  Routing
# normalises common scientific inflections before matching and keeps broad nouns (for example
# ``network`` or ``edges``) out of the generic index when they are ambiguous across packs.
# Dependency closure is applied after primary selection.
_OFFICIAL_PACK_ROUTE_INDEX: dict[str, tuple[str, ...]] = {
    "org.modellab.pack.multidimensional-mathematics": (
        "org modellab multidimensional", "array", "tensor", "tensor contraction", "matrix",
        "identity matrix", "matrix entries", "row major", "scientific quantity",
        "dimensioned quantity", "physical quantity", "standard uncertainty", "measurement uncertainty",
    ),
    "org.modellab.pack.probability-stochastic-systems": (
        "org modellab probability", "probability distribution", "discrete distribution", "finite distribution",
        "categorical distribution", "bernoulli distribution", "probability mass", "outcome probabilities", "stochastic process", "markov", "markov chain",
        "transition probabilities", "transition matrix",
    ),
    "org.modellab.pack.graphs-networks-discrete": (
        "org modellab graph", "graph", "directed network", "undirected network", "weighted network",
        "social network", "citation network", "transport network", "road network", "street network",
        "shortest path", "network centrality", "graph laplacian", "network laplacian",
        "laplacian of the network", "vertices and edges", "nodes and edges", "graph vertices", "graph edges", "arcs",
    ),
    "org.modellab.pack.generative-inference-decision-systems": (
        "org modellab generative", "hidden markov", "hidden state model", "hidden states", "hmm",
        "pomdp", "partially observable", "belief state", "active inference", "policy preferences",
        "predictive processing", "free energy principle", "markov blanket",
    ),
    "org.modellab.pack.dynamics-differential-equations-control": (
        "org modellab dynamics", "org modellab control", "ordinary differential equation", "ode",
        "ode system", "state space", "state space system", "continuous state space",
        "dynamical system", "differential equation", "spring oscillator", "spring mass", "mass spring",
        "damped oscillator", "harmonic oscillator", "initial state", "derivatives",
    ),
    "org.modellab.pack.spatial-fields-continuum-pdes": (
        "org modellab field", "org modellab pde", "scalar field", "vector field", "continuum model",
        "diffusion equation", "diffusion problem", "poisson equation", "poisson problem",
        "partial differential equation", "pde", "heat equation", "heat conduction",
        "temperature field", "velocity field", "field grid",
    ),
    "org.modellab.pack.geometry-meshes-spatial-computation": (
        "org modellab geometry", "point cloud", "cloud of points", "3d points", "spatial points",
        "mesh", "triangle mesh", "triangular mesh", "tetrahedral surface", "vertices and faces",
        "mesh geometry", "geometry",
    ),
    "org.modellab.pack.mechanics-structures-materials": (
        "org modellab material", "org modellab mechanics", "linear elastic material",
        "elastic material", "young s modulus", "elastic modulus", "poisson ratio", "material density",
        "truss", "truss structure", "structural frame", "supports and loads", "structural mechanics",
    ),
    "org.modellab.pack.statistical-inference-data-modelling": (
        "org modellab statistics", "statistics", "statistical model", "numerical dataset", "dataset",
        "data table", "columns and rows", "linear regression", "regression", "linear model study",
        "regression study", "grouped samples", "group test", "group comparison", "anova", "kruskal",
        "welch test", "pca", "principal component",
    ),
    "org.modellab.pack.optimisation-estimation-inverse-problems": (
        "org modellab optimisation", "org modellab optimization", "org modellab estimation",
        "org modellab inverse", "optimisation", "optimization", "estimation", "inverse",
        "constrained optimisation", "constrained optimization", "nonlinear optimisation",
        "nonlinear optimization", "nonlinear problem", "minimise", "minimize", "maximise", "maximize",
        "objective", "objective function", "objective and constraints",
        "inverse problem", "linear inverse", "parameter estimation", "residual fitting",
        "least squares estimation", "nonlinear least squares", "design matrix",
    ),
    "org.modellab.pack.electrical-electronic-electromagnetic-systems": (
        "org modellab electrical", "org modellab electronics", "org modellab electromagnetics",
        "electrical circuit", "electrical network", "circuit", "resistor", "voltage source", "ac circuit",
        "dc circuit", "capacitor", "inductor", "shockley diode", "diode", "saturation current", "point charge", "coulomb", "coulomb force", "electric charge",
        "electric field", "electrostatic",
    ),
    "org.modellab.pack.chemical-reaction-biological-systems": (
        "org modellab chemistry", "org modellab biological", "reaction network", "chemical reaction",
        "reaction rate", "rate constant", "species", "mass action", "chemical kinetics", "kinetics model",
        "compartment", "compartment model", "compartmental model", "compartment system", "two compartment",
        "transfer rates", "population", "population interaction",
        "intrinsic growth", "interaction matrix", "predator prey", "ecology", "ecological model", "lotka volterra",
    ),
    "org.modellab.pack.machine-learning-computational-intelligence": (
        "org modellab learning", "org modellab intelligence", "machine learning",
        "computational intelligence", "feature dataset", "feature matrix", "supervised study",
        "neural", "neural network", "feed forward network", "feedforward network", "mlp",
        "multilayer perceptron", "multi layer perceptron", "weights and biases",
        "fuzzy", "fuzzy rule system", "fuzzy controller", "membership sets", "k means", "kmeans",
        "logistic classifier", "logistic regression", "binary classification", "binary classifier", "classifier", "classification study", "clustering",
    ),
}


# Object-kind-level routing keeps scientific authoring schemas narrowly scoped.  Pack routing
# remains available for compatibility and capability selection, while this index determines
# the exact official object schemas exposed to the interpreter for creation/editing.
_OFFICIAL_KIND_ROUTE_INDEX: dict[str, tuple[str, ...]] = {
    "org.modellab.multidimensional.array": ("array", "tensor", "matrix", "matrix entries", "row major"),
    "org.modellab.multidimensional.quantity": ("quantity", "scientific quantity", "dimensioned quantity", "physical quantity", "measurement uncertainty", "standard uncertainty"),
    "org.modellab.probability.discrete-distribution": ("probability distribution", "discrete distribution", "finite distribution", "categorical distribution", "bernoulli distribution", "probability mass", "outcome probabilities"),
    "org.modellab.probability.markov-chain": ("markov", "markov chain", "transition matrix", "transition probabilities", "stochastic process"),
    "org.modellab.graph.network": ("graph", "directed network", "undirected network", "weighted network", "social network", "citation network", "transport network", "road network", "street network", "shortest path", "network centrality", "graph laplacian", "network laplacian", "laplacian of the network", "laplacian of this network", "nodes and edges", "vertices and edges", "nodes and links", "vertices and links", "graph vertices", "graph edges", "arcs"),
    "org.modellab.generative.hidden-markov-model": ("hidden markov", "hidden state model", "hidden states", "hmm"),
    "org.modellab.generative.pomdp": ("pomdp", "partially observable", "belief state"),
    "org.modellab.generative.active-inference-model": ("active inference", "predictive processing", "free energy principle", "markov blanket", "policy preferences"),
    "org.modellab.dynamics.ode-system": ("ordinary differential equation", "ode", "ode system", "dynamical system", "differential equation", "derivatives", "reaction kinetics", "spring oscillator", "harmonic oscillator", "spring mass", "mass spring"),
    "org.modellab.control.state-space-system": ("state space", "state space system", "continuous state space", "lti", "linear time invariant", "linear controller", "lti controller"),
    "org.modellab.field.structured-scalar-field": ("scalar field", "temperature field", "field grid"),
    "org.modellab.field.structured-vector-field": ("vector field", "velocity field"),
    "org.modellab.pde.diffusion-problem": ("diffusion equation", "diffusion problem", "heat equation", "heat conduction"),
    "org.modellab.pde.poisson-problem": ("poisson equation", "poisson problem", "laplace equation", "laplacian", "elliptic pde"),
    "org.modellab.geometry.point-cloud": ("point cloud", "cloud of points", "3d points", "spatial points"),
    "org.modellab.geometry.triangle-mesh": ("mesh", "triangle mesh", "triangular mesh", "tetrahedral surface", "vertices and faces", "mesh geometry"),
    "org.modellab.material.isotropic-linear-elastic": ("linear elastic material", "elastic material", "young s modulus", "elastic modulus", "poisson ratio", "material density"),
    "org.modellab.mechanics.truss-structure": ("truss", "truss structure", "structural frame", "supports and loads", "structural mechanics"),
    "org.modellab.statistics.dataset": ("numerical dataset", "dataset", "data table", "columns and rows", "pca", "principal component"),
    "org.modellab.statistics.linear-model-study": ("linear regression", "regression", "linear model study", "regression study"),
    "org.modellab.statistics.grouped-samples": ("grouped samples", "group test", "group comparison", "anova", "kruskal", "welch test"),
    "org.modellab.optimisation.nonlinear-problem": ("optimisation", "optimization", "optimise", "optimize", "minimise", "minimize", "maximise", "maximize", "objective", "objective function", "objective and constraints", "constrained optimisation", "constrained optimization", "nonlinear optimisation", "nonlinear optimization"),
    "org.modellab.estimation.nonlinear-least-squares": ("estimation", "estimate", "parameter estimation", "estimate parameters", "infer parameters", "infer parameter", "fit parameters", "residual fitting", "least squares estimation", "nonlinear least squares", "calibrate", "calibration", "identify parameter"),
    "org.modellab.inverse.linear-problem": ("inverse", "inverse problem", "linear inverse", "design matrix"),
    "org.modellab.electrical.linear-circuit": ("electrical circuit", "electrical network", "circuit", "resistor", "capacitor", "inductor", "voltage source", "ac circuit", "dc circuit", "rlc", "r l c", "resistor network", "circuit network"),
    "org.modellab.electronics.shockley-diode": ("shockley diode", "diode", "saturation current"),
    "org.modellab.electromagnetics.point-charge-system": ("point charge", "electric charge", "coulomb", "coulomb force", "electric field", "electrostatic"),
    "org.modellab.chemistry.mass-action-network": ("reaction network", "chemical reaction", "reaction rate", "rate constant", "species", "mass action", "chemical kinetics", "kinetics model"),
    "org.modellab.biological.compartment-system": ("compartment", "compartment model", "compartmental model", "compartment system", "two compartment", "transfer rates"),
    "org.modellab.biological.population-interaction-system": ("population", "population interaction", "intrinsic growth", "interaction matrix", "predator prey", "ecology", "ecological model", "lotka volterra"),
    "org.modellab.learning.feature-dataset": ("feature", "features", "feature dataset", "feature matrix", "ml data", "machine learning data", "clustering", "k means", "kmeans"),
    "org.modellab.learning.supervised-study": ("supervised study", "logistic classifier", "logistic regression", "binary classification", "binary classifier", "classifier", "classification study"),
    "org.modellab.learning.feedforward-network": ("neural", "neural network", "feed forward network", "feedforward network", "mlp", "multilayer perceptron", "multi layer perceptron", "weights and biases", "neural controller", "graph neural network"),
    "org.modellab.intelligence.fuzzy-rule-system": ("fuzzy", "fuzzy rule system", "fuzzy controller", "membership sets"),
}

_ROUTING_TOKEN_ALIASES: dict[str, str] = {
    "arrays": "array", "tensors": "tensor", "matrices": "matrix", "quantities": "quantity",
    "distributions": "distribution", "graphs": "graph", "networks": "network",
    "hmms": "hmm", "odes": "ode", "pdes": "pde", "equations": "equation",
    "fields": "field", "meshes": "mesh", "trusses": "truss", "datasets": "dataset",
    "classifiers": "classifier", "reactions": "reaction", "compartments": "compartment",
    "resistors": "resistor", "diodes": "diode", "charges": "charge", "populations": "population",
    "samples": "sample", "models": "model", "systems": "system", "controllers": "controller",
    "parameters": "parameter", "estimates": "estimate", "estimated": "estimate", "estimating": "estimate",
    "optimizes": "optimize", "optimized": "optimize", "optimizing": "optimize",
    "optimises": "optimise", "optimised": "optimise", "optimising": "optimise",
    "maximizes": "maximize", "maximized": "maximize", "maximizing": "maximize",
    "maximises": "maximise", "maximised": "maximise", "maximising": "maximise",
    "calibrates": "calibrate", "calibrated": "calibrate", "calibrating": "calibrate",
    "identifies": "identify", "identified": "identify", "identifying": "identify",
    "infers": "infer", "inferred": "infer", "inferring": "infer",
    "fitted": "fit", "fitting": "fit",
    "adding": "add", "added": "add", "adds": "add",
    "creating": "create", "created": "create", "creates": "create",
    "building": "build", "built": "build", "builds": "build",
    "constructing": "construct", "constructed": "construct", "constructs": "construct",
    "making": "make", "made": "make", "makes": "make",
    "defining": "define", "defined": "define", "defines": "define",
    "using": "use", "used": "use", "uses": "use",
    "including": "include", "included": "include", "includes": "include",
    "representing": "represent", "represented": "represent", "represents": "represent",
    "changing": "change", "changed": "change", "changes": "change",
    "modifying": "modify", "modified": "modify", "modifies": "modify",
    "removing": "remove", "removed": "remove", "removes": "remove",
    "deleting": "delete", "deleted": "delete", "deletes": "delete",
    "replacing": "replace", "replaced": "replace", "replaces": "replace",
    "attempting": "attempt", "attempted": "attempt", "tries": "try", "trying": "try",
}

_OFFICIAL_KIND_PREFIX_INDEX: dict[str, tuple[str, ...]] = {
    "org.modellab.pack.multidimensional-mathematics": ("org.modellab.multidimensional.",),
    "org.modellab.pack.probability-stochastic-systems": ("org.modellab.probability.",),
    "org.modellab.pack.graphs-networks-discrete": ("org.modellab.graph.",),
    "org.modellab.pack.generative-inference-decision-systems": ("org.modellab.generative.",),
    "org.modellab.pack.dynamics-differential-equations-control": ("org.modellab.dynamics.", "org.modellab.control."),
    "org.modellab.pack.spatial-fields-continuum-pdes": ("org.modellab.field.", "org.modellab.pde."),
    "org.modellab.pack.geometry-meshes-spatial-computation": ("org.modellab.geometry.",),
    "org.modellab.pack.mechanics-structures-materials": ("org.modellab.material.", "org.modellab.mechanics."),
    "org.modellab.pack.statistical-inference-data-modelling": ("org.modellab.statistics.",),
    "org.modellab.pack.optimisation-estimation-inverse-problems": ("org.modellab.optimisation.", "org.modellab.estimation.", "org.modellab.inverse."),
    "org.modellab.pack.electrical-electronic-electromagnetic-systems": ("org.modellab.electrical.", "org.modellab.electronics.", "org.modellab.electromagnetics."),
    "org.modellab.pack.chemical-reaction-biological-systems": ("org.modellab.chemistry.", "org.modellab.biological."),
    "org.modellab.pack.machine-learning-computational-intelligence": ("org.modellab.learning.", "org.modellab.intelligence."),
}

_DECODING_OPTIONS: dict[str, int | float] = {
    "seed": 0,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
}
GENERATION_PROFILES: tuple[dict[str, int | float], ...] = (
    {**_DECODING_OPTIONS, "num_ctx": 32768, "num_predict": 4096},
    {**_DECODING_OPTIONS, "num_ctx": 65536, "num_predict": 8192},
    {**_DECODING_OPTIONS, "num_ctx": 131072, "num_predict": 8192},
)
EVALUATION_GENERATION_OPTIONS: dict[str, int | float] = {
    **GENERATION_PROFILES[0],
    "temperature": 0.0,
}
# Kept as the ordinary short-request profile for callers which need a concrete default.
# Production requests are selected by ``generation_options_for_request`` below.
GENERATION_OPTIONS: dict[str, int | float] = dict(GENERATION_PROFILES[0])
_V111_PRODUCTION_GENERATION_OPTIONS: dict[str, int | float] = dict(GENERATION_PROFILES[-1])
_LEGACY_GENERATION_OPTIONS: dict[str, int | float] = {
    "seed": 0,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "num_ctx": 8192,
    "num_predict": 4096,
}

# The estimate deliberately includes the versioned prompt templates and JSON framing without
# importing the baseline module (which imports this compiler).  One estimated token per two
# UTF-8 bytes is conservative for normal instructions/YAML.  The largest profile is retained
# as a bounded fallback for adversarially dense text, so accepted source/instruction limits do
# not change merely to save memory for ordinary requests.
PROMPT_STATIC_OVERHEAD_BYTES = 12 * 1024
PROMPT_TOKEN_ESTIMATE_DENOMINATOR = 2
PROMPT_TOKEN_SAFETY_MARGIN = 4096

_PROVIDER_KEYS = {
    "provider",
    "endpoint",
    "runtime_version",
    "model_tag",
    "observed_model_digest",
    "identity_verification",
    "expected_base_model",
    "expected_quantization",
    "generation_options",
    "model_role",
    "registry_entry_sha256",
    "artifact_evidence",
}
_V122_PROVIDER_KEYS = _PROVIDER_KEYS - {"artifact_evidence"}
_V111_PROVIDER_KEYS = _V122_PROVIDER_KEYS - {"model_role", "registry_entry_sha256"}
_ARTIFACT_EVIDENCE_KEYS = {
    "identity_verification",
    "artifact_lock_sha256",
    "local_manifest_sha256",
    "model_blob_sha256",
    "verified_model_blob_size_bytes",
    "all_manifest_blobs_verified",
}
_LEGACY_PROVIDER_KEYS = {
    "provider",
    "endpoint",
    "runtime_version",
    "model",
    "model_digest",
    "base_model",
    "quantization",
    "generation_options",
}
_GENERATION_OPTION_KEYS = set(GENERATION_OPTIONS)
_RAW_OUTPUT_KEYS = {
    "schema",
    "schema_version",
    "action",
    "context_sha256",
    "context_requests",
    "operations",
    "explanation",
    "warnings",
    "clarification_question",
}
_OPERATION_KEYS = {"op", "path", "value"}
_ENTRY_MODEL_TYPES = {
    "variables": VariableSpec,
    "parameters": ParameterSpec,
    "constants": ConstantSpec,
    "derived_quantities": DerivedQuantitySpec,
    "functions": FunctionSpec,
    "vector_functions": VectorFunctionSpec,
    "matrix_functions": MatrixFunctionSpec,
    "constraints": ConstraintSpec,
    "assumptions": AssumptionSpec,
    "ambiguities": AmbiguitySpec,
    "objects": ModelObjectSpec,
    "relationships": ModelRelationshipSpec,
    "assets": StructuredAssetSpec,
}
_MODEL_SECTIONS = tuple(_ENTRY_MODEL_TYPES)
_TOP_LEVEL_ORDER = ("name", "metadata", *_MODEL_SECTIONS)
_ENTRY_FIELDS: dict[str, frozenset[str]] = {
    section: frozenset(model_type.model_fields)
    for section, model_type in _ENTRY_MODEL_TYPES.items()
}
_ENTRY_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    section: tuple(
        name for name, field_info in model_type.model_fields.items() if field_info.is_required()
    )
    for section, model_type in _ENTRY_MODEL_TYPES.items()
}
_METADATA_FIELDS = frozenset(ModelMetadataSpec.model_fields)
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class InterpreterProposalError(ValueError):
    """Raised when interpreter context, output, compilation, or provenance is invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InterpreterProposalError("Interpreter data must be finite JSON data.") from exc


def _bounded_text(
    value: object,
    field: str,
    maximum_bytes: int,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise InterpreterProposalError(f"{field} must be a string.")
    if "\x00" in value:
        raise InterpreterProposalError(f"{field} must not contain NUL characters.")
    if len(value.encode("utf-8")) > maximum_bytes:
        raise InterpreterProposalError(f"{field} exceeds its {maximum_bytes}-byte safety limit.")
    if not allow_empty and not value.strip():
        raise InterpreterProposalError(f"{field} must not be empty.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing:
        raise InterpreterProposalError(
            f"{field} is missing field(s): {', '.join(sorted(missing))}."
        )
    if extra:
        raise InterpreterProposalError(
            f"{field} contains unknown field(s): {', '.join(sorted(extra))}."
        )


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise InterpreterProposalError(f"{field} is not a lowercase SHA-256 digest.")
    return value


def _validated_warnings(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise InterpreterProposalError(f"{field} must be a list.")
    if len(value) > MAX_WARNINGS:
        raise InterpreterProposalError(f"{field} contains more than {MAX_WARNINGS} warnings.")
    return [
        _bounded_text(item, f"{field} item {index + 1}", MAX_WARNING_BYTES)
        for index, item in enumerate(value)
    ]


def validate_clarification_history(value: object) -> list[dict[str, str]]:
    """Validate the ordered, user-authored answers carried into a later interpreter turn."""
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise InterpreterProposalError("clarification_history must be a list.")
    if len(value) > MAX_CLARIFICATION_TURNS:
        raise InterpreterProposalError(
            f"clarification_history contains more than {MAX_CLARIFICATION_TURNS} turns."
        )
    result: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise InterpreterProposalError(
                f"clarification_history[{index}] must be an object."
            )
        _exact_keys(item, {"question", "answer"}, f"clarification_history[{index}]")
        result.append(
            {
                "question": _bounded_text(
                    item.get("question"),
                    f"clarification_history[{index}].question",
                    MAX_CLARIFICATION_QUESTION_BYTES,
                ),
                "answer": _bounded_text(
                    item.get("answer"),
                    f"clarification_history[{index}].answer",
                    MAX_CLARIFICATION_ANSWER_BYTES,
                ),
            }
        )
    if len(_canonical_json_bytes(result)) > MAX_CLARIFICATION_HISTORY_BYTES:
        raise InterpreterProposalError(
            "clarification_history exceeds its fixed 16 KiB safety budget."
        )
    return result


def _conversation_identifier(
    value: object,
    *,
    instruction: str,
    current_model_source: str,
) -> str:
    if value is None:
        value = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "model-laboratory-interpreter:"
                + _sha256_text(instruction)
                + ":"
                + _sha256_text(current_model_source),
            )
        )
    text = _bounded_text(value, "conversation_id", 128)
    try:
        return str(uuid.UUID(text))
    except (ValueError, AttributeError) as exc:
        raise InterpreterProposalError("conversation_id must be a UUID.") from exc


def validate_conversation_lineage(
    value: object,
    *,
    conversation_id: str,
    clarification_history: Sequence[Mapping[str, str]],
    provider_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate every prior clarification output and its immutable provider chain."""
    if value is None:
        value = []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise InterpreterProposalError("conversation_lineage must be a list.")
    if len(value) != len(clarification_history):
        raise InterpreterProposalError(
            "conversation_lineage must contain exactly one evidence record per answered clarification."
        )
    provider_chain_sha = canonical_json_sha256(
        {key: item for key, item in provider_identity.items() if key != "generation_options"}
    )
    result: list[dict[str, Any]] = []
    parent_sha: str | None = None
    required = {
        "schema", "schema_version", "conversation_id", "turn_index",
        "parent_turn_sha256", "context_sha256", "raw_output_sha256",
        "provider_identity", "provider_identity_sha256", "question_sha256", "turn_sha256",
    }
    for index, (item, history) in enumerate(zip(value, clarification_history, strict=True)):
        if not isinstance(item, Mapping):
            raise InterpreterProposalError(f"conversation_lineage[{index}] must be an object.")
        _exact_keys(item, required, f"conversation_lineage[{index}]")
        if (
            item.get("schema") != INTERPRETER_DIALOGUE_TURN_SCHEMA
            or item.get("schema_version") != INTERPRETER_DIALOGUE_TURN_SCHEMA_VERSION
            or item.get("conversation_id") != conversation_id
            or item.get("turn_index") != index
            or item.get("parent_turn_sha256") != parent_sha
        ):
            raise InterpreterProposalError(
                f"conversation_lineage[{index}] does not continue the expected dialogue chain."
            )
        for field in (
            "context_sha256", "raw_output_sha256", "provider_identity_sha256",
            "question_sha256", "turn_sha256",
        ):
            _require_sha256(item.get(field), f"conversation_lineage[{index}].{field}")
        prior_provider = validate_provider_identity(item.get("provider_identity"))
        if item["provider_identity_sha256"] != canonical_json_sha256(prior_provider):
            raise InterpreterProposalError(
                f"conversation_lineage[{index}] provider checksum is invalid."
            )
        prior_chain_sha = canonical_json_sha256(
            {key: candidate for key, candidate in prior_provider.items() if key != "generation_options"}
        )
        if prior_chain_sha != provider_chain_sha:
            raise InterpreterProposalError(
                "The local model/provider artifact changed during clarification. Start a new request."
            )
        if item["question_sha256"] != _sha256_text(str(history["question"])):
            raise InterpreterProposalError(
                f"conversation_lineage[{index}] is bound to a different clarification question."
            )
        computed = canonical_json_sha256(
            {key: candidate for key, candidate in item.items() if key != "turn_sha256"}
        )
        if item["turn_sha256"] != computed:
            raise InterpreterProposalError(
                f"conversation_lineage[{index}] checksum does not match its contents."
            )
        normalised = dict(item)
        result.append(normalised)
        parent_sha = computed
    return result


def _create_dialogue_turn_evidence(
    *,
    conversation_id: str,
    turn_index: int,
    parent_turn_sha256: str | None,
    context_sha256: str,
    raw_output: Mapping[str, Any],
    provider_identity: Mapping[str, Any],
    question: str,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "schema": INTERPRETER_DIALOGUE_TURN_SCHEMA,
        "schema_version": INTERPRETER_DIALOGUE_TURN_SCHEMA_VERSION,
        "conversation_id": conversation_id,
        "turn_index": turn_index,
        "parent_turn_sha256": parent_turn_sha256,
        "context_sha256": context_sha256,
        "raw_output_sha256": canonical_json_sha256(raw_output),
        "provider_identity": dict(provider_identity),
        "provider_identity_sha256": canonical_json_sha256(provider_identity),
        "question_sha256": _sha256_text(question),
    }
    evidence["turn_sha256"] = canonical_json_sha256(evidence)
    return evidence


def generation_options_for_request(
    *, instruction: str, context_document: Mapping[str, Any]
) -> dict[str, int | float]:
    """Select the smallest bounded KV-cache profile suitable for one exact request.

    The maximum 131K profile remains available, so the 64 KiB instruction and 512 KiB
    source acceptance contracts are not reduced.  Ordinary compact contexts use 32K and
    therefore do not reserve the maximum cache on every request.
    """
    instruction = _bounded_text(instruction, "Interpreter instruction", MAX_INSTRUCTION_BYTES)
    context_bytes = len(_canonical_json_bytes(context_document))
    if context_bytes > MAX_CONTEXT_PACKAGE_BYTES:
        raise InterpreterProposalError(
            "Interpreter context package exceeds its fixed transport budget."
        )
    prompt_bytes = (
        len(instruction.encode("utf-8"))
        + context_bytes
        + PROMPT_STATIC_OVERHEAD_BYTES
    )
    estimated_prompt_tokens = (
        prompt_bytes + PROMPT_TOKEN_ESTIMATE_DENOMINATOR - 1
    ) // PROMPT_TOKEN_ESTIMATE_DENOMINATOR
    for profile in GENERATION_PROFILES:
        required = (
            estimated_prompt_tokens
            + int(profile["num_predict"])
            + PROMPT_TOKEN_SAFETY_MARGIN
        )
        if required <= int(profile["num_ctx"]):
            return dict(profile)
    return dict(GENERATION_PROFILES[-1])


def _validate_json_value(value: object, field: str, *, depth: int = 0) -> Any:
    if depth > MAX_JSON_DEPTH:
        raise InterpreterProposalError(f"{field} exceeds the maximum JSON nesting depth.")
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and "\x00" in value:
            raise InterpreterProposalError(f"{field} must not contain NUL characters.")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InterpreterProposalError(f"{field} must contain only finite numbers.")
        return value
    if isinstance(value, list):
        return [
            _validate_json_value(item, f"{field}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or "\x00" in key:
                raise InterpreterProposalError(f"{field} object keys must be non-empty strings.")
            result[key] = _validate_json_value(item, f"{field}.{key}", depth=depth + 1)
        return result
    raise InterpreterProposalError(f"{field} contains a non-JSON value.")


def _normalized_observed_sha256(value: object, field: str) -> str:
    text = _bounded_text(value, field, 256).strip().lower()
    digest = text[7:] if text.startswith("sha256:") else text
    if _SHA256.fullmatch(digest) is None:
        raise InterpreterProposalError(f"{field} must contain a complete SHA-256 digest.")
    return digest


def _validate_artifact_evidence(
    value: object,
    *,
    role: str,
    identity_verification: str,
    observed_model_digest: str,
    registry_entry_sha256: str | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InterpreterProposalError("provider_identity.artifact_evidence must be an object.")
    _exact_keys(value, _ARTIFACT_EVIDENCE_KEYS, "provider_identity.artifact_evidence")
    evidence_identity = _bounded_text(
        value.get("identity_verification"),
        "provider_identity.artifact_evidence.identity_verification",
        256,
    )
    if evidence_identity != identity_verification:
        raise InterpreterProposalError(
            "Provider and artifact-evidence verification methods differ."
        )
    artifact_lock_sha = _require_sha256(
        value.get("artifact_lock_sha256"),
        "provider_identity.artifact_evidence.artifact_lock_sha256",
    )
    manifest_sha = _require_sha256(
        value.get("local_manifest_sha256"),
        "provider_identity.artifact_evidence.local_manifest_sha256",
    )
    model_blob_sha = _require_sha256(
        value.get("model_blob_sha256"),
        "provider_identity.artifact_evidence.model_blob_sha256",
    )
    size = value.get("verified_model_blob_size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise InterpreterProposalError(
            "provider_identity.artifact_evidence.verified_model_blob_size_bytes must be positive."
        )
    if value.get("all_manifest_blobs_verified") is not True:
        raise InterpreterProposalError(
            "Provider identity requires SHA-256 verification of every local manifest blob."
        )
    if manifest_sha != observed_model_digest:
        raise InterpreterProposalError(
            "The provider's observed model digest differs from the verified manifest bytes."
        )
    expected_lock = (
        FROZEN_BASE_ARTIFACT_LOCK_SHA256
        if role == "frozen_base"
        else registry_entry_sha256
    )
    if artifact_lock_sha != expected_lock:
        raise InterpreterProposalError(
            "Artifact evidence is bound to a different base lock or candidate registry entry."
        )
    if role == "frozen_base" and (
        manifest_sha != FROZEN_BASE_MANIFEST_SHA256
        or model_blob_sha != FROZEN_BASE_MODEL_BLOB_SHA256
        or size != FROZEN_BASE_MODEL_BLOB_SIZE_BYTES
    ):
        raise InterpreterProposalError(
            "Frozen-base artifact evidence differs from the exact bundled manifest/GGUF lock."
        )
    return {
        "identity_verification": evidence_identity,
        "artifact_lock_sha256": artifact_lock_sha,
        "local_manifest_sha256": manifest_sha,
        "model_blob_sha256": model_blob_sha,
        "verified_model_blob_size_bytes": size,
        "all_manifest_blobs_verified": True,
    }


def validate_provider_identity(
    value: object,
    *,
    generation_options: Mapping[str, int | float] | None = None,
    legacy: bool = False,
    allow_historical_current: bool = False,
) -> dict[str, Any]:
    """Validate observed provider facts without claiming an unverified weight identity."""
    if not isinstance(value, Mapping):
        raise InterpreterProposalError("provider_identity must be an object.")
    historical_current = False
    historical_artifact = False
    if legacy:
        _exact_keys(value, _LEGACY_PROVIDER_KEYS, "provider_identity")
    elif allow_historical_current and set(value) == _V122_PROVIDER_KEYS:
        historical_artifact = True
    elif allow_historical_current and set(value) == _V111_PROVIDER_KEYS:
        historical_current = True
    else:
        _exact_keys(value, _PROVIDER_KEYS, "provider_identity")
    expected: dict[str, Any] = (
        {
            "provider": LOCAL_PROVIDER,
            "endpoint": LOCAL_ENDPOINT,
            "model": LOCAL_MODEL,
            "base_model": LOCAL_BASE_MODEL,
            "quantization": LOCAL_QUANTIZATION,
        }
        if legacy
        else {
            "provider": LOCAL_PROVIDER,
            "endpoint": LOCAL_ENDPOINT,
            "expected_base_model": LOCAL_BASE_MODEL,
            "expected_quantization": LOCAL_QUANTIZATION,
        }
    )
    if not legacy:
        role = "frozen_base" if historical_current else value.get("model_role")
        if role == "frozen_base":
            expected.update(
                {
                    "model_tag": LOCAL_MODEL,
                    "identity_verification": (
                        "configured_tag_with_observed_digest"
                        if historical_current or historical_artifact
                        else FROZEN_BASE_IDENTITY_VERIFICATION
                    ),
                }
            )
            registry_sha: str | None = None
            if not historical_current and value.get("registry_entry_sha256") is not None:
                raise InterpreterProposalError(
                    "A frozen-base provider must not declare a registry entry."
                )
        elif role in {"approved_candidate", "evaluation_candidate"}:
            model_tag = _bounded_text(value.get("model_tag"), "provider_identity.model_tag", 256)
            if (
                model_tag == LOCAL_MODEL
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}", model_tag) is None
            ):
                raise InterpreterProposalError(
                    "An approved candidate must use a distinct valid local model tag."
                )
            expected.update(
                {
                    "model_tag": model_tag,
                    "identity_verification": (
                        (
                            "approved_registry_entry_with_observed_digest"
                            if role == "approved_candidate"
                            else "export_receipt_with_observed_digest"
                        )
                        if historical_artifact
                        else CANDIDATE_IDENTITY_VERIFICATIONS[role]
                    ),
                }
            )
            registry_sha = _require_sha256(
                value.get("registry_entry_sha256"),
                "provider_identity.registry_entry_sha256",
            )
        else:
            raise InterpreterProposalError(
                "provider_identity.model_role must be frozen_base, evaluation_candidate, or approved_candidate."
            )
    for field, required in expected.items():
        if value.get(field) != required:
            raise InterpreterProposalError(
                f"provider_identity.{field} must be {required!r}."
            )
    runtime_version = _bounded_text(
        value.get("runtime_version"), "provider_identity.runtime_version", 256
    )
    digest_field = "model_digest" if legacy else "observed_model_digest"
    if legacy or historical_current or historical_artifact:
        model_digest = _bounded_text(
            value.get(digest_field), f"provider_identity.{digest_field}", 256
        )
        normalized_model_digest = None
    else:
        model_digest = _bounded_text(
            value.get(digest_field), f"provider_identity.{digest_field}", 256
        )
        normalized_model_digest = _normalized_observed_sha256(
            model_digest, f"provider_identity.{digest_field}"
        )
    options = value.get("generation_options")
    if not isinstance(options, Mapping):
        raise InterpreterProposalError("provider_identity.generation_options must be an object.")
    _exact_keys(options, _GENERATION_OPTION_KEYS, "provider_identity.generation_options")
    if generation_options is not None:
        required_options = dict(generation_options)
    elif legacy:
        required_options = dict(_LEGACY_GENERATION_OPTIONS)
    elif historical_current:
        # Historical provider documents are accepted only with one of the actual historical
        # fixed/default profiles.  Their enclosing proposal/acceptance version decides which
        # one is required when stricter validation is available.
        candidate_options = dict(options)
        if candidate_options not in (
            _V111_PRODUCTION_GENERATION_OPTIONS,
            GENERATION_OPTIONS,
        ):
            raise InterpreterProposalError(
                "provider_identity.generation_options is not a supported historical profile."
            )
        required_options = candidate_options
    else:
        candidate_options = dict(options)
        if candidate_options not in (*GENERATION_PROFILES, EVALUATION_GENERATION_OPTIONS):
            raise InterpreterProposalError(
                "provider_identity.generation_options is not an approved production profile."
            )
        required_options = candidate_options
    for field, required in required_options.items():
        candidate = options.get(field)
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            raise InterpreterProposalError(
                f"provider_identity.generation_options.{field} must be numeric."
            )
        if float(candidate) != float(required):
            raise InterpreterProposalError(
                f"provider_identity.generation_options.{field} must be {required}."
            )
    result = {
        **expected,
        "runtime_version": runtime_version,
        digest_field: model_digest,
        "generation_options": required_options,
    }
    if not legacy and not historical_current:
        result["model_role"] = role
        result["registry_entry_sha256"] = registry_sha
    if not legacy and not historical_current and not historical_artifact:
        result["artifact_evidence"] = _validate_artifact_evidence(
            value.get("artifact_evidence"),
            role=role,
            identity_verification=str(expected["identity_verification"]),
            observed_model_digest=str(normalized_model_digest),
            registry_entry_sha256=registry_sha,
        )
    return result


def _validated_path(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= 3:
        raise InterpreterProposalError(f"{field} must contain one to three path components.")
    path: list[str] = []
    for index, item in enumerate(value):
        component = _bounded_text(item, f"{field}[{index}]", MAX_PATH_COMPONENT_BYTES)
        path.append(component)

    root = path[0]
    if root == "name":
        if len(path) != 1:
            raise InterpreterProposalError(f"{field} has an invalid name path.")
        return tuple(path)
    if root == "metadata":
        if len(path) == 2 and path[1] not in _METADATA_FIELDS:
            raise InterpreterProposalError(f"{field} has an unsupported metadata field.")
        if len(path) > 2:
            raise InterpreterProposalError(f"{field} has an invalid metadata path.")
        return tuple(path)
    if root not in _MODEL_SECTIONS or len(path) < 2:
        raise InterpreterProposalError(f"{field} has an unsupported model path.")
    if _IDENTIFIER.fullmatch(path[1]) is None:
        raise InterpreterProposalError(f"{field} contains an invalid model identifier.")
    if len(path) == 3 and path[2] not in _ENTRY_FIELDS[root]:
        raise InterpreterProposalError(
            f"{field} has an unsupported field for section '{root}'."
        )
    return tuple(path)


def _validated_context_request(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, list) and len(value) == 3 and value[0] == "@attachment":
        identifier = _bounded_text(value[1], f"{field}[1]", MAX_PATH_COMPONENT_BYTES)
        if len(identifier) != 64 or any(
            character not in "0123456789abcdef" for character in identifier
        ):
            raise InterpreterProposalError(
                f"{field} contains an invalid attachment identifier."
            )
        index_text = _bounded_text(value[2], f"{field}[2]", MAX_PATH_COMPONENT_BYTES)
        if not index_text.isdigit() or str(int(index_text)) != index_text:
            raise InterpreterProposalError(
                f"{field} attachment chunk index must be canonical non-negative decimal text."
            )
        return "@attachment", identifier, index_text
    return _validated_path(value, field)


def _path_exists(document: Mapping[str, Any], path: Sequence[str]) -> bool:
    current: object = document
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            return False
        current = current[component]
    return True


def _path_value(document: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: object = document
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            raise InterpreterProposalError(
                "Requested interpreter context path does not exist: " + ".".join(path) + "."
            )
        current = current[component]
    return deepcopy(current)


def _load_source_data(source: str) -> dict[str, Any]:
    if not source.strip():
        return {}
    try:
        data = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise InterpreterProposalError(
            "The current source must be valid YAML before the deterministic interpreter can edit it."
        ) from exc
    if not isinstance(data, dict):
        raise InterpreterProposalError(
            "The current source must be an empty document or a valid Model Laboratory mapping."
        )
    try:
        validate_model(parse_model_data(data))
    except (ModelParseError, ModelValidationError) as exc:
        raise InterpreterProposalError(
            "The current source must validate before deterministic interpreter edits are applied: "
            f"{exc}"
        ) from exc
    return _validate_json_value(data, "Current model source")


def _compact_schema(value: Any) -> Any:
    """Keep the validation-bearing subset of generated Pydantic JSON Schema."""
    if isinstance(value, list):
        return [_compact_schema(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    result: dict[str, Any] = {}
    for key in (
        "$ref",
        "type",
        "const",
        "enum",
        "required",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "pattern",
        "default",
    ):
        if key in value:
            result[key] = _compact_schema(value[key])
    for key in ("anyOf", "oneOf", "allOf", "prefixItems", "items", "additionalProperties"):
        if key in value:
            result[key] = _compact_schema(value[key])
    if isinstance(value.get("properties"), Mapping):
        result["properties"] = {
            str(key): _compact_schema(item)
            for key, item in sorted(value["properties"].items())
        }
    return result


def _build_model_schema_catalogue() -> dict[str, Any]:
    """Build the bounded catalogue derived from the authoritative Pydantic models."""
    complete = ModelSpec.model_json_schema()
    definitions = complete.get("$defs", {})
    return {
        "authoritative_schema_sha256": canonical_json_sha256(complete),
        "sections": {
            section: {
                "schema_sha256": canonical_json_sha256(model_type.model_json_schema()),
                "required": list(_ENTRY_REQUIRED_FIELDS[section]),
                "properties": {
                    name: _compact_schema(detail)
                    for name, detail in sorted(
                        model_type.model_json_schema().get("properties", {}).items()
                    )
                },
            }
            for section, model_type in _ENTRY_MODEL_TYPES.items()
        },
        "metadata": {
            "schema_sha256": canonical_json_sha256(ModelMetadataSpec.model_json_schema()),
            "properties": {
                name: _compact_schema(detail)
                for name, detail in sorted(
                    ModelMetadataSpec.model_json_schema().get("properties", {}).items()
                )
            },
        },
        "definitions": {
            str(name): _compact_schema(detail)
            for name, detail in sorted(definitions.items())
        },
    }


@lru_cache(maxsize=1)
def _model_schema_catalogue_json() -> str:
    # The authoritative schema catalogue is process-static.  Cache a serialized immutable
    # representation, then parse a fresh copy for each context so callers cannot mutate shared
    # state and change later context hashes.
    return json.dumps(_build_model_schema_catalogue(), separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _model_schema_catalogue() -> dict[str, Any]:
    value = json.loads(_model_schema_catalogue_json())
    if not isinstance(value, dict):  # pragma: no cover - defensive invariant
        raise InterpreterProposalError("Cached interpreter schema catalogue is malformed.")
    return value


def _build_context_contract(
    model: object | None = None, *, official_pack_ids: tuple[str, ...] = (),
    official_kind_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    from .builtin_packs import run_registry
    from .model_graph import CORE_KIND_REGISTRY
    from .official_packs import official_kind_catalogue

    if official_pack_ids:
        selected_packs = set(official_pack_ids)
        selected_kinds = set(official_kind_ids)
        official_rows = official_kind_catalogue(official_pack_ids)
        if selected_kinds:
            # Exact kind routing is authoritative for packs it resolves, but it must not
            # suppress another pack that was selected only by broader scientific language.
            # For those unresolved packs expose only their own primary kinds, never their
            # dependency closure.  This keeps mixed requests compositional without restoring
            # the pack-level context explosion that object-kind routing was introduced to fix.
            exact_packs = {
                pack_id
                for kind in selected_kinds
                if (pack_id := _pack_for_official_kind(kind)) is not None
            }
            broad_only_packs = selected_packs - exact_packs
            official_rows = [
                item
                for item in official_rows
                if item["kind"] in selected_kinds
                or _pack_for_official_kind(str(item["kind"])) in broad_only_packs
            ]
        else:
            # Broad pack-only requests still use object-kind routing: expose only primary
            # kinds belonging to the selected packs, not kinds pulled in by dependencies.
            official_rows = [
                item
                for item in official_rows
                if _pack_for_official_kind(str(item["kind"])) in selected_packs
            ]
        kind_catalogue = [
            {
                "kind": item["kind"],
                "version": item["version"],
                "title": item["title"],
                "executable": item["executable"],
                "property_schema": _compact_schema(item["property_schema"]),
            }
            for item in [*CORE_KIND_REGISTRY.catalogue(), *official_rows]
        ]
        capabilities = [
            {
                "id": item["id"],
                "version": item["version"],
                "applicable": item["applicable"],
            }
            for item in run_registry.catalogue(model)  # type: ignore[arg-type]
            if item["pack_id"] not in _all_official_pack_ids() or item["pack_id"] in selected_packs
        ]
    else:
        # This projection is intentionally byte-for-byte compatible with the v1.12
        # interpreter contract. Reviewed corpus records remain reconstructable even as
        # optional scientific packs extend the locally installed registry.
        kind_catalogue = CORE_KIND_REGISTRY.catalogue()
        capabilities = [
            item for item in run_registry.catalogue(model)  # type: ignore[arg-type]
            if item["pack_id"] not in _all_official_pack_ids()
        ]
    return {
        "output": {
            "schema": INTERPRETER_OUTPUT_SCHEMA,
            "schema_version": INTERPRETER_OUTPUT_SCHEMA_VERSION,
            "actions": ["request_context", "propose_edits", "needs_clarification", "unable"],
            "operations": ["set", "remove"],
            "maximum_operations": MAX_OPERATIONS,
            "path_form": "JSON array: [name], [metadata, field], or [section, identifier, field?]",
            "remove_requires_null_value": True,
        },
        "model": {
            "required": ["name"],
            "requires_content_in_one_of": [
                "functions",
                "vector_functions",
                "matrix_functions",
                "objects",
                "assets",
            ],
            "sections": list(_MODEL_SECTIONS),
            "entry_required_fields": {
                name: list(fields) for name, fields in _ENTRY_REQUIRED_FIELDS.items()
            },
            "entry_fields": {
                name: sorted(fields) for name, fields in _ENTRY_FIELDS.items()
            },
            "metadata_fields": sorted(_METADATA_FIELDS),
            "domain": "two finite numbers [lower, upper] with lower < upper",
            "constraint_relations": ["=", "==", "<", "<=", ">", ">="],
            "function_shorthand": "a function entry may be an expression string",
            "generated_schema_catalogue": _model_schema_catalogue(),
        },
        "expression_language": {
            "operators": ["+", "-", "*", "/", "**", "unary +", "unary -"],
            "constants": sorted(NAMED_CONSTANTS),
            "functions": sorted(FUNCTION_ARITIES),
        },
        "installed_model_object_kinds": kind_catalogue,
        "installed_capabilities": capabilities,
    }


@lru_cache(maxsize=1)
def _empty_context_contract_json() -> str:
    # Creation/clarification/unsupported requests have no current model and therefore share
    # exactly the same registry contract.  Serialize the cached value and parse a fresh copy on
    # use to preserve the historical no-shared-mutable-context property.
    return json.dumps(_build_context_contract(None), separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _context_contract(
    model: object | None = None, *, official_pack_ids: tuple[str, ...] = (),
    official_kind_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    if model is None and not official_pack_ids and not official_kind_ids:
        value = json.loads(_empty_context_contract_json())
        if not isinstance(value, dict):  # pragma: no cover - defensive invariant
            raise InterpreterProposalError("Cached interpreter context contract is malformed.")
        return value
    return _build_context_contract(
        model,
        official_pack_ids=official_pack_ids,
        official_kind_ids=official_kind_ids,
    )


def _all_official_pack_ids() -> set[str]:
    from .official_packs import OFFICIAL_PACK_MANIFESTS

    return {item.identifier for item in OFFICIAL_PACK_MANIFESTS}


def _official_pack_dependency_closure(pack_ids: tuple[str, ...]) -> set[str]:
    from .official_packs import OFFICIAL_PACK_MANIFESTS

    manifests = {item.identifier: item for item in OFFICIAL_PACK_MANIFESTS}
    selected = set(pack_ids)
    pending = list(pack_ids)
    while pending:
        for dependency in manifests[pending.pop()].dependencies:
            if dependency in manifests and dependency not in selected:
                selected.add(dependency)
                pending.append(dependency)
    return selected


_ROUTING_CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "shouldn't": "should not", "wouldn't": "would not",
    "couldn't": "could not", "won't": "will not", "can't": "can not",
    "cannot": "can not", "mustn't": "must not",
}
_ROUTING_SCOPE_BOUNDARIES = {"clausebreak", "but", "however", "instead", "yet", "then", "while", "whereas", "although", "though"}
_ROUTING_NEGATORS = {"not", "never", "without", "avoid", "avoiding", "no", "exclude", "excluding", "omit", "omitting"}
_ROUTING_OPERATION_TOKENS = {
    "optimise", "optimize", "optimisation", "optimization", "minimise", "minimize",
    "maximise", "maximize", "estimate", "estimation", "infer", "calibrate",
    "identify", "fit", "train", "training", "learn", "tune", "analyse", "analyze", "solve", "inverse",
}
_ROUTING_CREATION_TOKENS = {"create", "build", "construct", "make", "add", "define", "model", "setup", "set"}
_ROUTING_OBJECT_NEGATION_VERBS = {
    "create", "build", "construct", "make", "add", "define", "use", "include",
    "represent", "change", "modify", "remove", "delete", "replace", "convert",
}
_ROUTING_OBJECT_NEGATION_FILLERS = {"a", "an", "the", "this", "that", "any", "as", "to", "try", "attempt", "with", "via"}


def _routing_normalised_instruction(instruction: str) -> str:
    text = instruction.casefold().replace("’", "'")
    for contraction, expanded in _ROUTING_CONTRACTIONS.items():
        text = re.sub(rf"(?<![a-z0-9]){re.escape(contraction)}(?![a-z0-9])", expanded, text)
    # Alternative constructions explicitly exclude the following phrase.  Convert
    # them to the same local exclusion token used by other negators before tokenising.
    text = re.sub(r"\binstead\s+of\b", " exclude ", text)
    text = re.sub(r"\brather\s+than\b", " exclude ", text)
    text = re.sub(r"\bexcept\s+for\b", " exclude ", text)
    text = re.sub(r"\bas\s+opposed\s+to\b", " exclude ", text)
    text = re.sub(r"\bneither\b", " exclude ", text)
    # Preserve clause boundaries so local negation cannot suppress an affirmative
    # request in the next clause.  Commas are included because constructions such
    # as "without changing X, optimise Y" are common scientific instructions.
    text = re.sub(r"[.,;!?]+", " clausebreak ", text)
    tokens = re.findall(r"[a-z0-9]+", text)
    normalised = [_ROUTING_TOKEN_ALIASES.get(token, token) for token in tokens]
    return " ".join(normalised)


def _routing_term_present(normalised: str, term: str) -> bool:
    """Return whether a routing phrase occurs outside an explicit local negation scope."""
    tokens = normalised.split()
    needle = term.split()
    if not needle or len(needle) > len(tokens):
        return False
    for index in range(len(tokens) - len(needle) + 1):
        if tokens[index:index + len(needle)] != needle:
            continue
        boundary = 0
        for cursor in range(index - 1, -1, -1):
            if tokens[cursor] in _ROUTING_SCOPE_BOUNDARIES:
                boundary = cursor + 1
                break
        # A six-token local window covers ordinary formulations such as
        # "do not try to optimise", "no parameter estimation", "avoid graph
        # analysis", and "without attempting parameter estimation" without
        # letting negation leak across a long affirmative clause.
        start = max(boundary, index - 6)
        previous = tokens[start:index]
        needle_is_operation = any(token in _ROUTING_OPERATION_TOKENS for token in needle)
        negated = False
        for offset, token in enumerate(previous):
            if token not in _ROUTING_NEGATORS:
                continue
            # "not only X" is additive rather than a request to omit X.
            if token == "not" and offset + 1 < len(previous) and previous[offset + 1] == "only":
                continue
            between = previous[offset + 1:]
            if any(item in _ROUTING_CREATION_TOKENS for item in between):
                negated = True
                break
            if needle_is_operation:
                negated = True
                break
            # For domain/object nouns, distinguish a negated scientific operation
            # from a negated request to create/use/change the object itself.  Thus
            # "do not optimise the truss" still routes mechanics, while
            # "do not use a graph representation" and "avoid changing the graph"
            # suppress graph routing.
            meaningful = [item for item in between if item not in _ROUTING_OBJECT_NEGATION_FILLERS]
            coordinated_exclusion = any(item in {"and", "or", "nor"} for item in between)
            not_with_object = token == "not" and bool(between) and between[0] in {"with", "via"}
            if (
                len(between) <= 1
                or not meaningful
                or coordinated_exclusion
                or not_with_object
                or any(item in _ROUTING_OBJECT_NEGATION_VERBS for item in meaningful)
            ):
                negated = True
                break
        if negated:
            continue
        return True
    return False


def _existing_official_kind_selection(instruction: str, model: object | None) -> set[str]:
    """Select only existing official kinds that the edit request can actually target.

    Explicit object identifiers win.  Otherwise, a property-only edit may infer one
    existing kind when a property name uniquely identifies that kind in the current
    validated model.  This keeps rich-model edits narrow without falling back to every
    installed schema.
    """
    graph = getattr(model, "graph", None)
    objects = [
        item for item in getattr(graph, "objects", ())
        if _pack_for_official_kind(str(getattr(item, "kind", ""))) is not None
    ]
    if not objects:
        return set()

    lowered = instruction.casefold()
    mentioned: set[str] = set()
    for item in objects:
        identifier = str(getattr(item, "identifier", ""))
        if identifier and re.search(
            rf"(?<![A-Za-z0-9_.:-]){re.escape(identifier.casefold())}(?![A-Za-z0-9_.:-])",
            lowered,
        ):
            mentioned.add(str(getattr(item, "kind", "")))
    if mentioned:
        return mentioned

    normalised = _routing_normalised_instruction(instruction)
    padded = f" {normalised} "
    if any(
        phrase in padded
        for phrase in (
            " rename model ", " rename the model ", " rename this model ",
            " change model name ", " change the model name ", " model title ",
        )
    ):
        return set()

    # Infer a unique target from actual property names present on the existing objects.
    # Exact property spellings are intentionally preferred over stemming: in a mixed
    # generative model, ``emission`` identifies the HMM while ``emissions`` identifies
    # the POMDP. Shared properties such as ``states`` or ``initial`` remain ambiguous.
    property_candidates: set[str] = set()
    for item in objects:
        kind = str(getattr(item, "kind", ""))
        properties = getattr(item, "properties", {})
        if not isinstance(properties, Mapping):
            continue
        for property_name in properties:
            phrase = re.sub(r"[_-]+", " ", str(property_name).casefold()).strip()
            if phrase and _routing_term_present(normalised, phrase):
                property_candidates.add(kind)
                break
    if len(property_candidates) == 1:
        return property_candidates

    # A generic property edit is unambiguous only when all installed official
    # objects share one kind.  Rich mixed-pack models must not inject every
    # existing schema merely because those objects happen to be present.
    kinds = {str(getattr(item, "kind", "")) for item in objects}
    return kinds if len(kinds) == 1 else set()


def _pack_for_official_kind(kind: str) -> str | None:
    for pack_id, prefixes in _OFFICIAL_KIND_PREFIX_INDEX.items():
        if kind.startswith(prefixes):
            return pack_id
    return None


def _official_kind_context_selection(instruction: str, model: object | None) -> tuple[str, ...]:
    """Select exact official authoring kinds from ordinary scientific language."""
    normalised = _routing_normalised_instruction(instruction)
    padded = f" {normalised} "
    selected: set[str] = set()
    for kind, vocabulary in _OFFICIAL_KIND_ROUTE_INDEX.items():
        if any(_routing_term_present(normalised, term) for term in vocabulary):
            selected.add(kind)
    existing_selected = _existing_official_kind_selection(instruction, model)
    selected.update(existing_selected)

    graph_kind = "org.modellab.graph.network"
    electrical_kind = "org.modellab.electrical.linear-circuit"
    reaction_kind = "org.modellab.chemistry.mass-action-network"
    neural_kind = "org.modellab.learning.feedforward-network"
    pde_kind = "org.modellab.pde.poisson-problem"
    material_kind = "org.modellab.material.isotropic-linear-elastic"

    # Bare "network" is not a graph-theory signal.  Domain-qualified networks route only
    # to their scientific object kind unless explicit graph language is also present.
    graph_cues = (
        " graph ", " nodes and edges ", " vertices and edges ",
        " nodes and links ", " vertices and links ",
        " directed network ", " undirected network ", " weighted network ",
        " social network ", " citation network ", " transport network ", " road network ", " street network ",
        " shortest path ", " network centrality ", " graph laplacian ", " network laplacian ",
        " laplacian of the network ", " laplacian of this network ", " graph of this network ", " graph of the network ",
        " graph vertices ", " graph edges ",
    )
    routing_tokens = set(normalised.split())
    structural_node_edge_pair = (
        bool({"node", "nodes", "vertex", "vertices"} & routing_tokens)
        and bool({"edge", "edges", "link", "links"} & routing_tokens)
    )
    if structural_node_edge_pair:
        selected.add(graph_kind)
    if graph_kind not in existing_selected and not any(cue in padded for cue in graph_cues) and not structural_node_edge_pair:
        selected.discard(graph_kind)
    if any(_routing_term_present(normalised, phrase) for phrase in ("resistor network", "circuit network", "rlc", "r l c")):
        selected.add(electrical_kind)
        if not any(cue in padded for cue in graph_cues):
            selected.discard(graph_kind)
    if _routing_term_present(normalised, "reaction network"):
        selected.add(reaction_kind)
        if not any(cue in padded for cue in graph_cues):
            selected.discard(graph_kind)
    if (" hidden markov " in padded or " hmm " in padded) and " markov chain " not in padded:
        selected.discard("org.modellab.probability.markov-chain")

    # A transition matrix is normally a property of a Markov/HMM object rather than
    # a request to create a standalone multidimensional array.  Keep the array kind
    # only when the user separately asks for array/tensor storage semantics.
    explicit_array_cues = (" array ", " tensor ", " matrix entries ", " identity matrix ", " row major ")
    if any(cue in padded for cue in (" markov chain ", " hidden markov ", " hmm ")) and not any(
        cue in padded for cue in explicit_array_cues
    ):
        selected.discard("org.modellab.multidimensional.array")

    # "inverse temperature" is ordinary thermodynamic/statistical language, not an
    # inverse-problem request.
    if " inverse temperature " in padded and not any(
        cue in padded for cue in (" inverse problem ", " linear inverse ", " design matrix ")
    ):
        selected.discard("org.modellab.inverse.linear-problem")

    # Matrices that are fields of another scientific object do not imply a standalone
    # multidimensional-array authoring request.  Treat a matrix as standalone only when
    # the request explicitly creates/defines matrix storage or asks for array semantics.
    array_kind = "org.modellab.multidimensional.array"
    matrix_present = " matrix " in padded
    standalone_matrix_cue = any(
        _routing_term_present(normalised, cue)
        for cue in ("create matrix", "build matrix", "construct matrix", "make matrix", "add matrix", "define matrix")
    ) or any(cue in padded for cue in explicit_array_cues)
    other_domain_kind_selected = any(kind != array_kind for kind in selected)
    matrix_domain_context = other_domain_kind_selected or any(cue in padded for cue in (
        " statistical model ", " statistics ", " statistical ", " regression ", " pca ",
        " classification ", " classifier ", " truss ", " structural ", " mechanics ",
        " ode ", " differential equation ", " dynamical ", " optimisation ", " optimization ",
        " estimation ", " graph ", " network ", " node ", " vertex ", " edge ", " link ",
    ))
    if matrix_present and matrix_domain_context and not standalone_matrix_cue:
        selected.discard(array_kind)

    # A design matrix in ordinary regression is a statistical-model property, not
    # evidence that the user is formulating an inverse problem.
    if " design matrix " in padded and any(
        cue in padded for cue in (" linear regression ", " regression ", " linear model ")
    ) and not any(
        _routing_term_present(normalised, cue)
        for cue in ("inverse problem", "linear inverse", "parameter estimation", "estimate", "calibrate")
    ):
        selected.discard("org.modellab.inverse.linear-problem")

    # "species" is shared by ecology and chemistry.  Ecological/population context
    # wins unless reaction-specific chemistry language is also present.
    if any(cue in padded for cue in (" ecological ", " ecology ", " population ", " predator prey ")) and not any(
        cue in padded for cue in (" chemical reaction ", " reaction network ", " reaction rate ", " rate constant ", " mass action ", " chemical kinetics ")
    ):
        selected.discard("org.modellab.chemistry.mass-action-network")

    # "objective" can be an adjective in measurement language.  Do not treat
    # "objective measurement" as an optimisation problem without an optimisation cue.
    if " objective measurement " in padded and not any(
        _routing_term_present(normalised, cue)
        for cue in ("optimise", "optimize", "minimise", "minimize", "maximise", "maximize", "objective function", "constraints")
    ):
        selected.discard("org.modellab.optimisation.nonlinear-problem")

    # "graph" is also the ordinary verb/noun for plotting scientific results.
    # Plotting language must not expose graph theory unless network structure is
    # independently requested.
    explicit_network_graph_context = any(cue in padded for cue in (
        " graph of this network ", " graph of the network ", " social network ",
        " citation network ", " transport network ", " road network ", " street network ",
        " shortest path ", " network centrality ", " graph laplacian ", " network laplacian ",
        " laplacian of the network ", " laplacian of this network ",
    )) or structural_node_edge_pair
    plotting_targets = {"function", "curve", "trajectory", "residual", "residuals", "results", "result", "timeseries", "time", "portrait", "solution", "solutions", "data"}
    plotting_graph_context = (
        any(cue in padded for cue in (" plot the graph ", " plot a graph ", " draw the graph ", " draw a graph ", " sketch the graph ", " sketch a graph ", " graph this function ", " graph the function ", " graph y ", " graph x "))
        or (normalised.startswith(("graph ", "draw ", "sketch ")) and "graph" in routing_tokens and bool(plotting_targets & routing_tokens))
        or any(cue in padded for cue in (" graph the trajectory ", " graph the phase portrait ", " graph the residual ", " graph the residuals ", " graph the results "))
    )
    if plotting_graph_context and not explicit_network_graph_context:
        selected.discard(graph_kind)

    # "population" in statistical sampling language is not a biological population
    # interaction model.  Biological cues win when they are explicit.
    statistical_population_context = any(cue in padded for cue in (
        " population mean ", " population variance ", " population sample ", " population proportion ",
        " population standard deviation ", " population std deviation ", " population quantile ",
        " population percentile ", " population median ", " population estimator ",
        " statistical population ", " sample from the population ", " population parameter ",
    )) or (" population " in padded and any(cue in padded for cue in (" statistics ", " statistical ", " dataset ", " sample ", " confidence interval ", " standard error ", " estimate ", " estimation ", " proportion ", " quantile ", " percentile ")))
    biological_population_cues = (" population interaction ", " intrinsic growth ", " predator prey ", " ecology ", " ecological ", " lotka volterra ")
    if statistical_population_context and not any(cue in padded for cue in biological_population_cues):
        selected.discard("org.modellab.biological.population-interaction-system")
    positive_neural_cue = any(
        _routing_term_present(normalised, phrase)
        for phrase in ("neural network", "neural controller", "graph neural network", "feedforward network", "feed forward network")
    )
    if positive_neural_cue:
        selected.add(neural_kind)
        # "graph neural network" names a neural architecture, not a generic graph object,
        # unless the request separately asks for graph nodes/edges.
        if _routing_term_present(normalised, "graph neural network") and not any(cue in padded for cue in graph_cues[1:]):
            selected.discard(graph_kind)

    # Scientific homonyms and concise PDE language.
    if " poisson ratio " in padded and not any(
        phrase in padded for phrase in (" poisson equation ", " poisson problem ", " pde ")
    ):
        selected.discard(pde_kind)
        selected.add(material_kind)
    elif " poisson " in padded or " laplace equation " in padded or " laplace s equation " in padded or " laplacian " in padded:
        graph_laplacian_property = any(cue in padded for cue in (
            " laplacian matrix ", " graph laplacian ", " network laplacian ",
            " laplacian of the network ", " laplacian of this network ",
        )) and (graph_kind in selected or any(cue in padded for cue in graph_cues) or " network " in padded)
        if not graph_laplacian_property:
            selected.add(pde_kind)
        else:
            selected.add(graph_kind)
            selected.discard(pde_kind)

    linear_model_kind = "org.modellab.statistics.linear-model-study"
    if _routing_term_present(normalised, "linear model") and any(
        _routing_term_present(normalised, cue)
        for cue in ("fit", "dataset", "data", "measurement", "regression")
    ):
        selected.add(linear_model_kind)

    estimation_kind = "org.modellab.estimation.nonlinear-least-squares"
    if any(
        _routing_term_present(normalised, term)
        for term in (
            "calibrate", "calibration", "identify parameter",
            "infer parameters", "infer parameter",
        )
    ) or (
        _routing_term_present(normalised, "infer")
        and _routing_term_present(normalised, "parameter")
    ) or (
        _routing_term_present(normalised, "identify")
        and _routing_term_present(normalised, "parameter")
    ) or (
        _routing_term_present(normalised, "fit")
        and any(_routing_term_present(normalised, cue) for cue in ("ode", "differential equation", "dynamical system"))
    ):
        selected.add(estimation_kind)

    return tuple(kind for kind in _OFFICIAL_KIND_ROUTE_INDEX if kind in selected)


def _official_pack_context_selection(instruction: str, model: object | None) -> tuple[str, ...]:
    """Return primary official packs implied by exact kind routing plus broad pack cues."""
    normalised = _routing_normalised_instruction(instruction)
    padded = f" {normalised} "
    routed_kinds = _official_kind_context_selection(instruction, model)
    selected = {
        pack_id
        for kind in routed_kinds
        if (pack_id := _pack_for_official_kind(kind)) is not None
    }
    for pack_id, vocabulary in _OFFICIAL_PACK_ROUTE_INDEX.items():
        if any(_routing_term_present(normalised, term) for term in vocabulary):
            selected.add(pack_id)

    graph_pack = "org.modellab.pack.graphs-networks-discrete"
    structural_graph_cues = (
        " nodes and edges ", " vertices and edges ", " nodes and links ", " vertices and links ",
        " directed network ", " undirected network ", " weighted network ", " social network ", " citation network ", " transport network ", " road network ", " street network ",
        " shortest path ", " network centrality ", " graph laplacian ", " network laplacian ", " laplacian of the network ", " laplacian of this network ",
        " graph of this network ", " graph of the network ", " graph vertices ", " graph edges ", " arcs ",
    )
    graph_kind_selected = "org.modellab.graph.network" in routed_kinds
    routing_tokens = set(normalised.split())
    structural_node_edge_pair = (
        bool({"node", "nodes", "vertex", "vertices"} & routing_tokens)
        and bool({"edge", "edges", "link", "links"} & routing_tokens)
    )
    if structural_node_edge_pair:
        selected.add(graph_pack)
    if not graph_kind_selected and not (
        " graph " in padded or any(cue in padded for cue in structural_graph_cues) or structural_node_edge_pair
    ):
        selected.discard(graph_pack)
    if (
        not graph_kind_selected
        and " graph neural network " in padded
        and not any(cue in padded for cue in structural_graph_cues)
    ):
        selected.discard(graph_pack)

    # Verb forms are common in actual requests and should compose with domain packs.
    if any(
        _routing_term_present(normalised, term)
        for term in (
            "optimize", "optimise", "optimization", "optimisation",
            "minimize", "minimise", "maximize", "maximise",
        )
    ):
        selected.add("org.modellab.pack.optimisation-estimation-inverse-problems")
    if any(
        _routing_term_present(normalised, term)
        for term in (
            "estimate", "estimation", "fit parameters", "parameter estimation",
            "infer parameters", "infer parameter", "calibrate", "calibration",
            "identify parameter",
        )
    ) or (
        _routing_term_present(normalised, "infer")
        and _routing_term_present(normalised, "parameter")
    ) or (
        _routing_term_present(normalised, "identify")
        and _routing_term_present(normalised, "parameter")
    ) or (
        _routing_term_present(normalised, "fit")
        and any(_routing_term_present(normalised, cue) for cue in ("ode", "differential equation", "dynamical system"))
    ):
        selected.add("org.modellab.pack.optimisation-estimation-inverse-problems")
    if any(_routing_term_present(normalised, term) for term in ("laplace equation", "laplace s equation", "laplacian", "elliptic pde")):
        selected.add("org.modellab.pack.spatial-fields-continuum-pdes")
    if any(_routing_term_present(normalised, term) for term in ("lti", "linear time invariant", "lti controller")):
        selected.add("org.modellab.pack.dynamics-differential-equations-control")
    if any(_routing_term_present(normalised, term) for term in ("rlc", "r l c", "resistor network", "circuit network", "electrical network")):
        selected.add("org.modellab.pack.electrical-electronic-electromagnetic-systems")

    # Avoid pack-level over-routing from property vocabulary that belongs inside
    # another selected scientific object.
    if any(cue in padded for cue in (" markov chain ", " hidden markov ", " hmm ")) and not any(
        cue in padded for cue in (" array ", " tensor ", " matrix entries ", " identity matrix ", " row major ")
    ):
        selected.discard("org.modellab.pack.multidimensional-mathematics")
    if (" hidden markov " in padded or " hmm " in padded) and " markov chain " not in padded:
        selected.discard("org.modellab.pack.probability-stochastic-systems")
    if " inverse temperature " in padded and not any(
        cue in padded for cue in (" inverse problem ", " linear inverse ", " design matrix ")
    ):
        selected.discard("org.modellab.pack.optimisation-estimation-inverse-problems")

    array_pack = "org.modellab.pack.multidimensional-mathematics"
    array_kind_selected = "org.modellab.multidimensional.array" in routed_kinds
    if " matrix " in padded and not array_kind_selected and not any(
        cue in padded for cue in (" array ", " tensor ", " matrix entries ", " identity matrix ", " row major ")
    ):
        selected.discard(array_pack)
    if " design matrix " in padded and any(
        cue in padded for cue in (" linear regression ", " regression ", " linear model ")
    ) and not any(
        _routing_term_present(normalised, cue)
        for cue in ("inverse problem", "linear inverse", "parameter estimation", "estimate", "calibrate")
    ):
        # The statistics pack remains selected; remove the inverse/estimation pack
        # only when no separate positive estimation/optimisation language exists.
        selected.discard("org.modellab.pack.optimisation-estimation-inverse-problems")
    if " objective measurement " in padded and not any(
        _routing_term_present(normalised, cue)
        for cue in ("optimise", "optimize", "minimise", "minimize", "maximise", "maximize", "objective function", "constraints")
    ):
        selected.discard("org.modellab.pack.optimisation-estimation-inverse-problems")

    explicit_network_graph_context = any(cue in padded for cue in (
        " graph of this network ", " graph of the network ", " social network ",
        " citation network ", " transport network ", " road network ", " street network ",
        " shortest path ", " network centrality ", " graph laplacian ", " network laplacian ",
        " laplacian of the network ", " laplacian of this network ",
    )) or structural_node_edge_pair
    plotting_targets = {"function", "curve", "trajectory", "residual", "residuals", "results", "result", "timeseries", "time", "portrait", "solution", "solutions", "data"}
    plotting_graph_context = (
        any(cue in padded for cue in (" plot the graph ", " plot a graph ", " draw the graph ", " draw a graph ", " sketch the graph ", " sketch a graph ", " graph this function ", " graph the function ", " graph y ", " graph x "))
        or (normalised.startswith(("graph ", "draw ", "sketch ")) and "graph" in routing_tokens and bool(plotting_targets & routing_tokens))
        or any(cue in padded for cue in (" graph the trajectory ", " graph the phase portrait ", " graph the residual ", " graph the residuals ", " graph the results "))
    )
    if plotting_graph_context and not explicit_network_graph_context:
        selected.discard(graph_pack)

    statistical_population_context = any(cue in padded for cue in (
        " population mean ", " population variance ", " population sample ", " population proportion ",
        " population standard deviation ", " population std deviation ", " population quantile ",
        " population percentile ", " population median ", " population estimator ",
        " statistical population ", " sample from the population ", " population parameter ",
    )) or (" population " in padded and any(cue in padded for cue in (" statistics ", " statistical ", " dataset ", " sample ", " confidence interval ", " standard error ", " estimate ", " estimation ", " proportion ", " quantile ", " percentile ")))
    biological_population_cues = (" population interaction ", " intrinsic growth ", " predator prey ", " ecology ", " ecological ", " lotka volterra ")
    if statistical_population_context and not any(cue in padded for cue in biological_population_cues):
        selected.discard("org.modellab.pack.chemical-reaction-biological-systems")

    # A graph Laplacian is a graph property, not a PDE request unless a PDE cue is
    # independently present.
    graph_laplacian_property = any(cue in padded for cue in (
        " laplacian matrix ", " graph laplacian ", " network laplacian ",
        " laplacian of the network ", " laplacian of this network ",
    )) and (graph_pack in selected or " network " in padded or " graph " in padded)
    if graph_laplacian_property and not any(
        cue in padded for cue in (" pde ", " partial differential equation ", " poisson equation ", " laplace equation ", " elliptic pde ")
    ):
        selected.add(graph_pack)
        selected.discard("org.modellab.pack.spatial-fields-continuum-pdes")

    return tuple(pack_id for pack_id in _OFFICIAL_PACK_ROUTE_INDEX if pack_id in selected)


def scientific_request_boundary(instruction: str) -> str | None:
    """Return a deterministic reason when a request exceeds installed scientific contracts.

    Routing multiple object schemas into one prompt does not create a composed numerical
    capability. These checks prevent close-but-incorrect substitutions such as representing
    an SDE as an ODE, a graph neural network as a fixed dense feed-forward network, or a
    mesh-based PDE solve as an unrelated mesh plus uniform-grid Poisson problem.
    """
    normalised = _routing_normalised_instruction(instruction)
    padded = f" {normalised} "

    def present(*terms: str) -> bool:
        return any(_routing_term_present(normalised, term) for term in terms)

    if present("stochastic differential equation", "stochastic differential equations", "sde"):
        return (
            "Stochastic differential equations are not installed; an ordinary ODE would not "
            "preserve the requested stochastic dynamics."
        )

    unsupported_neural_specialisation = present(
        "graph neural network", "convolutional neural network", "convolutional network",
        "recurrent neural network", "recurrent network", "transformer network", "lstm", "gru",
    )
    neural_language = present("neural", "neural network", "feedforward network", "mlp")
    neural_training = neural_language and present(
        "train", "training", "learn weights", "fit weights", "tune weights",
    )
    if unsupported_neural_specialisation or neural_training:
        return (
            "The installed learning pack supports inference by an explicitly weighted dense "
            "feed-forward network, but not neural-network training or specialised CNN, GNN, "
            "RNN, LSTM, GRU, or transformer architectures."
        )
    if present("neural controller"):
        return (
            "A closed-loop neural-controller capability is not installed; fixed network "
            "inference and dynamical-system simulation cannot be coupled implicitly."
        )

    optimisation_requested = present(
        "optimise", "optimize", "optimisation", "optimization", "minimise", "minimize",
        "maximise", "maximize",
    )
    structural_model = present("truss", "truss structure", "structural frame", "structure")
    if optimisation_requested and structural_model:
        return (
            "Structural optimisation is not installed; the truss solver and generic nonlinear "
            "optimiser are independent capabilities and cannot be presented as a coupled solve."
        )

    parameter_operation = present("estimate", "infer", "calibrate", "calibration") or any(
        cue in padded
        for cue in (
            " identify parameter ", " identify the parameter ",
            " identify coefficient ", " identify the coefficient ",
            " identify rate ", " identify the rate ",
        )
    )
    parameter_language = present("parameter", "parameters", "coefficient", "coefficients", "rate", "rates")
    fit_to_evidence = present("fit") and any(
        cue in padded for cue in (
            " to data ", " to observations ", " to measurements ", " to a time series ",
            " to time series ", " from data ", " from observations ", " from measurements ",
        )
    )
    solver_model = present(
        "ode", "ordinary differential equation", "differential equation", "dynamical system",
        "state space", "reaction network", "mass action", "population interaction",
        "compartment model", "circuit", "rlc", "truss",
    )
    if solver_model and ((parameter_operation and parameter_language) or fit_to_evidence):
        return (
            "Solver-coupled parameter estimation is not installed; nonlinear least squares "
            "accepts explicit residual expressions but cannot call an ODE, reaction, circuit, "
            "state-space, population, compartment, or truss solver inside its residual loop."
        )

    pde_requested = present("laplace equation", "laplace s equation", "poisson equation", "elliptic pde")
    mesh_requested = present("mesh", "triangle mesh", "triangular mesh", "unstructured mesh", "finite element", "fem")
    if pde_requested and mesh_requested:
        return (
            "Mesh-based PDE or finite-element solving is not installed; the current Poisson "
            "solver uses a uniform rectilinear grid and does not consume triangle meshes."
        )
    return None


def _contract_budget_variant(contract: Mapping[str, Any], level: int) -> dict[str, Any]:
    """Return deterministic progressively smaller contract variants for transport fitting."""
    result = deepcopy(dict(contract))
    if level >= 1:
        result["installed_capabilities"] = []
    if level >= 2:
        model_contract = dict(result["model"])
        catalogue = model_contract.get("generated_schema_catalogue")
        if isinstance(catalogue, Mapping):
            model_contract["generated_schema_catalogue"] = {
                "authoritative_schema_sha256": catalogue.get("authoritative_schema_sha256"),
                "omitted_for_context_budget": True,
            }
        result["model"] = model_contract
    return result


def _clarification_history_projection(
    history: Sequence[Mapping[str, str]], *, compact_oldest: int
) -> list[dict[str, str]]:
    """Digest-project the oldest turns while preserving turn count and full-history binding."""
    if compact_oldest < 0 or compact_oldest > len(history):
        raise InterpreterProposalError("Invalid clarification-history compaction request.")
    projected: list[dict[str, str]] = []
    for index, item in enumerate(history):
        question = str(item["question"])
        answer = str(item["answer"])
        if index < compact_oldest:
            q_bytes = question.encode("utf-8")
            a_bytes = answer.encode("utf-8")
            question = (
                "[older clarification question omitted for context budget; "
                f"bytes={len(q_bytes)}; sha256={hashlib.sha256(q_bytes).hexdigest()}]"
            )
            answer = (
                "[older clarification answer omitted for context budget; "
                f"bytes={len(a_bytes)}; sha256={hashlib.sha256(a_bytes).hexdigest()}]"
            )
        projected.append({"question": question, "answer": answer})
    return projected


def _fit_contract_to_context_overhead(
    contract: Mapping[str, Any], *, common_without_contract: Mapping[str, Any], model_reserve_bytes: int = 4096
) -> dict[str, Any]:
    """Fit the contract after history/attachment overhead is known, not before it."""
    for level in range(3):
        candidate = _contract_budget_variant(contract, level)
        envelope = {**common_without_contract, "contract": candidate, "model": {"scope": "budget-reserve"}}
        if len(_canonical_json_bytes(envelope)) + model_reserve_bytes <= MAX_CONTEXT_PACKAGE_BYTES:
            return candidate
    raise InterpreterProposalError(
        "The fixed interpreter context cannot fit required clarification/attachment evidence even after deterministic contract compaction."
    )


def _finalise_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["context_sha256"] = canonical_json_sha256(result)
    if len(_canonical_json_bytes(result)) > MAX_CONTEXT_PACKAGE_BYTES:
        raise InterpreterProposalError(
            "The deterministic interpreter context exceeded its fixed transport budget."
        )
    return result


def _inventory(section: Mapping[str, Any], instruction_names: set[str]) -> dict[str, Any]:
    names = sorted(str(name) for name in section)
    eligible = [name for name in names if len(name.encode("utf-8")) <= MAX_PATH_COMPONENT_BYTES]
    mentioned = [name for name in eligible if name in instruction_names]
    samples: list[str] = []
    for name in [*mentioned, *eligible]:
        if name not in samples:
            samples.append(name)
        if len(samples) == 16:
            break
    return {
        "count": len(names),
        "names_sha256": _sha256_text("\x00".join(names)),
        "sample_names": samples,
        "unaddressable_long_name_count": len(names) - len(eligible),
    }


def _attachment_records(values: object) -> tuple[dict[str, Any], ...]:
    if values is None:
        return ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise InterpreterProposalError("Interpreter attachments must be a list of records.")
    if len(values) > MAX_ACTIVE_ATTACHMENTS:
        raise InterpreterProposalError(
            f"At most {MAX_ACTIVE_ATTACHMENTS} reference attachments may be active."
        )
    try:
        records = tuple(
            sorted(
                (validate_attachment_record(item) for item in values),
                key=lambda item: item["attachment_id"],
            )
        )
    except InterpreterAttachmentError as exc:
        raise InterpreterProposalError(str(exc)) from exc
    identifiers = [item["attachment_id"] for item in records]
    if len(identifiers) != len(set(identifiers)):
        raise InterpreterProposalError("Interpreter attachments contain duplicate files.")
    return records


def _attachment_manifest_evidence(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    manifests = [attachment_manifest(item) for item in records]
    return manifests, canonical_json_sha256(manifests)


def _attachment_projection(
    records: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "manifest_sha256": manifest_sha256,
        "documents": [attachment_manifest(item) for item in records],
        "request_path_form": ["@attachment", "attachment_id", "zero_based_chunk_index"],
        "full_text_sent_automatically": False,
        "included_chunks": [],
    }


def _validated_attachment_evidence(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "manifest_sha256",
        "documents",
        "disclosed_chunks",
        "disclosed_chunks_sha256",
    }:
        raise InterpreterProposalError("Interpreter attachment evidence fields are invalid.")
    try:
        documents = validate_attachment_manifests(value.get("documents"))
    except InterpreterAttachmentError as exc:
        raise InterpreterProposalError(str(exc)) from exc
    manifest_sha = _require_sha256(
        value.get("manifest_sha256"), "Interpreter attachment manifest_sha256"
    )
    if manifest_sha != canonical_json_sha256(documents):
        raise InterpreterProposalError("Interpreter attachment manifest checksum is invalid.")
    chunks = value.get("disclosed_chunks")
    if not isinstance(chunks, list):
        raise InterpreterProposalError("Interpreter disclosed attachment chunks must be a list.")
    by_id = {item["attachment_id"]: item for item in documents}
    normalised_chunks: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for index, item in enumerate(chunks):
        if not isinstance(item, Mapping) or set(item) != {
            "path", "text_bytes", "text_sha256"
        }:
            raise InterpreterProposalError(
                f"Interpreter disclosed attachment chunk {index + 1} is invalid."
            )
        path = _validated_context_request(
            item.get("path"), f"Interpreter disclosed attachment chunk {index + 1} path"
        )
        if path[0] != "@attachment" or path[1] not in by_id:
            raise InterpreterProposalError(
                "Interpreter disclosed attachment chunk is not bound to its manifest."
            )
        if int(path[2]) >= by_id[path[1]]["chunking"]["chunk_count"]:
            raise InterpreterProposalError(
                "Interpreter disclosed attachment chunk index is outside its manifest."
            )
        text_bytes = item.get("text_bytes")
        if (
            isinstance(text_bytes, bool)
            or not isinstance(text_bytes, int)
            or not 0 <= text_bytes <= ATTACHMENT_CHUNK_BYTES
        ):
            raise InterpreterProposalError(
                "Interpreter disclosed attachment chunk byte count is invalid."
            )
        text_sha = _require_sha256(
            item.get("text_sha256"),
            f"Interpreter disclosed attachment chunk {index + 1} text_sha256",
        )
        if path in seen:
            raise InterpreterProposalError("Interpreter attachment evidence has duplicate chunks.")
        seen.add(path)
        normalised_chunks.append(
            {"path": list(path), "text_bytes": text_bytes, "text_sha256": text_sha}
        )
    normalised_chunks.sort(key=lambda item: tuple(item["path"]))
    disclosed_sha = _require_sha256(
        value.get("disclosed_chunks_sha256"),
        "Interpreter attachment disclosed_chunks_sha256",
    )
    if disclosed_sha != canonical_json_sha256(normalised_chunks):
        raise InterpreterProposalError(
            "Interpreter disclosed attachment chunk checksum is invalid."
        )
    return {
        "manifest_sha256": manifest_sha,
        "documents": documents,
        "disclosed_chunks": normalised_chunks,
        "disclosed_chunks_sha256": disclosed_sha,
    }


def _context_attachment_evidence(context: Mapping[str, Any]) -> dict[str, Any]:
    projection = context.get("attachments")
    if projection is None:
        documents: list[dict[str, Any]] = []
        manifest_sha = canonical_json_sha256(documents)
        chunks: list[dict[str, Any]] = []
    else:
        if not isinstance(projection, Mapping):
            raise InterpreterProposalError("Interpreter attachment context is malformed.")
        documents = list(projection.get("documents", []))
        manifest_sha = str(projection.get("manifest_sha256", ""))
        included = projection.get("included_chunks")
        if not isinstance(included, list):
            raise InterpreterProposalError("Interpreter attachment chunks are malformed.")
        chunks = [
            {
                "path": item.get("path"),
                "text_bytes": item.get("text_bytes"),
                "text_sha256": item.get("text_sha256"),
            }
            for item in included
            if isinstance(item, Mapping)
        ]
    return _validated_attachment_evidence(
        {
            "manifest_sha256": manifest_sha,
            "documents": documents,
            "disclosed_chunks": chunks,
            "disclosed_chunks_sha256": canonical_json_sha256(
                sorted(chunks, key=lambda item: tuple(item["path"]))
            ),
        }
    )


def _attachment_chunk_candidates(
    *,
    records: Sequence[Mapping[str, Any]],
    requested: Sequence[tuple[str, ...]],
    instruction: str,
) -> list[tuple[str, int]]:
    candidates: list[tuple[str, int]] = [
        (path[1], int(path[2])) for path in requested if path[0] == "@attachment"
    ]
    terms = {
        item.casefold()
        for item in re.findall(r"[A-Za-z0-9_]{4,}", instruction)
        if item.casefold() not in {"this", "that", "with", "from", "model", "file"}
    }
    for record in records:
        identifier = str(record["attachment_id"])
        chunks = attachment_chunks(record)
        if len(chunks) <= 2:
            suggested = range(len(chunks))
        else:
            scored = sorted(
                (
                    (
                        sum(chunk.casefold().count(term) for term in terms),
                        index,
                    )
                    for index, chunk in enumerate(chunks)
                ),
                key=lambda item: (-item[0], item[1]),
            )
            suggested = [0, *[index for score, index in scored if score > 0][:2]]
        for index in suggested:
            candidate = (identifier, int(index))
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _context_with_attachment_chunks(
    payload: dict[str, Any],
    *,
    records: Sequence[Mapping[str, Any]],
    requested: Sequence[tuple[str, ...]],
    instruction: str,
    strict_requested: bool,
) -> dict[str, Any] | None:
    if not records:
        try:
            return _finalise_context(payload)
        except InterpreterProposalError:
            return None
    try:
        _finalise_context(payload)
    except InterpreterProposalError:
        return None
    by_id = {str(item["attachment_id"]): item for item in records}
    required = {
        (path[1], int(path[2])) for path in requested if path[0] == "@attachment"
    }
    included = payload["attachments"]["included_chunks"]
    for identifier, index in _attachment_chunk_candidates(
        records=records,
        requested=requested,
        instruction=instruction,
    ):
        text = attachment_chunks(by_id[identifier])[index]
        encoded = text.encode("utf-8")
        entry = {
            "path": ["@attachment", identifier, str(index)],
            "text": text,
            "text_bytes": len(encoded),
            "text_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        included.append(entry)
        try:
            _finalise_context(payload)
        except InterpreterProposalError:
            included.pop()
            if (identifier, index) in required:
                if strict_requested:
                    raise InterpreterProposalError(
                        "Requested attachment chunks do not fit the bounded context; request fewer chunks."
                    )
                return None
    return _finalise_context(payload)


def create_interpreter_context(
    *,
    instruction: str,
    current_model_source: str,
    requested_paths: Sequence[Sequence[str]] = (),
    round_index: int = 0,
    clarification_history: Sequence[Mapping[str, str]] = (),
    attachments: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Create the exact bounded context sent to the local model.

    A small source is included completely. A large source is represented by deterministic
    inventories, exact instruction-matched entries, and exact paths requested in prior
    rounds. The complete source is never truncated into apparently complete YAML.
    """
    instruction = _bounded_text(instruction, "Interpreter instruction", MAX_INSTRUCTION_BYTES)
    current_model_source = _bounded_text(
        current_model_source,
        "Current model source",
        MAX_MODEL_SOURCE_BYTES,
        allow_empty=True,
    )
    if isinstance(round_index, bool) or not isinstance(round_index, int):
        raise InterpreterProposalError("Interpreter context round must be an integer.")
    if not 0 <= round_index < MAX_CONTEXT_ROUNDS:
        raise InterpreterProposalError(
            f"Interpreter context round must be between 0 and {MAX_CONTEXT_ROUNDS - 1}."
        )
    if len(requested_paths) > MAX_CONTEXT_REQUESTS * MAX_CONTEXT_ROUNDS:
        raise InterpreterProposalError("Interpreter context request history is too large.")
    history = validate_clarification_history(clarification_history)
    attachment_records = _attachment_records(attachments)
    manifests, manifest_sha256 = _attachment_manifest_evidence(attachment_records)

    source_data = _load_source_data(current_model_source)
    context_model = (
        validate_model(parse_model_data(source_data)) if source_data else None
    )
    normalised_requests = sorted(
        {
            _validated_context_request(list(path), "Interpreter requested path")
            for path in requested_paths
        }
    )
    model_requests = [path for path in normalised_requests if path[0] != "@attachment"]
    attachment_requests = [
        path for path in normalised_requests if path[0] == "@attachment"
    ]
    if len(attachment_requests) > MAX_ATTACHMENT_CONTEXT_REQUESTS:
        raise InterpreterProposalError(
            f"At most {MAX_ATTACHMENT_CONTEXT_REQUESTS} attachment chunks may be requested per context history."
        )
    by_attachment_id = {
        str(record["attachment_id"]): record for record in attachment_records
    }
    for path in attachment_requests:
        record = by_attachment_id.get(path[1])
        if record is None:
            raise InterpreterProposalError(
                "Requested interpreter attachment is not active: " + path[1] + "."
            )
        if int(path[2]) >= int(record["chunking"]["chunk_count"]):
            raise InterpreterProposalError(
                "Requested interpreter attachment chunk does not exist: "
                + ".".join(path)
                + "."
            )
    for path in model_requests:
        if not _path_exists(source_data, path):
            raise InterpreterProposalError(
                "Requested interpreter context path does not exist: " + ".".join(path) + "."
            )

    selected_official_kinds = _official_kind_context_selection(instruction, context_model)
    selected_official_packs = _official_pack_context_selection(instruction, context_model)
    contract = _context_contract(
        context_model,
        official_pack_ids=selected_official_packs,
        official_kind_ids=selected_official_kinds,
    )
    boundary_reason = scientific_request_boundary(instruction)
    if boundary_reason is not None:
        contract = {
            **contract,
            "request_boundary": {
                "status": "unsupported-by-installed-capabilities",
                "required_action": "unable",
                "reason": boundary_reason,
            },
        }
    if attachment_records:
        contract = {
            **contract,
            "reference_attachments": {
                "maximum_active": MAX_ACTIVE_ATTACHMENTS,
                "request_path_form": [
                    "@attachment",
                    "attachment_id",
                    "zero_based_chunk_index",
                ],
                "rule": (
                    "Use context_requests only for exact omitted chunks. Attachment paths "
                    "are reference-only and must never be used as edit operation paths."
                ),
            },
        }
    history_sha256 = canonical_json_sha256(history)
    common_base: dict[str, Any] = {
        "schema": INTERPRETER_CONTEXT_SCHEMA,
        "schema_version": (
            INTERPRETER_CONTEXT_SCHEMA_VERSION
            if attachment_records
            else PREVIOUS_INTERPRETER_CONTEXT_SCHEMA_VERSION
        ),
        "round": round_index,
        "instruction_sha256": _sha256_text(instruction),
        "current_model_source_sha256": _sha256_text(current_model_source),
        "mode": "edit" if current_model_source.strip() else "create",
        "requested_paths": [list(path) for path in normalised_requests],
        # This always binds the complete validated history, even if some oldest model-visible
        # turns must later be digest-projected solely to satisfy the fixed transport budget.
        "clarification_history_sha256": history_sha256,
    }
    if attachment_records:
        common_base["attachments"] = _attachment_projection(
            attachment_records, manifest_sha256
        )

    # First exhaust deterministic contract compaction with the complete history.  Only when
    # accepted attachment/history inputs still cannot fit do we digest-project the minimum
    # number of oldest turns.  Turn count and the full-history SHA remain unchanged.
    fit_error: InterpreterProposalError | None = None
    common_without_contract: dict[str, Any] | None = None
    fitted_contract: dict[str, Any] | None = None
    for compact_oldest in range(len(history) + 1):
        projected_history = _clarification_history_projection(
            history, compact_oldest=compact_oldest
        )
        candidate_common = {
            **common_base,
            "clarification_history": projected_history,
        }
        try:
            candidate_contract = _fit_contract_to_context_overhead(
                contract,
                common_without_contract=candidate_common,
                model_reserve_bytes=(512 if not source_data else 4096),
            )
        except InterpreterProposalError as exc:
            fit_error = exc
            continue
        common_without_contract = candidate_common
        fitted_contract = candidate_contract
        break
    if common_without_contract is None or fitted_contract is None:
        if fit_error is not None:
            raise fit_error
        raise InterpreterProposalError("The fixed interpreter context cannot be constructed.")
    common: dict[str, Any] = {**common_without_contract, "contract": fitted_contract}

    complete = {
        **common,
        "model": {
            "scope": "complete",
            "document": source_data,
            "visible_paths": [],
            "unavailable_values": [],
        },
    }
    completed = _context_with_attachment_chunks(
        complete,
        records=attachment_records,
        requested=normalised_requests,
        instruction=instruction,
        strict_requested=False,
    )
    if completed is not None:
        return completed

    instruction_names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", instruction))
    sections: dict[str, Any] = {}
    for section_name in _MODEL_SECTIONS:
        section = source_data.get(section_name, {})
        sections[section_name] = _inventory(
            section if isinstance(section, Mapping) else {}, instruction_names
        )

    raw_name = source_data.get("name")
    if isinstance(raw_name, str) and len(raw_name.encode("utf-8")) <= 512:
        name_summary: object = raw_name
    elif isinstance(raw_name, str):
        name_summary = {
            "bytes": len(raw_name.encode("utf-8")),
            "sha256": _sha256_text(raw_name),
            "value_omitted": True,
        }
    else:
        name_summary = None

    partial: dict[str, Any] = {
        **common,
        "model": {
            "scope": "partial",
            "name": name_summary,
            "sections": sections,
            "included_values": [],
            "visible_paths": (
                [["name"]]
                if isinstance(raw_name, str)
                and not isinstance(name_summary, Mapping)
                else []
            ),
            "unavailable_values": [],
        },
    }

    candidates: list[tuple[str, ...]] = list(model_requests)
    if "metadata" in source_data:
        candidates.append(("metadata",))
    for section_name in _MODEL_SECTIONS:
        section = source_data.get(section_name, {})
        if not isinstance(section, Mapping):
            continue
        for name in sorted(section):
            if (
                name in instruction_names
                and isinstance(name, str)
                and len(name.encode("utf-8")) <= MAX_PATH_COMPONENT_BYTES
            ):
                candidates.append((section_name, name))

    unique_candidates: list[tuple[str, ...]] = []
    for path in candidates:
        if path not in unique_candidates:
            unique_candidates.append(path)

    model_context = partial["model"]
    for path in unique_candidates[: MAX_CONTEXT_REQUESTS * MAX_CONTEXT_ROUNDS]:
        value = _path_value(source_data, path)
        candidate = {"path": list(path), "value": value}
        model_context["included_values"].append(candidate)
        model_context["visible_paths"].append(list(path))
        try:
            _finalise_context(partial)
        except InterpreterProposalError:
            model_context["included_values"].pop()
            model_context["visible_paths"].pop()
            encoded = _canonical_json_bytes(value)
            if len(model_context["unavailable_values"]) < 16:
                unavailable = {
                    "path": list(path),
                    "bytes": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "reason": "value_exceeds_context_budget",
                }
                model_context["unavailable_values"].append(unavailable)
                try:
                    _finalise_context(partial)
                except InterpreterProposalError:
                    model_context["unavailable_values"].pop()

    completed = _context_with_attachment_chunks(
        partial,
        records=attachment_records,
        requested=normalised_requests,
        instruction=instruction,
        strict_requested=True,
    )
    if completed is None:
        raise InterpreterProposalError(
            "The bounded interpreter context could not include its required attachment evidence."
        )
    return completed


def _validated_context_document(
    value: object,
    *,
    instruction: str,
    current_model_source: str,
    clarification_history: Sequence[Mapping[str, str]] = (),
    attachments: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InterpreterProposalError("Interpreter context must be an object.")
    expected_context_version = (
        INTERPRETER_CONTEXT_SCHEMA_VERSION
        if attachments
        else PREVIOUS_INTERPRETER_CONTEXT_SCHEMA_VERSION
    )
    if value.get("schema") != INTERPRETER_CONTEXT_SCHEMA or value.get(
        "schema_version"
    ) != expected_context_version:
        raise InterpreterProposalError("Unsupported interpreter context schema.")
    requested = value.get("requested_paths")
    round_index = value.get("round")
    if not isinstance(requested, list):
        raise InterpreterProposalError("Interpreter context requested_paths must be a list.")
    expected = create_interpreter_context(
        instruction=instruction,
        current_model_source=current_model_source,
        requested_paths=requested,
        round_index=round_index,
        clarification_history=clarification_history,
        attachments=attachments,
    )
    if dict(value) != expected:
        raise InterpreterProposalError(
            "Interpreter context does not match the deterministic context for this source and instruction."
        )
    return expected


def _validated_operation(value: object, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InterpreterProposalError(f"Interpreter operation {index + 1} must be an object.")
    _exact_keys(value, _OPERATION_KEYS, f"Interpreter operation {index + 1}")
    operation = value.get("op")
    if operation not in {"set", "remove"}:
        raise InterpreterProposalError(
            f"Interpreter operation {index + 1} has an unsupported op."
        )
    path = _validated_path(value.get("path"), f"Interpreter operation {index + 1} path")
    if operation == "remove" and path == ("name",):
        raise InterpreterProposalError("The required model name cannot be removed.")
    raw_value = _validate_json_value(
        value.get("value"), f"Interpreter operation {index + 1} value"
    )
    if operation == "remove" and raw_value is not None:
        raise InterpreterProposalError("A remove operation must use a null value.")
    if operation == "set" and raw_value is None:
        raise InterpreterProposalError(
            "A set operation cannot use null; use remove for optional fields."
        )
    return {"op": str(operation), "path": list(path), "value": raw_value}


def validate_raw_interpreter_output(value: object) -> dict[str, Any]:
    """Validate Qwen's exact output grammar and action-specific invariants."""
    if not isinstance(value, Mapping):
        raise InterpreterProposalError("Interpreter output must be an object.")
    _exact_keys(value, _RAW_OUTPUT_KEYS, "Interpreter output")
    if value.get("schema") != INTERPRETER_OUTPUT_SCHEMA or value.get(
        "schema_version"
    ) != INTERPRETER_OUTPUT_SCHEMA_VERSION:
        raise InterpreterProposalError("Unsupported interpreter output schema.")
    action = value.get("action")
    if action not in {"request_context", "propose_edits", "needs_clarification", "unable"}:
        raise InterpreterProposalError("Interpreter output action is unsupported.")
    context_sha = _require_sha256(value.get("context_sha256"), "Interpreter context_sha256")
    requests_raw = value.get("context_requests")
    if not isinstance(requests_raw, list) or len(requests_raw) > MAX_CONTEXT_REQUESTS:
        raise InterpreterProposalError(
            f"Interpreter context_requests must contain at most {MAX_CONTEXT_REQUESTS} paths."
        )
    requests = sorted(
        {
            _validated_context_request(item, f"Interpreter context request {index + 1}")
            for index, item in enumerate(requests_raw)
        }
    )
    attachment_request_count = sum(path[0] == "@attachment" for path in requests)
    if attachment_request_count > MAX_ATTACHMENT_CONTEXT_REQUESTS:
        raise InterpreterProposalError(
            f"Interpreter output may request at most {MAX_ATTACHMENT_CONTEXT_REQUESTS} attachment chunks."
        )
    operations_raw = value.get("operations")
    if not isinstance(operations_raw, list) or len(operations_raw) > MAX_OPERATIONS:
        raise InterpreterProposalError(
            f"Interpreter operations must contain at most {MAX_OPERATIONS} edits."
        )
    operations = [
        _validated_operation(item, index) for index, item in enumerate(operations_raw)
    ]
    operations.sort(key=lambda item: (tuple(item["path"]), item["op"]))
    for index, left in enumerate(operations):
        left_path = tuple(left["path"])
        for right in operations[index + 1 :]:
            right_path = tuple(right["path"])
            if (
                left_path == right_path
                or left_path == right_path[: len(left_path)]
                or right_path == left_path[: len(right_path)]
            ):
                raise InterpreterProposalError(
                    "Interpreter operations contain duplicate or overlapping paths."
                )

    explanation = _bounded_text(
        value.get("explanation"),
        "Interpreter output explanation",
        MAX_EXPLANATION_BYTES,
        allow_empty=True,
    )
    warnings = _validated_warnings(value.get("warnings"), "Interpreter output warnings")
    clarification_question = _bounded_text(
        value.get("clarification_question"),
        "Interpreter clarification question",
        MAX_EXPLANATION_BYTES,
        allow_empty=True,
    )
    if action == "request_context":
        if not requests or operations or clarification_question:
            raise InterpreterProposalError(
                "request_context requires paths and forbids edit operations or a clarification question."
            )
    elif action == "propose_edits":
        if requests or not operations or clarification_question:
            raise InterpreterProposalError(
                "propose_edits requires operations and forbids context requests or a clarification question."
            )
    elif action == "needs_clarification":
        if requests or operations or not clarification_question.strip():
            raise InterpreterProposalError(
                "needs_clarification forbids requests and operations and requires one user-facing question."
            )
    elif requests or operations or clarification_question or not explanation.strip():
        raise InterpreterProposalError(
            "unable forbids requests, operations, and clarification questions and requires an explanation."
        )

    normalised = {
        "schema": INTERPRETER_OUTPUT_SCHEMA,
        "schema_version": INTERPRETER_OUTPUT_SCHEMA_VERSION,
        "action": action,
        "context_sha256": context_sha,
        "context_requests": [list(path) for path in requests],
        "operations": operations,
        "explanation": explanation,
        "warnings": warnings,
        "clarification_question": clarification_question,
    }
    if len(_canonical_json_bytes(normalised)) > MAX_EDIT_PROGRAM_BYTES:
        raise InterpreterProposalError("Interpreter output exceeds the edit-program limit.")
    return normalised


def _path_visible(context: Mapping[str, Any], path: Sequence[str]) -> bool:
    model = context["model"]
    if model.get("scope") == "complete":
        return True
    visible = [tuple(item) for item in model.get("visible_paths", [])]
    target = tuple(path)
    return any(
        target == item or target[: len(item)] == item
        for item in visible
    )


def _context_request_visible(context: Mapping[str, Any], path: Sequence[str]) -> bool:
    if path and path[0] == "@attachment":
        projection = context.get("attachments")
        chunks = projection.get("included_chunks", []) if isinstance(projection, Mapping) else []
        return any(
            isinstance(item, Mapping) and item.get("path") == list(path)
            for item in chunks
        )
    return _path_visible(context, path)


def _ensure_operation_is_grounded(
    operation: Mapping[str, Any],
    source_data: Mapping[str, Any],
    context: Mapping[str, Any],
) -> None:
    path = tuple(operation["path"])
    exists = _path_exists(source_data, path)
    if operation["op"] == "remove" and not exists:
        raise InterpreterProposalError(
            "A remove operation targets a path that does not exist: " + ".".join(path) + "."
        )
    if exists and not _path_visible(context, path):
        raise InterpreterProposalError(
            "An edit targets existing content that was not supplied to Qwen. "
            "Request that exact path first: " + ".".join(path) + "."
        )
    if len(path) == 3 and not _path_exists(source_data, path[:2]):
        raise InterpreterProposalError(
            "A field edit requires an existing entry; set the complete entry instead: "
            + ".".join(path[:2])
            + "."
        )


def _apply_operation(
    document: MutableMapping[str, Any], operation: Mapping[str, Any]
) -> None:
    path = tuple(operation["path"])
    op = operation["op"]
    value = _canonical_yaml_value(deepcopy(operation["value"]))
    if path == ("name",):
        document["name"] = value
        return
    if path == ("metadata",):
        if op == "remove":
            document.pop("metadata", None)
        else:
            document["metadata"] = value
        return
    if path[0] == "metadata":
        metadata = document.setdefault("metadata", {})
        if not isinstance(metadata, MutableMapping):
            raise InterpreterProposalError("Current model metadata is not editable as a mapping.")
        if op == "remove":
            if path[1] not in metadata:
                raise InterpreterProposalError("A remove operation targets a missing metadata field.")
            del metadata[path[1]]
        else:
            metadata[path[1]] = value
        return

    section_name, identifier = path[:2]
    section = document.setdefault(section_name, {})
    if not isinstance(section, MutableMapping):
        raise InterpreterProposalError(f"Model section '{section_name}' is not a mapping.")
    if len(path) == 2:
        if op == "remove":
            if identifier not in section:
                raise InterpreterProposalError("A remove operation targets a missing entry.")
            del section[identifier]
        else:
            section[identifier] = value
        return

    if identifier not in section:
        raise InterpreterProposalError("A field edit targets a missing model entry.")
    entry = section[identifier]
    if isinstance(entry, str) and section_name == "functions":
        entry = {"expression": entry}
        section[identifier] = entry
    if not isinstance(entry, MutableMapping):
        raise InterpreterProposalError("The targeted model entry is not an editable mapping.")
    field = path[2]
    if op == "remove":
        if field not in entry:
            raise InterpreterProposalError("A remove operation targets a missing entry field.")
        del entry[field]
    else:
        entry[field] = value


def _canonical_yaml_value(value: Any) -> Any:
    """Return JSON data with deterministic mapping order at every nesting level."""
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_yaml_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, list):
        return [_canonical_yaml_value(item) for item in value]
    return value


def _load_round_trip_model_data(source: str) -> MutableMapping[str, Any]:
    round_trip = RoundTripYAML(typ="rt")
    round_trip.preserve_quotes = True
    try:
        document = round_trip.load(source)
    except Exception as exc:
        raise InterpreterProposalError(
            "The current YAML cannot be represented by the round-trip edit compiler."
        ) from exc
    if not isinstance(document, MutableMapping):
        raise InterpreterProposalError("The current model source is not a YAML mapping.")
    return document


def _serialise_model_data(
    document: Mapping[str, Any],
    *,
    preserve_source_style: bool,
    original_source: str = "",
) -> str:
    if preserve_source_style:
        round_trip = RoundTripYAML(typ="rt")
        round_trip.preserve_quotes = True
        round_trip.width = 1_000_000
        # ruamel preserves comments, quotes, flow collections, and key order, but its
        # default sequence indentation can still rewrite untouched lines. Infer the
        # document's existing block-sequence convention before dumping.
        indentless = 0
        indented = 0
        lines = original_source.splitlines()
        for index, line in enumerate(lines[:-1]):
            match = re.match(r"^(\s*)[^#\s][^:]*:\s*(?:#.*)?$", line)
            if match is None:
                continue
            for following in lines[index + 1 :]:
                if not following.strip() or following.lstrip().startswith("#"):
                    continue
                sequence = re.match(r"^(\s*)-\s", following)
                if sequence is not None:
                    if len(sequence.group(1)) == len(match.group(1)):
                        indentless += 1
                    elif len(sequence.group(1)) > len(match.group(1)):
                        indented += 1
                break
        if indentless > indented:
            round_trip.indent(mapping=2, sequence=2, offset=0)
        else:
            round_trip.indent(mapping=2, sequence=4, offset=2)
        stream = StringIO()
        try:
            round_trip.dump(document, stream)
        except Exception as exc:
            raise InterpreterProposalError(
                "The edited YAML could not be serialised without discarding source structure."
            ) from exc
        source = stream.getvalue()
        return source if source.endswith("\n") else source + "\n"

    normalised = _canonical_yaml_value(
        json.loads(
            json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        )
    )
    ordered: dict[str, Any] = {}
    for key in _TOP_LEVEL_ORDER:
        if key in normalised:
            ordered[key] = normalised[key]
    for key in sorted(set(normalised) - set(ordered)):
        ordered[key] = normalised[key]
    source = yaml.safe_dump(
        ordered,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=1000,
    )
    return source if source.endswith("\n") else source + "\n"


def _edit_program_payload(output: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": output["schema"],
        "schema_version": output["schema_version"],
        "action": output["action"],
        "context_sha256": output["context_sha256"],
        "operations": output["operations"],
    }


def _compile_operations(
    *,
    output: Mapping[str, Any],
    context: Mapping[str, Any],
    current_model_source: str,
) -> tuple[str, Any, dict[str, Any]]:
    source_data = _load_source_data(current_model_source)
    for operation in output["operations"]:
        _ensure_operation_is_grounded(operation, source_data, context)
    compiled_data: MutableMapping[str, Any] = (
        _load_round_trip_model_data(current_model_source)
        if current_model_source.strip()
        else {"name": "Untitled model", "variables": {}, "functions": {}}
    )
    for operation in output["operations"]:
        _apply_operation(compiled_data, operation)
    source = _serialise_model_data(
        compiled_data,
        preserve_source_style=bool(current_model_source.strip()),
        original_source=current_model_source,
    )
    if len(source.encode("utf-8")) > MAX_MODEL_SOURCE_BYTES:
        raise InterpreterProposalError(
            "The compiled model exceeds the 512 KiB model-source safety limit."
        )
    try:
        model = validate_model(parse_model_text(source))
    except (ModelParseError, ModelValidationError) as exc:
        raise InterpreterProposalError(
            f"The deterministic edit program does not compile to a valid model: {exc}"
        ) from exc
    compiler = {
        "schema": INTERPRETER_COMPILER_SCHEMA,
        "schema_version": INTERPRETER_COMPILER_SCHEMA_VERSION,
        "context_schema_version": context["schema_version"],
        "output_schema_version": INTERPRETER_OUTPUT_SCHEMA_VERSION,
        "context_sha256": context["context_sha256"],
        "edit_program_sha256": canonical_json_sha256(_edit_program_payload(output)),
        "operation_count": len(output["operations"]),
        "context_round": context["round"],
        "attachment_evidence_sha256": canonical_json_sha256(
            _context_attachment_evidence(context)
        ),
    }
    return source, model, compiler


def proposal_diff(current_source: str, proposed_source: str) -> str:
    """Return the stable unified source diff shown during explicit review."""
    current = current_source.splitlines(keepends=True)
    proposed = proposed_source.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            current,
            proposed,
            fromfile="current/model.yaml",
            tofile="proposed/model.yaml",
            lineterm="\n",
        )
    )


def _proposal_payload_without_sha(proposal: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in proposal.items() if key != "proposal_sha256"}


def _acceptance_payload_without_sha(acceptance: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in acceptance.items() if key != "acceptance_sha256"}


def create_interpreter_proposal(
    *,
    raw_output: object,
    context_document: object,
    provider_identity: object,
    instruction: str,
    current_model_source: str,
    clarification_history: Sequence[Mapping[str, str]] = (),
    attachments: Sequence[Mapping[str, Any]] = (),
    conversation_id: str | None = None,
    conversation_lineage: Sequence[Mapping[str, Any]] = (),
    proposal_id: str | None = None,
    created_at_utc: str | None = None,
    expected_generation_options: Mapping[str, int | float] | None = None,
) -> dict[str, Any]:
    """Compile an exact edit program into a checksum-bound review proposal.

    ``expected_generation_options`` is a trusted caller contract used by the Stage-5
    evaluator when it runs a deterministic decoding profile. Ordinary application calls
    omit it and therefore retain the production generation profile.
    """
    instruction = _bounded_text(instruction, "Interpreter instruction", MAX_INSTRUCTION_BYTES)
    current_model_source = _bounded_text(
        current_model_source,
        "Current model source",
        MAX_MODEL_SOURCE_BYTES,
        allow_empty=True,
    )
    context = _validated_context_document(
        context_document,
        instruction=instruction,
        current_model_source=current_model_source,
        clarification_history=clarification_history,
        attachments=attachments,
    )
    output = validate_raw_interpreter_output(raw_output)
    if output["context_sha256"] != context["context_sha256"]:
        raise InterpreterProposalError(
            "Interpreter output was produced for a different context package."
        )
    boundary = context["contract"].get("request_boundary")
    if (
        isinstance(boundary, Mapping)
        and boundary.get("required_action") == "unable"
        and output["action"] != "unable"
    ):
        raise InterpreterProposalError(
            "The installed scientific capability boundary requires an unable response: "
            + str(boundary.get("reason", "the request is unsupported"))
        )
    if output["action"] != "propose_edits":
        raise InterpreterProposalError("Only propose_edits output can become a proposal.")
    required_generation_options = dict(
        expected_generation_options
        or generation_options_for_request(
            instruction=instruction, context_document=context
        )
    )
    provider = validate_provider_identity(
        provider_identity, generation_options=required_generation_options
    )
    dialogue_id = _conversation_identifier(
        conversation_id,
        instruction=instruction,
        current_model_source=current_model_source,
    )
    lineage = validate_conversation_lineage(
        conversation_lineage,
        conversation_id=dialogue_id,
        clarification_history=context["clarification_history"],
        provider_identity=provider,
    )
    source, model, compiler = _compile_operations(
        output=output,
        context=context,
        current_model_source=current_model_source,
    )
    attachment_evidence = _context_attachment_evidence(context)
    identifier = proposal_id or str(uuid.uuid4())
    try:
        uuid.UUID(identifier)
    except (ValueError, AttributeError) as exc:
        raise InterpreterProposalError("proposal_id must be a UUID.") from exc

    proposal: dict[str, Any] = {
        "schema": INTERPRETER_PROPOSAL_SCHEMA,
        "schema_version": INTERPRETER_PROPOSAL_SCHEMA_VERSION,
        "proposal_id": identifier,
        "created_at_utc": created_at_utc or _utc_now(),
        "provider": provider,
        "dialogue": {
            "conversation_id": dialogue_id,
            "turn_count": len(lineage),
            "turns": lineage,
            "lineage_sha256": canonical_json_sha256(lineage),
            "terminal_parent_turn_sha256": (
                lineage[-1]["turn_sha256"] if lineage else None
            ),
            "provider_identity_sha256": canonical_json_sha256(provider),
        },
        "request": {
            "instruction_sha256": _sha256_text(instruction),
            "current_model_source_sha256": _sha256_text(current_model_source),
            "clarification_history_sha256": context["clarification_history_sha256"],
            "clarification_turn_count": len(context["clarification_history"]),
            "attachments": attachment_evidence,
        },
        "compiler": compiler,
        "proposed_model": {
            "source": source,
            "source_sha256": _sha256_text(source),
            "canonical_model_ir_sha256": canonical_model_ir_sha256(model),
        },
        "explanation": output["explanation"],
        "warnings": output["warnings"],
        "requires_user_acceptance": True,
    }
    proposal["proposal_sha256"] = canonical_json_sha256(proposal)
    return proposal


def process_interpreter_output(
    *,
    raw_output: object,
    context_document: object,
    provider_identity: object,
    instruction: str,
    current_model_source: str,
    clarification_history: Sequence[Mapping[str, str]] = (),
    attachments: Sequence[Mapping[str, Any]] = (),
    conversation_id: str | None = None,
    conversation_lineage: Sequence[Mapping[str, Any]] = (),
    proposal_id: str | None = None,
    created_at_utc: str | None = None,
    expected_generation_options: Mapping[str, int | float] | None = None,
) -> dict[str, Any]:
    """Advance context negotiation or compile a final proposal without executing it."""
    context = _validated_context_document(
        context_document,
        instruction=instruction,
        current_model_source=current_model_source,
        clarification_history=clarification_history,
        attachments=attachments,
    )
    output = validate_raw_interpreter_output(raw_output)
    if output["context_sha256"] != context["context_sha256"]:
        raise InterpreterProposalError(
            "Interpreter output was produced for a different context package."
        )
    boundary = context["contract"].get("request_boundary")
    if (
        isinstance(boundary, Mapping)
        and boundary.get("required_action") == "unable"
        and output["action"] != "unable"
    ):
        raise InterpreterProposalError(
            "The installed scientific capability boundary requires an unable response: "
            + str(boundary.get("reason", "the request is unsupported"))
        )
    required_generation_options = dict(
        expected_generation_options
        or generation_options_for_request(
            instruction=instruction, context_document=context
        )
    )
    provider = validate_provider_identity(
        provider_identity, generation_options=required_generation_options
    )
    dialogue_id = _conversation_identifier(
        conversation_id,
        instruction=instruction,
        current_model_source=current_model_source,
    )
    lineage = validate_conversation_lineage(
        conversation_lineage,
        conversation_id=dialogue_id,
        clarification_history=context["clarification_history"],
        provider_identity=provider,
    )
    if output["action"] == "unable":
        return {
            "status": "unable",
            "explanation": output["explanation"],
            "warnings": output["warnings"],
            "provider": provider,
            "conversation_id": dialogue_id,
            "execution_performed": False,
        }
    if output["action"] == "needs_clarification":
        evidence = _create_dialogue_turn_evidence(
            conversation_id=dialogue_id,
            turn_index=len(lineage),
            parent_turn_sha256=lineage[-1]["turn_sha256"] if lineage else None,
            context_sha256=context["context_sha256"],
            raw_output=output,
            provider_identity=provider,
            question=output["clarification_question"],
        )
        return {
            "status": "needs_clarification",
            "question": output["clarification_question"],
            "explanation": output["explanation"],
            "warnings": output["warnings"],
            "provider": provider,
            "conversation_id": dialogue_id,
            "turn_evidence": evidence,
            "execution_performed": False,
        }
    if output["action"] == "request_context":
        if context["round"] + 1 >= MAX_CONTEXT_ROUNDS:
            raise InterpreterProposalError(
                "The interpreter exhausted its bounded context-request rounds without a proposal."
            )
        existing = [tuple(item) for item in context["requested_paths"]]
        additions = [tuple(item) for item in output["context_requests"]]
        if any(_context_request_visible(context, item) for item in additions):
            raise InterpreterProposalError(
                "The interpreter requested context that is already visible."
            )
        combined = [*existing, *[item for item in additions if item not in existing]]
        if len(combined) == len(existing):
            raise InterpreterProposalError("The interpreter requested no new context paths.")
        expanded = create_interpreter_context(
            instruction=instruction,
            current_model_source=current_model_source,
            requested_paths=combined,
            round_index=context["round"] + 1,
            clarification_history=clarification_history,
            attachments=attachments,
        )
        return {
            "status": "context_required",
            "context": expanded,
            "explanation": output["explanation"],
            "warnings": output["warnings"],
            "provider": provider,
            "conversation_id": dialogue_id,
            "execution_performed": False,
        }
    proposal = create_interpreter_proposal(
        raw_output=output,
        context_document=context,
        provider_identity=provider,
        instruction=instruction,
        current_model_source=current_model_source,
        clarification_history=clarification_history,
        attachments=attachments,
        conversation_id=dialogue_id,
        conversation_lineage=lineage,
        proposal_id=proposal_id,
        created_at_utc=created_at_utc,
        expected_generation_options=expected_generation_options,
    )
    return {"status": "proposal", "proposal": proposal, "execution_performed": False}


def _validated_compiler_record(
    value: object, field: str = "Interpreter compiler"
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InterpreterProposalError(f"{field} must be an object.")
    expected = {
        "schema",
        "schema_version",
        "context_schema_version",
        "output_schema_version",
        "context_sha256",
        "edit_program_sha256",
        "operation_count",
        "context_round",
    }
    if value.get("schema_version") == INTERPRETER_COMPILER_SCHEMA_VERSION:
        expected.add("attachment_evidence_sha256")
    _exact_keys(value, expected, field)
    compiler_version = value.get("schema_version")
    supported_version_triplets = {
        (
            INTERPRETER_COMPILER_SCHEMA_VERSION,
            INTERPRETER_CONTEXT_SCHEMA_VERSION,
            INTERPRETER_OUTPUT_SCHEMA_VERSION,
        ),
        (
            INTERPRETER_COMPILER_SCHEMA_VERSION,
            PREVIOUS_INTERPRETER_CONTEXT_SCHEMA_VERSION,
            INTERPRETER_OUTPUT_SCHEMA_VERSION,
        ),
        (
            PREVIOUS_INTERPRETER_COMPILER_SCHEMA_VERSION,
            PREVIOUS_INTERPRETER_CONTEXT_SCHEMA_VERSION,
            INTERPRETER_OUTPUT_SCHEMA_VERSION,
        ),
        (
            LEGACY_INTERPRETER_COMPILER_SCHEMA_VERSION,
            OLDER_INTERPRETER_CONTEXT_SCHEMA_VERSION,
            INTERPRETER_OUTPUT_SCHEMA_VERSION,
        ),
        (
            OLDER_INTERPRETER_COMPILER_SCHEMA_VERSION,
            OLDER_INTERPRETER_CONTEXT_SCHEMA_VERSION,
            PREVIOUS_INTERPRETER_OUTPUT_SCHEMA_VERSION,
        ),
        (
            INITIAL_INTERPRETER_COMPILER_SCHEMA_VERSION,
            LEGACY_INTERPRETER_CONTEXT_SCHEMA_VERSION,
            LEGACY_INTERPRETER_OUTPUT_SCHEMA_VERSION,
        ),
    }
    version_triplet = (
        compiler_version,
        value.get("context_schema_version"),
        value.get("output_schema_version"),
    )
    if value.get("schema") != INTERPRETER_COMPILER_SCHEMA or version_triplet not in supported_version_triplets:
        raise InterpreterProposalError("Unsupported interpreter compiler schema.")
    context_sha = _require_sha256(value.get("context_sha256"), f"{field} context_sha256")
    program_sha = _require_sha256(
        value.get("edit_program_sha256"), f"{field} edit_program_sha256"
    )
    attachment_evidence_sha = None
    if compiler_version == INTERPRETER_COMPILER_SCHEMA_VERSION:
        attachment_evidence_sha = _require_sha256(
            value.get("attachment_evidence_sha256"),
            f"{field} attachment_evidence_sha256",
        )
    operation_count = value.get("operation_count")
    context_round = value.get("context_round")
    if (
        isinstance(operation_count, bool)
        or not isinstance(operation_count, int)
        or not 1 <= operation_count <= MAX_OPERATIONS
    ):
        raise InterpreterProposalError("Interpreter compiler operation_count is invalid.")
    if (
        isinstance(context_round, bool)
        or not isinstance(context_round, int)
        or not 0 <= context_round < MAX_CONTEXT_ROUNDS
    ):
        raise InterpreterProposalError("Interpreter compiler context_round is invalid.")
    result = {
        "schema": INTERPRETER_COMPILER_SCHEMA,
        "schema_version": compiler_version,
        "context_schema_version": value["context_schema_version"],
        "output_schema_version": value["output_schema_version"],
        "context_sha256": context_sha,
        "edit_program_sha256": program_sha,
        "operation_count": operation_count,
        "context_round": context_round,
    }
    if attachment_evidence_sha is not None:
        result["attachment_evidence_sha256"] = attachment_evidence_sha
    return result


def _validated_dialogue_record(
    value: object,
    *,
    provider: Mapping[str, Any],
    expected_turn_count: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InterpreterProposalError("Interpreter dialogue provenance must be an object.")
    _exact_keys(
        value,
        {
            "conversation_id", "turn_count", "turns", "lineage_sha256",
            "terminal_parent_turn_sha256", "provider_identity_sha256",
        },
        "Interpreter dialogue provenance",
    )
    conversation_id = _conversation_identifier(
        value.get("conversation_id"), instruction="dialogue", current_model_source=""
    )
    turn_count = value.get("turn_count")
    turns = value.get("turns")
    if (
        isinstance(turn_count, bool)
        or not isinstance(turn_count, int)
        or turn_count != expected_turn_count
        or not isinstance(turns, list)
        or len(turns) != turn_count
    ):
        raise InterpreterProposalError("Interpreter dialogue turn count is inconsistent.")
    provider_sha = canonical_json_sha256(provider)
    if value.get("provider_identity_sha256") != provider_sha:
        raise InterpreterProposalError("Interpreter dialogue is bound to a different provider.")
    parent: str | None = None
    normalised_turns: list[dict[str, Any]] = []
    required_turn_keys = {
        "schema", "schema_version", "conversation_id", "turn_index",
        "parent_turn_sha256", "context_sha256", "raw_output_sha256",
        "provider_identity", "provider_identity_sha256", "question_sha256", "turn_sha256",
    }
    for index, turn in enumerate(turns):
        if not isinstance(turn, Mapping):
            raise InterpreterProposalError(f"Interpreter dialogue turn {index} is malformed.")
        _exact_keys(turn, required_turn_keys, f"Interpreter dialogue turn {index}")
        if (
            turn.get("schema") != INTERPRETER_DIALOGUE_TURN_SCHEMA
            or turn.get("schema_version") != INTERPRETER_DIALOGUE_TURN_SCHEMA_VERSION
            or turn.get("conversation_id") != conversation_id
            or turn.get("turn_index") != index
            or turn.get("parent_turn_sha256") != parent
        ):
            raise InterpreterProposalError(
                f"Interpreter dialogue turn {index} breaks the ordered provider chain."
            )
        for field in (
            "context_sha256", "raw_output_sha256", "provider_identity_sha256",
            "question_sha256", "turn_sha256",
        ):
            _require_sha256(turn.get(field), f"Interpreter dialogue turn {index} {field}")
        turn_provider = validate_provider_identity(turn.get("provider_identity"))
        if turn.get("provider_identity_sha256") != canonical_json_sha256(turn_provider):
            raise InterpreterProposalError(
                f"Interpreter dialogue turn {index} provider checksum is invalid."
            )
        if canonical_json_sha256(
            {key: item for key, item in turn_provider.items() if key != "generation_options"}
        ) != canonical_json_sha256(
            {key: item for key, item in provider.items() if key != "generation_options"}
        ):
            raise InterpreterProposalError(
                f"Interpreter dialogue turn {index} used a different runtime/model artifact."
            )
        computed = canonical_json_sha256(
            {key: item for key, item in turn.items() if key != "turn_sha256"}
        )
        if turn.get("turn_sha256") != computed:
            raise InterpreterProposalError(
                f"Interpreter dialogue turn {index} checksum is invalid."
            )
        normalised_turns.append(dict(turn))
        parent = computed
    lineage_sha = _require_sha256(
        value.get("lineage_sha256"), "Interpreter dialogue lineage_sha256"
    )
    if lineage_sha != canonical_json_sha256(normalised_turns):
        raise InterpreterProposalError("Interpreter dialogue lineage checksum is invalid.")
    terminal = value.get("terminal_parent_turn_sha256")
    if terminal != parent:
        raise InterpreterProposalError("Interpreter dialogue terminal parent is invalid.")
    return {
        "conversation_id": conversation_id,
        "turn_count": turn_count,
        "turns": normalised_turns,
        "lineage_sha256": lineage_sha,
        "terminal_parent_turn_sha256": terminal,
        "provider_identity_sha256": provider_sha,
    }


def validate_interpreter_proposal_document(value: object) -> dict[str, Any]:
    """Revalidate a complete compiled proposal and its checksums before acceptance."""
    if not isinstance(value, Mapping):
        raise InterpreterProposalError("Interpreter proposal must be an object.")
    expected_keys = {
        "schema",
        "schema_version",
        "proposal_id",
        "created_at_utc",
        "provider",
        "request",
        "compiler",
        "proposed_model",
        "explanation",
        "warnings",
        "requires_user_acceptance",
        "proposal_sha256",
    }
    version = value.get("schema_version")
    if version in {
        PREVIOUS_INTERPRETER_PROPOSAL_SCHEMA_VERSION,
        INTERPRETER_PROPOSAL_SCHEMA_VERSION,
    }:
        expected_keys.add("dialogue")
    _exact_keys(value, expected_keys, "Interpreter proposal")
    if value.get("schema") != INTERPRETER_PROPOSAL_SCHEMA or version not in {
        OLDER_INTERPRETER_PROPOSAL_SCHEMA_VERSION,
        LEGACY_INTERPRETER_PROPOSAL_SCHEMA_VERSION,
        PREVIOUS_INTERPRETER_PROPOSAL_SCHEMA_VERSION,
        INTERPRETER_PROPOSAL_SCHEMA_VERSION,
    }:
        raise InterpreterProposalError("Unsupported interpreter proposal schema.")
    identifier = value.get("proposal_id")
    try:
        uuid.UUID(str(identifier))
    except (ValueError, AttributeError) as exc:
        raise InterpreterProposalError("Interpreter proposal_id must be a UUID.") from exc
    created = _bounded_text(value.get("created_at_utc"), "created_at_utc", 128)
    try:
        parsed_created = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InterpreterProposalError("created_at_utc must be ISO-8601.") from exc
    if parsed_created.tzinfo is None:
        raise InterpreterProposalError("created_at_utc must include a timezone.")
    provider = validate_provider_identity(
        value.get("provider"),
        generation_options=(
            _V111_PRODUCTION_GENERATION_OPTIONS
            if version == OLDER_INTERPRETER_PROPOSAL_SCHEMA_VERSION
            else None
        ),
        allow_historical_current=version in {
            OLDER_INTERPRETER_PROPOSAL_SCHEMA_VERSION,
            LEGACY_INTERPRETER_PROPOSAL_SCHEMA_VERSION,
            PREVIOUS_INTERPRETER_PROPOSAL_SCHEMA_VERSION,
        },
    )
    request = value.get("request")
    if not isinstance(request, Mapping):
        raise InterpreterProposalError("Interpreter proposal request must be an object.")
    request_keys = {"instruction_sha256", "current_model_source_sha256"}
    if version in {
        LEGACY_INTERPRETER_PROPOSAL_SCHEMA_VERSION,
        PREVIOUS_INTERPRETER_PROPOSAL_SCHEMA_VERSION,
        INTERPRETER_PROPOSAL_SCHEMA_VERSION,
    }:
        request_keys |= {"clarification_history_sha256", "clarification_turn_count"}
    if version == INTERPRETER_PROPOSAL_SCHEMA_VERSION:
        request_keys.add("attachments")
    _exact_keys(request, request_keys, "Interpreter proposal request")
    for field in ("instruction_sha256", "current_model_source_sha256"):
        _require_sha256(request.get(field), f"Interpreter proposal {field}")
    if version in {
        LEGACY_INTERPRETER_PROPOSAL_SCHEMA_VERSION,
        PREVIOUS_INTERPRETER_PROPOSAL_SCHEMA_VERSION,
        INTERPRETER_PROPOSAL_SCHEMA_VERSION,
    }:
        _require_sha256(
            request.get("clarification_history_sha256"),
            "Interpreter proposal clarification_history_sha256",
        )
        turn_count = request.get("clarification_turn_count")
        if (
            isinstance(turn_count, bool)
            or not isinstance(turn_count, int)
            or not 0 <= turn_count <= MAX_CLARIFICATION_TURNS
        ):
            raise InterpreterProposalError(
                "Interpreter proposal clarification_turn_count is invalid."
            )
    attachments = None
    if version == INTERPRETER_PROPOSAL_SCHEMA_VERSION:
        attachments = _validated_attachment_evidence(request.get("attachments"))
    dialogue = None
    if version in {
        PREVIOUS_INTERPRETER_PROPOSAL_SCHEMA_VERSION,
        INTERPRETER_PROPOSAL_SCHEMA_VERSION,
    }:
        dialogue = _validated_dialogue_record(
            value.get("dialogue"),
            provider=provider,
            expected_turn_count=turn_count,
        )
    compiler = _validated_compiler_record(value.get("compiler"))
    if attachments is not None and compiler.get(
        "attachment_evidence_sha256"
    ) != canonical_json_sha256(attachments):
        raise InterpreterProposalError(
            "Interpreter compiler is bound to different attachment evidence."
        )
    proposed = value.get("proposed_model")
    if not isinstance(proposed, Mapping):
        raise InterpreterProposalError("Interpreter proposed_model must be an object.")
    _exact_keys(
        proposed,
        {"source", "source_sha256", "canonical_model_ir_sha256"},
        "Interpreter proposed_model",
    )
    source = _bounded_text(
        proposed.get("source"), "Interpreter proposed model source", MAX_MODEL_SOURCE_BYTES
    )
    if proposed.get("source_sha256") != _sha256_text(source):
        raise InterpreterProposalError("Interpreter proposed model source SHA-256 is invalid.")
    try:
        model = validate_model(parse_model_text(source))
    except (ModelParseError, ModelValidationError) as exc:
        raise InterpreterProposalError(
            f"The stored interpreter proposal is no longer a valid model: {exc}"
        ) from exc
    if proposed.get("canonical_model_ir_sha256") != canonical_model_ir_sha256(model):
        raise InterpreterProposalError("Interpreter proposal canonical Model IR SHA-256 is invalid.")
    explanation = _bounded_text(
        value.get("explanation"),
        "Interpreter proposal explanation",
        MAX_EXPLANATION_BYTES,
        allow_empty=True,
    )
    warnings = _validated_warnings(value.get("warnings"), "Interpreter proposal warnings")
    if value.get("requires_user_acceptance") is not True:
        raise InterpreterProposalError("Interpreter proposals must require explicit acceptance.")
    supplied_sha = value.get("proposal_sha256")
    computed_sha = canonical_json_sha256(_proposal_payload_without_sha(value))
    if supplied_sha != computed_sha:
        raise InterpreterProposalError("Interpreter proposal checksum does not match its contents.")
    result = {
        **_proposal_payload_without_sha(value),
        "proposal_id": str(identifier),
        "created_at_utc": created,
        "provider": provider,
        "request": dict(request),
        "compiler": compiler,
        "proposed_model": {
            "source": source,
            "source_sha256": proposed["source_sha256"],
            "canonical_model_ir_sha256": proposed["canonical_model_ir_sha256"],
        },
        "explanation": explanation,
        "warnings": warnings,
        "proposal_sha256": computed_sha,
    }
    if dialogue is not None:
        result["dialogue"] = dialogue
    return result


def accept_interpreter_proposal(
    proposal_document: object,
    *,
    expected_proposal_sha256: str,
    accepted_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create provenance only after an explicit checksum-bound acceptance."""
    proposal = validate_interpreter_proposal_document(proposal_document)
    if proposal["proposal_sha256"] != expected_proposal_sha256:
        raise InterpreterProposalError(
            "The reviewed proposal checksum does not match the proposal being accepted."
        )
    if "dialogue" not in proposal:
        raise InterpreterProposalError(
            "Historical interpreter proposals can be inspected but must be regenerated before acceptance because they lack the complete dialogue/provider chain."
        )
    acceptance: dict[str, Any] = {
        "schema": INTERPRETER_ACCEPTANCE_SCHEMA,
        "schema_version": INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
        "proposal_id": proposal["proposal_id"],
        "proposal_sha256": proposal["proposal_sha256"],
        "accepted_at_utc": accepted_at_utc or _utc_now(),
        "provider": proposal["provider"],
        "dialogue": proposal["dialogue"],
        "compiler": proposal["compiler"],
        "attachments": proposal["request"]["attachments"],
        "instruction_sha256": proposal["request"]["instruction_sha256"],
        "clarification_history_sha256": proposal["request"].get(
            "clarification_history_sha256", canonical_json_sha256([])
        ),
        "clarification_turn_count": proposal["request"].get(
            "clarification_turn_count", 0
        ),
        "previous_model_source_sha256": proposal["request"][
            "current_model_source_sha256"
        ],
        "accepted_model_source_sha256": proposal["proposed_model"]["source_sha256"],
        "accepted_model_ir_sha256": proposal["proposed_model"][
            "canonical_model_ir_sha256"
        ],
        "explanation": proposal["explanation"],
        "warnings": proposal["warnings"],
    }
    acceptance["acceptance_sha256"] = canonical_json_sha256(acceptance)
    return acceptance


def validate_interpreter_acceptance_document(value: object) -> dict[str, Any]:
    """Validate current compiler provenance and legacy v1.0 acceptance records."""
    if not isinstance(value, Mapping):
        raise InterpreterProposalError("Interpreter acceptance must be an object.")
    version = value.get("schema_version")
    base_keys = {
        "schema",
        "schema_version",
        "proposal_id",
        "proposal_sha256",
        "accepted_at_utc",
        "provider",
        "instruction_sha256",
        "previous_model_source_sha256",
        "accepted_model_source_sha256",
        "accepted_model_ir_sha256",
        "explanation",
        "warnings",
        "acceptance_sha256",
    }
    expected_keys = (
        base_keys | {"compiler"}
        if version in {
            EARLIER_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
            OLDER_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
            LEGACY_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
            PREVIOUS_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
            INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
        }
        else base_keys
    )
    if version in {
        LEGACY_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
        PREVIOUS_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
        INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
    }:
        expected_keys |= {"clarification_history_sha256", "clarification_turn_count"}
    if version == INTERPRETER_ACCEPTANCE_SCHEMA_VERSION:
        expected_keys |= {"dialogue", "attachments"}
    elif version == PREVIOUS_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION:
        expected_keys.add("dialogue")
    _exact_keys(value, expected_keys, "Interpreter acceptance")
    if value.get("schema") != INTERPRETER_ACCEPTANCE_SCHEMA or version not in {
        INITIAL_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
        EARLIER_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
        OLDER_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
        LEGACY_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
        PREVIOUS_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
        INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
    }:
        raise InterpreterProposalError("Unsupported interpreter acceptance schema.")
    try:
        uuid.UUID(str(value.get("proposal_id")))
    except (ValueError, AttributeError) as exc:
        raise InterpreterProposalError("Interpreter acceptance proposal_id must be a UUID.") from exc
    accepted_at = _bounded_text(value.get("accepted_at_utc"), "accepted_at_utc", 128)
    try:
        parsed = datetime.fromisoformat(accepted_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InterpreterProposalError("accepted_at_utc must be ISO-8601.") from exc
    if parsed.tzinfo is None:
        raise InterpreterProposalError("accepted_at_utc must include a timezone.")
    provider = validate_provider_identity(
        value.get("provider"),
        generation_options=(
            _LEGACY_GENERATION_OPTIONS
            if version == INITIAL_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION
            else _V111_PRODUCTION_GENERATION_OPTIONS
            if version in {
                EARLIER_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
                OLDER_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
            }
            else None
        ),
        legacy=version in {
            INITIAL_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
            EARLIER_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
        },
        allow_historical_current=version in {
            OLDER_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
            LEGACY_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
            PREVIOUS_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
        },
    )
    compiler = None
    if version in {
        EARLIER_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
        OLDER_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
        LEGACY_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
        PREVIOUS_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
        INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
    }:
        compiler = _validated_compiler_record(
            value.get("compiler"), "Interpreter acceptance compiler"
        )
    for field in (
        "proposal_sha256",
        "instruction_sha256",
        "previous_model_source_sha256",
        "accepted_model_source_sha256",
        "accepted_model_ir_sha256",
    ):
        _require_sha256(value.get(field), f"Interpreter acceptance {field}")
    turn_count = 0
    if version in {
        LEGACY_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
        PREVIOUS_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
        INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
    }:
        _require_sha256(
            value.get("clarification_history_sha256"),
            "Interpreter acceptance clarification_history_sha256",
        )
        turn_count = value.get("clarification_turn_count")
        if (
            isinstance(turn_count, bool)
            or not isinstance(turn_count, int)
            or not 0 <= turn_count <= MAX_CLARIFICATION_TURNS
        ):
            raise InterpreterProposalError(
                "Interpreter acceptance clarification_turn_count is invalid."
            )
    dialogue = None
    if version in {
        PREVIOUS_INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
        INTERPRETER_ACCEPTANCE_SCHEMA_VERSION,
    }:
        dialogue = _validated_dialogue_record(
            value.get("dialogue"),
            provider=provider,
            expected_turn_count=turn_count,
        )
    attachments = None
    if version == INTERPRETER_ACCEPTANCE_SCHEMA_VERSION:
        attachments = _validated_attachment_evidence(value.get("attachments"))
        if compiler is None or compiler.get(
            "attachment_evidence_sha256"
        ) != canonical_json_sha256(attachments):
            raise InterpreterProposalError(
                "Interpreter acceptance is bound to different attachment evidence."
            )
    explanation = _bounded_text(
        value.get("explanation"),
        "Interpreter acceptance explanation",
        MAX_EXPLANATION_BYTES,
        allow_empty=True,
    )
    warnings = _validated_warnings(value.get("warnings"), "Interpreter acceptance warnings")
    supplied_sha = value.get("acceptance_sha256")
    computed_sha = canonical_json_sha256(_acceptance_payload_without_sha(value))
    if supplied_sha != computed_sha:
        raise InterpreterProposalError("Interpreter acceptance checksum does not match its contents.")
    result = {
        **_acceptance_payload_without_sha(value),
        "proposal_id": str(value["proposal_id"]),
        "accepted_at_utc": accepted_at,
        "provider": provider,
        "explanation": explanation,
        "warnings": warnings,
        "acceptance_sha256": computed_sha,
    }
    if compiler is not None:
        result["compiler"] = compiler
    if dialogue is not None:
        result["dialogue"] = dialogue
    if attachments is not None:
        result["attachments"] = attachments
    return result
