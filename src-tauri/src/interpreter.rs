//! Loopback-only bridge to the persistent Ollama local-inference service.
//!
//! The webview never receives general network permission.  This native module is fixed to
//! one loopback endpoint and one versioned model tag, bounds every request/response, permits
//! only one inference operation at a time and exposes cancellation. The full model source
//! stays in the Python engine. Qwen receives a bounded deterministic context and returns a
//! typed edit program which is still untrusted until the Python compiler validates it.

use futures_util::StreamExt;
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tauri::async_runtime::Mutex as AsyncMutex;
use tokio::io::AsyncReadExt;
use tokio::sync::oneshot;
use uuid::Uuid;

pub const OLLAMA_ENDPOINT: &str = "http://127.0.0.1:11434";
pub const OLLAMA_MODEL: &str = "qwen3:4b-instruct-2507-q4_K_M";
pub const BASE_MODEL: &str = "Qwen/Qwen3-4B-Instruct-2507";
pub const QUANTIZATION: &str = "Q4_K_M";

const MAX_INSTRUCTION_BYTES: usize = 64 * 1024;
const MAX_CONTEXT_PACKAGE_BYTES: usize = 32 * 1024;
const MAX_EDIT_PROGRAM_BYTES: usize = 128 * 1024;
const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
const MAX_PULL_RESPONSE_BYTES: usize = 512 * 1024;
const STATUS_TIMEOUT: Duration = Duration::from_secs(5);
const INFERENCE_TIMEOUT: Duration = Duration::from_secs(20 * 60);
const PULL_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const PROMPT_STATIC_OVERHEAD_BYTES: usize = 12 * 1024;
const PROMPT_TOKEN_ESTIMATE_DENOMINATOR: usize = 2;
const PROMPT_TOKEN_SAFETY_MARGIN: u64 = 4_096;

const SYSTEM_PROMPT: &str = include_str!("../../model_lab/baseline/system_prompt_v1.3.txt");
const USER_PROMPT_TEMPLATE: &str = include_str!("../../model_lab/baseline/user_prompt_template_v1.0.txt");
const BASE_MODEL_LOCK_JSON: &str = include_str!("../../model_lab/baseline/base_model_lock_v1.1.json");
const FROZEN_BASE_IDENTITY_VERIFICATION: &str =
    "exact_local_manifest_and_all_blobs_sha256";
const CANDIDATE_IDENTITY_VERIFICATION: &str =
    "approved_registry_exact_manifest_and_model_blob_sha256";

#[derive(Debug)]
struct ActiveOperation {
    id: String,
    kind: &'static str,
    cancel: oneshot::Sender<()>,
}

pub struct InterpreterState {
    client: Client,
    active: AsyncMutex<Option<ActiveOperation>>,
    registry_path: PathBuf,
}

impl InterpreterState {
    pub fn new(registry_path: PathBuf) -> Self {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(3))
            .no_proxy()
            .build()
            .expect("could not construct the fixed local inference client");
        Self {
            client,
            active: AsyncMutex::new(None),
            registry_path,
        }
    }

    async fn begin(
        &self,
        kind: &'static str,
    ) -> Result<(String, oneshot::Receiver<()>), String> {
        let mut active = self.active.lock().await;
        if let Some(operation) = active.as_ref() {
            return Err(format!(
                "A local interpreter {} operation is already active ({}).",
                operation.kind, operation.id
            ));
        }
        let id = Uuid::new_v4().to_string();
        let (cancel, cancelled) = oneshot::channel();
        *active = Some(ActiveOperation {
            id: id.clone(),
            kind,
            cancel,
        });
        Ok((id, cancelled))
    }

    async fn finish(&self, id: &str) {
        let mut active = self.active.lock().await;
        if active.as_ref().map(|value| value.id.as_str()) == Some(id) {
            *active = None;
        }
    }

    async fn active_summary(&self) -> Option<Value> {
        self.active
            .lock()
            .await
            .as_ref()
            .map(|operation| {
                json!({"request_id": operation.id.as_str(), "operation": operation.kind})
            })
    }
}

#[derive(Debug, Deserialize)]
struct OllamaVersion {
    version: String,
}

#[derive(Debug, Clone, Deserialize)]
struct OllamaModel {
    #[serde(default)]
    name: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    digest: String,
    #[serde(default)]
    size: u64,
}

#[derive(Debug, Deserialize)]
struct OllamaTags {
    #[serde(default)]
    models: Vec<OllamaModel>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrozenArtifact {
    media_type: String,
    sha256: String,
    size_bytes: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrozenOllamaLock {
    model_tag: String,
    quantization: String,
    registry_manifest_sha256: String,
    config: FrozenArtifact,
    layers: Vec<FrozenArtifact>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BaseModelLockDocument {
    schema: String,
    schema_version: String,
    upstream: Value,
    ollama: FrozenOllamaLock,
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactEvidence {
    identity_verification: String,
    artifact_lock_sha256: String,
    local_manifest_sha256: String,
    model_blob_sha256: String,
    verified_model_blob_size_bytes: u64,
    all_manifest_blobs_verified: bool,
}

#[derive(Debug, Clone)]
struct ModelSelection {
    model_role: String,
    model_tag: String,
    expected_digest: Option<String>,
    expected_size: Option<u64>,
    expected_model_blob_sha256: Option<String>,
    frozen_base_lock: Option<FrozenOllamaLock>,
    registry_entry_sha256: Option<String>,
    export_receipt_sha256: Option<String>,
    training_report_sha256: Option<String>,
    candidate_evaluation_report_sha256: Option<String>,
    candidate_lifecycle_state: Option<String>,
    identity_verification: String,
}

fn frozen_ollama_lock() -> Result<FrozenOllamaLock, String> {
    let document: BaseModelLockDocument = serde_json::from_str(BASE_MODEL_LOCK_JSON)
        .map_err(|_| "Bundled frozen-base model lock is malformed JSON.".to_string())?;
    let _ = &document.upstream;
    if document.schema != "model-laboratory-base-model-lock"
        || document.schema_version != "1.1"
        || document.ollama.model_tag != OLLAMA_MODEL
        || document.ollama.quantization != QUANTIZATION
        || !valid_sha256(&document.ollama.registry_manifest_sha256)
        || document.ollama.config.media_type.is_empty()
        || !valid_sha256(&document.ollama.config.sha256)
        || document.ollama.config.size_bytes == 0
        || document.ollama.layers.is_empty()
        || document.ollama.layers.iter().any(|layer| {
            layer.media_type.is_empty() || !valid_sha256(&layer.sha256) || layer.size_bytes == 0
        })
    {
        return Err("Bundled frozen-base model lock failed validation.".to_string());
    }
    let media_types = document
        .ollama
        .layers
        .iter()
        .map(|layer| layer.media_type.as_str())
        .collect::<HashSet<_>>();
    if media_types.len() != document.ollama.layers.len()
        || !media_types.contains("application/vnd.ollama.image.model")
    {
        return Err("Bundled frozen-base layer contract is inconsistent.".to_string());
    }
    Ok(document.ollama)
}

impl ModelSelection {
    fn frozen_base() -> Result<Self, String> {
        Ok(Self {
            model_role: "frozen_base".to_string(),
            model_tag: OLLAMA_MODEL.to_string(),
            expected_digest: None,
            expected_size: None,
            expected_model_blob_sha256: None,
            frozen_base_lock: Some(frozen_ollama_lock()?),
            registry_entry_sha256: None,
            export_receipt_sha256: None,
            training_report_sha256: None,
            candidate_evaluation_report_sha256: None,
            candidate_lifecycle_state: None,
            identity_verification: FROZEN_BASE_IDENTITY_VERIFICATION.to_string(),
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryActive {
    role: String,
    entry_id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RegistryEntry {
    entry_id: String,
    lifecycle_state: String,
    training_status: String,
    candidate_id: Option<String>,
    candidate_tag: Option<String>,
    observed_model_digest: Option<String>,
    model_size_bytes: Option<u64>,
    quantization: Option<String>,
    base_model: Option<String>,
    base_snapshot_receipt_sha256: Option<String>,
    quantized_gguf_sha256: Option<String>,
    training_report_sha256: String,
    export_receipt_sha256: Option<String>,
    candidate_evaluation_report_sha256: Option<String>,
    eligible_for_promotion: Option<bool>,
    evaluation_purpose: Option<String>,
    promotion_campaign_id: Option<String>,
    evaluation_benchmark_sha256: Option<String>,
    promotion_seal_sha256: Option<String>,
    created_at_utc: String,
    evaluated_at_utc: Option<String>,
    approved_at_utc: Option<String>,
    entry_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryDocument {
    schema: String,
    schema_version: String,
    revision: u64,
    active: RegistryActive,
    entries: Vec<RegistryEntry>,
    history: Vec<Value>,
    registry_sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn canonical_json(value: &Value) -> Result<String, String> {
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(items) => {
                let ordered = items
                    .iter()
                    .map(|(key, item)| (key.clone(), sorted(item)))
                    .collect::<BTreeMap<_, _>>();
                Value::Object(ordered.into_iter().collect())
            }
            Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
            _ => value.clone(),
        }
    }
    serde_json::to_string(&sorted(value))
        .map_err(|_| "Interpreter model registry cannot be canonicalized.".to_string())
}

fn document_sha256(value: &Value, checksum_field: &str) -> Result<String, String> {
    let mut payload = value.clone();
    payload
        .as_object_mut()
        .ok_or_else(|| "Interpreter model registry must be an object.".to_string())?
        .remove(checksum_field);
    Ok(sha256_hex(canonical_json(&payload)?.as_bytes()))
}

fn normalized_digest(value: &str) -> Option<&str> {
    let digest = value.strip_prefix("sha256:").unwrap_or(value);
    if valid_sha256(digest) { Some(digest) } else { None }
}

fn verify_entry_sha(entry: &RegistryEntry) -> Result<(), String> {
    let value = serde_json::to_value(entry)
        .map_err(|_| "Interpreter registry candidate cannot be serialized.".to_string())?;
    if !valid_sha256(&entry.entry_sha256)
        || document_sha256(&value, "entry_sha256")? != entry.entry_sha256
    {
        return Err("Interpreter registry candidate checksum is invalid.".to_string());
    }
    if !valid_sha256(&entry.training_report_sha256)
        || entry.entry_id != format!("candidate-{}", &entry.training_report_sha256[..20])
    {
        return Err("Interpreter registry training identity is invalid.".to_string());
    }
    for digest in [
        entry.base_snapshot_receipt_sha256.as_deref(),
        entry.quantized_gguf_sha256.as_deref(),
        entry.export_receipt_sha256.as_deref(),
        entry.candidate_evaluation_report_sha256.as_deref(),
        entry.evaluation_benchmark_sha256.as_deref(),
        entry.promotion_seal_sha256.as_deref(),
    ].into_iter().flatten() {
        if !valid_sha256(digest) {
            return Err("Interpreter registry candidate contains a malformed artifact digest.".to_string());
        }
    }
    if entry.created_at_utc.is_empty()
        || !matches!(entry.lifecycle_state.as_str(), "experimental" | "trained" | "evaluated" | "approved")
        || !matches!(entry.training_status.as_str(), "candidate_adapter_experimental" | "candidate_adapter_unpromoted")
    {
        return Err("Interpreter registry candidate lifecycle identity is invalid.".to_string());
    }
    if entry.lifecycle_state == "experimental"
        && entry.training_status != "candidate_adapter_experimental"
    {
        return Err("Experimental interpreter registry state requires an experimental training report.".to_string());
    }
    if matches!(entry.lifecycle_state.as_str(), "trained" | "approved")
        && entry.training_status != "candidate_adapter_unpromoted"
    {
        return Err("Trained/approved interpreter registry state requires a non-experimental training report.".to_string());
    }
    let advanced = matches!(entry.lifecycle_state.as_str(), "evaluated" | "approved");
    if advanced {
        let candidate_id = entry.candidate_id.as_deref().unwrap_or("");
        let tag = entry.candidate_tag.as_deref().unwrap_or("");
        let digest = entry.observed_model_digest.as_deref().unwrap_or("");
        if !candidate_id.starts_with("candidate-")
            || tag.is_empty()
            || tag == OLLAMA_MODEL
            || normalized_digest(digest).is_none()
            || entry.model_size_bytes.unwrap_or(0) == 0
            || entry.quantization.as_deref() != Some(QUANTIZATION)
            || entry.base_model.as_deref() != Some(BASE_MODEL)
            || entry.quantized_gguf_sha256.is_none()
            || entry.export_receipt_sha256.is_none()
            || entry.candidate_evaluation_report_sha256.is_none()
            || entry.eligible_for_promotion.is_none()
            || !matches!(entry.evaluation_purpose.as_deref(), Some("development") | Some("promotion"))
            || entry.evaluation_benchmark_sha256.is_none()
            || entry.evaluated_at_utc.as_deref().unwrap_or("").is_empty()
        {
            return Err("Evaluated interpreter registry candidate is incomplete or incompatible.".to_string());
        }
    } else if entry.candidate_id.is_some()
        || entry.candidate_tag.is_some()
        || entry.observed_model_digest.is_some()
        || entry.model_size_bytes.is_some()
        || entry.quantization.is_some()
        || entry.base_model.is_some()
        || entry.quantized_gguf_sha256.is_some()
        || entry.export_receipt_sha256.is_some()
        || entry.candidate_evaluation_report_sha256.is_some()
        || entry.eligible_for_promotion.is_some()
        || entry.evaluation_purpose.is_some()
        || entry.promotion_campaign_id.is_some()
        || entry.evaluation_benchmark_sha256.is_some()
        || entry.promotion_seal_sha256.is_some()
        || entry.evaluated_at_utc.is_some()
    {
        return Err("Pre-evaluation interpreter registry candidate contains advanced lifecycle fields.".to_string());
    }
    if advanced {
        if entry.evaluation_purpose.as_deref() == Some("promotion") {
            let campaign = entry.promotion_campaign_id.as_deref().unwrap_or("");
            if Uuid::parse_str(campaign).is_err() || entry.promotion_seal_sha256.is_none() {
                return Err("Promotion evaluation has an invalid campaign UUID.".to_string());
            }
        } else if entry.promotion_campaign_id.is_some() || entry.promotion_seal_sha256.is_some() {
            return Err("Development evaluation cannot carry promotion campaign/seal evidence.".to_string());
        }
    }
    if entry.lifecycle_state == "approved" {
        if entry.eligible_for_promotion != Some(true)
            || entry.approved_at_utc.as_deref().unwrap_or("").is_empty()
        {
            return Err("Approved interpreter registry candidate lacks promotion evidence.".to_string());
        }
    } else if entry.approved_at_utc.is_some() {
        return Err("Only approved interpreter registry candidates may have an approval timestamp.".to_string());
    }
    if entry.lifecycle_state == "evaluated"
        && entry.training_status == "candidate_adapter_experimental"
        && entry.eligible_for_promotion != Some(false)
    {
        return Err("An evaluated experimental interpreter candidate must remain non-promotable.".to_string());
    }
    Ok(())
}

async fn model_selection(path: &Path) -> Result<ModelSelection, String> {
    if !path.exists() {
        return ModelSelection::frozen_base();
    }
    let bytes = tokio::fs::read(path)
        .await
        .map_err(|error| format!("Could not read interpreter model registry: {error}"))?;
    if bytes.len() > 1024 * 1024 {
        return Err("Interpreter model registry exceeds its 1 MiB safety limit.".to_string());
    }
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|_| "Interpreter model registry is malformed JSON.".to_string())?;
    let document: RegistryDocument = serde_json::from_value(value.clone())
        .map_err(|_| "Interpreter model registry has an unsupported structure.".to_string())?;
    if document.schema != "model-laboratory-interpreter-model-registry"
        || document.schema_version != "1.3"
        || !valid_sha256(&document.registry_sha256)
        || document_sha256(&value, "registry_sha256")? != document.registry_sha256
    {
        return Err("Interpreter model registry failed schema/checksum validation.".to_string());
    }
    let _ = document.revision;
    let _ = &document.history;
    let mut ids = HashSet::new();
    let mut tags = HashSet::new();
    let mut candidate_ids = HashSet::new();
    let mut promotion_benchmarks = HashSet::new();
    for entry in &document.entries {
        verify_entry_sha(entry)?;
        if !ids.insert(entry.entry_id.as_str()) {
            return Err("Interpreter model registry contains duplicate candidate entries.".to_string());
        }
        if let Some(tag) = entry.candidate_tag.as_deref() {
            if !tags.insert(tag) {
                return Err("Interpreter model registry contains duplicate evaluated tags.".to_string());
            }
        }
        if let Some(candidate_id) = entry.candidate_id.as_deref() {
            if !candidate_ids.insert(candidate_id) {
                return Err("Interpreter model registry contains duplicate evaluated candidate IDs.".to_string());
            }
        }
        if entry.evaluation_purpose.as_deref() == Some("promotion") {
            let benchmark = entry.evaluation_benchmark_sha256.as_deref().unwrap_or("");
            if !promotion_benchmarks.insert(benchmark) {
                return Err("A promotion benchmark was consumed by multiple candidates.".to_string());
            }
        }
    }
    if document.active.role == "frozen_base" {
        if document.active.entry_id.is_some() {
            return Err("Frozen-base registry selection names an unexpected candidate.".to_string());
        }
        return ModelSelection::frozen_base();
    }
    if document.active.role != "approved_candidate" {
        return Err("Interpreter model registry has an unsupported active role.".to_string());
    }
    let id = document.active.entry_id.as_deref()
        .ok_or_else(|| "Active candidate registry selection has no entry ID.".to_string())?;
    let entry = document.entries.iter().find(|item| item.entry_id == id)
        .ok_or_else(|| "Active candidate is absent from the interpreter registry.".to_string())?;
    if entry.lifecycle_state != "approved" {
        return Err("Active interpreter candidate is not in approved lifecycle state.".to_string());
    }
    Ok(ModelSelection {
        model_role: "approved_candidate".to_string(),
        model_tag: entry.candidate_tag.clone().ok_or_else(|| "Approved candidate has no model tag.".to_string())?,
        expected_digest: entry.observed_model_digest.clone(),
        expected_size: entry.model_size_bytes,
        expected_model_blob_sha256: entry.quantized_gguf_sha256.clone(),
        frozen_base_lock: None,
        registry_entry_sha256: Some(entry.entry_sha256.clone()),
        export_receipt_sha256: entry.export_receipt_sha256.clone(),
        training_report_sha256: Some(entry.training_report_sha256.clone()),
        candidate_evaluation_report_sha256: entry.candidate_evaluation_report_sha256.clone(),
        candidate_lifecycle_state: Some(entry.lifecycle_state.clone()),
        identity_verification: CANDIDATE_IDENTITY_VERIFICATION.to_string(),
    })
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    model: String,
    message: ResponseMessage,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    done_reason: String,
    #[serde(default)]
    total_duration: u64,
    #[serde(default)]
    load_duration: u64,
    #[serde(default)]
    prompt_eval_count: u64,
    #[serde(default)]
    eval_count: u64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StructuredOperation {
    op: String,
    path: Vec<String>,
    value: Value,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StructuredInterpreterOutput {
    schema: String,
    schema_version: String,
    action: String,
    context_sha256: String,
    context_requests: Vec<Vec<String>>,
    operations: Vec<StructuredOperation>,
    explanation: String,
    warnings: Vec<String>,
    clarification_question: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct GenerationOptions {
    seed: u64,
    temperature: f64,
    top_p: f64,
    top_k: u64,
    min_p: f64,
    num_ctx: u64,
    num_predict: u64,
}

fn generation_options(instruction_bytes: usize, context_bytes: usize) -> GenerationOptions {
    let prompt_bytes = instruction_bytes
        .saturating_add(context_bytes)
        .saturating_add(PROMPT_STATIC_OVERHEAD_BYTES);
    let estimated_prompt_tokens = prompt_bytes
        .saturating_add(PROMPT_TOKEN_ESTIMATE_DENOMINATOR - 1)
        / PROMPT_TOKEN_ESTIMATE_DENOMINATOR;
    let profile = [(32_768_u64, 4_096_u64), (65_536, 8_192), (131_072, 8_192)]
        .into_iter()
        .find(|(num_ctx, num_predict)| {
            (estimated_prompt_tokens as u64)
                .saturating_add(*num_predict)
                .saturating_add(PROMPT_TOKEN_SAFETY_MARGIN)
                <= *num_ctx
        })
        .unwrap_or((131_072, 8_192));
    GenerationOptions {
        seed: 0,
        temperature: 0.7,
        top_p: 0.8,
        top_k: 20,
        min_p: 0.0,
        num_ctx: profile.0,
        num_predict: profile.1,
    }
}

async fn bounded_json_response(response: Response, limit: usize) -> Result<Value, String> {
    let status = response.status();
    if response
        .content_length()
        .map(|length| length > limit as u64)
        .unwrap_or(false)
    {
        return Err("The local inference service response exceeded its safety limit.".to_string());
    }
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| {
            format!("Could not read the local inference service response: {error}")
        })?;
        if bytes.len().saturating_add(chunk.len()) > limit {
            return Err(
                "The local inference service response exceeded its safety limit.".to_string(),
            );
        }
        bytes.extend_from_slice(&chunk);
    }
    let value: Value = serde_json::from_slice(&bytes).map_err(|_| {
        "The local inference service returned a malformed JSON response.".to_string()
    })?;
    if !status.is_success() {
        let message = value
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("The local inference service rejected the request.");
        return Err(format!("Ollama returned {status}: {message}"));
    }
    Ok(value)
}


fn normalized_manifest_digest(value: &str) -> Result<String, String> {
    let value = value.trim().to_ascii_lowercase();
    let hex = value.strip_prefix("sha256:").unwrap_or(&value);
    if !valid_sha256(hex) {
        return Err("Ollama returned a malformed model digest.".to_string());
    }
    Ok(format!("sha256:{hex}"))
}

fn ollama_model_roots() -> Vec<PathBuf> {
    let mut roots = Vec::<PathBuf>::new();
    if let Some(value) = env::var_os("OLLAMA_MODELS") {
        if !value.is_empty() {
            roots.push(PathBuf::from(value));
        }
    }
    for variable in ["USERPROFILE", "HOME"] {
        if let Some(value) = env::var_os(variable) {
            if !value.is_empty() {
                roots.push(PathBuf::from(value).join(".ollama").join("models"));
            }
        }
    }
    roots.push(PathBuf::from("/usr/share/ollama/.ollama/models"));
    let mut seen = HashSet::new();
    roots
        .into_iter()
        .filter(|root| seen.insert(root.clone()))
        .collect()
}

fn safe_model_component(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')
        })
}

fn manifest_relative_path(model_tag: &str) -> Result<PathBuf, String> {
    if model_tag.contains('\\') {
        return Err("The selected Ollama tag has an unsafe path form.".to_string());
    }
    let (name, tag) = model_tag
        .rsplit_once(':')
        .ok_or_else(|| "The selected Ollama tag must include an explicit tag.".to_string())?;
    if !safe_model_component(tag) {
        return Err("The selected Ollama tag has an unsafe tag component.".to_string());
    }
    let components = name.split('/').collect::<Vec<_>>();
    if components.is_empty() || components.iter().any(|item| !safe_model_component(item)) {
        return Err("The selected Ollama tag has unsafe repository components.".to_string());
    }
    let explicit_registry = components.len() > 1
        && (components[0].contains('.') || components[0] == "localhost");
    let (registry, repository) = if explicit_registry {
        (components[0], &components[1..])
    } else {
        ("registry.ollama.ai", components.as_slice())
    };
    if repository.is_empty() {
        return Err("The selected Ollama tag has no repository component.".to_string());
    }
    let mut relative = PathBuf::from("manifests").join(registry);
    if repository.len() == 1 {
        relative.push("library");
    }
    for component in repository {
        relative.push(component);
    }
    relative.push(tag);
    Ok(relative)
}

fn manifest_candidates(model_tag: &str) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    let relative = manifest_relative_path(model_tag)?;
    Ok(ollama_model_roots()
        .into_iter()
        .map(|root| {
            let manifest = root.join(&relative);
            (root, manifest)
        })
        .collect())
}

async fn sha256_file(path: &Path) -> Result<(String, u64), String> {
    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|error| format!("Could not open local model artifact '{}': {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .await
            .map_err(|error| format!("Could not read local model artifact '{}': {error}", path.display()))?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .ok_or_else(|| "Local model artifact size overflowed u64.".to_string())?;
        digest.update(&buffer[..count]);
    }
    Ok((format!("{:x}", digest.finalize()), total))
}

async fn verify_blob_file(root: &Path, digest: &str) -> Result<u64, String> {
    let normalized = normalized_manifest_digest(digest)?;
    let hex = normalized
        .strip_prefix("sha256:")
        .ok_or_else(|| "Frozen-base blob digest normalization failed.".to_string())?;
    let path = root.join("blobs").join(format!("sha256-{hex}"));
    if !path.is_file() {
        return Err(format!(
            "Required local Ollama blob is missing: {}…",
            &normalized[..19]
        ));
    }
    let (observed, size) = sha256_file(&path).await?;
    if observed != hex {
        return Err(format!(
            "Local Ollama blob failed SHA-256 verification: {}…",
            &normalized[..19]
        ));
    }
    Ok(size)
}

async fn verify_selected_artifact(
    model: &OllamaModel,
    selection: &ModelSelection,
) -> Result<ArtifactEvidence, String> {
    let observed_manifest_digest = normalized_manifest_digest(&model.digest)?;
    let expected_manifest_sha256 = if let Some(lock) = selection.frozen_base_lock.as_ref() {
        lock.registry_manifest_sha256.as_str()
    } else {
        normalized_digest(selection.expected_digest.as_deref().unwrap_or(""))
            .ok_or_else(|| "Approved candidate has no exact manifest SHA-256.".to_string())?
    };
    if observed_manifest_digest != format!("sha256:{expected_manifest_sha256}") {
        return Err(
            "Installed Ollama tag digest does not match its exact locked manifest identity."
                .to_string(),
        );
    }
    let located = manifest_candidates(&selection.model_tag)?
        .into_iter()
        .find(|(_, manifest)| manifest.is_file())
        .ok_or_else(|| {
            "Cannot verify the selected model: the local Ollama manifest file was not found. Set OLLAMA_MODELS if Ollama uses a non-default model directory."
                .to_string()
        })?;
    let (root, manifest_path) = located;
    let raw = tokio::fs::read(&manifest_path)
        .await
        .map_err(|error| format!("Could not read local Ollama manifest: {error}"))?;
    if raw.len() > 1024 * 1024 {
        return Err("The local Ollama manifest exceeds its 1 MiB safety limit.".to_string());
    }
    let local_manifest_sha256 = sha256_hex(&raw);
    if format!("sha256:{local_manifest_sha256}") != observed_manifest_digest {
        return Err(
            "Ollama /api/tags digest does not match the SHA-256 of the local manifest bytes."
                .to_string(),
        );
    }
    let manifest: Value = serde_json::from_slice(&raw)
        .map_err(|_| "The local Ollama manifest is malformed JSON.".to_string())?;
    let object = manifest
        .as_object()
        .ok_or_else(|| "The local Ollama manifest must be an object.".to_string())?;
    let config = object
        .get("config")
        .and_then(Value::as_object)
        .ok_or_else(|| "The local Ollama manifest has no config identity.".to_string())?;
    let config_media_type = config
        .get("mediaType")
        .and_then(Value::as_str)
        .ok_or_else(|| "The local Ollama manifest config media type is malformed.".to_string())?;
    let config_digest = config
        .get("digest")
        .and_then(Value::as_str)
        .ok_or_else(|| "The local Ollama manifest config digest is malformed.".to_string())?;
    let config_size = config
        .get("size")
        .and_then(Value::as_u64)
        .ok_or_else(|| "The local Ollama manifest config size is malformed.".to_string())?;
    normalized_manifest_digest(config_digest)?;
    let layers = object
        .get("layers")
        .and_then(Value::as_array)
        .ok_or_else(|| "The local Ollama manifest has no layer list.".to_string())?;
    let mut by_type = BTreeMap::<String, (String, u64)>::new();
    for layer in layers {
        let layer = layer
            .as_object()
            .ok_or_else(|| "The local Ollama manifest contains a malformed layer.".to_string())?;
        let media_type = layer
            .get("mediaType")
            .and_then(Value::as_str)
            .ok_or_else(|| "The local Ollama manifest contains a malformed layer identity.".to_string())?;
        let digest = layer
            .get("digest")
            .and_then(Value::as_str)
            .ok_or_else(|| "The local Ollama manifest contains a malformed layer identity.".to_string())?;
        let size = layer
            .get("size")
            .and_then(Value::as_u64)
            .ok_or_else(|| "The local Ollama manifest contains a malformed layer size.".to_string())?;
        normalized_manifest_digest(digest)?;
        if size == 0 || by_type.insert(media_type.to_string(), (digest.to_string(), size)).is_some() {
            return Err(format!(
                "The local Ollama manifest duplicates layer type {media_type}."
            ));
        }
    }
    let model_record = by_type
        .get("application/vnd.ollama.image.model")
        .ok_or_else(|| "The local Ollama manifest has no model layer.".to_string())?;
    let expected_model_sha256 = if let Some(lock) = selection.frozen_base_lock.as_ref() {
        if config_media_type != lock.config.media_type
            || config_digest != format!("sha256:{}", lock.config.sha256)
            || config_size != lock.config.size_bytes
        {
            return Err("Installed Ollama config blob differs from the frozen lock.".to_string());
        }
        let actual_types = by_type.keys().cloned().collect::<HashSet<_>>();
        let expected_types = lock
            .layers
            .iter()
            .map(|layer| layer.media_type.clone())
            .collect::<HashSet<_>>();
        if actual_types != expected_types {
            return Err("Installed Ollama manifest has missing or extra frozen-base layers.".to_string());
        }
        for expected in &lock.layers {
            let (digest, size) = by_type.get(&expected.media_type).ok_or_else(|| {
                format!("Frozen-base manifest lacks required layer {}.", expected.media_type)
            })?;
            if digest != &format!("sha256:{}", expected.sha256) || *size != expected.size_bytes {
                return Err(format!(
                    "Installed Ollama {} layer differs from the frozen lock.", expected.media_type
                ));
            }
        }
        lock.layers
            .iter()
            .find(|layer| layer.media_type == "application/vnd.ollama.image.model")
            .map(|layer| layer.sha256.as_str())
            .ok_or_else(|| "Frozen lock has no model layer.".to_string())?
    } else {
        selection
            .expected_model_blob_sha256
            .as_deref()
            .filter(|digest| valid_sha256(digest))
            .ok_or_else(|| "Approved candidate has no exact GGUF SHA-256.".to_string())?
    };
    if model_record.0.as_str() != format!("sha256:{expected_model_sha256}") {
        return Err("Installed Ollama GGUF blob differs from its locked artifact.".to_string());
    }
    let mut verified_model_size = 0_u64;
    for (media_type, (digest, declared_size)) in &by_type {
        let size = verify_blob_file(&root, digest).await?;
        if size != *declared_size {
            return Err(format!("Local Ollama {media_type} blob size differs from its manifest."));
        }
        if media_type == "application/vnd.ollama.image.model" {
            verified_model_size = size;
        }
    }
    if verify_blob_file(&root, config_digest).await? != config_size {
        return Err("Local Ollama config blob size differs from its manifest.".to_string());
    }
    let artifact_lock_sha256 = selection
        .registry_entry_sha256
        .clone()
        .unwrap_or_else(|| sha256_hex(BASE_MODEL_LOCK_JSON.as_bytes()));
    Ok(ArtifactEvidence {
        identity_verification: selection.identity_verification.clone(),
        artifact_lock_sha256,
        local_manifest_sha256,
        model_blob_sha256: expected_model_sha256.to_string(),
        verified_model_blob_size_bytes: verified_model_size,
        all_manifest_blobs_verified: true,
    })
}

async fn runtime_identity(
    client: &Client,
    selection: &ModelSelection,
) -> Result<(String, OllamaModel, ArtifactEvidence), String> {
    let version_response = client
        .get(format!("{OLLAMA_ENDPOINT}/api/version"))
        .timeout(STATUS_TIMEOUT)
        .send()
        .await
        .map_err(|_| {
            "Ollama is not reachable on 127.0.0.1:11434. Install and start Ollama first."
                .to_string()
        })?;
    let version_value = bounded_json_response(version_response, MAX_PULL_RESPONSE_BYTES).await?;
    let version: OllamaVersion = serde_json::from_value(version_value)
        .map_err(|_| "Ollama returned an invalid version response.".to_string())?;
    if version.version.is_empty() || version.version.len() > 256 {
        return Err("Ollama returned an invalid runtime version.".to_string());
    }

    let tags_response = client
        .get(format!("{OLLAMA_ENDPOINT}/api/tags"))
        .timeout(STATUS_TIMEOUT)
        .send()
        .await
        .map_err(|error| format!("Could not inspect installed Ollama models: {error}"))?;
    let tags_value = bounded_json_response(tags_response, MAX_RESPONSE_BYTES).await?;
    let tags: OllamaTags = serde_json::from_value(tags_value)
        .map_err(|_| "Ollama returned an invalid model inventory.".to_string())?;
    let model = tags
        .models
        .into_iter()
        .find(|item| item.name == selection.model_tag || item.model == selection.model_tag)
        .ok_or_else(|| {
            format!(
                "The selected local model tag is not installed: {}.", selection.model_tag
            )
        })?;
    if model.digest.is_empty() || model.digest.len() > 256 {
        return Err("Ollama returned an invalid model digest.".to_string());
    }
    if selection.model_role == "frozen_base" && selection.frozen_base_lock.is_none() {
        return Err("Frozen-base selection has no bundled artifact lock.".to_string());
    }
    if selection.model_role != "frozen_base" {
        if let Some(expected) = selection.expected_digest.as_deref() {
            if normalized_digest(&model.digest) != normalized_digest(expected) {
                return Err(
                    "Installed candidate digest differs from its approved registry entry."
                        .to_string(),
                );
            }
        }
        if selection.expected_size.is_some_and(|expected| expected != model.size) {
            return Err(
                "Installed candidate size differs from its approved registry entry.".to_string(),
            );
        }
    }
    let evidence = verify_selected_artifact(&model, selection).await?;
    Ok((version.version, model, evidence))
}

fn provider_identity(
    runtime_version: &str,
    model: &OllamaModel,
    options: GenerationOptions,
    selection: &ModelSelection,
    evidence: &ArtifactEvidence,
) -> Value {
    json!({
        "provider": "ollama",
        "endpoint": OLLAMA_ENDPOINT,
        "runtime_version": runtime_version,
        "model_tag": selection.model_tag.as_str(),
        "observed_model_digest": model.digest.as_str(),
        "identity_verification": selection.identity_verification.as_str(),
        "expected_base_model": BASE_MODEL,
        "expected_quantization": QUANTIZATION,
        "generation_options": options,
        "model_role": selection.model_role.as_str(),
        "registry_entry_sha256": selection.registry_entry_sha256.as_deref(),
        "artifact_evidence": evidence
    })
}

fn output_schema() -> Value {
    serde_json::from_str(include_str!(
        "../../model_lab/baseline/output_schema_v1.2.json"
    ))
    .expect("bundled interpreter output schema must be valid JSON")
}


fn validate_input(
    value: &str,
    name: &str,
    maximum: usize,
    allow_empty: bool,
) -> Result<(), String> {
    if value.len() > maximum {
        return Err(format!("{name} exceeds its {maximum}-byte safety limit."));
    }
    if value.contains('\0') {
        return Err(format!("{name} must not contain NUL characters."));
    }
    if !allow_empty && value.trim().is_empty() {
        return Err(format!("{name} must not be empty."));
    }
    Ok(())
}

fn context_json(context_package: &Value) -> Result<String, String> {
    let object = context_package
        .as_object()
        .ok_or_else(|| "Interpreter context package must be an object.".to_string())?;
    if object.get("schema").and_then(Value::as_str)
        != Some("model-laboratory-interpreter-context")
        || object.get("schema_version").and_then(Value::as_str) != Some("1.2")
    {
        return Err("Unsupported interpreter context package.".to_string());
    }
    let digest = object
        .get("context_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "Interpreter context package has no checksum.".to_string())?;
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()) {
        return Err("Interpreter context package checksum is invalid.".to_string());
    }
    let encoded = serde_json::to_string(context_package)
        .map_err(|_| "Interpreter context package is not valid JSON.".to_string())?;
    validate_input(
        &encoded,
        "Interpreter context package",
        MAX_CONTEXT_PACKAGE_BYTES,
        false,
    )?;
    Ok(encoded)
}

fn user_prompt(instruction: &str, context: &str) -> String {
    USER_PROMPT_TEMPLATE
        .replacen("{{INSTRUCTION}}", instruction, 1)
        .replacen("{{CONTEXT_JSON}}", context, 1)
        .trim_end_matches('\n')
        .to_string()
}

#[tauri::command]
pub async fn interpreter_status(
    state: tauri::State<'_, InterpreterState>,
) -> Result<Value, String> {
    let active = state.active_summary().await;
    let selection = match model_selection(&state.registry_path).await {
        Ok(value) => value,
        Err(error) => return Ok(json!({
            "status": "registry_invalid",
            "runtime_available": false,
            "runtime": "ollama",
            "endpoint": OLLAMA_ENDPOINT,
            "model_tag": OLLAMA_MODEL,
            "model_role": "frozen_base",
            "expected_base_model": BASE_MODEL,
            "expected_quantization": QUANTIZATION,
            "model_installed": false,
            "active": active,
            "message": error
        })),
    };
    match runtime_identity(&state.client, &selection).await {
        Ok((version, model, artifact_evidence)) => {
            let verified_size = artifact_evidence.verified_model_blob_size_bytes;
            Ok(json!({
                "status": "ready",
                "runtime_available": true,
                "runtime": "ollama",
                "runtime_version": version,
                "endpoint": OLLAMA_ENDPOINT,
                "model_tag": selection.model_tag,
                "model_role": selection.model_role,
                "expected_base_model": BASE_MODEL,
                "expected_quantization": QUANTIZATION,
                "identity_verification": selection.identity_verification,
                "registry_entry_sha256": selection.registry_entry_sha256,
                "training_report_sha256": selection.training_report_sha256,
                "candidate_evaluation_report_sha256": selection.candidate_evaluation_report_sha256,
                "candidate_lifecycle_state": selection.candidate_lifecycle_state,
                "model_installed": true,
                "observed_model_digest": model.digest,
                "model_size_bytes": verified_size,
                "artifact_evidence": artifact_evidence,
                "active": active
            }))
        },
        Err(error) => {
            let model_missing = error.starts_with("The selected local model tag is not installed");
            Ok(json!({
                "status": if model_missing { "model_missing" } else { "runtime_unavailable" },
                "runtime_available": model_missing,
                "runtime": "ollama",
                "endpoint": OLLAMA_ENDPOINT,
                "model_tag": selection.model_tag,
                "model_role": selection.model_role,
                "expected_base_model": BASE_MODEL,
                "expected_quantization": QUANTIZATION,
                "identity_verification": selection.identity_verification,
                "registry_entry_sha256": selection.registry_entry_sha256,
                "training_report_sha256": selection.training_report_sha256,
                "candidate_evaluation_report_sha256": selection.candidate_evaluation_report_sha256,
                "candidate_lifecycle_state": selection.candidate_lifecycle_state,
                "model_installed": false,
                "active": active,
                "message": error
            }))
        }
    }
}

#[tauri::command]
pub async fn interpreter_pull_model(
    state: tauri::State<'_, InterpreterState>,
    confirmed: bool,
) -> Result<Value, String> {
    if !confirmed {
        return Err("Downloading the local model requires explicit confirmation.".to_string());
    }
    let selection = model_selection(&state.registry_path).await?;
    if selection.model_role != "frozen_base" {
        return Err(
            "Approved candidates cannot be pulled by tag. Reinstall the exact export or roll back to the frozen base model."
                .to_string(),
        );
    }
    let (request_id, cancelled) = state.begin("model pull").await?;
    let operation = async {
        let response = state
            .client
            .post(format!("{OLLAMA_ENDPOINT}/api/pull"))
            .timeout(PULL_TIMEOUT)
            .json(&json!({"model": OLLAMA_MODEL, "stream": false}))
            .send()
            .await
            .map_err(|_| {
                "Ollama is not reachable on 127.0.0.1:11434. Install and start Ollama first."
                    .to_string()
            })?;
        bounded_json_response(response, MAX_PULL_RESPONSE_BYTES).await
    };
    let result = tokio::select! {
        _ = cancelled => Err("The local model download was cancelled.".to_string()),
        value = operation => value,
    };
    state.finish(&request_id).await;
    result.map(|response| {
        json!({
            "request_id": request_id,
            "model": OLLAMA_MODEL,
            "response": response
        })
    })
}

#[tauri::command]
pub async fn interpreter_request(
    state: tauri::State<'_, InterpreterState>,
    instruction: String,
    context_package: Value,
) -> Result<Value, String> {
    validate_input(
        &instruction,
        "Interpreter instruction",
        MAX_INSTRUCTION_BYTES,
        false,
    )?;
    let context = context_json(&context_package)?;
    let context_digest = context_package["context_sha256"]
        .as_str()
        .ok_or_else(|| "Interpreter context package has no checksum.".to_string())?
        .to_string();
    let selection = model_selection(&state.registry_path).await?;
    let (runtime_version, model, artifact_evidence) = runtime_identity(&state.client, &selection).await?;
    let prompt = user_prompt(&instruction, &context);
    let options = generation_options(instruction.len(), context.len());
    let (request_id, cancelled) = state.begin("inference").await?;
    let operation = async {
        let response = state
            .client
            .post(format!("{OLLAMA_ENDPOINT}/api/chat"))
            .timeout(INFERENCE_TIMEOUT)
            .json(&json!({
                "model": selection.model_tag.as_str(),
                "messages": [
                    ChatMessage { role: "system", content: SYSTEM_PROMPT },
                    ChatMessage { role: "user", content: &prompt }
                ],
                "stream": false,
                "format": output_schema(),
                "keep_alive": "10m",
                "options": options
            }))
            .send()
            .await
            .map_err(|error| format!("Local inference failed: {error}"))?;
        let value = bounded_json_response(response, MAX_RESPONSE_BYTES).await?;
        let chat: ChatResponse = serde_json::from_value(value)
            .map_err(|_| "Ollama returned a malformed chat response.".to_string())?;
        if !chat.done || chat.model != selection.model_tag {
            return Err(
                "Ollama did not complete the request with the required model.".to_string(),
            );
        }
        validate_input(
            &chat.message.content,
            "Interpreter edit program",
            MAX_EDIT_PROGRAM_BYTES,
            false,
        )?;
        let output: StructuredInterpreterOutput = serde_json::from_str(&chat.message.content)
            .map_err(|_| {
                "The local model response did not satisfy the required structured-output schema."
                    .to_string()
            })?;
        if output.schema.as_str() != "model-laboratory-interpreter-output"
            || output.schema_version.as_str() != "1.2"
            || output.context_sha256.as_str() != context_digest.as_str()
        {
            return Err(
                "The local model response was not bound to the supplied interpreter context."
                    .to_string(),
            );
        }
        validate_input(
            &output.explanation,
            "Proposal explanation",
            16 * 1024,
            true,
        )?;
        validate_input(
            &output.clarification_question,
            "Clarification question",
            16 * 1024,
            true,
        )?;
        if output.warnings.len() > 32
            || output
                .warnings
                .iter()
                .any(|warning| {
                    validate_input(warning, "Proposal warning", 2 * 1024, false).is_err()
                })
        {
            return Err("The local model returned an invalid warning list.".to_string());
        }
        if output.context_requests.len() > 32 || output.operations.len() > 128 {
            return Err("The local model returned too many context requests or edits.".to_string());
        }
        Ok(json!({
            "output": output,
            "telemetry": {
                "done_reason": chat.done_reason,
                "total_duration_nanoseconds": chat.total_duration,
                "load_duration_nanoseconds": chat.load_duration,
                "prompt_tokens": chat.prompt_eval_count,
                "generated_tokens": chat.eval_count
            }
        }))
    };
    let result = tokio::select! {
        _ = cancelled => Err("The local interpreter request was cancelled.".to_string()),
        value = operation => value,
    };
    state.finish(&request_id).await;
    result.map(|mut response| {
        response["request_id"] = Value::String(request_id);
        response["provider_identity"] = provider_identity(
            &runtime_version,
            &model,
            options,
            &selection,
            &artifact_evidence,
        );
        response
    })
}

#[tauri::command]
pub async fn interpreter_cancel(
    state: tauri::State<'_, InterpreterState>,
) -> Result<Value, String> {
    let mut active = state.active.lock().await;
    let Some(operation) = active.take() else {
        return Ok(json!({"cancelled": false, "message": "No interpreter operation is active."}));
    };
    let id = operation.id;
    let kind = operation.kind;
    let _ = operation.cancel.send(());
    Ok(json!({"cancelled": true, "request_id": id, "operation": kind}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_schema_forbids_unreviewed_fields() {
        let schema = output_schema();
        assert_eq!(schema["additionalProperties"], Value::Bool(false));
        assert_eq!(schema["required"].as_array().map(Vec::len), Some(9));
        assert!(schema["properties"].get("model_source").is_none());
    }

    #[test]
    fn endpoint_and_model_are_fixed() {
        assert_eq!(OLLAMA_ENDPOINT, "http://127.0.0.1:11434");
        assert_eq!(OLLAMA_MODEL, "qwen3:4b-instruct-2507-q4_K_M");
    }

    #[test]
    fn frozen_base_selection_is_bound_to_the_bundled_artifact_lock() {
        let selection = ModelSelection::frozen_base().expect("bundled frozen-base lock");
        let lock = selection.frozen_base_lock.expect("frozen-base selection must carry a lock");
        assert_eq!(lock.model_tag, OLLAMA_MODEL);
        assert_eq!(lock.quantization, QUANTIZATION);
        assert_eq!(
            lock.layers.iter()
                .find(|layer| layer.media_type == "application/vnd.ollama.image.model")
                .map(|layer| layer.sha256.as_str()),
            Some("85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9")
        );
        assert_eq!(
            sha256_hex(BASE_MODEL_LOCK_JSON.as_bytes()),
            "1b73f807ea8c2a53088ae5e38b9b376f3cd9acd8213bf7c881af728018c7f38e"
        );
        assert_eq!(selection.model_role, "frozen_base");
    }

    #[test]
    fn ollama_manifest_digest_normalization_is_strict() {
        let digest = "a".repeat(64);
        assert_eq!(
            normalized_manifest_digest(&digest).expect("bare digest"),
            format!("sha256:{digest}")
        );
        assert_eq!(
            normalized_manifest_digest(&format!("sha256:{digest}")).expect("prefixed digest"),
            format!("sha256:{digest}")
        );
        assert!(normalized_manifest_digest("0edcdef34593").is_err());
    }

    #[tokio::test]
    #[ignore = "requires the exact frozen Ollama model to be installed locally"]
    async fn live_frozen_base_identity_smoke() {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(3))
            .no_proxy()
            .build()
            .expect("loopback client");
        let selection = ModelSelection::frozen_base().expect("bundled frozen-base lock");
        let (_version, model, evidence) = runtime_identity(&client, &selection)
            .await
            .expect("exact frozen-base artifact must verify");
        assert!(model.name == selection.model_tag || model.model == selection.model_tag);
        assert_eq!(
            evidence.identity_verification,
            FROZEN_BASE_IDENTITY_VERIFICATION
        );
        assert_eq!(
            evidence.model_blob_sha256,
            "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
        );
        assert!(evidence.verified_model_blob_size_bytes > 0);
    }

    #[test]
    fn structured_output_rejects_provider_fields() {
        let response = r#"{
            "schema":"model-laboratory-interpreter-output",
            "schema_version":"1.2",
            "action":"unable",
            "context_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "context_requests":[],
            "operations":[],
            "explanation":"proposal",
            "warnings":[],
            "clarification_question":"",
            "provider":"forged"
        }"#;
        assert!(serde_json::from_str::<StructuredInterpreterOutput>(response).is_err());
    }

    #[test]
    fn interpreter_inputs_are_bounded_before_network_use() {
        assert!(validate_input("request", "instruction", 64, false).is_ok());
        assert!(validate_input("", "instruction", 64, false).is_err());
        assert!(validate_input("too long", "instruction", 3, false).is_err());
        assert!(validate_input("bad\0value", "instruction", 64, false).is_err());
    }

    #[test]
    fn context_and_generation_budgets_are_independent_of_source_size() {
        assert_eq!(MAX_CONTEXT_PACKAGE_BYTES, 32 * 1024);
        let small = generation_options(1024, 8 * 1024);
        let medium = generation_options(50 * 1024, 1024);
        let large = generation_options(64 * 1024, 30 * 1024);
        assert_eq!((small.num_ctx, small.num_predict), (32_768, 4_096));
        assert_eq!((medium.num_ctx, medium.num_predict), (65_536, 8_192));
        assert_eq!((large.num_ctx, large.num_predict), (131_072, 8_192));
        assert!(!SYSTEM_PROMPT.contains("complete model source"));
        assert!(!SYSTEM_PROMPT.contains("model_source"));
    }
}
