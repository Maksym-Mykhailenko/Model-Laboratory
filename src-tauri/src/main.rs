#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod interpreter;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use interpreter::{
    interpreter_cancel, interpreter_pull_model, interpreter_request, interpreter_status,
    InterpreterState,
};
use serde::Serialize;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Mutex as StdMutex;
use std::time::{Duration, SystemTime};
use tokio::io::AsyncWriteExt;
use tauri::async_runtime::{Mutex as AsyncMutex, Receiver};
use tauri::{AppHandle, Manager};
#[cfg(target_os = "macos")]
use tauri::Emitter;
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};

// The native document ceiling matches the portable .mlab ceiling. Base64 expands a
// 128 MiB bundle to roughly 171 MiB; the JSON request boundary leaves bounded headroom.
// Embedded members are subsequently exposed to renderers as one-MiB content chunks.
const MAX_DOCUMENT_BYTES: u64 = 128 * 1024 * 1024;
const MAX_INTERPRETER_ATTACHMENT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ENGINE_REQUEST_BYTES: usize = 192 * 1024 * 1024;
const MAX_ENGINE_RESPONSE_BYTES: usize = 192 * 1024 * 1024;
const MAX_ENGINE_STDERR_BYTES: usize = 1024 * 1024;
const STALE_DOCUMENT_TEMP_AGE: Duration = Duration::from_secs(24 * 60 * 60);

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OpenedDocument {
    path: String,
    name: String,
    data_base64: String,
}

struct PendingDocuments(StdMutex<VecDeque<PathBuf>>);

struct EngineProcess {
    events: Receiver<CommandEvent>,
    child: Option<CommandChild>,
    stdout_buffer: Vec<u8>,
}

fn take_response_line(buffer: &mut Vec<u8>) -> Result<Option<String>, String> {
    let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') else {
        if buffer.len() > MAX_ENGINE_RESPONSE_BYTES {
            return Err("The scientific engine response exceeded the desktop safety limit."
                .to_string());
        }
        return Ok(None);
    };
    if newline > MAX_ENGINE_RESPONSE_BYTES {
        return Err(
            "The scientific engine response exceeded the desktop safety limit.".to_string(),
        );
    }
    let mut framed = buffer.drain(..=newline).collect::<Vec<_>>();
    framed.pop();
    if framed.last() == Some(&b'\r') {
        framed.pop();
    }
    if framed.is_empty() {
        return Err("The scientific engine returned an empty response frame.".to_string());
    }
    String::from_utf8(framed)
        .map(Some)
        .map_err(|_| "The scientific engine returned invalid UTF-8.".to_string())
}

fn validate_response_identity(request_json: &str, response_json: &str) -> Result<(), String> {
    let request: serde_json::Value = serde_json::from_str(request_json)
        .map_err(|_| "The desktop submitted an invalid scientific request.".to_string())?;
    let response: serde_json::Value = serde_json::from_str(response_json)
        .map_err(|_| "The scientific engine returned malformed JSON.".to_string())?;
    let request_id = request
        .get("id")
        .and_then(serde_json::Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or_else(|| "The desktop scientific request has an invalid id.".to_string())?;
    let response_id = response
        .get("id")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| "The scientific engine response has an invalid id.".to_string())?;
    if response_id != request_id {
        return Err("The scientific engine returned a response for the wrong request.".to_string());
    }
    if !response.get("ok").is_some_and(serde_json::Value::is_boolean) {
        return Err("The scientific engine response has no valid status.".to_string());
    }
    Ok(())
}

fn engine_request_timeout(request_json: &str) -> Duration {
    let action = serde_json::from_str::<serde_json::Value>(request_json)
        .ok()
        .and_then(|request| {
            request
                .get("action")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        });
    match action.as_deref() {
        Some(
            "health"
            | "example_catalogue"
            | "example_model"
            | "inspect_model"
            | "inspect_experiment"
            | "inspect_committed_experiment"
            | "read_content_chunk"
            | "release_interpreter_attachment",
        ) => Duration::from_secs(120),
        Some(
            "analyse_model"
            | "run_sweep"
            | "run_capability"
            | "reproduce_experiment"
            | "prepare_run_experiment"
            | "finalize_experiment",
        ) => Duration::from_secs(15 * 60),
        _ => Duration::from_secs(5 * 60),
    }
}

impl EngineProcess {
    async fn request(&mut self, request_json: &str) -> Result<String, String> {
        if self.stdout_buffer.iter().any(|byte| !byte.is_ascii_whitespace()) {
            return Err(
                "The scientific engine produced an unsolicited response frame.".to_string(),
            );
        }
        self.stdout_buffer.clear();
        let child = self
            .child
            .as_mut()
            .ok_or_else(|| "The scientific engine process is not available.".to_string())?;
        child
            .write(format!("{request_json}\n").as_bytes())
            .map_err(|error| format!("Could not send work to the scientific engine: {error}"))?;

        let mut stderr = Vec::new();
        while let Some(event) = self.events.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    let combined = self
                        .stdout_buffer
                        .len()
                        .checked_add(bytes.len())
                        .ok_or_else(|| {
                            "The scientific engine response exceeded the desktop safety limit."
                                .to_string()
                        })?;
                    if combined > MAX_ENGINE_RESPONSE_BYTES + 1 {
                        return Err(
                            "The scientific engine response exceeded the desktop safety limit."
                                .to_string(),
                        );
                    }
                    self.stdout_buffer.extend_from_slice(&bytes);
                    if let Some(response) = take_response_line(&mut self.stdout_buffer)? {
                        return Ok(response);
                    }
                }
                CommandEvent::Stderr(bytes) => {
                    if stderr.len() < MAX_ENGINE_STDERR_BYTES {
                        let remaining = MAX_ENGINE_STDERR_BYTES - stderr.len();
                        stderr.extend_from_slice(&bytes[..bytes.len().min(remaining)]);
                        stderr.push(b'\n');
                    }
                }
                CommandEvent::Terminated(payload) => {
                    return Err(format!(
                        "The scientific engine terminated{}: {}",
                        payload
                            .code
                            .map(|code| format!(" with exit code {code}"))
                            .unwrap_or_default(),
                        String::from_utf8_lossy(&stderr).trim()
                    ));
                }
                CommandEvent::Error(message) => {
                    return Err(format!("The scientific engine failed: {message}"));
                }
                _ => {}
            }
        }
        Err("The scientific engine closed its response channel.".to_string())
    }

    fn terminate(mut self) {
        if let Some(child) = self.child.take() {
            let _ = child.kill();
        }
    }
}

struct ScientificEngine(AsyncMutex<Option<EngineProcess>>);

fn extensions(kind: &str) -> (&'static str, &'static [&'static str]) {
    match kind {
        "model" => ("Model source", &["yaml", "yml"]),
        "experiment" => ("Model Laboratory experiment", &["mlab", "json"]),
        "commit" => ("Committed experiment receipt", &["json"]),
        "report-json" => ("JSON report", &["json"]),
        "report-text" => ("Text report", &["txt"]),
        "interpreter-attachment" => (
            "Interpreter reference",
            &["txt", "md", "yaml", "yml", "json", "csv", "tsv", "pdf"],
        ),
        _ => (
            "Model Laboratory document",
            &["mlab", "yaml", "yml", "json"],
        ),
    }
}

fn document_limit(kind: &str) -> u64 {
    if kind == "interpreter-attachment" {
        MAX_INTERPRETER_ATTACHMENT_BYTES
    } else {
        MAX_DOCUMENT_BYTES
    }
}

async fn read_document(path: PathBuf, maximum_bytes: u64) -> Result<OpenedDocument, String> {
    let metadata = tokio::fs::metadata(&path)
        .await
        .map_err(|error| format!("Could not inspect '{}': {error}", path.display()))?;
    if !metadata.is_file() {
        return Err("The selected path is not a file.".to_string());
    }
    if metadata.len() > maximum_bytes {
        return Err(format!(
            "The selected file is larger than the {} MiB desktop safety limit.",
            maximum_bytes / 1024 / 1024
        ));
    }
    let bytes = tokio::fs::read(&path)
        .await
        .map_err(|error| format!("Could not read '{}': {error}", path.display()))?;
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("document")
        .to_string();
    Ok(OpenedDocument {
        path: path.to_string_lossy().into_owned(),
        name,
        data_base64: BASE64.encode(bytes),
    })
}

#[tauri::command]
async fn open_document(kind: String) -> Result<Option<OpenedDocument>, String> {
    let (label, allowed) = extensions(&kind);
    let maximum_bytes = document_limit(&kind);
    let selected = rfd::AsyncFileDialog::new()
        .add_filter(label, allowed)
        .pick_file()
        .await;
    match selected {
        Some(file) => read_document(file.path().to_path_buf(), maximum_bytes).await.map(Some),
        None => Ok(None),
    }
}

#[tauri::command]
async fn save_document(
    kind: String,
    suggested_name: String,
    data_base64: String,
    path: Option<String>,
) -> Result<Option<String>, String> {
    let estimated_bytes = (data_base64.len() / 4).saturating_mul(3);
    if estimated_bytes > MAX_DOCUMENT_BYTES as usize {
        return Err(format!(
            "The document is larger than the {} MiB desktop safety limit.",
            MAX_DOCUMENT_BYTES / 1024 / 1024
        ));
    }
    let bytes = BASE64
        .decode(data_base64.as_bytes())
        .map_err(|_| "The document payload is not valid base64.".to_string())?;
    if bytes.len() > MAX_DOCUMENT_BYTES as usize {
        return Err(format!(
            "The decoded document is larger than the {} MiB desktop safety limit.",
            MAX_DOCUMENT_BYTES / 1024 / 1024
        ));
    }
    let (label, allowed) = extensions(&kind);
    let target = if let Some(existing) = path {
        let candidate = PathBuf::from(existing);
        let extension = candidate
            .extension()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase)
            .ok_or_else(|| "The save target has no supported extension.".to_string())?;
        if !allowed.contains(&extension.as_str()) {
            return Err(format!("The save target is not a supported {label} file."));
        }
        candidate
    } else {
        let selected = rfd::AsyncFileDialog::new()
            .add_filter(label, allowed)
            .set_file_name(&suggested_name)
            .save_file()
            .await;
        let Some(file) = selected else {
            return Ok(None);
        };
        file.path().to_path_buf()
    };
    atomic_write_document(&target, &bytes).await?;
    Ok(Some(target.to_string_lossy().into_owned()))
}

async fn atomic_write_document(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "The save target has no valid filename.".to_string())?;
    cleanup_stale_document_temps(parent, name).await;
    let temporary = parent.join(format!(".{name}.{}.tmp", uuid::Uuid::new_v4()));

    let result = async {
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .await
            .map_err(|error| format!("Could not create a temporary save file: {error}"))?;
        file.write_all(bytes)
            .await
            .map_err(|error| format!("Could not write the temporary save file: {error}"))?;
        file.sync_all()
            .await
            .map_err(|error| format!("Could not flush the temporary save file: {error}"))?;
        drop(file);
        replace_file(&temporary, path)
            .map_err(|error| format!("Could not safely replace '{}': {error}", path.display()))
    }
    .await;

    if result.is_err() {
        let _ = tokio::fs::remove_file(&temporary).await;
    }
    result
}

fn is_document_temp_file(target_name: &str, candidate_name: &str) -> bool {
    let prefix = format!(".{target_name}.");
    let Some(identifier) = candidate_name
        .strip_prefix(&prefix)
        .and_then(|value| value.strip_suffix(".tmp"))
    else {
        return false;
    };
    uuid::Uuid::parse_str(identifier).is_ok()
}

async fn cleanup_stale_document_temps(parent: &Path, target_name: &str) {
    let Ok(mut entries) = tokio::fs::read_dir(parent).await else {
        return;
    };
    let mut inspected = 0_u16;
    while inspected < 256 {
        let Ok(Some(entry)) = entries.next_entry().await else {
            break;
        };
        let Some(candidate_name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if !is_document_temp_file(target_name, &candidate_name) {
            continue;
        }
        inspected += 1;
        let Ok(metadata) = entry.metadata().await else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        let stale = metadata
            .modified()
            .ok()
            .and_then(|modified| SystemTime::now().duration_since(modified).ok())
            .is_some_and(|age| age >= STALE_DOCUMENT_TEMP_AGE);
        if stale {
            let _ = tokio::fs::remove_file(entry.path()).await;
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::rename(source, destination)
}

#[cfg(target_os = "windows")]
fn replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "Kernel32")]
    extern "system" {
        fn MoveFileExW(existing: *const u16, replacement: *const u16, flags: u32) -> i32;
    }

    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;
    let source_wide = source.as_os_str().encode_wide().chain(Some(0)).collect::<Vec<_>>();
    let destination_wide = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let moved = unsafe {
        MoveFileExW(
            source_wide.as_ptr(),
            destination_wide.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if moved == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn is_startup_document(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "mlab" | "yaml" | "yml" | "json"
            )
        })
        .unwrap_or(false)
}

#[tauri::command]
async fn startup_document(
    pending: tauri::State<'_, PendingDocuments>,
) -> Result<Option<OpenedDocument>, String> {
    let candidate = pending
        .0
        .lock()
        .map_err(|_| "The pending-document queue is unavailable.".to_string())?
        .front()
        .cloned();
    match candidate {
        Some(path) => read_document(path, MAX_DOCUMENT_BYTES).await.map(Some),
        None => Ok(None),
    }
}

fn acknowledge_pending_document(queue: &mut VecDeque<PathBuf>, expected_path: &str) -> bool {
    let matches = queue
        .front()
        .is_some_and(|path| path.to_string_lossy() == expected_path);
    if matches {
        queue.pop_front();
    }
    matches
}

#[tauri::command]
fn ack_startup_document(
    pending: tauri::State<'_, PendingDocuments>,
    path: String,
) -> Result<bool, String> {
    let mut queue = pending
        .0
        .lock()
        .map_err(|_| "The pending-document queue is unavailable.".to_string())?;
    Ok(acknowledge_pending_document(&mut queue, &path))
}

fn start_engine(app: &AppHandle) -> Result<EngineProcess, String> {
    let spawned = if cfg!(debug_assertions) {
        let executable = std::env::var("MODEL_LAB_PYTHON").unwrap_or_else(|_| {
            if cfg!(target_os = "windows") {
                "python".to_string()
            } else {
                "python3".to_string()
            }
        });
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or_else(|| "Could not resolve the project directory.".to_string())?
            .to_path_buf();
        app.shell()
            .command(executable)
            .arg(project_root.join("desktop_engine.py"))
            .current_dir(project_root)
            .set_raw_out(true)
            .spawn()
    } else {
        app.shell()
            .sidecar("model-lab-engine")
            .map_err(|error| format!("Could not locate the bundled scientific engine: {error}"))?
            .set_raw_out(true)
            .spawn()
    };
    let (events, child) =
        spawned.map_err(|error| format!("Could not start the scientific engine: {error}"))?;
    Ok(EngineProcess {
        events,
        child: Some(child),
        stdout_buffer: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        acknowledge_pending_document, engine_request_timeout, is_document_temp_file,
        take_response_line, validate_response_identity,
    };
    use std::collections::VecDeque;
    use std::path::PathBuf;
    use std::time::Duration;

    #[test]
    fn response_framing_waits_for_a_complete_line() {
        let mut buffer = br#"{"ok":tr"#.to_vec();
        assert!(take_response_line(&mut buffer).unwrap().is_none());
        buffer.extend_from_slice(b"ue}\n");
        assert_eq!(take_response_line(&mut buffer).unwrap().unwrap(), r#"{"ok":true}"#);
        assert!(buffer.is_empty());
    }

    #[test]
    fn response_framing_retains_surplus_bytes() {
        let mut buffer = b"first\r\nsecond\n".to_vec();
        assert_eq!(take_response_line(&mut buffer).unwrap().unwrap(), "first");
        assert_eq!(buffer, b"second\n");
    }

    #[test]
    fn response_framing_rejects_empty_frames() {
        let mut buffer = b"\n".to_vec();
        assert!(take_response_line(&mut buffer).is_err());
    }

    #[test]
    fn response_identity_must_match_the_request() {
        assert!(validate_response_identity(
            r#"{"id":7,"action":"health","payload":{}}"#,
            r#"{"id":7,"ok":true,"result":{}}"#,
        )
        .is_ok());
        assert!(validate_response_identity(
            r#"{"id":7,"action":"health","payload":{}}"#,
            r#"{"id":8,"ok":true,"result":{}}"#,
        )
        .is_err());
    }

    #[test]
    fn response_timeout_is_finite_and_scales_for_computation() {
        let health = engine_request_timeout(r#"{"id":1,"action":"health","payload":{}}"#);
        let analysis =
            engine_request_timeout(r#"{"id":2,"action":"analyse_model","payload":{}}"#);
        let unknown = engine_request_timeout(r#"{"id":3,"action":"future","payload":{}}"#);
        assert_eq!(health, Duration::from_secs(120));
        assert_eq!(analysis, Duration::from_secs(15 * 60));
        assert_eq!(unknown, Duration::from_secs(5 * 60));
    }

    #[test]
    fn pending_document_is_removed_only_after_matching_acknowledgement() {
        let mut queue =
            VecDeque::from([PathBuf::from("first.mlab"), PathBuf::from("second.yaml")]);
        assert!(!acknowledge_pending_document(&mut queue, "second.yaml"));
        assert_eq!(queue.len(), 2);
        assert!(acknowledge_pending_document(&mut queue, "first.mlab"));
        assert_eq!(queue.front(), Some(&PathBuf::from("second.yaml")));
    }

    #[test]
    fn document_temp_cleanup_matches_only_our_atomic_save_files() {
        assert!(is_document_temp_file(
            "model.yaml",
            ".model.yaml.123e4567-e89b-12d3-a456-426614174000.tmp",
        ));
        assert!(!is_document_temp_file(
            "model.yaml",
            ".other.yaml.123e4567-e89b-12d3-a456-426614174000.tmp",
        ));
        assert!(!is_document_temp_file(
            "model.yaml",
            ".model.yaml.not-a-uuid.tmp",
        ));
    }
}

/*
The scientific sidecar is deliberately persistent.  Python, SymPy, SciPy, and the
PyInstaller one-file payload are initialized once, while the mutex serializes the
newline-delimited request/response protocol.  A protocol failure discards and terminates
the process; the next request starts a clean replacement.
*/
#[tauri::command]
async fn engine_request(
    app: AppHandle,
    engine: tauri::State<'_, ScientificEngine>,
    request_json: String,
) -> Result<String, String> {
    if request_json.len() > MAX_ENGINE_REQUEST_BYTES {
        return Err("The scientific request exceeded the desktop safety limit.".to_string());
    }

    let mut process = engine.0.lock().await;
    if process.is_none() {
        *process = Some(start_engine(&app)?);
    }
    let request_timeout = engine_request_timeout(&request_json);
    let result = match tokio::time::timeout(
        request_timeout,
        process
            .as_mut()
            .expect("scientific engine initialized")
            .request(&request_json),
    )
    .await
    {
        Ok(response) => response.and_then(|response| {
            validate_response_identity(&request_json, &response)?;
            Ok(response)
        }),
        Err(_) => Err(format!(
            "The scientific engine timed out after {} seconds and was restarted.",
            request_timeout.as_secs()
        )),
    };
    if result.is_err() {
        if let Some(failed) = process.take() {
            failed.terminate();
        }
    }
    result
}

fn main() {
    let initial_documents = std::env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .filter(|path| path.is_file() && is_startup_document(path))
        .collect::<VecDeque<_>>();

    tauri::Builder::default()
        .manage(PendingDocuments(StdMutex::new(initial_documents)))
        .manage(ScientificEngine(AsyncMutex::new(None)))
        .setup(|app| {
            let registry_path = app
                .path()
                .app_data_dir()?
                .join("interpreter-model-registry.json");
            app.manage(InterpreterState::new(registry_path));
            Ok(())
        })
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            engine_request,
            open_document,
            save_document,
            startup_document,
            ack_startup_document,
            interpreter_status,
            interpreter_pull_model,
            interpreter_request,
            interpreter_cancel
        ])
        .build(tauri::generate_context!())
        .expect("error while building Model Laboratory")
        .run(|_app, _event| {
            #[cfg(target_os = "macos")]
            if let tauri::RunEvent::Opened { urls } = _event {
                let paths = urls
                    .into_iter()
                    .filter_map(|url| url.to_file_path().ok())
                    .filter(|path| path.is_file() && is_startup_document(path))
                    .collect::<Vec<_>>();
                if paths.is_empty() {
                    return;
                }
                if let Ok(mut pending) = _app.state::<PendingDocuments>().0.lock() {
                    pending.extend(paths);
                }
                let _ = _app.emit("external-document", ());
            }
        });
}
