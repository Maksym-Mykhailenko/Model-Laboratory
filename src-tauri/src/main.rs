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
}

impl EngineProcess {
    async fn request(&mut self, request_json: &str) -> Result<String, String> {
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
                    if bytes.len() > MAX_ENGINE_RESPONSE_BYTES {
                        return Err(
                            "The scientific engine response exceeded the desktop safety limit."
                                .to_string(),
                        );
                    }
                    return String::from_utf8(bytes)
                        .map_err(|_| "The scientific engine returned invalid UTF-8.".to_string());
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
    let selected = rfd::AsyncFileDialog::new()
        .add_filter(label, allowed)
        .set_file_name(&suggested_name)
        .save_file()
        .await;
    let Some(file) = selected else {
        return Ok(None);
    };
    tokio::fs::write(file.path(), bytes)
        .await
        .map_err(|error| format!("Could not save '{}': {error}", file.path().display()))?;
    Ok(Some(file.path().to_string_lossy().into_owned()))
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
        .pop_front();
    match candidate {
        Some(path) => read_document(path, MAX_DOCUMENT_BYTES).await.map(Some),
        None => Ok(None),
    }
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
            .spawn()
    } else {
        app.shell()
            .sidecar("model-lab-engine")
            .map_err(|error| format!("Could not locate the bundled scientific engine: {error}"))?
            .spawn()
    };
    let (events, child) =
        spawned.map_err(|error| format!("Could not start the scientific engine: {error}"))?;
    Ok(EngineProcess {
        events,
        child: Some(child),
    })
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
    let result = process
        .as_mut()
        .expect("scientific engine initialized")
        .request(&request_json)
        .await;
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
