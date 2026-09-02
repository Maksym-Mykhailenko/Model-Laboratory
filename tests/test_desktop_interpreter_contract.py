from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
import tomllib

from model_lab import __version__


ROOT = Path(__file__).resolve().parents[1]


class _IdParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.attributes: dict[str, dict[str, str | None]] = {}

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        identifier = attributes.get("id")
        if identifier is not None:
            self.ids.append(identifier)
            self.attributes[identifier] = attributes


def test_desktop_versions_are_consistent() -> None:
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    tauri = json.loads((ROOT / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8"))
    cargo = tomllib.loads((ROOT / "src-tauri" / "Cargo.toml").read_text(encoding="utf-8"))

    assert __version__ == package["version"] == tauri["version"] == cargo["package"]["version"] == "1.17.0"


def test_interpreter_review_controls_exist_and_are_not_placeholder_disabled() -> None:
    parser = _IdParser()
    parser.feed((ROOT / "frontend" / "index.html").read_text(encoding="utf-8"))
    required = {
        "interpreter-status",
        "interpreter-refresh-button",
        "interpreter-candidate-lifecycle",
        "interpreter-training-report",
        "interpreter-evaluation-report",
        "interpreter-pull-button",
        "interpreter-instruction",
        "interpreter-attach-button",
        "interpreter-attachment-list",
        "interpreter-request-button",
        "interpreter-cancel-button",
        "interpreter-clarification",
        "interpreter-clarification-answer",
        "interpreter-clarification-continue",
        "interpreter-clarification-cancel",
        "interpreter-proposal",
        "approve-interpreter-proposal",
        "interpreter-accept-button",
        "interpreter-reject-button",
    }

    assert len(parser.ids) == len(set(parser.ids))
    assert required <= set(parser.ids)
    assert "disabled" not in parser.attributes["interpreter-instruction"]
    assert "disabled" not in parser.attributes["interpreter-refresh-button"]


def test_live_and_committed_state_boundary_is_explicit_in_desktop_contract() -> None:
    parser = _IdParser()
    parser.feed((ROOT / "frontend" / "index.html").read_text(encoding="utf-8"))
    required = {
        "live-state-status",
        "committed-state-status",
        "committed-state-hash",
        "committed-experiment-hash",
        "committed-at",
        "committed-authority",
        "open-commit-button",
        "commit-state-button",
        "save-commit-button",
    }
    assert required <= set(parser.ids)

    frontend = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    engine = (ROOT / "desktop_engine.py").read_text(encoding="utf-8")
    assert 'engine("commit_experiment_state"' in frontend
    assert 'engine("inspect_committed_experiment"' in frontend
    assert 'native("open_document", { kind: "commit" })' in frontend
    assert 'parent_commit_sha256: state.committed?.commit_sha256 || null' in frontend
    assert '"live_state_authority": False' in (ROOT / "model_lab" / "commit.py").read_text(encoding="utf-8")
    assert 'if action == "commit_experiment_state"' in engine
    assert 'if action == "inspect_committed_experiment"' in engine
    assert '"commit" => ("Committed experiment receipt", &["json"])' in (
        ROOT / "src-tauri" / "src" / "main.rs"
    ).read_text(encoding="utf-8")


def test_webview_has_no_direct_ollama_or_general_network_permission() -> None:
    tauri = json.loads((ROOT / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8"))
    csp = tauri["app"]["security"]["csp"]
    connect_policy = next(
        directive for directive in csp.split(";") if directive.strip().startswith("connect-src")
    )

    assert "11434" not in csp
    assert "127.0.0.1" not in csp
    assert set(connect_policy.split()) == {"connect-src", "ipc:", "http://ipc.localhost"}


def test_native_provider_is_fixed_to_loopback_and_configured_model_tag() -> None:
    source = (ROOT / "src-tauri" / "src" / "interpreter.rs").read_text(encoding="utf-8")
    cargo = tomllib.loads((ROOT / "src-tauri" / "Cargo.toml").read_text(encoding="utf-8"))
    reqwest = cargo["dependencies"]["reqwest"]

    assert 'OLLAMA_ENDPOINT: &str = "http://127.0.0.1:11434"' in source
    assert 'OLLAMA_MODEL: &str = "qwen3:4b-instruct-2507-q4_K_M"' in source
    assert 'FROZEN_BASE_IDENTITY_VERIFICATION: &str =\n    "exact_local_manifest_and_all_blobs_sha256"' in source
    assert 'CANDIDATE_IDENTITY_VERIFICATION: &str =\n    "approved_registry_exact_manifest_and_model_blob_sha256"' in source
    assert 'BASE_MODEL_LOCK_JSON: &str = include_str!' in source
    assert 'verify_blob_file' in source
    assert 'Ollama /api/tags digest does not match the SHA-256 of the local manifest bytes.' in source
    assert 'Installed Ollama GGUF blob differs from its locked artifact.' in source
    assert 'verify_selected_artifact' in source
    assert 'live_frozen_base_identity_smoke' in source
    assert 'document.schema_version != "1.3"' in source
    assert "evaluation_benchmark_sha256" in source
    assert "promotion_seal_sha256" in source
    assert "training_report_sha256" in source
    assert "candidate_evaluation_report_sha256" in source
    assert "candidate_lifecycle_state" in source
    assert "official_digest" not in source
    assert ".no_proxy()" in source
    schema = json.loads((ROOT / "model_lab" / "baseline" / "output_schema_v1.2.json").read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert 'include_str!("../../model_lab/baseline/output_schema_v1.2.json")' in source.replace("\n", "").replace("    ", "")
    assert "MAX_CONTEXT_PACKAGE_BYTES: usize = 32 * 1024" in source
    assert "(32_768_u64, 4_096_u64)" in source
    assert "(65_536, 8_192)" in source
    assert "(131_072, 8_192)" in source
    assert 'Some("1.2")' in source
    assert "interpreter-model-registry.json" in (ROOT / "src-tauri" / "src" / "main.rs").read_text(encoding="utf-8")
    assert '"model_source"' not in source.split("#[cfg(test)]", 1)[0]
    assert reqwest["default-features"] is False
    assert set(reqwest["features"]) == {"json", "stream"}


def test_frontend_routes_inference_through_validation_and_explicit_acceptance() -> None:
    source = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

    assert 'native("interpreter_request"' in source
    assert 'engine("prepare_interpreter_context"' in source
    assert 'engine("process_interpreter_output"' in source
    assert 'engine("accept_interpreter_proposal"' in source
    assert 'engine("ingest_interpreter_attachment"' in source
    assert 'engine("release_interpreter_attachment"' in source
    assert "attachment_ids: state.interpreterConversation.attachmentIds" in source
    assert "contextPackage: context" in source
    assert "currentModelSource" not in source.split('native("interpreter_request"', 1)[1].split(");", 1)[0]
    assert "expected_proposal_sha256: reviewed.proposal.proposal_sha256" in source
    assert "current_model_source: elements.source.value" in source
    assert "No analysis was run" in source
    assert "interpreterTrainingReport" in source
    assert "interpreterEvaluationReport" in source
    assert "candidate_lifecycle_state" in source


def test_native_attachment_picker_is_type_and_size_bounded() -> None:
    source = (ROOT / "src-tauri" / "src" / "main.rs").read_text(encoding="utf-8")
    assert '"interpreter-attachment" => (' in source
    assert '&["txt", "md", "yaml", "yml", "json", "csv", "tsv", "pdf"]' in source
    assert "MAX_INTERPRETER_ATTACHMENT_BYTES: u64 = 16 * 1024 * 1024" in source
    assert "document_limit(&kind)" in source


def test_frontend_exposes_generic_capabilities_and_typed_experiment_actions() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    source = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

    for identifier in (
        "capability-select",
        "capability-settings",
        "capability-run-button",
        "capability-plan",
        "capability-result",
    ):
        assert f'id="{identifier}"' in html
    assert 'engine("run_capability"' in source
    assert 'engine("prepare_run_experiment"' in source
    assert "content_handle" in (ROOT / "desktop_engine.py").read_text(encoding="utf-8")


def test_windows_build_preflight_checks_required_versions_and_msvc_host() -> None:
    source = (ROOT / "scripts" / "build_windows.ps1").read_text(encoding="utf-8")

    assert '[Version]"3.11.0"' in source
    assert '[Version]"20.0.0"' in source
    assert "rustc -vV" in source
    assert "pc-windows-msvc" in source
    assert "run_model_graph_protocol_verification.py" in source


def test_sidecar_bundles_versioned_interpreter_assets() -> None:
    source = (ROOT / "scripts" / "build_sidecar.py").read_text(encoding="utf-8")
    assert "ROOT / 'model_lab' / 'baseline'" in source
    assert "model_lab/baseline" in source


def test_native_interpreter_compile_verification_is_mandatory_in_ci_and_builds() -> None:
    workflow = (ROOT / ".github" / "workflows" / "cross-platform-verification.yml").read_text(encoding="utf-8")
    windows = (ROOT / "scripts" / "build_windows.ps1").read_text(encoding="utf-8")
    unix = (ROOT / "scripts" / "build_unix.sh").read_text(encoding="utf-8")
    verifier = (ROOT / "scripts" / "verify_native_interpreter.ps1").read_text(encoding="utf-8")

    assert "native-interpreter:" in workflow
    assert "dtolnay/rust-toolchain@stable" in workflow
    assert "verify_native_interpreter.ps1" in workflow
    for source in (windows, unix, verifier):
        assert "cargo check --manifest-path src-tauri/Cargo.toml --locked" in source
        assert "cargo test --manifest-path src-tauri/Cargo.toml --locked interpreter::tests" in source
    assert "live_frozen_base_identity_smoke" in verifier
    assert "--ignored --nocapture" in verifier
