# Third-party distribution notes

Model Laboratory's original work is licensed under Apache-2.0 as described in `LICENSE`,
`NOTICE`, and `LICENSING.md`. The components listed below retain their own terms and are not
relicensed by Model Laboratory. The desktop stack itself does not require a Qt or other paid
GUI licence.

| Component | Role | Licence family |
|---|---|---|
| Tauri and official shell plugin | Native desktop host | MIT / Apache-2.0 |
| Plotly.js 3.7.0 | Offline 2D and WebGL 3D rendering | MIT |
| Python | Scientific sidecar runtime | PSF |
| NumPy | Array computation | BSD-3-Clause |
| SciPy | Numerical algorithms | BSD-3-Clause |
| SymPy | Symbolic mathematics | BSD-3-Clause |
| Pydantic | Schema validation | MIT |
| PyYAML | YAML parsing | MIT |
| ruamel.yaml | Comment- and style-preserving interpreter edits | MIT |
| pypdf | Local text extraction from interpreter PDF references | BSD-3-Clause |
| PyInstaller | Release build tool/bootloader | GPL with the PyInstaller bootloader exception |
| reqwest, serde, Tokio, uuid, base64 and most Tauri Rust dependencies | Native implementation | MIT and/or Apache-2.0; verify the resolved lockfile |
| Qwen3-4B-Instruct-2507 | Optional local interpreter model | Apache-2.0; external download, not redistributed |
| Ollama | Optional persistent local inference runtime | External installation; verify the selected Ollama release before distribution |
| PyTorch | Optional Stage-7 QLoRA training runtime | BSD-3-Clause; external Python dependency |
| Transformers | Optional Stage-7 model/tokenizer/trainer stack | Apache-2.0; external Python dependency |
| Accelerate | Optional Stage-7 device/training orchestration | Apache-2.0; external Python dependency |
| PEFT | Optional Stage-7 LoRA/QLoRA adapters | Apache-2.0; external Python dependency |
| bitsandbytes | Optional Stage-7 4-bit quantisation/optimizer backend | MIT; external Python dependency |
| huggingface-hub | Optional pinned upstream snapshot downloader | Apache-2.0; external Python dependency |
| safetensors | Optional Stage-7 weight format support | Apache-2.0; external Python dependency |
| Operating-system webview | Desktop renderer | Supplied/licensed by the target operating system |

`frontend/vendor/plotly.min.js` retains Plotly's copyright and MIT licence header. Before
each public binary release, generate a notice/SBOM from the exact Python, npm, and Cargo
lockfiles and inspect it for changed transitive dependencies. Package managers may resolve
new compatible releases within the declared Python version ranges. Model Laboratory's
archive does not contain Qwen weights, an Ollama binary, or the optional Stage-7 training stack;  the interpreter panel asks the
user's independently installed Ollama service to pull the configured versioned model tag.

These notes are an engineering distribution checklist, not legal advice.
