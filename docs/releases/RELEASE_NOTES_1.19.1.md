# Model Laboratory 1.19.1

Version 1.19.1 corrects cross-platform reproduction reporting and frozen-sidecar environment
metadata for the declarative analysis composition introduced in 1.19.0.

## Download

For most Windows users:

- **[Download the `.exe` installer](https://github.com/Maksym-Mykhailenko/Model-Laboratory/releases/download/v1.19.1/Model.Laboratory_1.19.1_x64-setup.exe)**
- [Download the MSI package](https://github.com/Maksym-Mykhailenko/Model-Laboratory/releases/download/v1.19.1/Model.Laboratory_1.19.1_x64_en-US.msi) for managed MSI deployment
- [SHA-256 checksums](https://github.com/Maksym-Mykhailenko/Model-Laboratory/releases/download/v1.19.1/SHA256SUMS.txt)
- [Installer smoke-test receipt](https://github.com/Maksym-Mykhailenko/Model-Laboratory/releases/download/v1.19.1/INSTALLER_SMOKE_TEST.json)

> [!WARNING]
> These installers are not Authenticode-signed. Windows may display an **Unknown publisher** or
> SmartScreen warning. Verify the downloaded file against `SHA256SUMS.txt` before installation.

## Corrected cross-platform reproduction

Composed analysis artifacts intentionally store SHA-256 identities for every intermediate value.
Those derived hashes correctly differ when an eigensolver produces a minute platform-dependent
rounding difference, but the original generic numerical comparator treated the changed hash text
as changed scientific structure. It therefore stopped before comparing the aligned numerical
values and could report `NOT REPRODUCED` with no deviation values.

The composed-analysis artifact now uses a dedicated comparator. Strict reproduction continues to
compare the complete artifact, including every value hash. If strict identity differs, tolerant
reproduction compares all recipe operations, input references, settings, outputs, views, and
underlying numerical values while excluding only `value_sha256`, which is derived from values
already being compared. A deviation inside the experiment's pre-frozen tolerances is reported as
`NUMERICAL REPRODUCTION`; a changed recipe or out-of-tolerance result still fails.

## Corrected frozen-sidecar metadata

PyInstaller now copies distribution metadata for NumPy, SciPy, SymPy, Pydantic, PyYAML, and Plotly.
The packaged sidecar exposes those versions through its health response, and the build aborts if
any required distribution is reported as `not installed`. This prevents an installed application
from incorrectly classifying bundled SciPy, SymPy, or PyYAML as missing.

## Compatibility

No persisted protocol version changes in 1.19.1. Existing v1.19.0 Gaussian hierarchy `.mlab`
bundles open directly and reproduce under the corrected comparator. Model IR remains 3.0,
expression AST 1.2, `.mlab` 2.0, experiment state 6, Run 1.1, Artifact 1.0, and sidecar protocol 8.

[Full changelog](https://github.com/Maksym-Mykhailenko/Model-Laboratory/compare/v1.19.0...v1.19.1)
