# Windows installer release

Model Laboratory produces both a per-user NSIS `.exe` installer and an MSI package from the same
tested source and bundled scientific sidecar. The release workflow runs the Rust, Python,
frontend, reference, bundle, reproduction, expression, protocol, and official-pack checks before
building either installer.

## Local release build

Use a Windows machine with Python 3.11+, Node.js 20+, and the Rust MSVC toolchain:

```powershell
./scripts/build_windows.ps1
```

The script writes installers and `SHA256SUMS.txt` under
`src-tauri/target/release/bundle`. Unsigned output is suitable for internal testing only.

For Authenticode signing, import a code-signing certificate into the current user's certificate
store and pass its thumbprint:

```powershell
./scripts/build_windows.ps1 -CertificateThumbprint "CERTIFICATE_THUMBPRINT" -RequireSigned
```

To install, launch, verify the `.mlab`, `.yaml`, and `.yml` associations, and uninstall both
packages on a disposable Windows test machine, add `-RunInstallerSmoke`. An earlier NSIS installer
may be supplied to test the upgrade path:

```powershell
./scripts/build_windows.ps1 -CertificateThumbprint "CERTIFICATE_THUMBPRINT" -RequireSigned `
  -RunInstallerSmoke -PreviousNsisInstaller "Model-Laboratory_1.17.0_x64-setup.exe"
```

Every smoke run writes `INSTALLER_SMOKE_TEST.json`, including installer SHA-256 values, runner
identity, individual clean-install/upgrade outcomes, timestamps, and any failure message. Uninstall
checks require both the installed executable and its registered open commands to be removed. The
smoke harness normalizes quoted registry paths written by either installer and attempts a quiet
cleanup if a later assertion fails, so a failed run still leaves actionable evidence without
contaminating subsequent package checks. Executable discovery accounts for Tauri's distinct
product and Cargo binary names: WiX installs the Cargo-named executable under the product-named
directory. A constrained scan of the recorded install directory handles future binary renames,
and a failure records every candidate path in the receipt. Association validation supports both
NSIS's literal open commands and WiX's advertised MSI associations. For advertised associations,
the harness asks the Windows Shell to resolve the registered ProgID and also recognizes the
product-owned Windows Installer command descriptor; uninstall validation rejects either form if
it remains registered.

## GitHub release build

Pushing the version tag `v1.18.0` runs `windows-release.yml`, verifies that the tag exactly matches
all release metadata, uploads the verified MSI,
NSIS installer, checksum manifest, and smoke-test receipt, then creates a draft GitHub release for
final review. The workflow looks up the newest earlier published release, downloads its NSIS
executable when one exists, and automatically exercises an in-place upgrade before the clean
installer tests. If no previous installer exists—as expected for a first binary release—the
receipt records the upgrade check as `NOT_RUN` without weakening the clean-install gates.
Cargo's Windows-target lock graph is resolved in the committed lockfile, and every native build or
verification command is checked immediately; the release stops at the first failed prerequisite.
The scientific sidecar is built before Cargo evaluates Tauri's external-binary configuration, so a
clean checkout does not depend on an ignored, pre-existing executable.

For public distribution, configure these repository secrets:

- `WINDOWS_CERTIFICATE_BASE64`: raw base64-encoded PFX bytes (without PEM headers), generated
  with `[Convert]::ToBase64String([IO.File]::ReadAllBytes("certificate.pfx"))`.
- `WINDOWS_CERTIFICATE_PASSWORD`: password for that PFX.

Tagged builds fail when either signing secret is absent. The workflow imports the certificate
temporarily, signs both installers with SHA-256 and a timestamp, requires valid signatures, and
runs clean-install smoke tests for NSIS and MSI before creating a draft. A manual run without
secrets may still produce an explicitly unsigned internal-test artifact. Workflow actions are
pinned to immutable commit SHAs; update those pins deliberately during dependency maintenance.
