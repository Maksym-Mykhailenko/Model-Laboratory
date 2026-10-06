# Security policy

Model Laboratory treats experiment files, model source, local attachments, interpreter output, and
extension declarations as potentially untrusted inputs.

## Supported release

Security maintenance currently targets the latest 1.19.x source and release line. Historical
formats remain covered by the compatibility and safe-open rules documented in the repository.

## Reporting a vulnerability

Use the repository's private security-advisory facility when it becomes available. Include:

- the affected version and platform;
- the relevant file, parser, IPC action, bundle member, capability, or workflow;
- a minimal reproduction or proof of concept;
- expected and observed behavior; and
- any known effect on confidentiality, integrity, availability, or reproducibility.

Please use a public issue only after a coordinated fix or disclosure has been prepared.

## Security boundaries

- Opening a `.mlab` performs bounded integrity and schema inspection without scientific execution.
- Experiment bundles declare capability requirements but do not carry executable extension code.
- Reproduction and capability execution are explicit actions using locally installed code.
- Interpreter output is parsed, bounded, schema-validated, compiled, and presented for acceptance.
- Local reference attachments use bounded extraction and content-addressed provenance.
- Workload estimation and capability settings validation occur before a runner starts.
- Committed experiment receipts bind the complete state and require their checksum for external
  execution interfaces.
- Network access is limited to explicit optional runtime operations such as obtaining a configured
  local model through the user's independently installed provider.

## Release handling

Release builds should be produced from a tagged source state, preserve dependency lockfiles,
generate build-identity evidence, and publish cryptographic checksums. Third-party dependencies and
vendored assets should be reviewed against the resolved release locks.

## Linux GLib dependency

The GTK3 desktop stack uses the 0.18 GLib API. `src-tauri/vendor/glib` backports the upstream
mutable-pointer fix for [RUSTSEC-2024-0429](https://rustsec.org/advisories/RUSTSEC-2024-0429.html).
The Cargo patch applies to the transitive GTK dependency, and CI runs the affected iterator
tests with release optimisations. The package retains its truthful 0.18.5 version, so a
version-only scanner can still flag it. The source and fix provenance are recorded in
[`MODEL_LAB_PATCH.md`](src-tauri/vendor/glib/MODEL_LAB_PATCH.md).
