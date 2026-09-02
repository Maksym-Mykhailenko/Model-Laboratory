# Security policy

Model Laboratory treats experiment files, model source, local attachments, interpreter output, and
extension declarations as potentially untrusted inputs.

## Supported release

Security maintenance currently targets the latest 1.17.x source and release line. Historical
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
