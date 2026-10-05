# Reproducible examples

## Six-level Gaussian hierarchy

[Read the complete scientific case](../docs/cases/gaussian-hierarchy-dispersion.md), or open its
individual assets:

- Model and 27-step analysis recipe:
  [`../models/gaussian-hierarchy-dispersion.yaml`](../models/gaussian-hierarchy-dispersion.yaml)
- Openable experiment: [`gaussian-hierarchy-dispersion.mlab`](gaussian-hierarchy-dispersion.mlab)
- Compact reference results:
  [`gaussian-hierarchy-dispersion-reference-results.json`](gaussian-hierarchy-dispersion-reference-results.json)
- Reference figure:
  [`../docs/assets/gaussian-hierarchy-dispersion.png`](../docs/assets/gaussian-hierarchy-dispersion.png)
- Asset checksums:
  [`gaussian-hierarchy-dispersion-SHA256SUMS.txt`](gaussian-hierarchy-dispersion-SHA256SUMS.txt)
- Recipe language: [Declarative analysis composition](../docs/science/analysis-composition.md)

The `.mlab` is a validated draft/interchange bundle produced by the 1.19.0 recipe capability. It
contains the model, completed Run Record, composed-analysis artifact, environment receipt,
provenance, and integrity fingerprints. Opening it performs inspection only; choose **Reproduce**
to rerun and compare the artifact. Model Laboratory 1.19.1 can report minute cross-platform
eigensolver differences as `NUMERICALLY REPRODUCED` while retaining strict identity checking and
reporting environment differences separately.

The checked-in bundle is deliberately not marked `author_approved_for_publication`. Publication
approval is an explicit author action in Model Laboratory: review the frozen experiment, approve
its exact review digest, and save the resulting publication bundle.

The complete release-source hashes are recorded in the case's checksum manifest.
