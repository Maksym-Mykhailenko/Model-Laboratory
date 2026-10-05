# Model Laboratory 1.19.0

Version 1.19.0 introduces declarative scientific analysis composition. A model can now combine
installed numerical primitives, derive model-specific observables, and specify a multi-panel figure
without adding a one-off Python capability. The first flagship case reproduces the six-level
linear-Gaussian hierarchy from *A scale-free measure of relaxation anisotropy in precision-weighted
variational inference*.

## Download

For most Windows users:

- **[Download the `.exe` installer](https://github.com/Maksym-Mykhailenko/Model-Laboratory/releases/download/v1.19.0/Model.Laboratory_1.19.0_x64-setup.exe)**
- [Download the MSI package](https://github.com/Maksym-Mykhailenko/Model-Laboratory/releases/download/v1.19.0/Model.Laboratory_1.19.0_x64_en-US.msi) for managed MSI deployment
- [SHA-256 checksums](https://github.com/Maksym-Mykhailenko/Model-Laboratory/releases/download/v1.19.0/SHA256SUMS.txt)
- [Installer smoke-test receipt](https://github.com/Maksym-Mykhailenko/Model-Laboratory/releases/download/v1.19.0/INSTALLER_SMOKE_TEST.json)

> [!WARNING]
> These installers are not Authenticode-signed. Windows may display an **Unknown publisher** or
> SmartScreen warning. Verify the downloaded file against `SHA256SUMS.txt` before installation.

## Declarative analysis recipes

The new `org.modellab.composition.analysis-recipe@1.0` object contains an ordered acyclic graph of
installed operations. Version 1.19 includes finite arrays, linear/logarithmic coordinates, safe array formulas,
scalar and matrix-function evaluation, symmetric generalised eigenspectra, interpolated threshold
crossings, named outputs, and one declarative view with up to four panels.

Each result step carries its operation, settings, input references, input content hashes, numerical
value, and SHA-256 identity. The composed result uses the existing Run, Artifact, comparator,
reproduction, and `.mlab` contracts; it is not a parallel execution path.

Documents still cannot ship executable pack code. Formula parsing permits a compact numerical
vocabulary but rejects attribute access, imports, indexing, comprehensions, arbitrary calls,
statements, non-finite results, oversized expressions, and oversized intermediate or stored arrays.
Broadcast and matrix-product sizes are checked before allocation.

## Six-level Gaussian hierarchy

The new `models/gaussian-hierarchy-dispersion.yaml` example declares a six-state quadratic free
energy, its precision-weighted Hessian, two matched spectral profiles, and a 27-step recipe. It
reproduces:

- relaxation spectra `(1.000, 5.129, 9.482, 12.402, 15.325, 24.160)` and
  `(1.000, 1.730, 15.015, 15.094, 23.575, 24.160)`;
- log-spectral dispersions `1.031` and `1.282`;
- 90% and 10% residual-energy threshold times;
- recovery breadths `4.134` and `4.741`; and
- effective residual dimensions `1.076` and `1.851` at time `0.4`.

The desktop exposes the model in the example catalogue, runs the generic recipe capability, and
renders residual energy and effective dimension in a two-panel local Plotly view. Repeated runs
produce identical artifact hashes, and the saved experiment reproduces exactly. The source release
includes the model, a static reference figure, and an openable draft `.mlab` containing the
completed run and composed artifact.

## Capability boundary

This release raises the installed catalogue to 14 manifests, 35 official object kinds, and 46
capabilities. A new model does not require new native code when its analysis fits the recipe
vocabulary. A genuinely new solver, scientific data structure, stochastic contract, or comparator
still requires a reviewed installed capability.

The local interpreter's frozen reviewed corpus continues to cover the 34 pre-existing scientific
object kinds. It does not author recipes in 1.19. Recipes are edited as validated YAML or adapted
from the flagship example; execution, rendering, bundling, and reproduction are fully available.

## Compatibility

The serialized core protocols remain unchanged from 1.18.0: Model IR 3.0, expression AST 1.2,
`.mlab` 2.0, experiment state 6, Run 1.1, Artifact 1.0, and desktop sidecar protocol 8. Version
1.19 adds a namespaced object kind, capability, artifact type, and renderer. Existing models and
experiments continue to open under the previous compatibility rules.

## Verification

The release source passes:

- 460 Python tests, including formula sandbox, allocation-bound, Gaussian reference, renderer,
  desktop-boundary, and exact-reproduction tests;
- frontend contract and syntax checks;
- analytical reference, bundle, reproduction, expression, and Model Graph protocol campaigns; and
- 57/57 expanded official-pack checks with 14 manifests, 35 kinds, and 46 capabilities.

Native installer signing status, hashes, and clean-install results are reported by the Windows
release workflow rather than inferred from source verification.

[Full changelog](https://github.com/Maksym-Mykhailenko/Model-Laboratory/compare/v1.18.0...v1.19.0)
