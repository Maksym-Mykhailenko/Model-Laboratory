# Model Laboratory 1.14.0

Version 1.14.0 implements the second group of four official optional scientific packs. It
does not add discipline-specific fields to the experiment container. Every new method uses
the existing Model Graph → Capability Run → Scientific Artifact → View boundary and therefore
freezes and reproduces through `.mlab` 2.0 without a container migration.

## New official packs

### Dynamical Systems, Differential Equations and Control

- strict nonlinear ODE-system objects with named states, initial state, interval, equations,
  time symbol and fixed parameters;
- explicit-method/tolerance ODE integration and canonical sampling;
- bounded equilibrium solving, symbolic Jacobians and eigenvalue stability classification;
- continuous state-space objects; and
- poles, controllability, observability and constant-input trajectory analysis.

### Spatial Fields, Continuum Models and PDEs

- coordinate-aware rectilinear scalar and vector fields in supported dimensions;
- gradient, Laplacian, integral, extrema, divergence and curl;
- reproducible 1D method-of-lines diffusion; and
- sparse 2D uniform-grid Poisson solution with explicit sign and Dirichlet conventions.

### Geometry, Meshes and Spatial Computation

- labelled Euclidean point-cloud and indexed triangle-mesh objects;
- principal geometry, affine rank, bounding boxes and exact diameter;
- mesh incidence, components, manifold/boundary checks, Euler characteristic, area, enclosed
  volume, normals and quality; and
- deterministic edge-geodesic shortest paths.

### Mechanics, Structures and Materials

- first-class isotropic elastic material records and derived constitutive matrices;
- 2D/3D truss nodes, elements, supports, named load cases and mass data;
- assembled linear static response, reactions and element resultants; and
- generalised stiffness/mass modal analysis.

## Desktop and interpreter integration

The Capabilities workspace discovers all eight official packs from the installed registry and
uses the same generic settings editor, run-plan, artifact display and renderer path. New Plotly
views cover trajectories, spectra, fields, PDE solutions, point clouds, meshes, paths,
constitutive matrices, truss deformations and modes.

Interpreter schema context is now dependency-routed by requested discipline and current model
objects. Core-only authoring retains its historical contract hash; Active-Inference authoring
retains the original first-four catalogue; each new pack receives only its own schema and
declared dependency closure. Every tested route remains within the fixed 32 KiB context-package
limit without lowering the accepted 512 KiB model-source limit.

## Compatibility and scope

The application version advances to 1.14.0. Model IR 3.0, expression AST 1.2, experiment state
6, Run 1.1, Artifact 1.0 and `.mlab` 2.0 do not change. Existing official kind/capability
contracts remain at 1.0 and are not redefined.

These are bounded first capability versions, not claims of exhaustive disciplinary coverage.
SDE/DAE solvers, adaptive/unstructured FEM, CAD solids, nonlinear/contact/plastic mechanics
and specialised control design remain outside these exact contracts.
