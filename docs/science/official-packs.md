# Official scientific packs

Model Laboratory 1.16.0 implements the first thirteen official structured scientific packs.
They use the same four-layer boundary as every future extension:

1. a strict namespaced Model Graph object schema;
2. locally installed capability code with explicit workload estimation;
3. an immutable renderer-independent scientific artifact and comparator; and
4. a renderer-owned view that is excluded from scientific identity.

An `.mlab` may require a pack and preserve its data, runs, artifacts and views. It cannot
contain Python, JavaScript, a native library, or another executable pack payload. Opening a
bundle never runs a capability.

The v1.16 full distribution bundles and activates these thirteen pack implementations.
Their manifests already declare `official-optional` distribution and dependencies; moving
later large packs into separately downloaded, signed packages does not require a new Model
Graph or `.mlab` representation.

## Pack catalogue

### Multidimensional Mathematics and Scientific Quantities

- Pack: `org.modellab.pack.multidimensional-mathematics@1.1`
- Objects: `org.modellab.multidimensional.array@1.0`,
  `org.modellab.multidimensional.quantity@1.0`
- Capabilities: array analysis, tensor contraction, quantity normalisation
- Results: values and shapes remain explicit; matrix-only quantities such as determinant,
  rank and singular values are present only where mathematically defined. Tensor contraction
  preserves uncontracted axis labels, combines physical dimensions and propagates independent
  standard uncertainties.
- Units: seven SI base-dimension exponents identify compatibility. Conversion scale,
  affine offset and standard uncertainty are stored explicitly. Official physical packs consume
  the same registry and convert declared inputs to canonical SI before numerical work.

### Probability, Stochastic Processes and Simulation

- Pack: `org.modellab.pack.probability-stochastic-systems@1.1`
- Objects: finite discrete distributions and row-stochastic Markov chains
- Capabilities: probability analysis, exact Markov evolution, seeded Markov simulation
- Reducible chains return every closed-class stationary basis distribution and explicitly report
  non-uniqueness; no arbitrary vector is presented as the stationary distribution.
- Randomness: PCG64 seed, NumPy version, implementation, chain/draw layout and sampling
  settings are stored with the artifact.
- Reproduction: identical samples can reproduce exactly. A changed stream may be only
  `STATISTICALLY REPRODUCED`, using the significance, mean, variance and minimum-sample
  criteria frozen in the saved reference.

### Graphs, Networks and Discrete Structures

- Pack: `org.modellab.pack.graphs-networks-discrete@1.1`
- Object: `org.modellab.graph.network@1.0`
- Capabilities: deterministic network analysis and view generation
- Semantics: node identity is explicit; edge direction, multiplicity, weight and label are
  preserved. Graph-theoretic degree counts parallel edges and counts an undirected self-loop
  twice; unique-neighbour counts are reported separately. Analysis also includes density,
  weak/strong components, cycle/DAG status, adjacency spectrum and optional non-negative
  weighted shortest path.

### Generative Models, Inference and Decision Systems

- Pack: `org.modellab.pack.generative-inference-decision-systems@1.1`
- Objects: hidden Markov models, finite POMDPs, active-inference generative models, plus
  directed networks from the graph pack
- Capabilities:
  - scaled HMM filtering and backward smoothing, log evidence and structural-zero-preserving
    Viterbi decoding;
  - exact bounded finite-horizon POMDP belief recursion with observation-contingent actions;
  - Markov-blanket discovery as parents, children and co-parents;
  - active-inference policy propagation with preference risk, likelihood ambiguity,
    expected free energy, policy priors/precision, likelihood precision and exact finite
    event-reach recursion.

The pack title is deliberately broader than “Discrete Active Inference.” Its ontology can
represent the common finite-state foundations used by predictive processing, the free-energy
principle, decision processes, POMDPs, Markov chains and Markov blankets without pretending
that these theories are identical.

### Dynamical Systems, Differential Equations and Control

- Pack: `org.modellab.pack.dynamics-differential-equations-control@1.0`
- Objects: autonomous or time-dependent finite-dimensional ODE systems and continuous linear
  state-space systems with explicit state/input/output identity
- Capabilities:
  - SciPy ODE integration on the declared interval with explicit method, `rtol`, `atol` and
    canonical sample coordinates;
  - bounded multistart equilibrium solving for autonomous systems, symbolic Jacobians and
    eigenvalue-based local stability classification;
  - pole, spectral-abscissa, controllability and observability analysis; and
  - constant-input continuous state-space simulation.
- Results preserve equation source, fixed parameter values, solver settings, evaluation count
  and the exact NumPy/SciPy/SymPy backend identity recorded by the Run Record.

### Spatial Fields, Continuum Models and PDEs

- Pack: `org.modellab.pack.spatial-fields-continuum-pdes@1.1`
- Objects: one- to three-dimensional rectilinear scalar fields, two/three-dimensional vector
  fields, a 1D constant-diffusivity Dirichlet problem and a 2D uniform-grid Poisson problem
- Capabilities:
  - grid-aware gradient, gradient magnitude, Laplacian, integral and extrema;
  - divergence and dimension-appropriate curl;
  - method-of-lines diffusion using a declared stiff/non-stiff integrator and tolerances; and
  - sparse finite-difference solution of `-Laplacian(u) = source` with constant edge data and
    an explicit bottom/top-edge ownership convention at corner nodes.
- Coordinates are first-class, converted from declared source units to metres, and must be finite
  and strictly increasing. Field values are likewise converted to canonical SI; solvers do not
  infer spacing, boundary conditions or sign conventions from a plot.

### Geometry, Meshes and Spatial Computation

- Pack: `org.modellab.pack.geometry-meshes-spatial-computation@1.1`
- Objects: labelled 2D/3D Euclidean point clouds and indexed, oriented 3D triangle meshes
- Capabilities:
  - centroid, covariance, principal directions, affine rank, bounds and exact diameter;
  - edge incidence, connected face components, boundary/non-manifold detection, Euler
    characteristic, surface area, enclosed volume, normals and triangle quality; and
  - deterministic Euclidean shortest paths along mesh edges.
- Mesh validation rejects invalid indices, repeated face vertices and degenerate triangles
  before any analysis or renderer is invoked. Principal subspaces are identified by canonical
  orthogonal projectors, so degenerate eigenspaces do not acquire platform-dependent scientific
  identities from arbitrary LAPACK basis rotations.

### Mechanics, Structures and Materials

- Pack: `org.modellab.pack.mechanics-structures-materials@1.1`
- Objects: isotropic linear-elastic material records and 2D/3D linear pin-jointed trusses with
  explicit node/element identity, supports, named load cases and nodal/element mass
- Capabilities:
  - derived elastic moduli plus 3D Voigt, plane-stress and plane-strain constitutive matrices;
  - assembled linear static response, reactions, element strain/stress/axial force and strain
    energy; and
  - restrained generalised stiffness/mass eigenanalysis with reproducible mode shapes and
    frequencies.
- A singular or numerically unstable structure is reported as unable to solve; it is never
  regularised silently. Declared geometry, force, stress, modulus, mass and density units are
  converted before analysis. Static and modal artifacts remain independent of deformation views.

### Statistical Inference and Data Modelling

- Pack: `org.modellab.pack.statistical-inference-data-modelling@1.1`
- Objects: complete finite numerical datasets, weighted linear-model studies and independent
  grouped samples
- Capabilities:
  - descriptive summaries, covariance/correlation and principal-component decomposition with
    explicit degrees-of-freedom convention;
  - ordinary or positive-weighted least squares with coefficient uncertainty, confidence
  intervals, residuals, rank, conditioning, intercept-appropriate centred or through-origin
  uncentred R², and explicitly defined information criteria; and
  - one-way ANOVA, Kruskal-Wallis, eta-squared and Holm-corrected pairwise Welch tests.
- Version 1.0 requires complete finite observations. Missing-data handling, mixed models,
  Bayesian estimation and causal identification are not silently inferred.

### Optimisation, Estimation and Inverse Problems

- Pack: `org.modellab.pack.optimisation-estimation-inverse-problems@1.1`
- Objects: bounded smooth nonlinear programmes, bounded nonlinear residual systems and weighted
  linear inverse problems with optional reference parameters
- Capabilities:
  - constrained SLSQP minimisation/maximisation using exact expression-language derivatives;
  - trust-region nonlinear least squares with exact residual Jacobians, identifiability and
    covariance diagnostics; and
  - SVD-diagnosed weighted linear inversion with optional zero-order Tikhonov regularisation,
    covariance and resolution matrices.
- Bounds, constraint relations, residual definitions, weighting and regularisation are frozen
  inputs. Rank-deficient parameterisations expose an unidentifiable subspace and do not report
  misleading finite covariance/standard errors. Failed convergence is reported; a solver result
  is never relabelled as an optimum.

### Electrical, Electronic and Electromagnetic Systems

- Pack: `org.modellab.pack.electrical-electronic-electromagnetic-systems@1.1`
- Objects: linear lumped RLC/source circuits, ideal Shockley diodes and two/three-dimensional
  electrostatic point-charge systems
- Capabilities:
  - DC modified-nodal analysis with explicit open-capacitor/short-inductor semantics;
  - complex AC phasor sweeps at declared positive frequencies;
  - temperature- and ideality-aware diode current, differential conductance and resistance; and
  - electric potential/field, net charge, dipole moment and pair potential energy.
- Circuit topology, reference node, element orientation, units and source phase remain explicit.
  Circuit values, charge coordinates, charges and permittivity are converted to canonical SI
  before the equations are evaluated.
  Semiconductor circuits, transmission lines and full-wave Maxwell solvers are later contracts.

### Chemical, Reaction and Biological Systems

- Pack: `org.modellab.pack.chemical-reaction-biological-systems@1.1`
- Objects: irreversible mass-action networks, linear biological compartment systems and
  generalised Lotka-Volterra population-interaction systems
- Capabilities:
  - reaction stoichiometry, conservation-law, complex/linkage and deficiency analysis;
  - explicit-tolerance deterministic mass-action simulation;
  - compartment transfer/loss stability, residence-time, conservation and trajectory analysis;
    and
  - coexistence-equilibrium stability and trajectories for population interaction systems.
- Declared concentration, amount and time units are converted before rates are evaluated. Reaction
  stoichiometry is integral, the mass-action ODE uses its unclipped polynomial vector field at
  internal solver stages, and only bounded post-integration round-off may be projected.
- Solver records preserve convergence separately from presentation-only work counts.

### Machine Learning and Computational Intelligence

- Pack: `org.modellab.pack.machine-learning-computational-intelligence@1.0`
- Objects: labelled feature datasets, supervised regression/classification studies, explicitly
  weighted feed-forward networks and bounded zero-order Sugeno fuzzy-rule systems
- Capabilities:
  - deterministic ridge regression and reference-class multinomial classification with explicit
    in-sample fit diagnostics;
  - seeded k-means with canonical cluster ordering, assignments, centres, inertia and convergence;
  - feed-forward inference with identity/ReLU/tanh/sigmoid/softmax activations; and
  - product-conjunction rule aggregation with weighted-singleton defuzzification.
- This v1.0 pack evaluates and reproduces explicit studies and fixed networks; it does not silently
  train neural-network weights, select architectures, tune hyperparameters or claim that fuzzy
  inference and statistical learning are the same scientific method.

## Validation and limits

JSON Schema rejects unknown fields, invalid primitive types and unbounded container sizes.
Capability settings are validated generically against the registered schema, with defaults applied
before both workload estimation and execution; runners cannot receive unknown or out-of-range
settings by bypassing their UI.
Pack semantic validators additionally enforce probability normalisation, aligned labels and
shapes, finite values, row-stochastic transitions, graph endpoint validity, common policy
horizons, compatible action/state/observation dimensions and unique identities.

Capability workload is estimated before execution. Individual runners also impose hard
result bounds. Visualization is dimension- and result-specific; the existence of an
arbitrary-rank tensor does not imply that every rank has a misleading spatial rendering.

These are bounded pack contracts, not exhaustive disciplinary coverage. In particular, this
release does not yet implement continuous probability distributions, missing-data models,
mixed/hierarchical statistics, SDE/DAE solvers, MCMC, global optimisation certificates,
adaptive or unstructured PDE/FEM methods, CAD/solid kernels, nonlinear/contact/plastic
mechanics, nonlinear semiconductor circuits, transmission lines, full-wave electromagnetics,
reversible/thermodynamic reaction laws, stochastic chemical kinetics, symbolic index calculus,
  hypergraphs, graph isomorphism, continuous-state POMDP solvers, variational message passing,
  continuous active inference, neural-network training, deep-learning frameworks, reinforcement
  learning, automatic differentiation frameworks or probabilistic programming.

## Examples

- `models/multidimensional.yaml`
- `models/probability.yaml`
- `models/network.yaml`
- `models/generative-systems.yaml`
- `models/dynamics-control.yaml`
- `models/spatial-fields.yaml`
- `models/geometry-mesh.yaml`
- `models/mechanics-structure.yaml`
- `models/statistical-data.yaml`
- `models/optimisation-inverse.yaml`
- `models/electrical-electromagnetic.yaml`
- `models/chemical-biological.yaml`
- `models/machine-learning.yaml`

Every example validates without execution. Its applicable capabilities appear only after
validation; running one remains an explicit user action.
