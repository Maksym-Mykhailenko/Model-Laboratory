# Model Laboratory 1.15.0

Version 1.15.0 implements the third group of four official optional scientific packs. The
installed catalogue now contains twelve packs, thirty official Model Graph object kinds and
forty-one official capabilities, all executed through the existing Run → Artifact protocol.

## New official packs

### Statistical Inference and Data Modelling

- Complete finite numerical datasets with descriptive statistics, covariance, correlation and
  principal-component analysis.
- Ordinary and positive-weighted linear regression with coefficient inference, confidence
  intervals, residual diagnostics, rank/conditioning, R², AIC and BIC.
- Independent-group ANOVA, Kruskal-Wallis, eta-squared and Holm-corrected pairwise Welch tests.

### Optimisation, Estimation and Inverse Problems

- Bounded smooth minimisation/maximisation with explicit equality and inequality constraints,
  SLSQP and symbolic exact derivatives.
- Bounded nonlinear least-squares estimation with exact residual Jacobians and covariance.
- Weighted linear inversion with optional Tikhonov regularisation, singular values, covariance
  and resolution matrices.

### Electrical, Electronic and Electromagnetic Systems

- Linear DC and complex-frequency AC modified-nodal analysis for RLC circuits and independent
  current/voltage sources.
- Ideal Shockley diode characteristics with explicit temperature, ideality and area scaling.
- Two/three-dimensional electrostatic point-charge potential, field, dipole and energy analysis.

### Chemical, Reaction and Biological Systems

- Irreversible mass-action networks with stoichiometry, conservation-law, complex/linkage,
  deficiency and explicit-tolerance kinetic simulation.
- Linear compartment transfer/loss systems with residence times, stability, conservation,
  steady-state and trajectory analysis.
- Generalised Lotka-Volterra interaction systems with coexistence/stability and trajectories.

## Integration and compatibility

The desktop Capabilities workspace discovers all twelve packs without pack-specific pages.
Every new result has a deterministic Plotly renderer, workload estimate and numeric artifact
comparator. The AI interpreter derives the new object schemas and capabilities from the
installed registry and routes them through bounded dependency-specific contexts below 32 KiB.

Model IR remains 3.0, `.mlab` remains 2.0, experiment state remains 6, Run remains 1.1 and
Artifact remains 1.0. No historical model or experiment migration is required.

## Examples

- `models/statistical-data.yaml`
- `models/optimisation-inverse.yaml`
- `models/electrical-electromagnetic.yaml`
- `models/chemical-biological.yaml`

The v1.0 contracts are deliberately bounded. Missing-data/hierarchical statistical models,
global optimisation certificates, nonlinear semiconductor circuits, full-wave electromagnetics,
reversible thermodynamics and stochastic reaction kinetics remain future capability versions.
