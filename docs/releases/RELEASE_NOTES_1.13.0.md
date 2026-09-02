# Model Laboratory 1.13.0

Version 1.13.0 is the first release to implement structured official scientific packs on
top of the Model Graph and generic Run → Artifact architecture introduced in 1.9.

## Four official packs

1. **Multidimensional Mathematics and Scientific Quantities**
   - labelled arrays and rank-1–8 tensors with finite shape/value contracts;
   - matrix rank, determinant, singular values and norms;
   - deterministic tensor contraction;
   - SI dimension vectors, affine base-unit conversion and standard uncertainty.
2. **Probability, Stochastic Processes and Simulation**
   - finite probability distributions with support, entropy, concentration and moments;
   - exact finite Markov evolution, stationary distribution, spectrum and communicating classes;
   - bounded PCG64 trajectory ensembles with complete RNG/sampling identity and frozen
     statistical-equivalence criteria.
3. **Graphs, Networks and Discrete Structures**
   - directed/undirected weighted graphs and bounded multigraphs;
   - degrees, density, components, cycle/DAG status, adjacency spectrum and weighted paths;
   - deterministic circular network views.
4. **Generative Models, Inference and Decision Systems**
   - scaled hidden-Markov filtering, smoothing, evidence and Viterbi paths;
   - exact bounded finite-horizon POMDP belief planning with observation contingencies;
   - Markov-blanket discovery over directed dependency networks;
   - active-inference policy propagation, preference risk, ambiguity, expected free energy,
     posterior policy probabilities and exact finite event-reach probabilities.

## Integration and reproducibility

- Eight strict namespaced Model Graph object kinds are registered locally and receive
  semantic validation beyond JSON shape checking.
- Eleven capabilities use the existing workload gate and produce immutable typed artifacts.
- Official artifacts can be added to generic experiment run plans, stored in `.mlab` 2.0,
  and reproduced without adding discipline-specific fields to the bundle format.
- A stochastic artifact comparator distinguishes exact sample-stream reproduction from
  statistical equivalence under criteria frozen before reproduction.
- Reproduction report schema 1.2 adds `STATISTICALLY REPRODUCED`, so stochastic equivalence
  is not mislabeled as deterministic numerical reproduction.
- Renderer-owned Plotly views cover arrays, distributions, Markov probabilities, networks,
  hidden-state posteriors, decision values and active-inference policies.
- The AI interpreter's bounded contract derives the installed object schemas from the same
  registry while retaining its 32 KiB context-package ceiling.
- Existing legacy Model Graph kinds remain opaque and non-executable; historical `.mlab`
  parsing continues to use the historical core registry.

## Desktop and examples

- The Capabilities workspace now displays the official pack catalogue and applicability.
- JSON settings receive registry-schema defaults, and official object selectors are populated
  from the validated Model Graph where possible.
- Portable result previews are bounded in the UI; complete artifacts remain available to
  experiment storage and reproduction.
- Four example models demonstrate every new object family.

The four packs are deliberately versioned first implementations. They do not yet claim all
methods in multidimensional mathematics, stochastic modelling, graph theory, or generative
systems; later additions can extend the packs without changing Model IR or `.mlab` containers.
