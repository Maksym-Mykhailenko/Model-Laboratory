# Model Laboratory 1.16.0

Version 1.16.0 completes the initial thirteen-pack scientific roadmap and corrects the numerical,
dimensional and identity defects found during review of the first twelve packs.

## Machine Learning and Computational Intelligence

The new official-optional pack adds four first-class objects and four capabilities:

- labelled finite feature datasets with optional sample identifiers and per-feature unit labels;
- supervised regression and classification studies with explicit in-sample fit diagnostics;
- fixed feed-forward networks with dimension-checked layers and declared activations;
- bounded zero-order Sugeno fuzzy-rule systems with explicit membership functions and rules;
- deterministic ridge regression and multinomial logistic classification;
- seeded, canonically labelled k-means clustering;
- feed-forward network evaluation; and
- product-conjunction fuzzy inference with weighted-singleton defuzzification.

Every result is an immutable generic artifact with a deterministic renderer, comparator, workload
estimate and reproducible Run Record. The app now installs 13 official manifests, 34 official object
kinds and 45 official capabilities.

## Scientific correctness

- HMM backward recursion is scaled and structural zero probabilities remain `-inf` in Viterbi.
- Reducible Markov chains return their closed recurrent classes and stationary basis rather than an
  arbitrary single distribution.
- Rank-deficient nonlinear and linear inverse estimators report unidentifiability and withhold
  ordinary finite covariance/standard errors.
- No-intercept regression uses uncentred R² and its corresponding adjusted formula.
- Multigraph degree counts edge multiplicity and undirected self-loops twice; unique-neighbour
  counts are separate fields.
- Mass-action rates use the actual polynomial ODE state at internal stages; bounded non-negativity
  projection is post-integration only.
- Degenerate eigenspaces/null spaces use canonical projectors for artifact identity while retaining
  a human-readable basis for presentation.

## Operational physical units

A shared SI unit registry is now consumed by quantities, fields, geometry, mechanics, electrical/
electromagnetic, chemical and biological packs. Declared centimetres, nanocoulombs, kilonewtons,
megapascals, molar concentrations and other supported units are converted before equations run.
Canonical outputs and source-unit provenance are both retained. Unknown units fail explicitly.

Tensor contraction now preserves surviving labels, combines dimensional exponents/units and
propagates independent standard uncertainties.

## Protocol hardening

Registered capability settings schemas are enforced centrally before workload estimation and
execution. Defaults become part of the effective Run settings; unknown keys, invalid types,
out-of-range values and excessive arrays fail before a runner starts. HMM workload includes sequence
length.

The Stage-6 5,000-record interpreter corpus was deterministically regenerated against the expanded
catalogue so all stored contexts and targets remain checksum-valid. The corpus schema, seed, split,
review requirement and held-out benchmark exclusion are unchanged.

## Compatibility

Model IR 3.0, expression AST 1.2, experiment state 6, `.mlab` 2.0, Run 1.1 and the generic Artifact
envelope 1.0 remain unchanged. Model-object kinds remain 1.0. Corrected pack implementations and
their affected capability/artifact payload contracts advance to 1.1; unchanged capabilities retain
1.0. Existing bundles open without migration and never execute merely by being opened. Archived
artifacts retain their historical identities; a missing corrected 1.0 executor yields `UNABLE TO
REPRODUCE` instead of silently using 1.1 mathematics under an old label.

Live Tauri/Ollama, platform installer, QLoRA training and cross-machine verification remain target-
environment work. This archive contains no fabricated empirical result for those unavailable runs.
