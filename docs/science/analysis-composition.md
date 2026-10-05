# Declarative analysis composition

Model Laboratory 1.19 adds a bounded recipe language for analyses that do not warrant a new
Python capability. A recipe is a typed Model Graph object containing an ordered graph of installed
numerical operations. It may derive new quantities and a figure, but it cannot import code, access
files or the network, call arbitrary Python, or change the model.

This layer addresses an important capability boundary: a new scientific model can often be handled
by composing existing mathematical primitives instead of adding a model-specific pack. New native
code remains appropriate when a genuinely new solver, data structure, comparator, or scientific
contract is required.

## Execution model

The object kind is `org.modellab.composition.analysis-recipe@1.0`. The installed capability
`org.modellab.composition.run-analysis-recipe@1.0` executes its steps in declaration order.
Inputs may reference only earlier step identifiers, so the graph is acyclic by construction.

Every completed step records:

- its installed operation identifier;
- input references and the content hashes of those inputs;
- validated settings;
- its finite numerical value; and
- a SHA-256 digest over that complete record.

Named outputs point to those immutable step records. Running the same model and settings produces
the same composed-analysis artifact, and a saved experiment carries that artifact through the
normal Run Record and `.mlab` reproduction path.

## Installed operations

| Operation | Purpose |
|---|---|
| `array.literal` | Introduce a finite numerical scalar, vector, or array. |
| `array.linspace` | Create a bounded, uniformly spaced one-dimensional coordinate. |
| `array.logspace` | Create a bounded logarithmically spaced one-dimensional coordinate. |
| `array.expression` | Evaluate a whitelisted numerical formula over prior step values. |
| `matrix.evaluate` | Evaluate a declared Model Laboratory matrix function at one model point. |
| `matrix.generalized-eigenvalues` | Compute the real symmetric spectrum of \(H v = \mu G v\); \(G\) must be positive definite. |
| `scalar.evaluate-points` | Evaluate a declared scalar function at explicit model points. |
| `series.first-crossing` | Interpolate the first above- or below-threshold crossing of an ordered series. |

The formula evaluator supports arithmetic, bounded powers, matrix multiplication, constants `pi`
and `e`, and these functions: `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `sinh`, `cosh`,
`tanh`, `sum`, `mean`, `std`, `min`, `max`, `prod`, `outer`, `dot`, `transpose`, `trace`, and
`det`. Reductions accept an optional integer axis. Attribute access, indexing, comprehensions,
conditionals, keywords, arbitrary function calls, and executable statements are not part of the
language.

## Minimal recipe

```yaml
objects:
  spectrum-summary:
    kind: org.modellab.composition.analysis-recipe
    properties:
      steps:
        - id: H
          operation: matrix.evaluate
          settings: {matrix: hessian, point: [0, 0]}
        - id: rates
          operation: matrix.generalized-eigenvalues
          inputs: {matrix: H}
          settings: {require_positive: true}
        - id: dispersion
          operation: array.expression
          inputs: {mu: rates}
          settings: {expression: "std(log(mu))"}
      outputs:
        relaxation rates: rates
        log-spectral dispersion: dispersion
```

The recipe declares *what* is composed. The Python implementation of each operation remains part
of the installed, versioned application.

## Declarative figure

A recipe may declare one view containing up to four panels and twelve series per panel. Each panel
references a one-dimensional x step and one or more aligned y steps. Linear or logarithmic x axes
and a small fixed set of line styles may be selected. Labels, axes, and panel titles are presentation
data; the underlying step values remain part of the composed scientific artifact.
The desktop renderer builds the Plotly figure locally and does not execute document-supplied
JavaScript.

## Six-level Gaussian flagship

The complete [flagship scientific case](../cases/gaussian-hierarchy-dispersion.md) presents the
question, equations, declared 27-step computation, full results, interpretation, assets,
checksums, limitations, and reproduction procedure. The implementation summary follows.

[`models/gaussian-hierarchy-dispersion.yaml`](../../models/gaussian-hierarchy-dispersion.yaml)
implements the six-level linear-Gaussian example from
[*A scale-free measure of relaxation anisotropy in precision-weighted variational inference*](https://doi.org/10.2139/ssrn.5853487).
One recipe evaluates the tridiagonal
free-energy Hessian for two precision profiles, computes both positive relaxation spectra, derives
the reported summary quantities, samples ensemble relaxation, and renders a two-panel comparison.

| Quantity | Profile A | Profile B |
|---|---:|---:|
| Relaxation rates \(\mu\) | 1.000, 5.129, 9.482, 12.402, 15.325, 24.160 | 1.000, 1.730, 15.015, 15.094, 23.575, 24.160 |
| Log-spectral dispersion \(D\) | 1.031 | 1.282 |
| \(t_{0.90}\) | 0.00479 | 0.00403 |
| \(t_{0.10}\) | 0.299 | 0.461 |
| Recovery breadth \(W\) | 4.134 | 4.741 |
| Effective dimension at \(t=0.4\) | 1.076 | 1.851 |

The automated test tolerances account for the four-decimal precision profiles reported in the
example. The model, recipe, outputs, figure specification, content hashes, strict same-environment
identity path, and tolerant cross-platform reproduction path are exercised by the release suite.

![Ensemble relaxation in the matched six-level Gaussian hierarchies](../assets/gaussian-hierarchy-dispersion.png)

The repository also includes an openable
[`gaussian-hierarchy-dispersion.mlab`](../../examples/gaussian-hierarchy-dispersion.mlab) bundle
containing the completed run and composed artifact.

## Safety and workload bounds

Recipes fail closed under the following limits:

- 128 steps, 64 named outputs, one view, and four panels;
- 4,096 expression bytes and 512 expression-syntax nodes;
- 200,000 values in any stored or intermediate array;
- 1,000,000 values across all stored step results;
- 5,000,000 scalar operations for any formula-level matrix product or determinant; and
- 512 rows or columns for a generalised eigenvalue problem, subject also to the value limit.

Broadcast, outer-product, dot-product, and matrix-multiplication result shapes are checked before
allocation. Every numerical result must be real, finite, and non-empty. Generalised spectral work
checks matrix shape, symmetry, metric shape, and positive definiteness; recipes may additionally
require a strictly positive spectrum.

## Current boundary

The reviewed local-interpreter corpus covers the 34 pre-existing scientific object kinds. It does
not yet author or rewrite analysis recipes. In 1.19, recipes are created by editing validated YAML
or adapting the bundled flagship model. This prevents an unreviewed generation path from being
presented as supported while leaving recipe execution, rendering, bundling, and reproduction fully
available.

The first release intentionally provides a compact numerical vocabulary. It does not yet chain
artifacts from separate prior runs, branch on data, iterate, solve arbitrary optimisation problems,
or load plugins from a document. Those additions require explicit protocol and provenance designs;
they are not silently emulated by formulas.
