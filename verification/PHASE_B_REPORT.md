# Model Laboratory Phase-B reproduction verification

**Result: 51/51 checks passed.**

The checks exercise exact fingerprint reproduction, explicit numerical tolerance comparison, result-specific semantic comparison, status classification, environment separation, explicit execution refusal, `.mlab` reproduction and JSON/text reproduction reports.

- **PASS** — quadratic.yaml: exact status
- **PASS** — quadratic.yaml: every result strict
- **PASS** — quadratic.yaml: environment captured
- **PASS** — quadratic.yaml: .mlab explicit reproduction
- **PASS** — quadratic.yaml: JSON report schema
- **PASS** — quadratic.yaml: text report
- **PASS** — surface.yaml: exact status
- **PASS** — surface.yaml: every result strict
- **PASS** — surface.yaml: environment captured
- **PASS** — surface.yaml: .mlab explicit reproduction
- **PASS** — surface.yaml: JSON report schema
- **PASS** — surface.yaml: text report
- **PASS** — structured.yaml: exact status
- **PASS** — structured.yaml: every result strict
- **PASS** — structured.yaml: environment captured
- **PASS** — structured.yaml: .mlab explicit reproduction
- **PASS** — structured.yaml: JSON report schema
- **PASS** — structured.yaml: text report
- **PASS** — issues.yaml: exact status
- **PASS** — issues.yaml: every result strict
- **PASS** — issues.yaml: environment captured
- **PASS** — issues.yaml: .mlab explicit reproduction
- **PASS** — issues.yaml: JSON report schema
- **PASS** — issues.yaml: text report
- **PASS** — exact result survives different environment
- **PASS** — different environment is explicit
- **PASS** — laboratory version difference is explicit
- **PASS** — within-tolerance result is numerical reproduction
- **PASS** — numerical result differs strictly
- **PASS** — numerical comparator accepts saved tolerance
- **PASS** — numerical deviation is reported
- **PASS** — outside-tolerance result is rejected
- **PASS** — mixed reproduced/non-reproduced result is partial
- **PASS** — outside-tolerance deviation is reported
- **PASS** — strict-only symbolic mismatch is rejected
- **PASS** — strict-only mismatch prevents numerical overall status
- **PASS** — reordered sweep loses strict fingerprint
- **PASS** — reordered sweep reproduces numerically
- **PASS** — per-result inability has UNABLE status
- **PASS** — mixed inability produces partial overall status
- **PASS** — execution failure detail preserved
- **PASS** — over-budget reproduction is refused
- **PASS** — over-budget reproduction performs no evaluation
- **PASS** — over-budget report explains refusal
- **PASS** — no successful result with execution failures is unable
- **PASS** — diagnostic message wording is presentation-only
- **PASS** — strict diagnostic fingerprint excludes message wording
- **PASS** — diagnostic detail order is scientifically irrelevant
- **PASS** — saved diagnostic message remains preserved
- **PASS** — different scientific diagnostic details are rejected
- **PASS** — Hessian eigenvalue order is semantically irrelevant
