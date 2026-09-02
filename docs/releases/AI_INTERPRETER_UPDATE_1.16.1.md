# Model Laboratory 1.16.1 — AI interpreter update v11

This update keeps the product version at 1.16.1 and replaces the current interpreter prompt,
development benchmark, training corpus, and QLoRA configuration. Earlier versioned artifacts remain
in the tree as provenance and are not current defaults.

## Current identities

| Artifact | Current identity |
|---|---|
| System prompt | `model_lab/baseline/system_prompt_v1.3.txt` |
| Few-shot registry | `model_lab/baseline/few_shot_examples_v1.2.json` |
| Development benchmark | `verification/interpreter_baseline_v1.6.json` — 150 cases — SHA-256 `e84cf6c7bab332253dc33a4e2fc5c30d97667e8bc41bbfcd1d97e117c44ecde0` |
| Training corpus | `training/interpreter_corpus_v1.5/` — 6,300 records; 5,039 train / 1,261 validation — content SHA-256 `bfee076b5501a75311f0d85247a3109a7e8d66910a11ba3dfb59ee72931df28b` |
| Train JSONL | SHA-256 `5f396dec5eceb14f4c7aa9cf1cc9c58c92e8ac8a3bef7a2508191e44e0f01576` |
| Validation JSONL | SHA-256 `8e6b05ce1b669fd39897889b93e7794ce04786210d5cbeb0a6b6a4078fdc43eb` |
| Human-review queue | 725 pending rows — SHA-256 `5a6bfd5a6131a91859ede689cc2e3addc14b84a7b4bb2645cefd4d6469bf69af` |
| QLoRA configuration | `training/interpreter_qlora_v1.7.json` — SHA-256 `47eddd1b58b8951d596575e9e3ea88c2143020d5832dc2bdfc822def882b208f` |

## Corrections

- Scientific requests now have a deterministic, compiler-enforced capability boundary. SDEs,
  specialised or trainable neural networks, neural controllers, structural optimisation,
  solver-coupled parameter estimation, and mesh/FEM PDE solving return `unable` until an exact
  installed capability exists.
- Routing multiple schemas into one context is no longer presented as an executable composite
  solver. The same boundary is enforced before native inference and again during output compilation.
- The 150-case held-out benchmark adds ten negative cases for these near-neighbour failures.
- The regenerated 6,300-record corpus adds split-isolated negative archetypes while preserving the
  existing size, family totals, compiler grounding, and mandatory 725-row human review gate.
- Saved immutable commit receipts can be reopened from the desktop. Receipt, state, source, and
  model identities are validated first; live controls are restored without numerical execution.
- CI now includes an Ubuntu CPython 3.13 regression lane.

## Verification

- Python regression suite: 444/444 passed.
- Frontend suite: 8/8 passed; both JavaScript modules pass syntax checking.
- Reference, bundle, reproduction, expression-AST, Model Graph/Run protocol, and official-pack
  suites: 49/49, 37/37, 51/51, 33/33, 33/33, and 54/54 passed.
- Deep corpus replay: 6,300/6,300 passed; all 725 review rows remain validly bound.
- The 150-case development benchmark validates, but no live Qwen score is claimed.
- Normal QLoRA preflight is `BLOCKED` as intended: human review, the live untouched-base run,
  immutable environment receipt, pinned ML dependencies, and CUDA hardware are still absent.
- Rust/Tauri compilation and platform installers require a target Rust toolchain and remain
  runtime-dependent; Python/static checks are not substituted for them.
