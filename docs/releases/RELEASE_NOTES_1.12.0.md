# Model Laboratory 1.12.0 release notes

Version 1.12.0 completes the implementation needed to perform, evaluate, export, approve, select,
and reverse a local Qwen interpreter fine-tune. It deliberately does not claim that the pending
human or model-running work has occurred.

## Interpreter changes

### Adaptive production memory

Production requests select the smallest safe 32K/4K, 64K/8K, or 128K/8K Ollama profile from
the exact bounded instruction and context size. The 64 KiB instruction and 512 KiB source limits
are unchanged; the complete source stays local and Qwen receives at most 32 KiB of structured
context. Ordinary requests therefore avoid reserving the maximum KV cache.

### Real clarification dialogue

`needs_clarification` now opens a desktop question/answer step. Up to eight ordered turns are
fed back into the deterministic context with a 16 KiB total history ceiling. Source or
instruction changes invalidate the conversation. Proposal/acceptance provenance records the
history checksum and turn count without persisting answer text there.

### Stage-6 corpus v1.1

The regenerated 5,000-record corpus includes paired clarification conversations: question in
turn zero, explicit user answer and compiled proposal in turn one. Context-negotiation pairs
remain present. All records are compiler-grounded and the Stage-5 benchmark remains excluded.
The 400-item review queue remains honestly pending.

### Verified export and candidate evaluation

The new export pipeline verifies the training report and every adapter byte, verifies the exact
base snapshot, merges PEFT weights, invokes explicitly identified GGUF conversion and Q4_K_M
quantisation tools, imports a distinct Ollama tag, and writes a checksum-bound receipt. A
plan-only mode performs all inexpensive gates without loading weights.

The candidate evaluator reuses the complete Stage-5 benchmark, prompt, compiler, scorer, and
32K/4K deterministic evaluation profile. It supports atomic checkpoint/resume and retains raw
evidence. Report validation replays the raw responses rather than trusting a self-hashed metric
summary. Eligibility requires no regression in any aggregate scientific/safety metric or any
family semantic rate.

### Explicit promotion and rollback

The checksum-bound model registry records immutable approved candidates and event history.
Promotion requires explicit confirmation of the reviewed candidate-report SHA-256 and current
registry SHA-256, revalidates every gate and the installed candidate digest, and updates the
registry atomically. Rollback can select any prior approved entry or the frozen base. The native
desktop bridge fails closed on an invalid registry and verifies an approved candidate's Ollama
digest and size before use.

## Compatibility and packaging

Interpreter context/compiler/proposal/acceptance contracts advance to 1.2/1.3/2.2/1.3, and
the baseline contract/report advance to 1.3 to identify adaptive production profiles, while
historical accepted provenance remains loadable. The sidecar now explicitly bundles the complete
versioned prompt/schema asset directory as PyInstaller data. Existing Model IR, expression AST,
experiment-state, `.mlab`, and desktop sidecar protocol versions are unchanged.

## What remains empirical

The supplied archive contains no trained adapter and no fabricated performance result. Before
promotion, a workstation must complete all 400 genuine review decisions, run the untouched
80-case baseline, download/verify the frozen base snapshot, train with the optional CUDA stack,
export the candidate, run its complete evaluation, and review both exact reports.
