# Model Laboratory 1.12.6

Version 1.12.6 corrects an unnecessarily strict pre-empirical QLoRA environment requirement.

## Training-runtime policy

- The supported training runtime is CPython `>=3.10,<3.14`, restoring the compatibility range
  used before 1.12.5.
- CPython 3.12.10 is no longer required merely to create a training environment.
- Direct ML package versions, CUDA requirements, deterministic process settings, and the complete
  environment receipt remain fail-closed.
- Freezing a receipt records the exact Python version and executable hash actually selected.
- Preflight and training compare the current environment with that receipt byte-for-byte at the
  identity layer. Switching, for example, from 3.13.15 to 3.13.16 after freezing is rejected.

This separates compatibility from replay: several supported Python releases may establish a new
training environment, but a frozen empirical run still has one exact environment identity.

## Contract versions

- QLoRA configuration: 1.4
- QLoRA training report: 1.4
- Training-environment policy: 1.2
- Training-environment receipt: unchanged at 1.0

No model was trained and no empirical result was fabricated in this release.
