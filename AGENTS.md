# Repository Guidance for Codex

## Project context
This repository reproduces DCCL for few-source domain generalization.
The current innovation branch develops CSR-DCCL incrementally.

## Current implementation stage
Only implement Stage 1:
- causal semantic reliability estimation
- reliability-weighted DCCL positive-pair loss

Do not implement yet:
- XDomainMix or feature mixing
- GGA or gradient-guided scheduling
- online LLM/VLM inference
- large architectural refactors

## Engineering constraints
- Preserve the original DCCL execution path when new features are disabled.
- New functionality must be guarded by configuration flags.
- Do not commit datasets, model weights, checkpoints, logs, outputs, API keys, or secrets.
- Prefer minimal, localized changes over refactoring.
- Add shape comments for new tensor operations.
- Add tests or a minimal smoke test for all new functionality.

## Acceptance criteria
- Original DCCL can still run with the new module disabled.
- Reliable DCCL loss runs with fake causal/spurious embeddings.
- No NaN loss or non-finite gradients in the smoke test.
- Provide commands for reproducing the tests and launching both baseline and new-mode training.