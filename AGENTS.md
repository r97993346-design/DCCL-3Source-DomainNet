# ICR-DCCL Development Guidance

## Project objective
This branch implements an intervention-calibrated reliable connection mechanism for DCCL in few-source domain generalization.

## Stage 1 only
Implement only:
- Fourier amplitude intervention for domain-style perturbation
- intervention stability score
- pair-wise reliability matrix
- reliable weighting of the original DCCL cross-domain positive contrastive branch

## Do not implement in Stage 1
- LLM/VLM prompts
- CLIP embeddings
- causal/spurious JSON files
- CIRL factorization loss
- CIRL adversarial masker or Gumbel-Softmax
- XDomainMix
- GGA or gradient scheduling
- large refactors

## Compatibility requirements
- Preserve the original DCCL execution path when the new flag is disabled.
- Keep all original DCCL loss terms except replacing the relevant sup_cl branch when the new module is enabled.
- Do not double-count original and weighted contrastive losses.
- Handle image normalization correctly before Fourier intervention.
- Do not commit datasets, checkpoints, embeddings, logs, or model weights.

## Acceptance criteria
- Baseline DCCL runs unchanged with the new module disabled.
- Fourier intervention is unit-tested.
- Reliable weights only affect same-class cross-domain positive pairs.
- Minimal smoke test runs without NaN or non-finite gradients.
