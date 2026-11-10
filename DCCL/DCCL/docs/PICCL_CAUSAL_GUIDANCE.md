# PICCL as causal guidance for original DCCL

Base: `feature/piccl-final-experiment-v6`. This branch adds three mutually exclusive, switchable paths without changing the original `DCCL` class.

## Shared causal estimator

* PIRE: `delta_aug = (z_aug-z) - stopgrad(z0_aug-z0)`.
* CDRM: EMA of class/domain adaptation residuals `z-z0`, centered within each class.
* ISSL: orthogonal rank-`r` basis `Q=qr(B).Q`, trained with ISR coverage (augmentation + domain responses).
* CMP: `m=z-beta*(z@Q)@Q.T`; `beta` ramps from zero.

Always use only source-domain data for EMA updates, and do not inspect the held-out target to choose training parameters. Strong augmentation `x_2` comes from the existing DCCL pipeline. For the first experiments keep `TN=false`, `aug=0` (no CutMix), `sample_d=false`, and `re_w=false`.

## Modes

| mode | CE | SupCon | pre_cl_loss | GT/VAE | inference |
|---|---|---|---|---|---|
| DCCL baseline (`--algorithm DCCL`) | z | z | z vs z0 | keep | z |
| `piccl_guidance_mode: mediator` (v6 compatibility) | m | m | m vs z0 | keep | m |
| `piccl_guidance_mode: guide_contrast` (recommended first) | z | m | z vs z0 | keep | z |
| `piccl_guidance_mode: replace_gt` | z | m | z vs z0 | **remove** | z |

Both guidance modes still learn Q through ISR and run SupCon on m. They do *not* add a second SupCon or change the original CE/classifier. Thus DICM guides training but adds no inference-time module in these two modes. The original DCCL GT branch is left untouched for `guide_contrast`, but explicitly skipped in `replace_gt`. The original pre-trained anchor contrast always sees z in guidance modes.

The optional v6 corrected cross-domain pair-reliability weight can be enabled with `piccl_use_reliable_contrast=true` in a *separate* ablation. It is applied only for the normal (non-re_w, non-sample_d) SupCon branch. It computes `1 - ||Proj_Q(z_i-z_j)||^2 / ||z_i-z_j||^2` for same-class, cross-domain positives. It is detached; ordinary self-augmented and same-domain positive pairs keep weight one. Do not use old v5 reliability logs as evidence for this corrected weighting.

## Train commands (run inside DCCL/DCCL)

Recommended first comparison, with all four PACS targets and identical seed/split/settings:

```bash
python -u train_all.py dccl_baseline \
  --dataset PACS --algorithm DCCL \
  --data_dir /path/to/data --model resnet50 \
  --steps 5000 --checkpoint_freq 100 --seed 0 --trial_seed 0 \
  --deterministic --l 1 --l_d 0.05 --l_layer 1 --aug 0

python -u train_all.py piccl_guide_contrast \
  configs/piccl_dccl_causal_guidance.yaml \
  --dataset PACS --algorithm PICCL \
  --data_dir /path/to/data --model resnet50 \
  --steps 5000 --checkpoint_freq 100 --seed 0 --trial_seed 0 \
  --deterministic --l 1 --l_d 0.05 --l_layer 1 --aug 0

python -u train_all.py piccl_replace_gt \
  configs/piccl_dccl_causal_guidance.yaml \
  --dataset PACS --algorithm PICCL \
  --data_dir /path/to/data --model resnet50 \
  --steps 5000 --checkpoint_freq 100 --seed 0 --trial_seed 0 \
  --deterministic --l 1 --l_d 0.05 --l_layer 1 --aug 0 \
  --piccl_guidance_mode replace_gt

python -u train_all.py piccl_mediator_v6 \
  configs/piccl_dccl_causal_guidance.yaml \
  --dataset PACS --algorithm PICCL \
  --data_dir /path/to/data --model resnet50 \
  --steps 5000 --checkpoint_freq 100 --seed 0 --trial_seed 0 \
  --deterministic --l 1 --l_d 0.05 --l_layer 1 --aug 0 \
  --piccl_guidance_mode mediator
```

`train_all.py` consumes known PICCL keys from YAML or sconf CLI overrides; the new key is registered in `hparams_registry.py`.

## Required diagnostics

Verify at every checkpoint:
* `piccl_beta` grows past zero after warmup;
* `valid_domain_response_count > 0` and `loss_isr_aug`, `loss_isr_dom` are finite;
* `projection_ratio` and `feature_delta_ratio` confirm nontrivial interventions;
* `guide_contrast_mode`/`guide_replace_gt_mode` indicate the requested path;
* GT `reg_loss == 0` for `replace_gt`;
* no target environment is used to update EMA or choose hyperparameters.

For fair model selection, use source-domain validation (IID selection) and source-selected SWAD, not target-domain oracle tuning. Evaluate `test_out` only after configuration/selection is fixed. Run at least seeds 0,1,2 with matched splits and report mean/std and per-domain values. Do not compare only the final step or cherry-pick the maximum target-domain checkpoint.

## Semantics and caution

This implementation is a *training-only causal guidance* path, not a claim that Q perfectly identifies causality. The supervised contrastive gradients backpropagate through `m` to the trainable DCCL featurizer, while Q is updated through detached PIRE/CDRM targets via ISR. CE and pre-trained anchors preserve the original semantic representation. Because these modes infer directly from `z`, their SWAD inference does not depend on averaging the Q basis.

The new branch has not been trained or empirically validated yet. Existing v1/v5 PACS results are historical reference data, not results for this code.
