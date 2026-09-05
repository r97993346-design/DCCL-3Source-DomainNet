# CIPT + Direct Causal Contrastive Learning

This branch isolates a simpler causal-contrastive fusion on top of CIPT.
The CLIP image encoder, causal/spurious decomposition, prompt bank, TDA,
optimizer family, SWAD selection and inference path stay unchanged.

## Pure CIPT

Set `--cipt_pure true`.

The objective is exactly:

`L_CIPT = L_cls + beta * L_de + gamma * L_ind`

Only the original CLIP-preprocessed image is consumed.

## Direct causal contrastive fusion

With `cipt_pure: false`, each training image produces two views:

- `x`: official CLIP Resize + CenterCrop + CLIP normalization;
- `x_2`: RandomResizedCrop + horizontal flip + ColorJitter + random grayscale + CLIP normalization.

The original view follows the normal CIPT path:

`x -> frozen CLIP -> (e, s) -> L_cls + beta*L_de + gamma*L_ind`

The augmented view is used only to create a contrastive positive:

`x_2 -> frozen CLIP -> causal decomposition -> e_aug`

There is no contrastive projection head. The direct causal representations are
normalized and sent to the existing supervised contrastive objective:

`L_con = SupCon(normalize(e), normalize(e_aug), labels)`

The augmented view is deliberately not used for augmented decomposition,
causal-consistency, classification, independence, pre-CL, or representation
regularization. Both inherited projection heads are removed and the optimizer
is rebuilt without their parameters. The unused Gaussian regularizer parameter
is frozen as well.

The fusion objective is therefore:

`L_total = L_CIPT + lambda_eff * L_con`

The new coefficient is `cipt_causal_contrastive_weight`, with default maximum
value `0.1`. To avoid an abrupt contrastive gradient directly on the causal
decomposition, it is linearly warmed up for 500 steps:

`lambda_eff = lambda_max * min(1, step / warmup_steps)`

Recommended first sweep on PACS:

- `lambda_max = 0.05`
- `lambda_max = 0.10` (default)
- `lambda_max = 0.25`
- `lambda_max = 0.50`

Keep `beta`, `gamma`, temperature, prompts, augmentation, seed and SWAD fixed
while doing this sweep. The main comparison should be Pure CIPT versus direct
causal contrastive under identical settings.

## B5c prompt-selection ablations

The class-agnostic B5c bank contains 42 environment/style prompts. This branch
supports three selection modes through `--cipt_selector_mode`:

- `random`: preserve the legacy baseline exactly. Training samples one shared
  random K-subset per step; evaluation uses the deterministic first K prompts.
- `all`: use all 42 B5c prompts. `cipt_k` is ignored for the intervention count.
- `adaptive`: perform label-free, per-sample safe/diverse selection. This is the
  default on this branch.

Adaptive selection first extracts each prompt's actual one-token TDA residual:

`delta_k = W_out(W_value(p_k))`

It shortlists Top-L candidates using the existing causal/spurious decomposition:

`relevance(i,k) = cos(s_i, delta_k) - lambda * abs(cos(e_i, delta_k))`

The shortlist is reranked by Jensen-Shannon consistency between the causal-only
class distribution and each intervened distribution. Greedy MMR then penalizes
redundant prompt effects and returns K unique prompts per image. Selection never
uses ground-truth labels. The final selected interventions remain differentiable,
so the original CIPT classification gradients still update TDA and the causal
branch.

The first `cipt_selector_warmup_steps` training updates retain the legacy
random-K path so the causal/spurious features stabilize. Evaluation always uses
the configured mode; adaptive evaluation does not require the training-step
buffer, which keeps SWAD inference correct.

Recommended initial adaptive settings:

```bash
--cipt_selector_mode adaptive \
--cipt_selector_candidates 8 \
--cipt_selector_causal_penalty 0.5 \
--cipt_selector_js_weight 1.0 \
--cipt_selector_diversity_weight 0.1 \
--cipt_selector_warmup_steps 500
```

For a controlled ablation, change only `cipt_selector_mode` between `random`,
`all`, and `adaptive`; keep every other option and seed fixed.

The synthetic selector tests can be run without a dataset or CLIP checkpoint:

```bash
cd DCCL/DCCL
python -m unittest discover -s tests -p 'test_cipt_prompt_selector.py'
```

## Example PACS run

```bash
cd DCCL/DCCL
CUDA_VISIBLE_DEVICES=0 python train_all.py pacs_direct_causal_cl \
  --dataset PACS --algorithm CIPTDCCL --data_dir /path/to/data \
  --deterministic --trial_seed 0 --seed 0 \
  --cipt_clip_backbone ViT-B/16 --cipt_clip_path /path/to/ViT-B-16.pt \
  --cipt_beta 4 --cipt_gamma 5 --cipt_k 4 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --cipt_template_mode b5c --cipt_selector_mode adaptive \
  --cipt_causal_contrastive_weight 0.1 \
  --cipt_contrastive_warmup_steps 500
```
